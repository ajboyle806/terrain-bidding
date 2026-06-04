"""Process GrandTour ANYmal dataset into our mechanism pipeline format.

Downloads missions, extracts traversal costs + terrain features, saves as HDF5.

Usage:
  python -m terrain_bidding.grandtour.process --missions 3
"""
import os
import tarfile
import numpy as np
import h5py
import zarr
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_tree


REPO_ID = "leggedrobotics/grand_tour_dataset"
SEGMENT_DURATION = 5.0  # seconds per cost segment


def list_missions(max_n=None):
    """List available mission folders."""
    items = list(list_repo_tree(REPO_ID, repo_type="dataset"))
    missions = [f.path for f in items if hasattr(f, 'path') and f.path.startswith("202")]
    if max_n:
        missions = missions[:max_n]
    return missions


def download_mission(mission_id: str, local_dir: str = "grandtour_data"):
    """Download actuator + odometry data for one mission."""
    patterns = [
        f"{mission_id}/data/anymal_state_actuator*",
        f"{mission_id}/data/anymal_state_odometry*",
    ]
    for pat in patterns:
        snapshot_download(REPO_ID, repo_type="dataset",
                         allow_patterns=pat, local_dir=local_dir)


def extract_tar(tar_path: str, extract_dir: str):
    """Extract a tar file to directory."""
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(extract_dir)


def compute_costs_from_mission(mission_path: str, segment_duration: float = SEGMENT_DURATION):
    """Extract per-segment traversal costs from actuator data.

    Returns dict with costs, timestamps, and optional odometry.
    """
    # Extract actuator tar
    actuator_tar = os.path.join(mission_path, "data", "anymal_state_actuator.tar")
    extract_dir = os.path.join(mission_path, "extracted")
    if not os.path.exists(os.path.join(extract_dir, "anymal_state_actuator")):
        extract_tar(actuator_tar, extract_dir)

    store = zarr.open(os.path.join(extract_dir, "anymal_state_actuator"), mode="r")
    timestamps = store["timestamp"][:]
    dt = np.median(np.diff(timestamps))
    n_samples = len(timestamps)

    # Compute instantaneous power (all 12 joints)
    total_power = np.zeros(n_samples)
    for j in range(12):
        torque = store[f"{j:02d}_state_joint_torque"][:]
        velocity = store[f"{j:02d}_state_joint_velocity"][:]
        total_power += np.abs(torque * velocity)

    # Segment into windows
    window = int(segment_duration / dt)
    n_segments = n_samples // window

    costs = np.zeros(n_segments)
    mean_powers = np.zeros(n_segments)
    for i in range(n_segments):
        seg_power = total_power[i * window:(i + 1) * window]
        energy = seg_power.sum() * dt
        costs[i] = 1.0 * energy + 0.1 * segment_duration  # alpha*energy + beta*time
        mean_powers[i] = seg_power.mean()

    # Try to get odometry for distance/elevation
    distances = np.zeros(n_segments)
    elevations = np.zeros(n_segments)
    odom_tar = os.path.join(mission_path, "data", "anymal_state_odometry.tar")
    if os.path.exists(odom_tar):
        if not os.path.exists(os.path.join(extract_dir, "anymal_state_odometry")):
            extract_tar(odom_tar, extract_dir)
        try:
            odom = zarr.open(os.path.join(extract_dir, "anymal_state_odometry"), mode="r")
            # Check for position data
            pos_keys = [k for k in odom.keys() if "position" in k or "pose" in k]
            if "pose_position" in odom:
                pos = odom["pose_position"][:]  # (N, 3)
                odom_ts = odom["timestamp"][:]
                # Resample to match actuator segments
                for i in range(n_segments):
                    t_start = timestamps[i * window]
                    t_end = timestamps[min((i + 1) * window, n_samples - 1)]
                    mask = (odom_ts >= t_start) & (odom_ts <= t_end)
                    if mask.sum() > 1:
                        seg_pos = pos[mask]
                        distances[i] = np.sqrt(np.sum(np.diff(seg_pos[:, :2], axis=0)**2, axis=1)).sum()
                        elevations[i] = seg_pos[-1, 2] - seg_pos[0, 2]
        except Exception as e:
            print(f"  Warning: odometry extraction failed: {e}")

    return {
        "costs": costs,
        "mean_powers": mean_powers,
        "distances": distances,
        "elevations": elevations,
        "n_segments": n_segments,
        "duration": n_samples * dt,
    }


def process_missions(n_missions: int = 5, local_dir: str = "grandtour_data",
                     output_dir: str = "data/grandtour"):
    """Download and process multiple missions into HDF5."""
    os.makedirs(output_dir, exist_ok=True)
    missions = list_missions(max_n=n_missions * 2)  # get extra in case some fail

    all_costs = []
    all_distances = []
    all_elevations = []
    all_powers = []
    all_mission_ids = []

    processed = 0
    for mission_id in missions:
        if processed >= n_missions:
            break
        print(f"\nProcessing mission: {mission_id}")
        try:
            download_mission(mission_id, local_dir)
            mission_path = os.path.join(local_dir, mission_id)
            result = compute_costs_from_mission(mission_path)

            all_costs.extend(result["costs"])
            all_distances.extend(result["distances"])
            all_elevations.extend(result["elevations"])
            all_powers.extend(result["mean_powers"])
            all_mission_ids.extend([processed] * result["n_segments"])

            print(f"  {result['n_segments']} segments, "
                  f"mean cost={result['costs'].mean():.1f}, "
                  f"CV={result['costs'].std() / max(result['costs'].mean(), 1):.3f}")
            processed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    # Save as HDF5
    costs = np.array(all_costs)
    distances = np.array(all_distances)
    elevations = np.array(all_elevations)
    powers = np.array(all_powers)
    mission_ids = np.array(all_mission_ids)

    # Create dummy heightmap (16x16 zeros — we don't have terrain heightmaps per segment)
    # Use power profile as a proxy feature instead
    n = len(costs)
    heightmaps = np.zeros((n, 16, 16), dtype=np.float32)
    slopes = powers / max(powers.max(), 1)  # normalized power as terrain difficulty proxy
    roughness = np.abs(np.diff(np.concatenate([[0], costs]))) / max(costs.std(), 1)
    friction = np.ones(n) * 0.8  # unknown, use constant

    # Split by mission: first missions = train, last = test
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    splits = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, None),
    }

    for name, slc in splits.items():
        path = f"{output_dir}/{name}.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("heightmap", data=heightmaps[slc])
            f.create_dataset("slope", data=slopes[slc])
            f.create_dataset("roughness", data=roughness[slc])
            f.create_dataset("friction", data=friction[slc])
            f.create_dataset("distance", data=distances[slc])
            f.create_dataset("elevation_change", data=elevations[slc])
            f.create_dataset("cost", data=costs[slc])
            f.create_dataset("terrain_type", data=mission_ids[slc])
        print(f"  {name}: {len(costs[slc])} samples -> {path}")

    print(f"\nTotal: {n} segments from {processed} missions")
    print(f"Cost stats: mean={costs.mean():.1f} std={costs.std():.1f} CV={costs.std()/costs.mean():.3f}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    process_missions(n_missions=n)
