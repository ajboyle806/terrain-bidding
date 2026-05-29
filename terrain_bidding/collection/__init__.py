"""Phase 2: Rollout data collection with frozen policy.

Collects terrain observations + realized costs for ensemble training.
"""
import os
import torch
import numpy as np
import h5py
from pathlib import Path
from terrain_bidding.configs import CollectionConfig, CostWeights, PolicyConfig
from terrain_bidding.policy import ActorCritic
from terrain_bidding.envs import TERRAIN_NAMES, DISCRETE_OBSTACLES

try:
    from terrain_bidding.envs import TerrainBiddingEnv
    HAS_ISAAC = True
except Exception:
    HAS_ISAAC = False


def collect(cfg: CollectionConfig = CollectionConfig(),
            policy_path: str = "checkpoints/policy_best.pt",
            save_dir: str = "data",
            device: str = "cuda"):
    """Collect rollout dataset with frozen policy."""
    if not HAS_ISAAC:
        raise RuntimeError("Isaac Gym required. Run on GPU machine.")

    os.makedirs(save_dir, exist_ok=True)
    policy_cfg = PolicyConfig()
    policy = ActorCritic(policy_cfg).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    # Storage
    data = {
        "heightmap": [],       # (16, 16)
        "slope": [],           # scalar
        "roughness": [],       # scalar
        "friction": [],        # scalar
        "distance": [],        # scalar
        "elevation_change": [], # scalar
        "cost": [],            # scalar
        "terrain_type": [],    # int
    }

    # Collect in-distribution (5 terrain types, excluding discrete obstacles)
    in_dist_types = [i for i in range(6) if i != DISCRETE_OBSTACLES]
    per_terrain = cfg.num_rollouts // len(in_dist_types)

    for terrain_idx in in_dist_types:
        collected = 0
        failures = 0
        while collected < per_terrain:
            result = _run_rollout(policy, terrain_idx, cfg, device)
            if result is None:
                failures += 1
                continue
            for k, v in result.items():
                data[k].append(v)
            collected += 1
            if (collected + failures) % 1000 == 0:
                print(f"  {TERRAIN_NAMES[terrain_idx]}: {collected}/{per_terrain} "
                      f"(failures: {failures})")
        print(f"  {TERRAIN_NAMES[terrain_idx]}: done. Failure rate: "
              f"{failures / (collected + failures):.1%}")

    # Collect held-out terrain (discrete obstacles)
    held_out = {k: [] for k in data}
    collected, failures = 0, 0
    while collected < cfg.held_out_rollouts:
        result = _run_rollout(policy, DISCRETE_OBSTACLES, cfg, device)
        if result is None:
            failures += 1
            continue
        for k, v in result.items():
            held_out[k].append(v)
        collected += 1
    print(f"  held_out: done. Failure rate: {failures / (collected + failures):.1%}")

    # Save splits
    _save_splits(data, cfg, save_dir)
    _save_hdf5(held_out, f"{save_dir}/held_out.hdf5")
    print(f"Dataset saved to {save_dir}/")


def _run_rollout(policy, terrain_idx, cfg, device):
    """Execute one rollout. Returns feature dict or None on failure."""
    # Sample terrain patch and goal
    heightmap, slope, roughness, friction = _sample_terrain_patch(terrain_idx)
    distance = np.random.uniform(*cfg.distance_range)
    elevation = np.random.uniform(*cfg.elevation_range)

    # Run episode with frozen policy, record torques and velocities
    trajectory = _execute_trajectory(policy, terrain_idx, distance, elevation, cfg, device)
    if trajectory is None:  # fall or timeout
        return None

    # Compute cost from ground-truth telemetry
    cost = _compute_cost(trajectory, cfg.cost_weights)

    return {
        "heightmap": heightmap,
        "slope": slope,
        "roughness": roughness,
        "friction": friction,
        "distance": distance,
        "elevation_change": elevation,
        "cost": cost,
        "terrain_type": terrain_idx,
    }


def _sample_terrain_patch(terrain_idx):
    """Sample a 16x16 heightmap patch from terrain. Returns (heightmap, slope, roughness, friction)."""
    # Actual implementation queries Isaac Gym terrain mesh
    raise NotImplementedError("Requires Isaac Gym runtime")


def _execute_trajectory(policy, terrain_idx, distance, elevation, cfg, device):
    """Run frozen policy to goal. Returns trajectory dict or None on failure."""
    # Returns {"torques": (T, 12), "joint_vels": (T, 12), "dt": float, "duration": float}
    raise NotImplementedError("Requires Isaac Gym runtime")


def _compute_cost(trajectory, weights: CostWeights) -> float:
    """Compute c = α·∫τ·q̇ dt + β·T from trajectory telemetry."""
    torques = trajectory["torques"]      # (T, 12)
    joint_vels = trajectory["joint_vels"]  # (T, 12)
    dt = trajectory["dt"]

    # Energy: integral of |torque * joint_velocity| over time
    power = np.abs(torques * joint_vels).sum(axis=1)  # (T,)
    energy = power.sum() * dt

    duration = trajectory["duration"]
    return weights.alpha * energy + weights.beta * duration


def _save_splits(data, cfg, save_dir):
    """Split data into train/val/test and save as HDF5."""
    n = len(data["cost"])
    indices = np.random.permutation(n)
    n_train = int(n * cfg.train_split)
    n_val = int(n * cfg.val_split)

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }
    for name, idx in splits.items():
        split_data = {k: [v[i] for i in idx] for k, v in data.items()}
        _save_hdf5(split_data, f"{save_dir}/{name}.hdf5")


def _save_hdf5(data: dict, path: str):
    """Save data dict to HDF5 file."""
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=np.array(v))


if __name__ == "__main__":
    collect()
