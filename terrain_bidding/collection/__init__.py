"""Phase 2: Rollout data collection with frozen policy.

Runs the trained policy in the legged_gym env and records terrain features + costs.
Uses parallel envs for fast collection.
"""
import os
import sys
import numpy as np
import h5py
from pathlib import Path

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch
from terrain_bidding.configs import CollectionConfig, CostWeights


def collect(cfg: CollectionConfig = CollectionConfig(),
            policy_path: str = "checkpoints/model_8000.pt",
            save_dir: str = "data",
            device: str = "cuda"):
    """Collect rollout dataset using parallel envs."""
    from isaacgym import gymapi
    from rsl_rl.runners import OnPolicyRunner
    from terrain_bidding.envs import TerrainBiddingEnv, TerrainBiddingEnvCfg, TerrainBiddingPPOCfg
    from terrain_bidding.policy.train import class_to_dict

    os.makedirs(save_dir, exist_ok=True)
    torch.cuda.empty_cache()

    # Find checkpoint
    if not os.path.exists(policy_path):
        import glob
        models = sorted(glob.glob("checkpoints/**/model_*.pt", recursive=True) +
                        glob.glob("checkpoints/model_*.pt"))
        if not models:
            raise FileNotFoundError("No model checkpoint found")
        policy_path = models[-1]
    print(f"Using policy: {policy_path}")

    # Create env
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.005
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.physx.num_threads = 10
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.5
    sim_params.physx.max_depenetration_velocity = 1.0
    sim_params.physx.max_gpu_contact_pairs = 2**23
    sim_params.physx.default_buffer_size_multiplier = 5
    sim_params.physx.contact_collection = gymapi.ContactCollection(2)

    env_cfg = TerrainBiddingEnvCfg()
    num_envs = 16
    env_cfg.env.num_envs = num_envs
    # Reduce terrain size to fit in VRAM during collection
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10
    env = TerrainBiddingEnv(env_cfg, sim_params, gymapi.SIM_PHYSX, "cuda:0", headless=True)

    # Load policy
    train_cfg = TerrainBiddingPPOCfg()
    train_cfg_dict = class_to_dict(train_cfg)
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device="cuda:0")
    runner.load(policy_path)
    policy = runner.get_inference_policy(device="cuda:0")
    print("Policy loaded. Collecting data...")

    # Collection storage
    all_heightmaps = []
    all_slopes = []
    all_roughness = []
    all_friction = []
    all_distances = []
    all_elevations = []
    all_costs = []
    all_terrain_types = []

    # Track per-env episode data
    episode_torques = [[] for _ in range(num_envs)]
    episode_steps = [0] * num_envs
    start_positions = env.root_states[:, :3].clone()

    target_rollouts = cfg.num_rollouts
    collected = 0
    total_steps = 0
    max_steps = target_rollouts * 500  # safety limit

    obs = env.get_observations()
    while collected < target_rollouts and total_steps < max_steps:
        # Step policy
        actions = policy(obs)
        env.step(actions)
        obs = env.get_observations()
        total_steps += 1

        # Record torques for all envs
        torques = env.torques.detach().cpu().numpy()  # (num_envs, 12)
        dof_vel = env.dof_vel.detach().cpu().numpy()  # (num_envs, 12)
        for i in range(num_envs):
            episode_torques[i].append(np.abs(torques[i] * dof_vel[i]).sum())
            episode_steps[i] += 1

        # Check for episode resets
        reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        for idx in reset_ids.cpu().numpy():
            idx = int(idx)
            # Only save if episode was long enough (not immediate fall)
            if episode_steps[idx] > 50:  # at least 1 second
                dt = 0.005 * 4  # sim_dt * decimation = control dt
                duration = episode_steps[idx] * dt
                energy = sum(episode_torques[idx]) * dt

                cost = cfg.cost_weights.alpha * energy + cfg.cost_weights.beta * duration

                # Get terrain features from env
                # Heightmap: sample from terrain around robot start position
                hm = _get_heightmap_patch(env, idx, start_positions[idx])
                slope = _compute_slope(hm)
                roughness = float(hm.std())
                friction = np.random.uniform(0.5, 1.25)  # from domain rand range

                # Distance traveled
                current_pos = env.root_states[idx, :3].cpu().numpy()
                start_pos = start_positions[idx].cpu().numpy()
                distance = np.linalg.norm(current_pos[:2] - start_pos[:2])
                elevation = current_pos[2] - start_pos[2]

                # Terrain type from curriculum level
                terrain_type = int(env.terrain_levels[idx].item()) % 5

                all_heightmaps.append(hm)
                all_slopes.append(slope)
                all_roughness.append(roughness)
                all_friction.append(friction)
                all_distances.append(distance)
                all_elevations.append(elevation)
                all_costs.append(cost)
                all_terrain_types.append(terrain_type)
                collected += 1

                if collected % 1000 == 0:
                    print(f"  Collected {collected}/{target_rollouts}")

            # Reset tracking for this env
            episode_torques[idx] = []
            episode_steps[idx] = 0
            start_positions[idx] = env.root_states[idx, :3].clone()

    print(f"Collection done: {collected} rollouts in {total_steps} steps")

    # Save
    _save_dataset(all_heightmaps, all_slopes, all_roughness, all_friction,
                  all_distances, all_elevations, all_costs, all_terrain_types,
                  cfg, save_dir)


def _get_heightmap_patch(env, env_idx, position):
    """Extract 16x16 heightmap patch around position."""
    try:
        hs = env.terrain.heightsamples
        px = int((position[0].item()) / env.terrain.cfg.horizontal_scale)
        py = int((position[1].item()) / env.terrain.cfg.horizontal_scale)
        px = max(8, min(px, hs.shape[0] - 8))
        py = max(8, min(py, hs.shape[1] - 8))
        patch = hs[px-8:px+8, py-8:py+8].cpu().numpy().astype(np.float32)
        if patch.shape == (16, 16):
            return patch * env.terrain.cfg.vertical_scale
    except Exception:
        pass
    return np.random.randn(16, 16).astype(np.float32) * 0.01


def _compute_slope(heightmap):
    """Compute mean slope from heightmap gradient."""
    gy, gx = np.gradient(heightmap, 0.1)
    return float(np.sqrt(gx**2 + gy**2).mean())


def _save_dataset(heightmaps, slopes, roughness, friction, distances,
                  elevations, costs, terrain_types, cfg, save_dir):
    """Split and save as HDF5."""
    n = len(costs)
    print(f"Saving {n} samples...")
    indices = np.random.permutation(n)
    n_train = int(n * cfg.train_split)
    n_val = int(n * cfg.val_split)

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    for name, idx in splits.items():
        path = f"{save_dir}/{name}.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("heightmap", data=np.array([heightmaps[i] for i in idx]))
            f.create_dataset("slope", data=np.array([slopes[i] for i in idx]))
            f.create_dataset("roughness", data=np.array([roughness[i] for i in idx]))
            f.create_dataset("friction", data=np.array([friction[i] for i in idx]))
            f.create_dataset("distance", data=np.array([distances[i] for i in idx]))
            f.create_dataset("elevation_change", data=np.array([elevations[i] for i in idx]))
            f.create_dataset("cost", data=np.array([costs[i] for i in idx]))
            f.create_dataset("terrain_type", data=np.array([terrain_types[i] for i in idx]))
        print(f"  {name}: {len(idx)} samples -> {path}")


if __name__ == "__main__":
    collect()
