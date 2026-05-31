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
            device: str = "cuda",
            terrain_mode: str = "in_distribution"):
    """Collect rollout dataset using parallel envs.

    Args:
        terrain_mode: "in_distribution" (types 0-3, for ensemble training)
                      "held_out" (type 4 = discrete obstacles, for testing)
                      "all" (all types, original behavior)
    """
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
    num_envs = 2048
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.num_rows = 20
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.max_init_terrain_level = 19
    env_cfg.terrain.curriculum = False

    # Set terrain proportions based on mode
    if terrain_mode == "in_distribution":
        env_cfg.terrain.terrain_proportions = [0.25, 0.25, 0.25, 0.25, 0.0]
        print(f"Mode: IN-DISTRIBUTION (types 0-3)")
    elif terrain_mode == "held_out":
        env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0]
        print(f"Mode: HELD-OUT (type 4 = discrete obstacles)")
    else:
        env_cfg.terrain.terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        print(f"Mode: ALL terrain types")

    print(f"Creating env: {num_envs} envs, 20x20 terrain...")
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
    all_terrain_levels = []

    # Track per-env episode data (running sums, not lists)
    episode_energy = torch.zeros(num_envs, device="cuda:0")
    episode_steps = np.zeros(num_envs, dtype=int)
    start_positions = env.root_states[:, :3].clone()

    # Goal-directed: set a goal 15m ahead for each robot
    goal_distance = 15.0  # meters
    goal_tolerance = 0.5
    goal_directions = torch.randn(num_envs, 2, device="cuda:0")
    goal_directions = goal_directions / goal_directions.norm(dim=1, keepdim=True)
    goal_positions = start_positions[:, :2] + goal_directions * goal_distance

    target_rollouts = cfg.held_out_rollouts if terrain_mode == "held_out" else cfg.num_rollouts
    collected = 0
    total_steps = 0
    max_steps = target_rollouts * 500  # safety limit
    max_episode_steps = int(15.0 / (0.005 * 4))  # 15s timeout

    obs = env.get_observations()
    print(f"Target: {target_rollouts} rollouts. Starting collection loop...")
    while collected < target_rollouts and total_steps < max_steps:
        # Override commands to point toward goal
        with torch.no_grad():
            pos_2d = env.root_states[:, :2]
            to_goal = goal_positions - pos_2d
            dist_to_goal = to_goal.norm(dim=1, keepdim=True).clamp(min=0.1)
            direction = to_goal / dist_to_goal
            # Set forward velocity command (1.5 m/s toward goal)
            env.commands[:, 0] = direction[:, 0] * 1.5  # vx
            env.commands[:, 1] = direction[:, 1] * 0.5  # vy
            env.commands[:, 2] = 0.0  # no yaw rate

        # Step policy (no grad to prevent memory buildup)
        with torch.no_grad():
            actions = policy(obs)
        env.step(actions)
        obs = env.get_observations()
        total_steps += 1

        if total_steps % 500 == 0:
            print(f"  step {total_steps} | collected {collected}/{target_rollouts}", flush=True)

        # Record torques for all envs (running sum on GPU, no cpu transfer)
        with torch.no_grad():
            power = (env.torques * env.dof_vel).abs().sum(dim=1)
            episode_energy += power
        episode_steps += 1

        # Periodic CUDA cache clear
        if total_steps % 5000 == 0:
            torch.cuda.empty_cache()

        # Check goal reached or timeout/fall
        with torch.no_grad():
            dist_to_goal_now = (goal_positions - env.root_states[:, :2]).norm(dim=1)
            reached_goal = dist_to_goal_now < goal_tolerance
            timed_out = torch.tensor(episode_steps >= max_episode_steps, device="cuda:0")
            fell = env.reset_buf.bool() & ~timed_out

        # Process completed episodes (goal reached, timeout, or fall)
        done_mask = reached_goal | env.reset_buf.bool()
        done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_ids.cpu().numpy():
            idx = int(idx)
            # Only save successful goal reaches
            if reached_goal[idx] and episode_steps[idx] > 20:
                dt = 0.005 * 4  # sim_dt * decimation = control dt
                duration = episode_steps[idx] * dt
                energy = episode_energy[idx].item() * dt

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

                # Terrain type from terrain column mapped to category
                # With 20 cols and proportions [0.2]*5: cols 0-3=type0, 4-7=type1, etc.
                col = int(env.terrain_types[idx].item())
                terrain_type = min(col // 4, 4)
                terrain_level = int(env.terrain_levels[idx].item())

                all_heightmaps.append(hm)
                all_slopes.append(slope)
                all_roughness.append(roughness)
                all_friction.append(friction)
                all_distances.append(distance)
                all_elevations.append(elevation)
                all_costs.append(cost)
                all_terrain_types.append(terrain_type)
                all_terrain_levels.append(terrain_level)
                collected += 1

                if collected % 1000 == 0:
                    print(f"  Collected {collected}/{target_rollouts}")

            # Reset tracking for this env
            episode_energy[idx] = 0.0
            episode_steps[idx] = 0
            start_positions[idx] = env.root_states[idx, :3].clone().detach()
            # New random goal for this env
            new_dir = torch.randn(2, device="cuda:0")
            new_dir = new_dir / new_dir.norm().clamp(min=0.1)
            goal_positions[idx] = env.root_states[idx, :2].detach() + new_dir * goal_distance

    print(f"Collection done: {collected} rollouts in {total_steps} steps")

    # Save
    _save_dataset(all_heightmaps, all_slopes, all_roughness, all_friction,
                  all_distances, all_elevations, all_costs, all_terrain_types,
                  all_terrain_levels, cfg, save_dir)


def _get_heightmap_patch(env, env_idx, position):
    """Extract 16x16 heightmap patch around position."""
    try:
        hs = env.height_samples  # legged_gym stores this as (rows, cols) tensor
        h_scale = env.terrain.cfg.horizontal_scale
        v_scale = env.terrain.cfg.vertical_scale
        border = env.terrain.cfg.border_size

        # World position to heightmap index (account for border)
        px = int((position[0].item() + border) / h_scale)
        py = int((position[1].item() + border) / h_scale)
        px = max(8, min(px, hs.shape[0] - 9))
        py = max(8, min(py, hs.shape[1] - 9))
        patch = hs[px-8:px+8, py-8:py+8].cpu().numpy().astype(np.float32)
        if patch.shape == (16, 16):
            return patch * v_scale
    except Exception:
        pass
    # Fallback: try env.terrain.heightsamples directly
    try:
        hs = env.terrain.heightsamples
        h_scale = env.terrain.cfg.horizontal_scale
        v_scale = env.terrain.cfg.vertical_scale
        border = getattr(env.terrain.cfg, 'border_size', 0)
        px = int((position[0].item() + border) / h_scale)
        py = int((position[1].item() + border) / h_scale)
        px = max(8, min(px, hs.shape[0] - 9))
        py = max(8, min(py, hs.shape[1] - 9))
        patch = hs[px-8:px+8, py-8:py+8]
        if hasattr(patch, 'cpu'):
            patch = patch.cpu().numpy()
        patch = patch.astype(np.float32)
        if patch.shape == (16, 16):
            return patch * v_scale
    except Exception:
        pass
    return np.random.randn(16, 16).astype(np.float32) * 0.01


def _compute_slope(heightmap):
    """Compute mean slope from heightmap gradient."""
    gy, gx = np.gradient(heightmap, 0.1)
    return float(np.sqrt(gx**2 + gy**2).mean())


def _save_dataset(heightmaps, slopes, roughness, friction, distances,
                  elevations, costs, terrain_types, terrain_levels, cfg, save_dir):
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
            f.create_dataset("terrain_level", data=np.array([terrain_levels[i] for i in idx]))
        print(f"  {name}: {len(idx)} samples -> {path}")


if __name__ == "__main__":
    collect()
