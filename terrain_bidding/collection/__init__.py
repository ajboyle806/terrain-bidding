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
            policy_path: str = "checkpoints/model_8000.pt",
            save_dir: str = "data",
            device: str = "cuda"):
    """Collect rollout dataset with frozen policy."""
    if not HAS_ISAAC:
        raise RuntimeError("Isaac Gym required. Run on GPU machine.")

    os.makedirs(save_dir, exist_ok=True)

    # Load policy using rsl_rl's format
    from rsl_rl.runners import OnPolicyRunner
    from terrain_bidding.envs import TerrainBiddingEnvCfg, TerrainBiddingPPOCfg, TerrainBiddingEnv
    from isaacgym import gymapi
    from terrain_bidding.policy.train import class_to_dict

    # Find checkpoint if default doesn't exist
    if not os.path.exists(policy_path):
        import glob
        models = sorted(glob.glob("checkpoints/**/model_*.pt", recursive=True))
        if models:
            policy_path = models[-1]
        else:
            models = sorted(glob.glob("checkpoints/model_*.pt"))
            if models:
                policy_path = models[-1]
            else:
                raise FileNotFoundError("No model checkpoint found in checkpoints/")
    print(f"Using policy: {policy_path}")

    # Create env for collection (headless, single env)
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
    env_cfg.env.num_envs = 64  # parallel collection
    env = TerrainBiddingEnv(env_cfg, sim_params, gymapi.SIM_PHYSX, "cuda:0", headless=True)

    train_cfg = TerrainBiddingPPOCfg()
    train_cfg_dict = class_to_dict(train_cfg)
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=None, device="cuda:0")
    runner.load(policy_path)
    policy = runner.get_inference_policy(device="cuda:0")
    print("Policy loaded successfully.")

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
    """Sample a 16x16 heightmap patch from terrain. Returns (heightmap, slope, roughness, friction).

    Requires Isaac Gym runtime. Uses the global _ENV singleton.
    """
    if not HAS_ISAAC:
        raise RuntimeError("Isaac Gym required. Run on GPU machine.")

    env = _get_env_singleton()
    # Pick a random sub-region of the terrain matching terrain_idx
    terrain = env.terrain
    row = np.random.randint(0, terrain.heightsamples.shape[0] - 16)
    col = np.random.randint(0, terrain.heightsamples.shape[1] - 16)
    patch = terrain.heightsamples[row:row+16, col:col+16].cpu().numpy() * terrain.vertical_scale

    # Compute features from patch
    gy, gx = np.gradient(patch, terrain.horizontal_scale)
    slope = np.sqrt(gx**2 + gy**2).mean()
    roughness = patch.std()

    # Friction from domain randomization range for this terrain
    friction = np.random.uniform(0.5, 1.25)

    return patch.astype(np.float32), float(slope), float(roughness), float(friction)


def _execute_trajectory(policy, terrain_idx, distance, elevation, cfg, device):
    """Run frozen policy to goal. Returns trajectory dict or None on failure.

    Requires Isaac Gym runtime. Runs a single-env episode with the frozen policy.
    """
    if not HAS_ISAAC:
        raise RuntimeError("Isaac Gym required. Run on GPU machine.")

    env = _get_env_singleton()
    dt = 1.0 / 50.0  # control frequency
    max_steps = int(cfg.max_episode_time / dt)

    # Set goal for env 0
    goal_dir = np.random.randn(2)
    goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-8)
    env.commands[0, 0] = goal_dir[0] * 1.0  # vx command
    env.commands[0, 1] = goal_dir[1] * 0.3  # vy command
    env.commands[0, 2] = 0.0  # yaw rate

    torques_list, joint_vels_list = [], []
    start_pos = env.root_states[0, :3].clone()

    for step in range(max_steps):
        obs = env.obs_buf[0:1]
        with torch.no_grad():
            action = policy.act(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]

        env.step(action)

        # Record telemetry
        torques_list.append(env.torques[0].cpu().numpy().copy())
        joint_vels_list.append(env.dof_vel[0].cpu().numpy().copy())

        # Check termination
        if env.reset_buf[0]:
            if env.time_out_buf[0]:
                return None  # timeout
            return None  # fall

        # Check goal reached
        pos = env.root_states[0, :3]
        dist_traveled = torch.norm(pos[:2] - start_pos[:2]).item()
        if dist_traveled >= distance - cfg.goal_tolerance:
            duration = (step + 1) * dt
            return {
                "torques": np.array(torques_list),
                "joint_vels": np.array(joint_vels_list),
                "dt": dt,
                "duration": duration,
            }

    return None  # timeout


# Singleton env for data collection (avoids recreating per rollout)
_ENV_SINGLETON = None


def _get_env_singleton():
    """Lazy-init a single Isaac Gym env for data collection."""
    global _ENV_SINGLETON
    if _ENV_SINGLETON is None:
        from terrain_bidding.envs import TerrainBiddingEnv, TerrainBiddingEnvCfg
        from isaacgym import gymapi
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 200.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        env_cfg = TerrainBiddingEnvCfg()
        env_cfg.env.num_envs = 1  # single env for sequential collection
        _ENV_SINGLETON = TerrainBiddingEnv(
            env_cfg, sim_params, gymapi.SIM_PHYSX, "cuda:0", headless=True)
    return _ENV_SINGLETON


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
