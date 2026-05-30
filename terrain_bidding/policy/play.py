"""Visualize trained policy in Isaac Gym viewer."""
import sys
import os


def play(checkpoint_path: str = None):
    """Run trained policy with visualization."""
    import isaacgym  # noqa: F401
    from isaacgym import gymapi
    from rsl_rl.runners import OnPolicyRunner
    from terrain_bidding.envs import TerrainBiddingEnv, TerrainBiddingEnvCfg, TerrainBiddingPPOCfg

    def class_to_dict(obj):
        """Convert nested config class to dict (avoids legged_gym circular import)."""
        result = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue
            val = getattr(obj, key)
            if callable(val):
                continue
            if isinstance(val, type):
                result[key] = class_to_dict(val)
            else:
                result[key] = val
        return result

    # Find latest checkpoint if not specified
    if checkpoint_path is None:
        import glob
        models = sorted(glob.glob("checkpoints/**/model_*.pt", recursive=True))
        if not models:
            print("No checkpoints found. Train first.")
            return
        checkpoint_path = models[-1]
        print(f"Using: {checkpoint_path}")

    # Create env with rendering (headless=False)
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
    env_cfg.env.num_envs = 16  # small for visualization on 8GB VRAM
    env = TerrainBiddingEnv(env_cfg, sim_params, gymapi.SIM_PHYSX, "cuda:0", headless=False)

    # Load policy
    train_cfg = TerrainBiddingPPOCfg()
    train_cfg_dict = class_to_dict(train_cfg)
    # Older rsl_rl expects flat dict with keys from runner/policy/algorithm merged
    runner_cfg = {**train_cfg_dict.get("runner", {}),
                  **train_cfg_dict.get("policy", {}),
                  **train_cfg_dict.get("algorithm", {})}
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device="cuda:0")
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device="cuda:0")

    # Run
    obs = env.get_observations()
    while True:
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    play(path)
