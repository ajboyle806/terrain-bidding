"""PPO training using legged_gym's OnPolicyRunner.

This delegates to legged_gym/rsl_rl's battle-tested PPO implementation
rather than our custom loop.
"""
import os
from pathlib import Path

try:
    import isaacgym  # noqa: F401 — must be imported before torch
    from isaacgym import gymapi
    from legged_gym.envs import LeggedRobot
    from legged_gym.utils.helpers import class_to_dict, get_args, update_cfg_from_args
    from rsl_rl.runners import OnPolicyRunner
    HAS_ISAAC = True
except ImportError:
    HAS_ISAAC = False

from terrain_bidding.configs import PolicyConfig


def train(save_dir: str = "checkpoints"):
    """Train locomotion policy using legged_gym's runner."""
    if not HAS_ISAAC:
        raise RuntimeError("Isaac Gym + legged_gym + rsl_rl required. Run on GPU machine.")

    from terrain_bidding.envs import TerrainBiddingEnv, TerrainBiddingEnvCfg, TerrainBiddingPPOCfg

    os.makedirs(save_dir, exist_ok=True)

    # Create sim params
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

    # Create environment
    env_cfg = TerrainBiddingEnvCfg()
    env = TerrainBiddingEnv(env_cfg, sim_params, gymapi.SIM_PHYSX, "cuda:0", headless=True)

    # Create runner
    train_cfg = TerrainBiddingPPOCfg()
    train_cfg_dict = class_to_dict(train_cfg)

    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=save_dir, device="cuda:0")
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                 init_at_random_ep_len=True)

    # Save final policy in our expected format
    import torch
    policy_state = runner.get_inference_policy(device="cuda:0").state_dict()
    torch.save(policy_state, f"{save_dir}/policy_best.pt")
    print(f"Policy saved to {save_dir}/policy_best.pt")


if __name__ == "__main__":
    train()
