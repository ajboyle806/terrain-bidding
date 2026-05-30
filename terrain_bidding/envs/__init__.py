"""Isaac Gym environment for legged locomotion with terrain curriculum.

Extends legged_gym's LeggedRobot with our specific terrain types,
reward structure, and domain randomization.
"""
from __future__ import annotations

import torch
import numpy as np

try:
    from isaacgym import gymapi, gymtorch
    from legged_gym.envs import LeggedRobot
    from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
    from legged_gym.utils.terrain import Terrain
    HAS_ISAAC = True
except (ImportError, AttributeError):
    HAS_ISAAC = False

from terrain_bidding.configs import PolicyConfig, TerrainConfig


# Terrain type indices
FLAT = 0
LOW_ROUGH = 1
HIGH_ROUGH = 2
MILD_SLOPE = 3
STEEP_SLOPE = 4
DISCRETE_OBSTACLES = 5  # held out from cost estimator training
TERRAIN_NAMES = ["flat", "low_rough", "high_rough", "mild_slope", "steep_slope", "discrete_obstacles"]


if HAS_ISAAC:
    class TerrainBiddingEnvCfg(LeggedRobotCfg):
        """Inherits full legged_gym config, overrides what we need."""

        class env(LeggedRobotCfg.env):
            num_envs = 1024
            num_observations = 235
            num_actions = 12
            episode_length_s = 15.0

        class terrain(LeggedRobotCfg.terrain):
            mesh_type = "trimesh"
            num_rows = 20
            num_cols = 20
            terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
            curriculum = True
            measure_heights = True

        class commands(LeggedRobotCfg.commands):
            num_commands = 4
            heading_command = True
            class ranges:
                lin_vel_x = [-1.0, 1.5]
                lin_vel_y = [-0.5, 0.5]
                ang_vel_yaw = [-1.0, 1.0]
                heading = [-3.14, 3.14]

        class init_state(LeggedRobotCfg.init_state):
            pos = [0.0, 0.0, 0.42]
            default_joint_angles = {
                "FL_hip_joint": 0.1, "FR_hip_joint": -0.1,
                "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
                "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
                "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
                "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
                "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
            }

        class control(LeggedRobotCfg.control):
            control_type = 'P'
            stiffness = {"joint": 20.0}
            damping = {"joint": 0.5}
            action_scale = 0.25
            decimation = 4

        class asset(LeggedRobotCfg.asset):
            file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf"
            name = "a1"
            foot_name = "foot"
            penalize_contacts_on = ["thigh", "calf"]
            terminate_after_contacts_on = ["base"]
            self_collisions = 1

        class domain_rand(LeggedRobotCfg.domain_rand):
            randomize_friction = True
            friction_range = [0.5, 1.25]
            randomize_base_mass = True
            added_mass_range = [-1.0, 3.0]
            push_robots = True
            push_interval_s = 15
            max_push_vel_xy = 1.0

        class rewards(LeggedRobotCfg.rewards):
            soft_dof_pos_limit = 0.9
            base_height_target = 0.25
            class scales(LeggedRobotCfg.rewards.scales):
                torques = -0.0002
                dof_pos_limits = -10.0

        class normalization(LeggedRobotCfg.normalization):
            class obs_scales(LeggedRobotCfg.normalization.obs_scales):
                lin_vel = 2.0
                ang_vel = 0.25
                dof_pos = 1.0
                dof_vel = 0.05
                height_measurements = 5.0
            clip_observations = 100.
            clip_actions = 100.

    class TerrainBiddingPPOCfg(LeggedRobotCfgPPO):
        """PPO runner config."""
        class policy(LeggedRobotCfgPPO.policy):
            init_noise_std = 1.0
            actor_hidden_dims = [512, 256, 128]
            critic_hidden_dims = [512, 256, 128]
            activation = 'elu'

        class algorithm(LeggedRobotCfgPPO.algorithm):
            entropy_coef = 0.01
            clip_param = 0.2
            learning_rate = 1e-3
            schedule = 'adaptive'
            desired_kl = 0.01
            lam = 0.95
            num_learning_epochs = 5
            num_mini_batches = 4

        class runner(LeggedRobotCfgPPO.runner):
            policy_class_name = 'ActorCritic'
            algorithm_class_name = 'PPO'
            num_steps_per_env = 24
            max_iterations = 10000
            save_interval = 500
            experiment_name = 'terrain_bidding'
            run_name = 'a1'

else:
    # Stub for non-Isaac environments
    class TerrainBiddingEnvCfg:
        class env:
            num_envs = 1024
            num_observations = 235
            num_actions = 12
            episode_length_s = 15.0
    class TerrainBiddingPPOCfg:
        pass


class TerrainBiddingEnv(LeggedRobot if HAS_ISAAC else object):
    """Legged robot environment with terrain curriculum for cost data collection.

    Inherits all standard behavior from LeggedRobot (rewards, termination,
    terrain curriculum). We only keep the init override for our config objects.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        if not HAS_ISAAC:
            raise RuntimeError("Isaac Gym not available. Install from NVIDIA.")
        self.terrain_cfg = TerrainConfig()
        self.policy_cfg = PolicyConfig()
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
