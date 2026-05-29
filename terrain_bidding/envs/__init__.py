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
    from legged_gym.utils.terrain import Terrain
    HAS_ISAAC = True
except ImportError:
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


class TerrainBiddingEnvCfg:
    """Configuration matching legged_gym's expected format."""

    class env:
        num_envs = 4096
        num_observations = 95
        num_actions = 12
        episode_length_s = 15.0

    class terrain:
        mesh_type = "trimesh"
        num_rows = 20
        num_cols = 20
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]  # no obstacles initially
        curriculum = True
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 5.0
        static_friction = 1.0
        dynamic_friction = 1.0

    class commands:
        num_commands = 3  # vx, vy, yaw_rate
        heading_command = False

    class init_state:
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            "FL_hip": 0.0, "FR_hip": 0.0, "RL_hip": 0.0, "RR_hip": 0.0,
            "FL_thigh": 0.8, "FR_thigh": 0.8, "RL_thigh": 1.0, "RR_thigh": 1.0,
            "FL_calf": -1.5, "FR_calf": -1.5, "RL_calf": -1.5, "RR_calf": -1.5,
        }

    class control:
        stiffness = {"joint": 20.0}
        damping = {"joint": 0.5}
        action_scale = 0.5
        decimation = 4  # 50Hz control at 200Hz sim

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        actuator_strength_range = [0.9, 1.1]
        latency_range_ms = [0, 20]

    class rewards:
        class scales:
            lin_vel_tracking = 1.0
            ang_vel_tracking = 0.5
            torque = -0.0002
            joint_vel = -0.0001
            orientation = -0.2
            survival = 1.0
            fall = -10.0

    class termination:
        base_height_min = 0.15
        max_pitch_roll = 1.047  # 60 degrees in radians
        contact_grace_period = 0.1  # seconds before non-foot contact terminates


class TerrainBiddingEnv(LeggedRobot if HAS_ISAAC else object):
    """Legged robot environment with terrain curriculum for cost data collection."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        if not HAS_ISAAC:
            raise RuntimeError("Isaac Gym not available. Install from NVIDIA.")
        self.terrain_cfg = TerrainConfig()
        self.policy_cfg = PolicyConfig()
        self.unlocked_terrains = [FLAT]  # curriculum starts with flat
        self.terrain_success_buffer = {i: [] for i in range(6)}
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _compute_observations(self):
        """95-dim observation: joints(12) + joint_vel(12) + base_lin_vel(3) +
        base_ang_vel(3) + gravity(3) + prev_action(12) + terrain_heights(50)."""
        self.obs_buf = torch.cat([
            self.dof_pos,                          # 12
            self.dof_vel,                          # 12
            self.base_lin_vel,                     # 3
            self.base_ang_vel,                     # 3
            self.projected_gravity,                # 3
            self.actions,                          # 12
            self._get_terrain_heights(),           # 50
        ], dim=-1)

    def _get_terrain_heights(self):
        """Sample 50 height points on 0.1m grid around base."""
        points = self._get_height_sample_points()  # (num_envs, 50, 3)
        heights = self._get_heights_at_points(points)  # (num_envs, 50)
        return heights - self.root_states[:, 2:3]  # relative to base height

    def _get_height_sample_points(self):
        """Generate 50 sample points in a pattern around the robot base."""
        # 5x10 grid, 0.1m spacing, centered on base
        x = torch.linspace(-0.2, 0.2, 5, device=self.device)
        y = torch.linspace(-0.5, 0.4, 10, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        offsets = torch.stack([grid_x.flatten(), grid_y.flatten(),
                               torch.zeros(50, device=self.device)], dim=-1)
        # Rotate by base yaw and translate to base position
        base_pos = self.root_states[:, :3]  # (N, 3)
        points = base_pos.unsqueeze(1) + offsets.unsqueeze(0)  # (N, 50, 3)
        return points

    def _get_heights_at_points(self, points):
        """Query terrain heightmap at given world coordinates."""
        # Convert world coords to heightmap indices
        px = ((points[:, :, 0] - self.terrain.border_size) /
              self.terrain.horizontal_scale).long()
        py = ((points[:, :, 1] - self.terrain.border_size) /
              self.terrain.horizontal_scale).long()
        px = torch.clamp(px, 0, self.terrain.heightsamples.shape[0] - 1)
        py = torch.clamp(py, 0, self.terrain.heightsamples.shape[1] - 1)
        return self.terrain.heightsamples[px, py] * self.terrain.vertical_scale

    def compute_reward(self):
        """Reward function with terrain-dependent energy-time tradeoffs."""
        s = self.cfg.rewards.scales

        # Velocity tracking (exponential kernel)
        lin_vel_error = torch.sum((self.commands[:, :2] - self.base_lin_vel[:, :2]) ** 2, dim=1)
        ang_vel_error = (self.commands[:, 2] - self.base_ang_vel[:, 2]) ** 2
        r_lin = torch.exp(-lin_vel_error / 0.25) * s.lin_vel_tracking
        r_ang = torch.exp(-ang_vel_error / 0.25) * s.ang_vel_tracking

        # Penalties
        r_torque = s.torque * torch.sum(self.torques ** 2, dim=1)
        r_joint_vel = s.joint_vel * torch.sum(self.dof_vel ** 2, dim=1)
        gravity_xy = self.projected_gravity[:, :2]
        r_orient = s.orientation * torch.sum(gravity_xy ** 2, dim=1)

        # Survival bonus
        r_survival = torch.ones(self.num_envs, device=self.device) * s.survival

        self.rew_buf = r_lin + r_ang + r_torque + r_joint_vel + r_orient + r_survival

    def check_termination(self):
        """Fall detection: height, orientation, non-foot contact."""
        cfg_t = self.cfg.termination

        # Base too low
        height_term = self.root_states[:, 2] < cfg_t.base_height_min

        # Excessive pitch/roll
        pitch = torch.atan2(self.projected_gravity[:, 0], self.projected_gravity[:, 2])
        roll = torch.atan2(self.projected_gravity[:, 1], self.projected_gravity[:, 2])
        orient_term = (torch.abs(pitch) > cfg_t.max_pitch_roll) | \
                      (torch.abs(roll) > cfg_t.max_pitch_roll)

        # Timeout
        timeout = self.episode_length_buf >= self.max_episode_length

        self.reset_buf = height_term | orient_term | timeout
        self.time_out_buf = timeout

        # Apply fall penalty
        fall_mask = (height_term | orient_term) & ~timeout
        self.rew_buf[fall_mask] += self.cfg.rewards.scales.fall

    def update_terrain_curriculum(self, env_ids):
        """Advance curriculum when success rate > 80% on current terrain."""
        for env_id in env_ids:
            terrain_type = self.terrain_types[env_id].item()
            success = not self.time_out_buf[env_id] and not (
                self.root_states[env_id, 2] < self.cfg.termination.base_height_min)
            self.terrain_success_buffer[terrain_type].append(float(success))

            buf = self.terrain_success_buffer[terrain_type]
            if len(buf) >= 100:
                rate = sum(buf[-100:]) / 100
                if rate >= self.terrain_cfg.curriculum_advance_threshold:
                    next_terrain = terrain_type + 1
                    if next_terrain < 6 and next_terrain not in self.unlocked_terrains:
                        self.unlocked_terrains.append(next_terrain)

    def _sample_terrain_for_env(self, env_id):
        """Sample uniformly from unlocked terrain types."""
        return np.random.choice(self.unlocked_terrains)
