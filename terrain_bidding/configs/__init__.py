"""Shared configuration dataclasses for all phases."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class CostWeights:
    alpha: float = 1.0  # energy weight
    beta: float = 0.1   # time weight


@dataclass
class TerrainConfig:
    num_rows: int = 20
    num_cols: int = 20
    horizontal_scale: float = 0.1  # meters per cell
    vertical_scale: float = 0.005
    # Curriculum thresholds per terrain type
    flat_threshold: float = 0.0
    low_roughness_range: tuple = (0.0, 0.02)
    high_roughness_range: tuple = (0.02, 0.06)
    mild_slope_range: tuple = (0.0, 15.0)   # degrees
    steep_slope_range: tuple = (15.0, 30.0)
    obstacle_height_range: tuple = (0.05, 0.15)
    curriculum_advance_threshold: float = 0.8  # 80% success


@dataclass
class PolicyConfig:
    num_envs: int = 1024
    obs_dim: int = 95
    act_dim: int = 12
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    action_clip: float = 0.5
    kp: float = 20.0
    kd: float = 0.5
    control_freq: float = 50.0
    # PPO
    lr: float = 3e-4
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    minibatch_size: int = 2048
    epochs_per_batch: int = 4
    max_steps: int = 500_000_000
    # Domain randomization
    friction_range: tuple = (0.5, 1.25)
    restitution_range: tuple = (0.0, 0.4)
    mass_perturbation_range: tuple = (-1.0, 3.0)
    actuator_strength_range: tuple = (0.9, 1.1)
    latency_range_ms: tuple = (0, 20)


@dataclass
class CollectionConfig:
    num_rollouts: int = 30_000
    held_out_rollouts: int = 1_500
    distance_range: tuple = (3.0, 15.0)
    elevation_range: tuple = (-2.0, 2.0)
    max_episode_time: float = 15.0
    goal_tolerance: float = 0.5
    cost_weights: CostWeights = field(default_factory=CostWeights)
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class EnsembleConfig:
    K: int = 5
    terrain_embed_dim: int = 64
    mlp_hidden: int = 128
    dropout: float = 0.1
    log_var_clip: tuple = (-4.0, 4.0)
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10
    # Larger ablation variant
    large_terrain_embed_dim: int = 256
    large_mlp_hidden: int = 256


@dataclass
class MechanismConfig:
    N: int = 4                  # fleet size
    R: float = 5.0              # completion reward (normalized cost space, costs ~ 0±1)
    kappa: float = 2.0          # scoring penalty coefficient
    gamma: float = 1.0          # epistemic weighting in allocation
    S_baseline: float = None    # set from validation performance


@dataclass
class ExperimentConfig:
    episodes_per_condition: int = 5000
    fleet_sizes: List[int] = field(default_factory=lambda: [4, 8])
    kappa_sweep: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0])
    gamma_sweep: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    strategic_counts_n4: List[int] = field(default_factory=lambda: [0, 1, 2])
    strategic_counts_n8: List[int] = field(default_factory=lambda: [0, 2, 4])
    seed: int = 42
