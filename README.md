# Terrain Bidding

Incentive-compatible task allocation for legged robots via uncertainty decomposition.

## Setup

**GPU machine (Phases 1–3):**
```bash
# Install Isaac Gym from https://developer.nvidia.com/isaac-gym
# Clone legged_gym and rsl_rl:
git clone https://github.com/leggedrobotics/legged_gym.git
git clone https://github.com/leggedrobotics/rsl_rl.git
pip install -e legged_gym/ rsl_rl/
pip install -e .
```

**Any machine (Phases 4–6):**
```bash
pip install -e .
```

## Project Structure

```
terrain_bidding/
├── envs/          # Phase 1: Isaac Gym environment + terrain curriculum
├── policy/        # Phase 1: Actor-critic + PPO training
├── collection/    # Phase 2: Rollout data collection
├── estimator/     # Phase 3: Ensemble cost estimator
├── mechanism/     # Phase 4: Allocator + scoring rule
├── adversaries/   # Phase 5: Strategic agent implementations
├── experiments/   # Phase 6: Grid runner + metrics + plotting
└── configs/       # Shared configuration dataclasses
```

## Running

```bash
# Phase 1: Train locomotion policy (GPU machine)
python -m terrain_bidding.policy.train

# Phase 1: Evaluate policy against gate criteria
python -m terrain_bidding.policy.evaluate

# Phase 2: Collect rollout dataset
python -m terrain_bidding.collection.collect

# Phase 3: Train ensemble cost estimator
python -m terrain_bidding.estimator.train

# Phase 4-6: Run experiments
python -m terrain_bidding.experiments.run
```
