# Terrain Bidding

Incentive-compatible task allocation for legged robots via adaptive uncertainty decomposition.

## Paper Results

All experiment outputs are stored in:
- `outputs/` — pickled experiment results (.pkl)
- `figures/` — generated paper figures (.png)
- `results.txt` — raw terminal output from experiment runs

## Data

- `data/` — Isaac Gym simulation data (A1 robot)
  - `data/train_types01/` — ensemble training data (terrain types 0-1)
  - `data/ood_type2/` — mild OOD (stairs up)
  - `data/ood_type3/` — moderate OOD (stairs down)
  - `data/ood_type4/` — strong OOD (discrete obstacles)
- `data/grandtour/` — Real ANYmal data (GrandTour dataset, 8 missions)
- `checkpoints/ensemble/` — trained ensemble (types 0-1)
- `checkpoints/ensemble_grandtour/` — ensemble trained on real ANYmal data

## Running Experiments

```bash
# Simulation experiments (Mac or any machine with PyTorch + numpy + scipy)
python3 -m terrain_bidding.experiments.focused_experiments
python3 -m terrain_bidding.experiments.best_paper_experiments
python3 -m terrain_bidding.experiments.additional_experiments
python3 -m terrain_bidding.experiments.fleet_experiments

# Real-data validation (downloads from HuggingFace)
python3 -m terrain_bidding.grandtour 8

# Generate figures
python3 -m terrain_bidding.experiments.generate_figures
```

## Key Results

| Experiment | Finding |
|---|---|
| Adaptive mechanism | FPR stays 9-11% across all OOD levels (static full: 67%) |
| Unbiased decomposition | 176% better detection (sep +5.18 vs +1.88) |
| Deterrence | All κ > 0 make manipulation unprofitable |
| Real ANYmal validation | sep=+0.92, FPR=0%, deterrence at κ=10 |
| Reputation baseline | Detects but doesn't deter (gain still +200%) |
| Scaling | Stable detection/FPR at N=4, 8, 16 |
| Learning adversary | Converges to honesty from all initializations |

## Project Structure

```
terrain_bidding/
├── envs/              # Isaac Gym environment (Phase 1)
├── policy/            # PPO training + evaluation (Phase 1)
├── collection/        # Rollout data collection (Phase 2)
├── estimator/         # Ensemble cost estimator (Phase 3)
├── mechanism/         # Allocator + scoring rule (Phase 4)
├── adversaries/       # Strategic agents (Phase 5)
├── experiments/       # All experiment scripts (Phase 6)
│   ├── focused_experiments.py      # Core paper results
│   ├── best_paper_experiments.py   # Adaptive switcher, γ/κ, fleet
│   ├── additional_experiments.py   # Robustness, welfare
│   ├── fleet_experiments.py        # Heterogeneous fleet, reputation
│   ├── real_sampler.py             # Private observations + real data
│   ├── held_out.py                 # OOD proxy sampler
│   └── generate_figures.py         # Paper figure generation
├── grandtour/         # Real ANYmal data pipeline
└── configs/           # Shared configuration
```

## Setup

**GPU machine (Phases 1-2, Isaac Gym):**
```bash
pip install -e .
# + Isaac Gym, legged_gym, rsl_rl
python -m terrain_bidding.policy
python -m terrain_bidding.collection in_distribution
```

**Any machine (Phases 3-6, experiments):**
```bash
pip install torch numpy scipy h5py matplotlib
python3 -m terrain_bidding.experiments.focused_experiments
```
