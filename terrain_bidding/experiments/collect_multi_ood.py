"""Collect data with multi-OOD split for distribution shift benchmark.

Run on laptop:
  python -m terrain_bidding.experiments.collect_multi_ood

Collects 3 datasets:
  data/train_types012/  — ensemble training (types 0-2: smooth slope, rough slope, stairs up)
  data/ood_mild/        — mildly OOD (type 3: stairs down, similar to stairs up)
  data/ood_strong/      — strongly OOD (type 4: discrete obstacles, very different)
"""
import sys
import os

try:
    import isaacgym  # noqa
except ImportError:
    pass

from terrain_bidding.collection import collect
from terrain_bidding.configs import CollectionConfig


def main():
    cfg = CollectionConfig()
    cfg.num_rollouts = 30000
    cfg.held_out_rollouts = 5000

    # Step 1: In-distribution (types 0-2)
    print("="*60)
    print("STEP 1: Collecting IN-DISTRIBUTION (types 0-2)")
    print("="*60)
    collect(cfg=cfg, save_dir="data/train_types012", terrain_mode="in_distribution_012")

    # Step 2: Mildly OOD (type 3 = stairs down)
    print("\n" + "="*60)
    print("STEP 2: Collecting MILDLY OOD (type 3 = stairs down)")
    print("="*60)
    cfg_mild = CollectionConfig()
    cfg_mild.num_rollouts = 5000
    collect(cfg=cfg_mild, save_dir="data/ood_mild", terrain_mode="ood_type3")

    # Step 3: Strongly OOD (type 4 = discrete obstacles)
    print("\n" + "="*60)
    print("STEP 3: Collecting STRONGLY OOD (type 4 = discrete obstacles)")
    print("="*60)
    cfg_strong = CollectionConfig()
    cfg_strong.num_rollouts = 5000
    collect(cfg=cfg_strong, save_dir="data/ood_strong", terrain_mode="ood_type4")

    print("\n\nDone! Now train ensemble on data/train_types012/ only:")
    print("  python -m terrain_bidding.estimator --data-dir data/train_types012")


if __name__ == "__main__":
    main()
