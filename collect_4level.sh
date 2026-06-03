#!/bin/bash
# Collect data for 4-level OOD gradient:
#   Train: types 0-1 only
#   OOD mild: type 2 (stairs up)
#   OOD moderate: type 3 (stairs down)
#   OOD strong: type 4 (discrete obstacles)
#
# Run each step one at a time:
#   bash collect_4level.sh 1
#   bash collect_4level.sh 2
#   bash collect_4level.sh 3
#   bash collect_4level.sh 4
#   bash collect_4level.sh 5

set -e
cd ~/terrain-bidding
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

STEP=${1:-1}

if [ "$STEP" = "1" ]; then
    echo "=== Step 1: Training data (types 0-1, 30K rollouts) ==="
    python -c "
import isaacgym
from terrain_bidding.collection import collect
from terrain_bidding.configs import CollectionConfig
cfg = CollectionConfig(); cfg.num_rollouts = 30000
collect(cfg=cfg, save_dir='data/train_types01', terrain_mode='in_distribution_01')
"
    echo "Done. Run: bash collect_4level.sh 2"

elif [ "$STEP" = "2" ]; then
    echo "=== Step 2: OOD mild (type 2 = stairs up, 5K) ==="
    python -c "
import isaacgym
from terrain_bidding.collection import collect
from terrain_bidding.configs import CollectionConfig
cfg = CollectionConfig(); cfg.num_rollouts = 5000
collect(cfg=cfg, save_dir='data/ood_type2', terrain_mode='ood_type2')
"
    echo "Done. Run: bash collect_4level.sh 3"

elif [ "$STEP" = "3" ]; then
    echo "=== Step 3: OOD moderate (type 3 = stairs down, 5K) ==="
    python -c "
import isaacgym
from terrain_bidding.collection import collect
from terrain_bidding.configs import CollectionConfig
cfg = CollectionConfig(); cfg.num_rollouts = 5000
collect(cfg=cfg, save_dir='data/ood_type3', terrain_mode='ood_type3')
"
    echo "Done. Run: bash collect_4level.sh 4"

elif [ "$STEP" = "4" ]; then
    echo "=== Step 4: OOD strong (type 4 = discrete obstacles, 5K) ==="
    python -c "
import isaacgym
from terrain_bidding.collection import collect
from terrain_bidding.configs import CollectionConfig
cfg = CollectionConfig(); cfg.num_rollouts = 5000
collect(cfg=cfg, save_dir='data/ood_type4', terrain_mode='ood_type4')
"
    echo "Done. Run: bash collect_4level.sh 5"

elif [ "$STEP" = "5" ]; then
    echo "=== Step 5: Train ensemble on types 0-1 only ==="
    python -c "
import isaacgym
from terrain_bidding.estimator import train_ensemble
train_ensemble(data_dir='data/train_types01')
"
    echo "Done. Now push:"
    echo "  git add -f data/ checkpoints/ensemble/"
    echo "  git commit -m 'data: 4-level OOD gradient (train on types 0-1)'"
    echo "  git push"

else
    echo "Usage: bash collect_4level.sh [1|2|3|4|5]"
fi
