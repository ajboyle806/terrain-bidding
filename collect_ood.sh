#!/bin/bash
# Run each step separately (Isaac Gym can't reinitialize in same process)
# Usage: bash collect_ood.sh [step]
#   step 1: collect mild OOD (type 3)
#   step 2: collect strong OOD (type 4)
#   step 3: train ensemble on types 0-2

set -e
cd ~/terrain-bidding
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

STEP=${1:-1}

if [ "$STEP" = "1" ]; then
    echo "=== Step 1: Collecting mild OOD (type 3 = stairs down) ==="
    python -c "
import isaacgym
from terrain_bidding.collection import collect
from terrain_bidding.configs import CollectionConfig
cfg = CollectionConfig(); cfg.num_rollouts = 5000
collect(cfg=cfg, save_dir='data/ood_mild', terrain_mode='ood_type3')
"
    echo "Done. Now run: bash collect_ood.sh 2"

elif [ "$STEP" = "2" ]; then
    echo "=== Step 2: Collecting strong OOD (type 4 = discrete obstacles) ==="
    python -c "
import isaacgym
from terrain_bidding.collection import collect
from terrain_bidding.configs import CollectionConfig
cfg = CollectionConfig(); cfg.num_rollouts = 5000
collect(cfg=cfg, save_dir='data/ood_strong', terrain_mode='ood_type4')
"
    echo "Done. Now run: bash collect_ood.sh 3"

elif [ "$STEP" = "3" ]; then
    echo "=== Step 3: Training ensemble on types 0-2 only ==="
    python -m terrain_bidding.estimator
    echo "Done. Now push:"
    echo "  git add -f data/ checkpoints/ensemble/"
    echo "  git commit -m 'data: multi-OOD split'"
    echo "  git push"

else
    echo "Usage: bash collect_ood.sh [1|2|3]"
fi
