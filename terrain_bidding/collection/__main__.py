"""Entry point for: python -m terrain_bidding.collection [mode]

Modes:
  in_distribution  - types 0-3 (for ensemble training) [default]
  held_out         - type 4 only (discrete obstacles, for testing)
  all              - all types
"""
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import sys
from terrain_bidding.collection import collect

mode = sys.argv[1] if len(sys.argv) > 1 else "in_distribution"
save_dir = "data" if mode == "in_distribution" else f"data/{mode}"
collect(terrain_mode=mode, save_dir=save_dir)
