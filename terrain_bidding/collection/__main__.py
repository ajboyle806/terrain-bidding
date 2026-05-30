"""Entry point for: python -m terrain_bidding.collection"""
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

from terrain_bidding.collection import collect

collect()
