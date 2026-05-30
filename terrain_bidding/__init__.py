# Isaac Gym must be imported before torch in any module
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass