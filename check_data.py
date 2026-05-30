import h5py
import numpy as np

f = h5py.File("data/train.hdf5", "r")
costs = f["cost"][:]
terrain = f["terrain_type"][:]
slopes = f["slope"][:]
roughness = f["roughness"][:]
distances = f["distance"][:]
f.close()

print(f"Cost: mean={costs.mean():.3f}, std={costs.std():.3f}, min={costs.min():.3f}, max={costs.max():.3f}")
print(f"CV: {costs.std()/costs.mean():.3f}")
print("\nBy terrain type:")
for t in range(5):
    c = costs[terrain == t]
    if len(c) > 0:
        print(f"  Terrain {t}: n={len(c)}, mean={c.mean():.3f}, std={c.std():.3f}")

print("\nBy slope (binned):")
for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 100.0)]:
    mask = (slopes >= lo) & (slopes < hi)
    c = costs[mask]
    if len(c) > 0:
        print(f"  Slope [{lo:.1f},{hi:.1f}): n={len(c)}, mean={c.mean():.3f}, std={c.std():.3f}")

print("\nBy roughness (binned):")
for lo, hi in [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, 10.0)]:
    mask = (roughness >= lo) & (roughness < hi)
    c = costs[mask]
    if len(c) > 0:
        print(f"  Rough [{lo:.2f},{hi:.2f}): n={len(c)}, mean={c.mean():.3f}, std={c.std():.3f}")

print("\nBy distance (binned):")
for lo, hi in [(0, 1), (1, 3), (3, 5), (5, 10), (10, 100)]:
    mask = (distances >= lo) & (distances < hi)
    c = costs[mask]
    if len(c) > 0:
        print(f"  Dist [{lo},{hi}): n={len(c)}, mean={c.mean():.3f}, std={c.std():.3f}")

print("\nEnergy-only analysis (cost - 0.1*duration proxy):")
# Duration ~ episode_steps * dt. Since cost = energy + 0.1*time,
# and all episodes are similar length, let's check if cost correlates with anything.
# Actually, let's just look at cost variance decomposition:
print(f"\nCorrelations with cost:")
from numpy import corrcoef
print(f"  slope vs cost: r={corrcoef(slopes, costs)[0,1]:.4f}")
print(f"  roughness vs cost: r={corrcoef(roughness, costs)[0,1]:.4f}")
print(f"  distance vs cost: r={corrcoef(distances, costs)[0,1]:.4f}")
