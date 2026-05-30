import h5py
import numpy as np

f = h5py.File("data/train.hdf5", "r")
costs = f["cost"][:]
terrain = f["terrain_type"][:]
f.close()

print(f"Cost: mean={costs.mean():.3f}, std={costs.std():.3f}, min={costs.min():.3f}, max={costs.max():.3f}")
print(f"CV: {costs.std()/costs.mean():.3f}")
for t in range(5):
    c = costs[terrain == t]
    if len(c) > 0:
        print(f"  Terrain {t}: n={len(c)}, mean={c.mean():.3f}, std={c.std():.3f}")
