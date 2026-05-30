"""Quick diagnostic: check what terrain data looks like."""
import h5py
import numpy as np

f = h5py.File("data/train.hdf5", "r")
hm = f["heightmap"][:10]  # first 10 heightmaps
slopes = f["slope"][:100]
roughness = f["roughness"][:100]
f.close()

print("First 10 heightmap stats:")
for i in range(10):
    print(f"  [{i}] min={hm[i].min():.4f} max={hm[i].max():.4f} std={hm[i].std():.4f} range={hm[i].max()-hm[i].min():.4f}")

print(f"\nSlope stats: min={slopes.min():.4f} max={slopes.max():.4f} mean={slopes.mean():.4f}")
print(f"Roughness stats: min={roughness.min():.4f} max={roughness.max():.4f} mean={roughness.mean():.4f}")
