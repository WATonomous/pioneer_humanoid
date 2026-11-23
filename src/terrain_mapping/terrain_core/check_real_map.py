import numpy as np
from terrain_core.traversability import compute_traversability
from terrain_core.traversability.policy_scan import foothold_layer

elevation = np.load(r"C:\Users\Ritha\OneDrive\Desktop\WATO\HUMANOID\pioneer_humanoid\terrain mapping bags\elevation.npy")
print("shape", elevation.shape, "NaN fraction", np.isnan(elevation).mean())

trav = compute_traversability(elevation, 0.04)
res = foothold_layer(trav["traversable"], 0.04, trav["traversability"])

# the return type depends on how the new chat wrote it; handle the likely cases
good = res["foothold"] if isinstance(res, dict) else res[0]

print("traversable cells:", int(trav["traversable"].sum()))
print("good cells:", int(good.sum()))
print("kept: %.2f%%" % (100 * good.sum() / trav["traversable"].sum()))