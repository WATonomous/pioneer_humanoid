import sys
from pathlib import Path

import matplotlib.pyplot as plt

from terrain_core.preprocess.crop import in_depth_range
from terrain_core.preprocess.depth import depth_to_points

# tum.py lives in terrain_tools/, not next to this script, so add that folder
# to the places Python looks for imports (parents[1] = the terrain/ folder).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "terrain_tools"))
from tum import load_tum_frame  # noqa: E402

frame = load_tum_frame(450)              # a frame that sees far into the room
points = depth_to_points(frame.depth, frame.K)

keep = in_depth_range(points, z_min=0.3, z_max=3.0)
kept, removed = points[keep], points[~keep]
print("points before crop:", len(points))
print("kept:", len(kept), " removed:", len(removed),
      "(%.1f%%)" % (100 * len(removed) / len(points)))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
k, r = kept[::20], removed[::20]         # every 20th point, so the plot stays fast
ax.scatter(k[:, 0], k[:, 1], k[:, 2], s=1, c="tab:blue", label="kept (0.3-3 m)")
ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=1, c="tab:red", label="removed (too far)")
ax.set_xlabel("x right (m)")
ax.set_ylabel("y down (m)")
ax.set_zlabel("z forward (m)")
ax.set_aspect("equal")
ax.view_init(elev=-30, azim=-90)
ax.legend()
plt.show()
