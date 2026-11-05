import sys
from pathlib import Path

import matplotlib.pyplot as plt

from terrain_core.preprocess.crop import in_depth_range
from terrain_core.preprocess.depth import depth_to_points
from terrain_core.preprocess.downsample import voxel_downsample

# tum.py lives in terrain_tools/, not next to this script, so add that folder
# to the places Python looks for imports (parents[1] = the terrain/ folder).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "terrain_tools"))
from tum import load_tum_frame  # noqa: E402

frame = load_tum_frame(450)
points = depth_to_points(frame.depth, frame.K)
points = points[in_depth_range(points, z_min=0.3, z_max=3.0)]

for size in [0.01, 0.02, 0.05, 0.10]:
    thinned = voxel_downsample(points, size)
    print(f"voxel {size * 100:.0f} cm: {len(points)} -> {len(thinned)} points")

thinned = voxel_downsample(points, 0.02)

fig = plt.figure(figsize=(12, 5))
for i, (pts, title) in enumerate([(points[::10], "after crop (every 10th point)"),
                                  (thinned, "after 2 cm voxel downsample")]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=pts[:, 2], cmap="turbo")
    ax.set_title(title)
    ax.set_xlabel("x right (m)")
    ax.set_ylabel("y down (m)")
    ax.set_zlabel("z forward (m)")
    ax.set_aspect("equal")
    ax.view_init(elev=-30, azim=-90)
plt.show()
