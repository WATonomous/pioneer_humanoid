import sys
from pathlib import Path

import matplotlib.pyplot as plt

from terrain_core.preprocess.depth import depth_to_points

# tum.py lives in terrain_tools/, not next to this script, so add that folder
# to the places Python looks for imports (parents[1] = the terrain/ folder).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "terrain_tools"))
from tum import load_tum_frame  # noqa: E402


frame = load_tum_frame(450)
points = depth_to_points(frame.depth, frame.K)
print("image size:", frame.width, "x", frame.height, "=", frame.width * frame.height, "pixels")
print("points:", len(points))
print("z range (m): %.2f to %.2f" % (points[:, 2].min(), points[:, 2].max()))

# Left: the depth image. Right: the same thing as 3D points.
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(frame.depth, cmap="turbo")
fig.colorbar(im, ax=ax1, label="depth (m)")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
p = points[::20]                       # every 20th point, so the plot stays fast
ax2.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c=p[:, 2], cmap="turbo")
ax2.set_xlabel("x right (m)")
ax2.set_ylabel("y down (m)")
ax2.set_zlabel("z forward (m)")
ax2.set_aspect("equal")
ax2.view_init(elev=-75, azim=-90)      # look roughly from behind the camera
plt.show()
