import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from terrain_core.preprocess.pipeline import PreprocessConfig, preprocess

# tum.py lives in terrain_tools/, not next to this script, so add that folder
# to the places Python looks for imports (parents[1] = the terrain/ folder).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "terrain_tools"))
from tum import camera_pose_at, load_tum_frame  # noqa: E402


CONFIG = PreprocessConfig.from_yaml(Path(__file__).resolve().parents[1] / "config" / "terrain.yaml")


def cloud_in_world(index):
    """The whole TR-4 chain for one image, through preprocess()."""
    frame = load_tum_frame(index)
    T_world_cam = camera_pose_at(frame.stamp)        # where the camera was when THIS image was taken
    return preprocess(frame, T_world_cam, CONFIG).points, T_world_cam


a, T_a = cloud_in_world(0)
b, T_b = cloud_in_world(5)
print("camera at frame 0 (m):", np.round(T_a[:3, 3], 2))
print("camera at frame 5 (m):", np.round(T_b[:3, 3], 2))
print("height range in world (m): %.2f to %.2f" % (a[:, 2].min(), a[:, 2].max()))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(a[:, 0], a[:, 1], a[:, 2], s=1, c="tab:blue", label="frame 0")
ax.scatter(b[:, 0], b[:, 1], b[:, 2], s=1, c="tab:orange", label="frame 5")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z up (m)")
ax.set_aspect("equal")
ax.legend()
plt.show()
