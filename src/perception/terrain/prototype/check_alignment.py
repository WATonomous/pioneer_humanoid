import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from terrain_core.preprocess.pipeline import PreprocessConfig, preprocess

# tum.py lives in terrain_tools/, not next to this script, so add that folder
# to the places Python looks for imports (parents[1] = the terrain/ folder).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "terrain_tools"))
from tum import camera_pose_at, load_tum_frame  # noqa: E402


CONFIG = PreprocessConfig.from_yaml(Path(__file__).resolve().parents[1] / "config" / "terrain.yaml")


def cloud_in_world(index, pose_stamp=None):
    """preprocess() on one TUM image. pose_stamp: use the pose at a different time (to test mistakes)."""
    frame = load_tum_frame(index)
    stamp = frame.stamp if pose_stamp is None else pose_stamp
    return preprocess(frame, camera_pose_at(stamp, max_gap=1.0), CONFIG).points


def alignment(a, b, max_dist=0.03):
    """For each point in b, the distance to the closest point in a.

    Returns (share of b's points within max_dist of a, median distance in metres).
    """
    dist, _ = cKDTree(a).query(b)        # KD-tree: fast "closest point" lookup
    return np.mean(dist < max_dist), np.median(dist)


a = cloud_in_world(0)
stamp_0 = load_tum_frame(0).stamp

for j in [5, 10, 20]:
    stamp_j = load_tum_frame(j).stamp
    tests = {
        "right pose": cloud_in_world(j),
        "pose 50 ms late": cloud_in_world(j, pose_stamp=stamp_j + 0.05),
        "frame 0's pose": cloud_in_world(j, pose_stamp=stamp_0),
    }
    for name, b in tests.items():
        share, median = alignment(a, b)
        print(f"frame 0 vs {j:2d}, {name:16s}: {share:4.0%} within 3 cm, median {median * 100:.1f} cm")
    print()
