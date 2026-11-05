import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from terrain_core.preprocess.crop import in_depth_range
from terrain_core.preprocess.depth import depth_to_points
from terrain_core.preprocess.downsample import voxel_downsample
from terrain_core.preprocess.transform import transform_points

# tum.py lives in terrain_tools/, not next to this script, so add that folder
# to the places Python looks for imports (parents[1] = the terrain/ folder).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "terrain_tools"))
from tum import camera_pose_at, load_tum_frame  # noqa: E402


def cloud_in_world(index, pose_stamp=None):
    """TR-4 chain for one TUM image. pose_stamp: use the pose at a different time (to test mistakes)."""
    frame = load_tum_frame(index)
    points = depth_to_points(frame.depth, frame.K)
    points = points[in_depth_range(points, z_min=0.3, z_max=3.0)]
    points = voxel_downsample(points, 0.02)
    stamp = frame.stamp if pose_stamp is None else pose_stamp
    return transform_points(camera_pose_at(stamp, max_gap=1.0), points)


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
