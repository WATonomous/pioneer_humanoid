"""Load VO text outputs and construct Keyframe objects.

This first step only converts keyframes.txt and kf_vo.txt into the shared
Keyframe data type. Backend replay and optimized trajectory output come later.
"""

from pathlib import Path

from slam_core.types import Keyframe
from slam_core.types.pose import covariance_from_sigmas, from_tum


def load_vo_keyframes(keyframes_path, odom_path):
    odom_poses = []
    for line in Path(odom_path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = line.split()
        odom_poses.append(from_tum(values[1:8]))

    keyframes = []
    for line in Path(keyframes_path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = line.split()
        keyframe_id = int(values[0])
        keyframes.append(
            Keyframe(
                id=keyframe_id,
                prev_id=int(values[1]),
                stamp=float(values[2]),
                T_prev_kf=from_tum(values[3:10]),
                covariance=covariance_from_sigmas(
                    float(values[12]),
                    float(values[13]),
                ),
                T_odom_kf=odom_poses[keyframe_id],
                tracking_gap=bool(int(values[11])),
                n_inliers=int(values[10]),
            )
        )
    return keyframes
