"""Load VO text outputs and construct backend data objects.

This helper currently converts keyframes.txt and kf_vo.txt into Keyframe
objects. LoopClosure objects are imported for the next replay stage.
"""

from pathlib import Path

from slam_core.types import Keyframe, LoopClosure
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


def load_vo_loops(loops_path, rot_sigma=0.05, trans_sigma=0.05):
    covariance = covariance_from_sigmas(rot_sigma, trans_sigma)
    loops = []
    for line in Path(loops_path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = line.split()
        loops.append(
            LoopClosure(
                old_id=int(values[1]),
                new_id=int(values[0]),
                stamp=float(values[2]),
                T_old_new=from_tum(values[3:10]),
                covariance=covariance.copy(),
                n_inliers=int(values[10]),
            )
        )
    return loops
