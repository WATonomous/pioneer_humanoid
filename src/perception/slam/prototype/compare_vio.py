#!/usr/bin/env python3
# Compares pose_graph.py (VO only) with vio_graph.py (VO + IMU) on one run
#
# Run pose_graph.py <run> and vio_graph.py <run> first. Reads <run>/kf_after.txt and <run>/vio/kf_after.txt.
#
# Levelness: the accelerometer, turned into each trajectory's odom frame by its own keyframe
# rotations and averaged (gravity.py's trick), should point straight along +y. Its tilt is how far
# the map would lean in map.py. Done per third of the run too, so a tilt that drifts shows up.
# Loops: how far each trajectory is from agreeing with every loop in loops.txt, and the end to start gap.
import argparse
import os

import numpy as np

from pose_graph import rows
from slam_core.backend.imu_fusion.gravity import gravity_in_frame
from slam_core.backend.imu_fusion.imu import read_extrinsics, read_imu
from slam_core.backend.imu_fusion.preintegration import make_params
from slam_core.types.pose import from_tum, inv_T, rotation_angle


# kf_after.txt (stamp tx ty tz qx qy qz qw) into stamps and 4x4 poses
def read_kf_poses(path):
    v = np.loadtxt(path, ndmin=2)
    return v[:, 0], [from_tum(r[1:8]) for r in v]


# Angle of gravity away from +y, in degrees
def tilt_deg(g):
    return np.degrees(np.arccos(np.clip(g[1] / np.linalg.norm(g), -1, 1)))


# Tilt over the whole run and over each third
def levelness(imu, params, stamps, poses):
    n = len(stamps)
    parts = [(0, n)] + [(i * n // 3, (i + 1) * n // 3) for i in range(3)]
    return [tilt_deg(gravity_in_frame(*imu, stamps[a:b], poses[a:b], params)) for a, b in parts]


# Rotation (deg) and translation (m) error of each loop: measured T_old_new vs what the poses say
def loop_errors(loops, stamps, poses):
    index = {round(s, 6): i for i, s in enumerate(stamps)}
    errs = []
    for old_stamp, new_stamp, T_meas in loops:
        T_est = inv_T(poses[index[old_stamp]]) @ poses[index[new_stamp]]
        E = inv_T(T_meas) @ T_est
        errs.append((np.degrees(rotation_angle(E[:3, :3])), np.linalg.norm(E[:3, 3])))
    return np.array(errs).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="out", help="vo.py output directory")
    ap.add_argument("--vio", help="vio_graph.py output directory (default <run>/vio)")
    args = ap.parse_args()
    vio_dir = args.vio or os.path.join(args.run, "vio")

    imu = read_imu(os.path.join(args.run, "imu.txt"))
    params = make_params(read_extrinsics(os.path.join(args.run, "imu_extrinsics.txt")))
    results = {"VO (pose_graph)": read_kf_poses(os.path.join(args.run, "kf_after.txt")),
               "VIO (vio_graph)": read_kf_poses(os.path.join(vio_dir, "kf_after.txt"))}

    # loops.txt rows use keyframe ids; keyframes.txt maps ids to stamps, which both outputs share
    kf_stamp = {int(v[0]): round(float(v[2]), 6) for v in rows(os.path.join(args.run, "keyframes.txt"))}
    loops = [(kf_stamp[int(v[1])], kf_stamp[int(v[0])], from_tum([float(x) for x in v[3:10]]))
             for v in rows(os.path.join(args.run, "loops.txt"))]

    print("Levelness: tilt of gravity from straight down in the output frame (deg, 0 = level map)")
    print(f"{'':18}{'whole run':>11}{'1st third':>11}{'2nd third':>11}{'3rd third':>11}")
    for name, (stamps, poses) in results.items():
        print(f"{name:18}" + "".join(f"{t:11.2f}" for t in levelness(imu, params, stamps, poses)))

    print(f"\nLoops: {len(loops)} in loops.txt")
    for name, (stamps, poses) in results.items():
        gap = np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3])
        line = f"{name:18} last KF to first KF {gap:.3f} m"
        if loops:
            e = loop_errors(loops, stamps, poses)
            line += (f"   loop error median {np.median(e[:, 0]):.2f} deg {np.median(e[:, 1]):.3f} m"
                     f"   max {e[:, 0].max():.2f} deg {e[:, 1].max():.3f} m")
        print(line)


if __name__ == "__main__":
    main()
