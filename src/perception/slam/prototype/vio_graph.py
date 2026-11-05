#!/usr/bin/env python3
# Batch VIO over a vo.py run: pose_graph.py plus the IMU (slam_core.backend.imu_fusion)
#
# Needs <run>/keyframes.txt, imu.txt, imu_extrinsics.txt; uses loops.txt and frames.txt if present.
# Writes <out>/kf_before.txt (leveled VO), kf_after.txt, trajectory_after.txt (TUM format), the same
# files as pose_graph.py, so map.py and evo work unchanged. Default <out> is <run>/vio so it doesn't
# overwrite pose_graph.py's output.
import argparse
import os

import numpy as np

from pose_graph import read_keyframes, rows, write_tum
from slam_core.backend.imu_fusion.gravity import estimate_gravity, level_rotation
from slam_core.backend.imu_fusion.imu import read_extrinsics, read_imu
from slam_core.backend.imu_fusion.preintegration import make_params
from slam_core.backend.imu_fusion.vio_graph import solve_vio
from slam_core.types.pose import from_tum, rotation_angle


# Angle between the camera's "down" (y) in odom and gravity (+y): 0 = camera level
def tilt_deg(T):
    return np.degrees(np.arccos(np.clip(T[1, 1], -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="out", help="vo.py output directory")
    ap.add_argument("--out", help="output directory (default <run>/vio)")
    ap.add_argument("--loop-rot", type=float, default=0.02, help="rad, loop closure sigma")
    ap.add_argument("--loop-trans", type=float, default=0.03, help="m, loop closure sigma")
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "vio")
    os.makedirs(out, exist_ok=True)

    kfs = read_keyframes(os.path.join(args.run, "keyframes.txt"))
    if not kfs:
        raise SystemExit(f"no keyframes in {args.run}/keyframes.txt")
    imu = read_imu(os.path.join(args.run, "imu.txt"))
    params = make_params(read_extrinsics(os.path.join(args.run, "imu_extrinsics.txt")))
    loops = [(int(v[1]), int(v[0]), from_tum([float(x) for x in v[3:10]]))
             for v in rows(os.path.join(args.run, "loops.txt"))]

    # Which way is down at KF0 (3c). estimate_gravity takes keyframes.txt as a plain array
    kf_array = np.loadtxt(os.path.join(args.run, "keyframes.txt"), ndmin=2)
    R_odom_kf0 = level_rotation(estimate_gravity(*imu, kf_array, params))

    graph, initial, result, before, after = solve_vio(kfs, loops, imu, params, R_odom_kf0,
                                                      args.loop_rot, args.loop_trans)

    first, last = kfs[0].id, kfs[-1].id

    def gap(P):
        return np.linalg.norm(P[last][:3, 3] - P[first][:3, 3])

    print(f"{len(kfs)} keyframes, {len(loops)} loops")
    print(f"KF0 tilt from gravity alignment {tilt_deg(before[first]):.2f} deg, after solving {tilt_deg(after[first]):.2f} deg")
    print(f"graph error   before {graph.error(initial):.3f}   after {graph.error(result):.3f}")
    print(f"last KF to first KF   before {gap(before):.3f} m   after {gap(after):.3f} m")
    moved = [np.degrees(rotation_angle(before[k][:3, :3].T @ after[k][:3, :3])) for k in after]
    print(f"rotation change vs leveled VO   median {np.median(moved):.2f} deg   max {np.max(moved):.2f} deg")

    stamps = [kf.stamp for kf in kfs]
    write_tum(os.path.join(out, "kf_before.txt"), stamps, [before[kf.id] for kf in kfs])
    write_tum(os.path.join(out, "kf_after.txt"), stamps, [after[kf.id] for kf in kfs])

    # Moves every frame along with its optimized keyframe: T_odom_cur = T_odom_kf(after) @ T_kf_cur
    fs, fT = [], []
    for v in rows(os.path.join(args.run, "frames.txt")):
        fs.append(float(v[0]))
        fT.append(after[int(v[1])] @ from_tum([float(x) for x in v[2:9]]))
    write_tum(os.path.join(out, "trajectory_after.txt"), fs, fT)
    print(f"wrote {out}/kf_before.txt, kf_after.txt, trajectory_after.txt")


if __name__ == "__main__":
    main()
