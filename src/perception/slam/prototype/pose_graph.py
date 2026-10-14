#!/usr/bin/env python3
# batch GTSAM pose graph over the keyframes written by vo.py
#
# the graph has a prior pinning keyframe 0 at the origin, one odometry edge per keyframe
# (T_prev_kf with kf.covariance as noise), and one edge per loop in loops.txt. the loops are
# what correct the drift: with none, the result is the same as the VO trajectory
#
# writes <run>/kf_before.txt, kf_after.txt, trajectory_after.txt (TUM format)
# plot with: evo_traj tum <run>/trajectory_vo.txt <run>/trajectory_after.txt -p --plot_mode xz
#
# the real backend will be incremental (iSAM2) and take Keyframes straight from the VO
import argparse
import os

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X   # X(i) = graph key for keyframe i's pose

from slam_core.types import Keyframe
from slam_core.types.pose import covariance_from_sigmas, from_tum, to_tum


# reads a text file into split rows, skipping blank lines and # comments (missing file = no rows)
def rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [line.split() for line in fh if line.strip() and not line.startswith("#")]


# reads keyframes.txt back into Keyframe objects, rebuilding each T_odom_kf by chaining T_prev_kf
def read_keyframes(path):
    kfs, odom = [], {}
    for v in rows(path):
        kf_id, prev_id = int(v[0]), int(v[1])
        T = from_tum([float(x) for x in v[3:10]])
        odom[kf_id] = np.eye(4) if prev_id < 0 else odom[prev_id] @ T
        kfs.append(Keyframe(id=kf_id, prev_id=prev_id, stamp=float(v[2]), T_prev_kf=T,
                            covariance=covariance_from_sigmas(float(v[12]), float(v[13])),
                            T_odom_kf=odom[kf_id], n_inliers=int(v[10]), tracking_gap=bool(int(v[11]))))
    return kfs


# writes poses as "stamp tx ty tz qx qy qz qw" lines
def write_tum(path, stamps, poses):
    with open(path, "w") as fh:
        for s, T in zip(stamps, poses):
            fh.write(f"{s:.6f} " + " ".join(f"{x:.6f}" for x in to_tum(T)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="out", help="vo.py output directory")
    ap.add_argument("--loop-rot", type=float, default=0.02, help="rad, loop closure sigma")
    ap.add_argument("--loop-trans", type=float, default=0.03, help="m, loop closure sigma")
    args = ap.parse_args()

    kfs = read_keyframes(os.path.join(args.run, "keyframes.txt"))
    if not kfs:
        raise SystemExit(f"no keyframes in {args.run}/keyframes.txt")

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    # GTSAM's Pose3 noise is rotation first (rx ry rz tx ty tz), the same order as
    # Keyframe.covariance, so it can be passed straight in
    sigmas = lambda rot, trans: gtsam.noiseModel.Diagonal.Sigmas(np.array([rot] * 3 + [trans] * 3))  # noqa: E731

    # one pose per keyframe, starting from the raw VO pose
    for kf in kfs:
        initial.insert(X(kf.id), gtsam.Pose3(kf.T_odom_kf))
        if kf.is_first:
            # pins the first keyframe, without a prior the whole graph could slide and rotate freely
            graph.add(gtsam.PriorFactorPose3(X(kf.id), gtsam.Pose3(kf.T_odom_kf), sigmas(1e-6, 1e-6)))
        else:
            # odometry edge to the previous keyframe; tracking gaps already carry a huge covariance (~1 m / 1 rad)
            graph.add(gtsam.BetweenFactorPose3(X(kf.prev_id), X(kf.id), gtsam.Pose3(kf.T_prev_kf),
                                               gtsam.noiseModel.Gaussian.Covariance(kf.covariance)))

    # loop edges say "keyframe new is at T_old_new in keyframe old's frame". Huber
    # down-weights a loop that disagrees strongly with everything else (likely false),
    # so one wrong loop can't drag the whole map
    loops = rows(os.path.join(args.run, "loops.txt"))
    huber = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
    for v in loops:
        new_id, old_id = int(v[0]), int(v[1])
        T_old_new = from_tum([float(x) for x in v[3:10]])
        noise = gtsam.noiseModel.Robust.Create(huber, sigmas(args.loop_rot, args.loop_trans))
        graph.add(gtsam.BetweenFactorPose3(X(old_id), X(new_id), gtsam.Pose3(T_old_new), noise))

    first, last = kfs[0].id, kfs[-1].id

    # solves the whole graph at once with Levenberg-Marquardt
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()

    # prints how much the graph error and the end to start distance dropped
    before = {kf.id: kf.T_odom_kf for kf in kfs}
    after = {kf.id: result.atPose3(X(kf.id)).matrix() for kf in kfs}
    gap = lambda P: np.linalg.norm(P[last][:3, 3] - P[first][:3, 3])   # noqa: E731
    n_gaps = sum(kf.tracking_gap for kf in kfs)
    print(f"{len(kfs)} keyframes, {n_gaps} tracking gaps, {len(loops)} loops")
    print(f"graph error   before {graph.error(initial):.3f}   after {graph.error(result):.3f}")
    print(f"last KF to first KF   before {gap(before):.3f} m   after {gap(after):.3f} m")

    # saves the keyframe poses before and after optimizing
    stamps = [kf.stamp for kf in kfs]
    write_tum(os.path.join(args.run, "kf_before.txt"), stamps, [before[kf.id] for kf in kfs])
    write_tum(os.path.join(args.run, "kf_after.txt"), stamps, [after[kf.id] for kf in kfs])

    # moves every frame along with its optimized keyframe: T_odom_cur = T_odom_kf(after) @ T_kf_cur
    fs, fT = [], []
    for v in rows(os.path.join(args.run, "frames.txt")):
        fs.append(float(v[0]))
        fT.append(after[int(v[1])] @ from_tum([float(x) for x in v[2:9]]))
    write_tum(os.path.join(args.run, "trajectory_after.txt"), fs, fT)
    print(f"wrote {args.run}/kf_before.txt, kf_after.txt, trajectory_after.txt")


if __name__ == "__main__":
    main()
