#!/usr/bin/env python3
"""How good could loop closure possibly be on this VO? Feed the pose graph PERFECT loops (from the ground truth).

    python slam_tools/oracle_loops.py DATASET_DIR RUN_DIR --out OUT_DIR

RUN_DIR is a run_tum_loop.py output (VO only is fine). The loops come from the ground truth poses, never from images:
for every keyframe that revisits an earlier one (view within 1 m and 40 deg after at least 5 m of true travel) it adds a
loop to the closest such keyframe, with a 10-keyframe spacing. The pose graph then runs as usual and the trajectory error is
reported. That number is the ceiling for any loop detector on this visual odometry: loop closure removes drift, not VO noise.
Needs numpy and gtsam (via prototype/pose_graph.py).
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "slam_core"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from run_tum_loop import ate, read_traj  # noqa: E402
from slam_core.types.pose import from_tum, inv_T, rotation_angle, to_tum  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-gt-path", type=float, default=5.0)
    ap.add_argument("--spacing", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for f in ("keyframes.txt", "frames.txt", "kf_vo.txt", "trajectory_vo.txt"):
        shutil.copy(os.path.join(args.run, f), args.out)

    kf = np.loadtxt(os.path.join(args.run, "kf_vo.txt"), comments="#")          # stamp + pose, index = keyframe id
    gt = np.loadtxt(os.path.join(args.dataset, "groundtruth.txt"), comments="#")
    G = [from_tum(gt[int(np.argmin(np.abs(gt[:, 0] - t))), 1:8]) for t in kf[:, 0]]
    arc = np.concatenate([[0.0], np.cumsum([np.linalg.norm(G[i][:3, 3] - G[i - 1][:3, 3]) for i in range(1, len(G))])])

    loops, last = [], -10 ** 9
    for k in range(len(G)):
        if k - last < args.spacing:
            continue
        best = None
        for j in range(k):
            if k - j < 20 or arc[k] - arc[j] < args.min_gt_path:
                continue
            T = inv_T(G[j]) @ G[k]
            d, a = np.linalg.norm(T[:3, 3]), np.degrees(rotation_angle(T[:3, :3]))
            if d < 1.0 and a < 40 and (best is None or d < best[0]):
                best = (d, j, T)
        if best:
            loops.append((k, best[1], kf[k, 0], best[2]))
            last = k
    with open(os.path.join(args.out, "loops.txt"), "w") as fh:
        fh.write("# new_kf_id old_kf_id stamp tx ty tz qx qy qz qw inliers  (pose = T_old_new, from the ground truth)\n")
        for k, j, t, T in loops:
            fh.write(f"{k} {j} {t:.6f} " + " ".join(f"{v:.6f}" for v in to_tum(T)) + " 100\n")

    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "slam_core"))
    subprocess.run([sys.executable, os.path.join(HERE, "..", "prototype", "pose_graph.py"), args.out], check=True,
                   env=env, stdout=subprocess.DEVNULL)
    gt_all = read_traj(os.path.join(args.dataset, "groundtruth.txt"))
    raw = ate(read_traj(os.path.join(args.run, "trajectory_vo.txt")), gt_all)[0]
    opt = ate(read_traj(os.path.join(args.out, "trajectory_after.txt")), gt_all)[0]
    print(f"{len(loops)} oracle loops: ATE VO only {raw:.3f} m -> with perfect loops {opt:.3f} m")


if __name__ == "__main__":
    main()
