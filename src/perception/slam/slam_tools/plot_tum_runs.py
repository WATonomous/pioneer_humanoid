#!/usr/bin/env python3
"""Draw TUM runs from run_tum_loop.py: ground truth vs VO vs after loop closure, with the loops.

    python slam_tools/plot_tum_runs.py --out runs.png \\
        --seq fr1/room GT_FILE RUN_NONE RUN_RADIUS RUN_MODULE \\
        --seq fr2/desk GT_FILE RUN_NONE RUN_RADIUS RUN_MODULE

One row per --seq, three panels: VO only, old radius search, new module. Top-down (x, y) view. The estimates are
rigidly aligned to the ground truth (the same alignment the error number uses). Grey = ground truth, red = VO only,
blue = after loop closure, green lines = accepted loops (between the two keyframes they connect).
Needs numpy and matplotlib only.
"""
import argparse
import os

import numpy as np


def read(path):
    return np.loadtxt(path, comments="#", ndmin=2)


# Rigid alignment (rotation + translation) of est positions onto gt positions by timestamp. Returns (R, t, rmse)
def align(est, gt, max_dt=0.02):
    idx = np.clip(np.searchsorted(gt[:, 0], est[:, 0]), 1, len(gt) - 1)
    idx = np.where(np.abs(gt[idx - 1, 0] - est[:, 0]) < np.abs(gt[idx, 0] - est[:, 0]), idx - 1, idx)
    ok = np.abs(gt[idx, 0] - est[:, 0]) < max_dt
    A, B = est[ok, 1:4], gt[idx[ok], 1:4]
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    S = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    t = cb - R @ ca
    err = (R @ A.T).T + t - B
    return R, t, float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", nargs=5, action="append", required=True,
                    metavar=("NAME", "GT", "NONE", "RADIUS", "MODULE"))
    ap.add_argument("--out", default="tum_runs.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(args.seq), 3, figsize=(15, 4.6 * len(args.seq)), squeeze=False)
    for row, (name, gt_file, *runs) in zip(axes, args.seq):
        gt = read(gt_file)
        for ax, run, label in zip(row, runs, ("VO only", "old radius search", "new module")):
            final = "trajectory_vo.txt" if label == "VO only" else "trajectory_after.txt"
            est = read(os.path.join(run, final))
            R, t, rmse = align(est, gt)
            P = (R @ est[:, 1:4].T).T + t
            ax.plot(gt[:, 1], gt[:, 2], color="0.75", lw=5, label="ground truth", solid_capstyle="round")
            ax.plot(P[:, 0], P[:, 1], color="tab:red" if label == "VO only" else "tab:blue", lw=1.4,
                    label="VO" if label == "VO only" else "after loop closure")
            loops_file = os.path.join(run, "loops.txt")
            rows = [line.split() for line in open(loops_file) if line.strip() and not line.startswith("#")] if os.path.exists(loops_file) else []
            if rows:
                kf = read(os.path.join(run, "kf_after.txt"))                # one pose per keyframe, id order
                K = (R @ kf[:, 1:4].T).T + t
                for v in rows:
                    a, b = K[int(v[1])], K[int(v[0])]
                    ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:green", lw=1.6, alpha=0.9)
            ax.set_title(f"{name}: {label}\n{len(rows)} loops, error {rmse * 100:.1f} cm")
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
        row[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    print("wrote", args.out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
