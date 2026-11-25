# Run VO offiline on a TUM sequence. Write trajectory in TUM format

import argparse
import time
import numpy as np
from slam_core.frontend.tracker import Tracker
from slam_core.io.tum import TUMDataset


def mat_to_quat(R):
    # 3x3 rotation amtrix -> (qx, qy, qz, qw)
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sequence", help="path to a TUM Sequence folder")
    p.add_argument("--out", default="traj_vo.txt")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = p.parse_args()

    ds = TUMDataset(args.sequence)
    tr = Tracker()
    if args.max_frames <= 0:
        n = len(ds)
    else:
        n = min(args.max_frames, len(ds))
    n_ok = n_kf = 0
    t0 = time.perf_counter()
    with open(args.out, "w") as f:
        for i in range(n):
            frame = ds[i]
            r = tr.process(frame)
            n_kf += r.keyframe is not None
            if not r.ok:
                print(f"frame {i}: lost ({r.reason})")
                continue
            n_ok += 1
            t = r.T_odom_cur[:3, 3]
            q = mat_to_quat(r.T_odom_cur[:3, :3])
            f.write(f"{frame.stamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
        dt = time.perf_counter() - t0
        print(f"\n{n_ok}/{n} frames tracked, {n_kf} keyframes, {n / dt:.1f} fps -> {args.out}")


if __name__ == "__main__":
    main()
