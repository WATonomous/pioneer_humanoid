#!/usr/bin/env python3
"""Run VO + loop closure on a TUM RGB-D sequence and score it against the ground truth.

    python slam_tools/run_tum_loop.py DATASET_DIR --loop-mode module --out runs/room_module

DATASET_DIR is an unpacked TUM sequence (e.g. rgbd_dataset_freiburg1_room, which has several loop closures).
--loop-mode: none (VO only), radius (the old brute force search), module (slam_core.loop_closure).
Writes the usual vo.py outputs to --out, runs prototype/pose_graph.py (needs gtsam), then prints the trajectory
error (ATE RMSE, rigid alignment like evo's default) of the raw VO trajectory and of the optimized one.
Needs torch (XFeat is fetched with torch.hub on first use), opencv, numpy, gtsam.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "slam_core"))
sys.path.insert(0, os.path.join(HERE, "..", "prototype"))

import numpy as np  # noqa: E402


# Reads "stamp tx ty tz qx qy qz qw" lines into an (N, 8) array
def read_traj(path):
    return np.loadtxt(path, comments="#", ndmin=2)


# ATE RMSE (m): match poses by timestamp, rigidly align the estimate to the ground truth (Umeyama without scale),
# RMS of the remaining position error. Returns (rmse, number of matched poses)
def ate(est, gt, max_dt=0.02):
    idx = np.searchsorted(gt[:, 0], est[:, 0])
    idx = np.clip(idx, 1, len(gt) - 1)
    idx = np.where(np.abs(gt[idx - 1, 0] - est[:, 0]) < np.abs(gt[idx, 0] - est[:, 0]), idx - 1, idx)
    ok = np.abs(gt[idx, 0] - est[:, 0]) < max_dt
    A, B = est[ok, 1:4], gt[idx[ok], 1:4]
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    S = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    err = (R @ (A - ca).T).T + cb - B
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--loop-mode", choices=["none", "radius", "module"], default="module")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=2, help="use every Nth frame (speed)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = all)")
    ap.add_argument("--set", nargs="*", default=[], help="VO Params overrides, e.g. loop_min_inliers=30")
    ap.add_argument("--isam2", action="store_true",
                    help="module mode: also replay the run through the incremental iSAM2 backend, as a live system would")
    args = ap.parse_args()

    import vo
    from slam_core.io.tum import TUMDataset, load_groundtruth

    over = dict(kv.split("=") for kv in args.set)
    fields = vo.Params.__dataclass_fields__
    over = {k: fields[k].type(v) if callable(fields[k].type) else v for k, v in over.items()}
    if args.loop_mode == "none":
        over.update(loop_mode="radius", loop_min_time=1e12)      # no keyframe is ever old enough: VO only
    else:
        over["loop_mode"] = args.loop_mode
    p = vo.Params(**over)

    data = TUMDataset(args.dataset)
    V = vo.VO(vo.load_xfeat(p.top_k), p)
    n, lost, t0 = 0, 0, time.time()
    for i in range(0, len(data), args.step):
        res = V.process(data[i])
        n += 1
        lost += res.state == "lost"
        if res.loop_to >= 0:
            print(f"[{res.new_kf.stamp:.1f}] loop to KF {res.loop_to} (now {len(V.loops)} kept)", flush=True)
        if n % 100 == 0:
            print(f"  {n} frames, {len(V.refs)} keyframes, {len(V.loops)} loops, {time.time() - t0:.0f}s", flush=True)
        if args.max_frames and n >= args.max_frames:
            break
    V.write(args.out)
    print(f"{n} frames, {lost} lost, {len(V.refs)} keyframes, {len(V.loops)} loops, {time.time() - t0:.0f}s")

    gt = load_groundtruth(args.dataset)
    raw = read_traj(os.path.join(args.out, "trajectory_vo.txt"))
    print(f"ATE raw VO: {ate(raw, gt)[0]:.3f} m over {ate(raw, gt)[1]} poses")

    # The optimized trajectory: the same GTSAM pose graph as before, run on what vo.py just wrote
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "..", "slam_core"))
    subprocess.run([sys.executable, os.path.join(HERE, "..", "prototype", "pose_graph.py"), args.out], check=True, env=env)
    opt = read_traj(os.path.join(args.out, "trajectory_after.txt"))
    print(f"ATE after loop closure: {ate(opt, gt)[0]:.3f} m ({args.loop_mode}, {len(V.loops)} loops)")

    if args.isam2:
        if args.loop_mode != "module":
            raise SystemExit("--isam2 needs --loop-mode module")
        replay_isam2(V, args.out, gt)


# Replays the keyframes through a fresh loop detector and the incremental iSAM2 backend, one keyframe at a time like a live
# system: each keyframe's odometry edge goes in, then whatever loops the detector adds or retracts at that moment. Writes
# trajectory_after_isam2.txt (every frame moved with its corrected keyframe) and prints the error and the backend's cost.
# The detector is re-run with the same verifier, so its loops can differ from the VO run's by RANSAC randomness
def replay_isam2(V, out, gt):
    from slam_core.backend.incremental import IncrementalBackend
    from slam_core.loop_closure import LoopClosureDetector, LoopParams
    from slam_core.types.pose import to_tum

    backend, detector = IncrementalBackend(), LoopClosureDetector(V._verify_loop, LoopParams())
    active, rejected, ms = {}, 0, []
    for r in V.refs:
        update = detector.process(r.kf, V.encoder.encode(r.kf.descriptors))
        t = time.perf_counter()
        backend.add_keyframe(r.kf)
        bad = backend.apply(update)
        ms.append((time.perf_counter() - t) * 1000)
        for L in update.added:
            active[L.id] = L
        for i in update.retracted + bad:
            active.pop(i, None)
        rejected += len(bad)

    poses = {r.kf.id: backend.pose(r.kf.id) for r in V.refs}
    path = os.path.join(out, "trajectory_after_isam2.txt")
    with open(path, "w") as fh:
        for stamp, kf_id, T in V.frames:
            fh.write(f"{stamp:.6f} " + " ".join(f"{v:.6f}" for v in to_tum(poses[kf_id] @ T)) + "\n")
    print(f"ATE with iSAM2 (live replay): {ate(read_traj(path), gt)[0]:.3f} m ({len(active)} loops kept, {rejected} rejected "
          f"by the backend's check); backend {np.median(ms):.1f} ms median, {np.max(ms):.0f} ms max per keyframe")


if __name__ == "__main__":
    main()
