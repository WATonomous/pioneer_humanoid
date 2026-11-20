#!/usr/bin/env python3
"""Score loop closures from our module and from RTAB-Map on a TUM sequence with the same ground-truth rules.

    python slam_tools/compare_loops.py DATASET_DIR --ours RUN_DIR [--rtabmap rtabmap.db]

RUN_DIR is a run_tum_loop.py output (loops.txt + kf_vo.txt). rtabmap.db is the database RTAB-Map writes.
The two systems make keyframes differently, so everything is scored by TIME, not by keyframe id:
  - a loop is CORRECT if its relative pose is within 10 cm and 5 deg of the ground truth between the two images;
  - the ground-truth trajectory is sampled every 0.5 s; a sample is a REVISIT if an earlier one, at least 5 m of true travel
    before it, was within 1 m and 40 deg; COVERAGE is the fraction of revisit samples with a correct loop within 2 s.
RTAB-Map stores loop transforms in its robot base frame and may store a link in either direction; one convention (frame and
direction) is chosen for all of RTAB-Map's loops, the one that fits the ground truth best, so the comparison is not rigged
against it by a frame mix-up. Loop types counted for RTAB-Map: global loop closures and proximity (local space) closures.
"""
import argparse
import os
import sqlite3
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "slam_core"))

import numpy as np  # noqa: E402

from slam_core.types.pose import from_tum, inv_T, rotation_angle  # noqa: E402

# Camera optical frame (x right, y down, z forward) expressed in a robot base frame (x forward, y left, z up)
OPTICAL_IN_BASE = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


def gt_pose_at(gt, t):
    return from_tum(gt[int(np.argmin(np.abs(gt[:, 0] - t))), 1:8])


# Error (cm, deg) of a loop transform T (pose of the newer image in the older one's frame) against the ground truth
def loop_error(gt, t_old, t_new, T):
    E = inv_T(inv_T(gt_pose_at(gt, t_old)) @ gt_pose_at(gt, t_new)) @ T
    return np.linalg.norm(E[:3, 3]) * 100, np.degrees(rotation_angle(E[:3, :3]))


# Our loops: [(t_old, t_new, T_old_new)] from loops.txt + keyframe stamps
def load_ours(run):
    stamps = np.loadtxt(os.path.join(run, "kf_vo.txt"), comments="#", ndmin=2)[:, 0]
    out = []
    for line in open(os.path.join(run, "loops.txt")):
        if line.strip() and not line.startswith("#"):
            v = line.split()
            out.append((stamps[int(v[1])], stamps[int(v[0])], from_tum([float(x) for x in v[3:10]])))
    return out


# RTAB-Map's loop links: [(t_a, t_b, T_a_b)] as stored (frame and direction are sorted out by fit_rtabmap_convention)
def load_rtabmap(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    stamps = dict(con.execute("SELECT id, stamp FROM Node"))
    out = []
    for a, b, blob in con.execute("SELECT from_id, to_id, transform FROM Link WHERE type IN (1, 2)"):
        raw = bytes(blob)
        if len(raw) != 48:
            raw = zlib.decompress(raw)
        T = np.eye(4)
        T[:3, :] = np.frombuffer(raw, dtype=np.float32).reshape(3, 4)
        out.append((stamps[a], stamps[b], T))
    return out


# Picks the one frame/direction convention that best explains all of RTAB-Map's loops, and returns them in our convention
def fit_rtabmap_convention(gt, links):
    C = OPTICAL_IN_BASE
    options = {
        "as stored": lambda ta, tb, T: (ta, tb, T),
        "reversed": lambda ta, tb, T: (tb, ta, inv_T(T)),
        "base->optical": lambda ta, tb, T: (ta, tb, inv_T(C) @ T @ C),
        "base->optical, reversed": lambda ta, tb, T: (tb, ta, inv_T(C) @ inv_T(T) @ C),
    }
    best = None
    for name, f in options.items():
        loops = [f(*link) for link in links]
        med = np.median([loop_error(gt, *L)[0] for L in loops]) if loops else 0
        if best is None or med < best[0]:
            best = (med, name, loops)
    return best[1], best[2]


def revisit_samples(gt, step=0.5, min_travel=5.0):
    t = np.arange(gt[0, 0], gt[-1, 0], step)
    P = [gt_pose_at(gt, s) for s in t]
    arc = np.concatenate([[0.0], np.cumsum([np.linalg.norm(P[i][:3, 3] - P[i - 1][:3, 3]) for i in range(1, len(P))])])
    rev = []
    for i in range(len(P)):
        for j in range(i):
            if arc[i] - arc[j] < min_travel:
                break
            T = inv_T(P[j]) @ P[i]
            if np.linalg.norm(T[:3, 3]) < 1.0 and np.degrees(rotation_angle(T[:3, :3])) < 40:
                rev.append(t[i])
                break
    return np.array(rev)


def report(name, gt, loops, revisits, note=""):
    errs = [loop_error(gt, *L) for L in loops]
    good = [L for L, (c, d) in zip(loops, errs) if c < 10 and d < 5]
    far = sum(1 for c, _ in errs if c > 100)
    cov = np.mean([any(abs(r - L[1]) <= 2.0 for L in good) for r in revisits]) if len(revisits) else float("nan")
    prec = f"{len(good) / len(loops):.2f}" if loops else "-"
    med = f"{np.median([c for c, _ in errs]):.1f} cm" if errs else "-"
    print(f"{name:10s} loops {len(loops):4d}  correct {len(good):4d}  precision {prec:>5s}  off by >1 m {far:3d}  "
          f"median error {med:>8s}  coverage {cov:.2f}  {note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--ours", help="run_tum_loop.py output directory (module mode)")
    ap.add_argument("--rtabmap", help="rtabmap.db written by RTAB-Map on the same sequence")
    args = ap.parse_args()

    gt = np.loadtxt(os.path.join(args.dataset, "groundtruth.txt"), comments="#")
    revisits = revisit_samples(gt)
    print(f"{os.path.basename(os.path.normpath(args.dataset))}: {len(revisits)} revisit samples (every 0.5 s)")
    if args.ours:
        report("ours", gt, load_ours(args.ours), revisits)
    if args.rtabmap:
        convention, loops = fit_rtabmap_convention(gt, load_rtabmap(args.rtabmap))
        report("RTAB-Map", gt, loops, revisits, f"(transform convention: {convention})")


if __name__ == "__main__":
    main()
