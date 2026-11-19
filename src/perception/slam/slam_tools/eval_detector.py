#!/usr/bin/env python3
"""The whole loop closure detector (retrieval, verification, gate, confirmation, consistency) on a TUM sequence,
scored against the ground truth, for several parameter settings.

    python slam_tools/eval_detector.py DATASET_DIR --cache cache_room.pt

--cache is the file eval_loops_pr.py writes (the VO keyframes with features and depth). The real XFeat verifier is used;
verification results are shared between settings, so the sweep is much faster than the first run.
Ground truth decides:
  - a loop is CORRECT if its relative pose is within 10 cm and 5 deg of the truth;
  - a keyframe is a REVISIT if an earlier keyframe at least 20 odometry edges and --min-gt-path metres of true travel before it
    was within 1 m and 40 deg;
    a run of consecutive revisit keyframes is an EVENT; an event is CLOSED if a correct accepted loop lands on it
    (or within 5 keyframes of it);
  - coverage = fraction of revisit keyframes inside a closed event window.
Reports precision (correct / accepted), event recall, coverage, and the time the first (uncached) run took per keyframe.
"""
import argparse
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "slam_core"))
sys.path.insert(0, os.path.join(HERE, "..", "prototype"))

import numpy as np  # noqa: E402

# name -> LoopParams overrides. The first is the default
VARIANTS = {
    "default": {},
    "old default": dict(min_inliers=40, confirm_count=3, top_k=5, min_score=0.2),
    "min_inliers=30": dict(min_inliers=30),
    "min_inliers=20": dict(min_inliers=20),
    "confirm=2": dict(confirm_count=2),
    "confirm=1": dict(confirm_count=1),
    "no gate": dict(gate=False),
    "no pcm": dict(pcm=False),
    "no defenses": dict(gate=False, pcm=False, confirm_count=1),
    "gate x3": dict(gate_inflation=3.0),
    "top_k=10": dict(top_k=10),
    "min_score=0.1": dict(min_score=0.1),
    "cooldown=5": dict(cooldown=5),
    "mi30+confirm2": dict(min_inliers=30, confirm_count=2),
    "mi20+confirm2": dict(min_inliers=20, confirm_count=2),
    "mi20+confirm2+top10": dict(min_inliers=20, confirm_count=2, top_k=10),
    # speed: fewer candidates verified, higher retrieval similarity floor
    "top_k=3": dict(top_k=3),
    "top_k=2": dict(top_k=2),
    "min_score=0.3": dict(min_score=0.3),
    "min_score=0.4": dict(min_score=0.4),
    "min_score=0.5": dict(min_score=0.5),
    "mi30+c2+top3": dict(min_inliers=30, confirm_count=2, top_k=3),
    "mi30+c2+top3+s0.3": dict(min_inliers=30, confirm_count=2, top_k=3, min_score=0.3),
    "mi30+c2+top2+s0.3": dict(min_inliers=30, confirm_count=2, top_k=2, min_score=0.3),
    "mi20+c2+top3+s0.3": dict(min_inliers=20, confirm_count=2, top_k=3, min_score=0.3),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--encoder", choices=["bow", "vlad"], default="bow")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--min-gt-path", type=float, default=5.0, help="a revisit needs this many metres of true travel in between")
    ap.add_argument("--vocab-kfs", type=int, default=40, help="train the vocabulary on the first N keyframes (0 = all)")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), help="names from VARIANTS (default: all)")
    args = ap.parse_args()

    import torch
    import vo
    from slam_core.io.tum import load_groundtruth
    from slam_core.loop_closure import BagOfWords, KeyframeGraph, LoopClosureDetector, LoopParams, Vlad
    from slam_core.types.pose import from_tum, inv_T, rotation_angle

    p = vo.Params(loop_min_inliers=15)          # the verifier returns anything with 15+ inliers; the detector filters
    V = vo.VO(vo.load_xfeat(p.top_k), p)
    V.refs, V.K, V.dist = torch.load(args.cache, weights_only=False)
    kfs = V.keyframes
    print(f"{len(kfs)} keyframes")

    gt = load_groundtruth(args.dataset)
    G = [from_tum(gt[int(np.argmin(np.abs(gt[:, 0] - k.stamp))), 1:8]) for k in kfs]

    def rel_err(i, j, T):
        E = inv_T(inv_T(G[i]) @ G[j]) @ T
        return np.linalg.norm(E[:3, 3]), np.degrees(rotation_angle(E[:3, :3]))

    graph = KeyframeGraph()
    for k in kfs:
        graph.add(k)
    # distance the camera really travelled (ground truth) up to each keyframe: a camera standing still revisits nothing
    arc = np.concatenate([[0.0], np.cumsum([np.linalg.norm(G[i][:3, 3] - G[i - 1][:3, 3]) for i in range(1, len(kfs))])])
    revisit = []
    for k in kfs:
        for j in range(k.id):
            if graph.hops(j, k.id) < 20 or arc[k.id] - arc[j] < args.min_gt_path:
                continue
            T = inv_T(G[j]) @ G[k.id]
            if np.linalg.norm(T[:3, 3]) < 1.0 and np.degrees(rotation_angle(T[:3, :3])) < 40:
                revisit.append(k.id)
                break
    events, cur = [], []
    for i in revisit:
        if cur and i - cur[-1] > 3:
            events.append((cur[0], cur[-1]))
            cur = []
        cur.append(i)
    if cur:
        events.append((cur[0], cur[-1]))
    print(f"revisit keyframes: {len(revisit)}, events: {len(events)}")

    n_fit = args.vocab_kfs or len(kfs)
    enc = Vlad(args.k or 64) if args.encoder == "vlad" else BagOfWords(args.k or 500)
    enc.fit([k.descriptors for k in kfs[:n_fit]])
    descs = [enc.encode(k.descriptors) for k in kfs]

    memo, calls, cost = {}, [0], []

    def verify(old_id, new_id):
        calls[0] += 1
        if (old_id, new_id) not in memo:
            t = time.time()
            memo[(old_id, new_id)] = V._verify_loop(old_id, new_id)
            cost.append(time.time() - t)
        return memo[(old_id, new_id)]

    print(f"\n{'variant':22s} {'loops':>5s} {'correct':>7s} {'precision':>9s} {'events closed':>14s} {'coverage':>9s} {'verifies':>9s}  rejections")
    for name in args.variants:
        det = LoopClosureDetector(verify, LoopParams(**VARIANTS[name]))
        calls[0] = 0
        active, t0, lat = {}, time.time(), []
        for kf, d in zip(kfs, descs):
            t = time.time()
            up = det.process(kf, d)
            lat.append(time.time() - t)
            for L in up.added:
                active[L.id] = L
            for i in up.retracted:
                active.pop(i, None)
        loops = list(active.values())
        ok = [rel_err(L.old_id, L.new_id, L.T_old_new) for L in loops]
        good = [L for L, (t, r) in zip(loops, ok) if t < 0.10 and r < 5.0]
        closed = [e for e in events if any(e[0] - 5 <= L.new_id <= e[1] + 5 for L in good)]
        covered = sum(1 for i in revisit if any(abs(i - L.new_id) <= 5 for L in good))
        rej = Counter(r.reason for r in det.rejected)
        prec = f"{len(good) / len(loops):.2f}" if loops else "-"
        print(f"{name:22s} {len(loops):5d} {len(good):7d} {prec:>9s} {len(closed):>6d}/{len(events):<7d} "
              f"{covered / max(len(revisit), 1):9.2f} {calls[0]:9d}  gate {rej['gate']}, unconfirmed {rej['unconfirmed']}, "
              f"inconsistent {rej['inconsistent']}, verify {rej['verify']}")
        if name == args.variants[0]:
            print(f"  (uncached run: {1000 * np.mean(lat):.0f} ms per keyframe on average, {1000 * np.percentile(lat, 95):.0f} ms p95, "
                  f"{len(memo)} distinct pairs verified, {1000 * np.mean(cost):.0f} ms per verification)")
        else:
            print(f"      ~{1000 * calls[0] * np.mean(cost) / len(kfs):.0f} ms per keyframe (verifications x average cost)")


if __name__ == "__main__":
    main()
