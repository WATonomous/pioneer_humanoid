#!/usr/bin/env python3
"""How good is loop closure on its own, on a TUM sequence? (Precision / recall against the ground truth.)

    python slam_tools/eval_loops_pr.py DATASET_DIR [--step 2]

Runs VO once (no loops) to get keyframes with features and depth. Then for every keyframe it asks the loop module's
retrieval for candidates among older keyframes and verifies them with the real XFeat + PnP verifier. Ground truth
decides what is right:
  - a keyframe HAS A REVISIT if some keyframe at least --min-hops odometry edges earlier was within 1 m and 40 deg
    of it (so the views overlap);
  - a verified loop is CORRECT if its relative pose is within 10 cm and 5 deg of the ground truth.
It reports retrieval recall@k (is a true revisit among the candidates?) and, sweeping the inlier threshold,
precision and recall of the best verified loop per keyframe. The headline number in the literature is max recall
at 100% precision. The detector's gate / confirmation / consistency are NOT part of this sweep (see run_tum_loop.py).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "slam_core"))
sys.path.insert(0, os.path.join(HERE, "..", "prototype"))

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--min-hops", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=0.2)
    ap.add_argument("--encoder", choices=["vlad", "bow"], default="vlad", help="global descriptor: VLAD (k=64) or bag of words (k=500)")
    ap.add_argument("--k", type=int, default=0, help="vocabulary size (0 = 64 for vlad, 500 for bow)")
    ap.add_argument("--cache-only", action="store_true", help="save the VO cache and stop")
    ap.add_argument("--group", type=int, default=0, help="retrieval group re-ranking window (0 = off)")
    ap.add_argument("--cache", default="", help="file to save/load the VO keyframes (skips re-running VO)")
    ap.add_argument("--set", nargs="*", default=[], help="VO Params overrides, e.g. loop_guided=1")
    ap.add_argument("--vocab-kfs", type=int, default=0, help="train the vocabulary on the first N keyframes (0 = all)")
    args = ap.parse_args()

    import vo
    from slam_core.io.tum import TUMDataset, load_groundtruth
    from slam_core.loop_closure import BagOfWords, DescriptorDatabase, KeyframeGraph, Vlad, group_rerank
    from slam_core.types.pose import from_tum, inv_T, rotation_angle

    over = dict(kv.split("=") for kv in args.set)
    fields = vo.Params.__dataclass_fields__
    over = {k: fields[k].type(v) for k, v in over.items()}
    p = vo.Params(loop_mode="radius", loop_min_time=1e12, **over)       # VO only
    V = vo.VO(vo.load_xfeat(p.top_k), p)
    if args.cache and os.path.exists(args.cache):
        import torch
        V.refs, V.K, V.dist = torch.load(args.cache, weights_only=False)
    else:
        data = TUMDataset(args.dataset)
        for i in range(0, len(data), args.step):
            V.process(data[i])
        if args.cache:
            import torch
            torch.save((V.refs, V.K, V.dist), args.cache)
    if args.cache_only:
        print(f"cached {len(V.refs)} keyframes")
        return
    kfs = V.keyframes
    print(f"{len(kfs)} keyframes")

    gt = load_groundtruth(args.dataset)
    pose = lambda t: from_tum(gt[int(np.argmin(np.abs(gt[:, 0] - t))), 1:8])
    G = [pose(k.stamp) for k in kfs]

    def rel_err(i, j, T):                # error of T_i_j against ground truth
        E = inv_T(inv_T(G[i]) @ G[j]) @ T
        return np.linalg.norm(E[:3, 3]), np.degrees(rotation_angle(E[:3, :3]))

    graph = KeyframeGraph()
    for k in kfs:
        graph.add(k)

    n_fit = args.vocab_kfs or len(kfs)
    vlad = Vlad(args.k or 64) if args.encoder == "vlad" else BagOfWords(args.k or 500)
    vlad.fit([k.descriptors for k in kfs[:n_fit]])
    db = DescriptorDatabase()

    rows = []         # (has_revisit, retrieved_true_revisit, best_inliers, best_correct)
    for k in kfs:
        desc = vlad.encode(k.descriptors)
        cands = [] if len(db) == 0 else group_rerank(
            db.query(desc, k=4 * args.top_k, min_score=args.min_score, exclude=lambda j: graph.hops(j, k.id) < args.min_hops),
            args.group)[:args.top_k]
        db.add(k.id, desc)
        true_olds = []
        for j in range(k.id):
            if graph.hops(j, k.id) < args.min_hops:
                continue
            T_gt = inv_T(G[j]) @ G[k.id]
            if np.linalg.norm(T_gt[:3, 3]) < 1.0 and np.degrees(rotation_angle(T_gt[:3, :3])) < 40:
                true_olds.append(j)
        best_n, best_ok = 0, False
        for old_id, _ in cands:
            v = V._verify_loop(old_id, k.id)
            if v is not None and v.n_inliers > best_n:
                t, r = rel_err(old_id, k.id, v.T_old_new)
                best_n, best_ok = v.n_inliers, bool(t < 0.10 and r < 5.0)
        rows.append((bool(true_olds), any(o in {c for c, _ in cands} for o in true_olds), best_n, best_ok))

    R = np.array(rows, dtype=float)
    has = R[:, 0] == 1
    print(f"keyframes with a true revisit: {int(has.sum())} of {len(kfs)}")
    if has.sum():
        print(f"retrieval recall@{args.top_k}: {R[has, 1].mean():.2f}  (a true revisit is among the candidates)")
    print(f"\n{'min inliers':>11s} {'accepted':>9s} {'correct':>8s} {'precision':>10s} {'recall':>7s}")
    best_recall = 0.0
    for tau in (10, 15, 20, 30, 40, 60, 80, 100, 150):
        acc = R[:, 2] >= tau
        ok = acc & (R[:, 3] == 1)
        prec = ok.sum() / acc.sum() if acc.sum() else float("nan")
        rec = (ok & has).sum() / has.sum() if has.sum() else float("nan")
        if acc.sum() and ok.sum() == acc.sum():
            best_recall = max(best_recall, rec)
        print(f"{tau:11d} {int(acc.sum()):9d} {int(ok.sum()):8d} {prec:10.2f} {rec:7.2f}")
    print(f"\nmax recall at 100% precision: {best_recall:.2f}")


if __name__ == "__main__":
    main()
