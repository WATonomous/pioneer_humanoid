#!/usr/bin/env python3
"""How long do the expensive parts of loop closure take on this machine? Run it on the GPU and again on the CPU.

    python slam_tools/bench_matching.py DATASET_DIR                       # uses the GPU if torch sees one
    CUDA_VISIBLE_DEVICES= python slam_tools/bench_matching.py DATASET_DIR # forces the CPU (set the variable empty)

Times, per call, after warm-up (median and 90th percentile, milliseconds):
  detect    XFeat keypoints + descriptors on one image
  match     LighterGlue matching of two images' features
  verify    one full loop candidate verification: match + depth lookup + PnP RANSAC (vo.estimate_pose), the unit the
            loop detector pays for every candidate, about 3 per keyframe with the default settings
Needs torch, kornia, tqdm, opencv, numpy; XFeat is fetched with torch.hub on first use.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "slam_core"))
sys.path.insert(0, os.path.join(HERE, "..", "prototype"))

import numpy as np  # noqa: E402


def stats(ms):
    return f"{np.median(ms):7.1f} ms median, {np.percentile(ms, 90):7.1f} ms p90  (n={len(ms)})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--frames", type=int, default=60, help="how many frames to time (spread over the sequence)")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    import torch
    import vo
    from slam_core.io.tum import TUMDataset

    dev = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    print(f"torch {torch.__version__}, running on {dev}, {os.cpu_count()} CPU threads")

    p = vo.Params()
    xfeat = vo.load_xfeat(p.top_k)
    data = TUMDataset(args.dataset)
    idx = np.linspace(0, len(data) - 6, args.frames).astype(int)
    frames = [data[i] for i in idx]
    pairs_with = [data[min(i + 5, len(data) - 1)] for i in idx]       # a second view a few frames later (overlapping)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    t_detect, t_match, t_verify = [], [], []
    for n, (f1, f2) in enumerate(zip(frames, pairs_with)):
        sync(); t = time.perf_counter()
        d1 = vo.detect(xfeat, f1.image, p.top_k)
        sync(); t1 = time.perf_counter() - t
        d2 = vo.detect(xfeat, f2.image, p.top_k)
        sync(); t = time.perf_counter()
        vo.match(xfeat, d1, d2)
        sync(); t2 = time.perf_counter() - t
        ref = vo.Ref(None, d1, f1.depth, f1.image)
        sync(); t = time.perf_counter()
        vo.estimate_pose(xfeat, ref, d2, f1.K, f1.dist, p, 15)
        sync(); t3 = time.perf_counter() - t
        if n >= args.warmup:
            t_detect.append(t1 * 1000)
            t_match.append(t2 * 1000)
            t_verify.append(t3 * 1000)
    print(f"detect  {stats(t_detect)}")
    print(f"match   {stats(t_match)}")
    print(f"verify  {stats(t_verify)}")
    print(f"=> about {3 * np.median(t_verify):.0f} ms of verification per keyframe at 3 candidates")


if __name__ == "__main__":
    main()
