"""One NYU Depth v2 shard (sayakpaul/nyu_depth_v2 on Hugging Face: a tar of .h5 frames, each 'rgb' (3,480,640) uint8 and
'depth' (480,640) float metres) streamed on stdin -> npz of real RGB-D frames for distill.py --real_rgbd.
Nothing is written to disk but the output. Keeps every --every-th frame (the shards are video: neighbours are near-copies).

    curl -sL <shard url> | python3 -I convert_nyu.py <out.npz> [--every 3]
"""
import argparse
import io
import sys
import tarfile

import cv2
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--max", type=int, default=0, help="stop after this many kept frames (tests)")
    args = ap.parse_args()
    rgbs, deps, k = [], [], 0
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as tar:
        for m in tar:
            if not (m.isfile() and m.name.endswith(".h5")):
                continue
            k += 1
            if (k - 1) % args.every:
                continue
            with h5py.File(io.BytesIO(tar.extractfile(m).read()), "r") as f:
                rgb = np.transpose(f["rgb"][()], (1, 2, 0))
                dep = f["depth"][()].astype(np.float32)
            rgbs.append(cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA))
            dep = cv2.resize(dep, (320, 240), interpolation=cv2.INTER_NEAREST)
            deps.append(np.clip(np.nan_to_num(dep) * 1000, 0, 65535).round().astype(np.uint16))
            if args.max and len(rgbs) >= args.max:
                break
    rgb, dep = np.stack(rgbs), np.stack(deps)
    np.savez(args.out, rgb=rgb, depth=dep)
    print(f"saved {args.out}: {len(rgb)} of {k} frames, {(dep > 0).mean():.0%} depth valid, "
          f"median {np.median(dep[dep > 0]) / 1000:.2f} m")


if __name__ == "__main__":
    sys.exit(main())
