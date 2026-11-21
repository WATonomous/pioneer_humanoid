"""TUM RGB-D sequences -> one npz of real RGB-D frames for distill.py --real_rgbd.

Pairs each colour frame with the nearest depth frame (<= 20 ms apart, as TUM's associate.py), keeps every
--every-th pair, resizes 640x480 -> 320x240 (the bench's render size, same 4:3), and stores depth in millimetres
(TUM depth PNGs are metres x 5000; 0 = no reading, kept as 0).

    python3 -I convert_tum.py <raw dir> <out.npz> [--every 2]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def read_list(p):
    out = []
    for line in open(p):
        if line.startswith("#") or not line.strip():
            continue
        t, f = line.split()[:2]
        out.append((float(t), f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=2)
    args = ap.parse_args()
    rgbs, deps, seq_ids, names = [], [], [], []
    for si, seq in enumerate(sorted(p for p in Path(args.raw).iterdir() if p.is_dir())):
        rgb_l, dep_l = read_list(seq / "rgb.txt"), read_list(seq / "depth.txt")
        dt = np.array([t for t, _ in dep_l])
        pairs = []
        for t, f in rgb_l:
            j = int(np.argmin(np.abs(dt - t)))
            if abs(dt[j] - t) <= 0.02:
                pairs.append((f, dep_l[j][1]))
        pairs = pairs[::args.every]
        for fr, fd in pairs:
            c = cv2.imread(str(seq / fr), cv2.IMREAD_COLOR)
            d = cv2.imread(str(seq / fd), cv2.IMREAD_UNCHANGED)
            if c is None or d is None or d.dtype != np.uint16:
                continue
            rgbs.append(cv2.resize(c, (320, 240), interpolation=cv2.INTER_AREA)[:, :, ::-1])
            deps.append((cv2.resize(d, (320, 240), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 5.0)
                        .round().astype(np.uint16))
            seq_ids.append(si)
        names.append(seq.name)
        print(f"{seq.name}: {len(pairs)} frames", flush=True)
    rgb, dep = np.stack(rgbs), np.stack(deps)
    valid = (dep > 0).mean()
    np.savez(args.out, rgb=rgb, depth=dep, seq=np.array(seq_ids), names=np.array(names))
    print(f"saved {args.out}: {len(rgb)} frames from {len(names)} sequences, {valid:.0%} of depth pixels valid, "
          f"depth median {np.median(dep[dep > 0]) / 1000:.2f} m")


if __name__ == "__main__":
    sys.exit(main())
