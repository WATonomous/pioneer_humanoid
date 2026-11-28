"""SUN RGB-D (SUNRGBD.zip, read in place) -> npz of real RGB-D frames for distill.py --real_rgbd.

10,335 frames from 4 sensors (Kinect v1 and v2, Intel RealSense, Asus Xtion). Uses each frame's RAW sensor depth
(depth/, not the filled depth_bfx/). SUN RGB-D stores depth as a bit-rotated uint16: mm = (v >> 3) | (v << 13).
Each image is centre-cropped to 4:3 and resized to 320x240.

    python3 -I convert_sun.py <SUNRGBD.zip> <out.npz>
"""
import sys
import zipfile
from collections import defaultdict

import cv2
import numpy as np


def crop43(x):
    h, w = x.shape[:2]
    if w * 3 > h * 4:
        nw = h * 4 // 3
        return x[:, (w - nw) // 2:(w - nw) // 2 + nw]
    nh = w * 3 // 4
    return x[(h - nh) // 2:(h - nh) // 2 + nh]


def main():
    src, out = sys.argv[1], sys.argv[2]
    z = zipfile.ZipFile(src)
    frames = defaultdict(dict)                      # frame dir -> {"image": name, "depth": name}
    for n in z.namelist():
        parts = n.split("/")
        if len(parts) >= 3 and parts[-2] in ("image", "depth") and parts[-1].lower().endswith((".jpg", ".png")):
            frames["/".join(parts[:-2])][parts[-2]] = n
    rgbs, deps, sensors = [], [], []
    for fdir, f in sorted(frames.items()):
        if "image" not in f or "depth" not in f:
            continue
        c = cv2.imdecode(np.frombuffer(z.read(f["image"]), np.uint8), cv2.IMREAD_COLOR)
        d = cv2.imdecode(np.frombuffer(z.read(f["depth"]), np.uint8), cv2.IMREAD_UNCHANGED)
        if c is None or d is None or d.dtype != np.uint16:
            continue
        d = ((d >> 3) | (d << 13)).astype(np.uint16)
        d[d > 10000] = 0                             # beyond 10 m: not a valid reading
        rgbs.append(cv2.resize(crop43(c), (320, 240), interpolation=cv2.INTER_AREA)[:, :, ::-1])
        deps.append(cv2.resize(crop43(d), (320, 240), interpolation=cv2.INTER_NEAREST))
        sensors.append(fdir.split("/")[1] if "/" in fdir else "?")
    rgb, dep = np.stack(rgbs), np.stack(deps)
    np.savez(out, rgb=rgb, depth=dep, sensor=np.array(sensors))
    s, c = np.unique(sensors, return_counts=True)
    print(f"saved {out}: {len(rgb)} frames, sensors {dict(zip(s, c))}, {(dep > 0).mean():.0%} depth valid, "
          f"median {np.median(dep[dep > 0]) / 1000:.2f} m")


if __name__ == "__main__":
    sys.exit(main())
