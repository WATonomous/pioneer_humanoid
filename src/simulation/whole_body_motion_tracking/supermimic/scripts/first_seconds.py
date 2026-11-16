"""The first N seconds of each SuperMimic motion npz (the benchmark set: the first 20 s of every clip).

  python first_seconds.py <npz dir> <out dir> <seconds>
"""
import sys
from pathlib import Path

import numpy as np

src, dst, seconds = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
files = sorted(src.glob("*.npz"))
if not files:
    raise FileNotFoundError(f"no .npz in {src}")
for f in files:
    d = np.load(f)
    frames = d["joint_pos"].shape[0]
    keep = int(round(seconds * float(d["fps"][0])))
    # per-frame arrays are cut, everything else (fps, names) is copied
    np.savez(dst / f.name, **{k: d[k][:keep] if d[k].ndim and d[k].shape[0] == frames else d[k] for k in d.files})
print(f"{len(files)} clips -> {dst}")
