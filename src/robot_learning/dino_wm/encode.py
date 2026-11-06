"""Encode a recorded peg_insert dataset (LeRobot, one episode per data file) with frozen DINOv2: every 5th frame
(5 Hz), the 5 Hz arm states, the 25 Hz actions grouped per 0.2 s, and a 96x128 thumbnail per frame for the decoder.

  python encode.py <dataset dir> <out .pt> --tok 16
"""
import argparse
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import ACT_DIM, STRIDE, DinoEncoder

p = argparse.ArgumentParser()
p.add_argument("dataset", type=Path)
p.add_argument("out")
p.add_argument("--tok", type=int, required=True, help="token grid per frame: 16 (all patches) or 8 (pooled)")
p.add_argument("--camera", default="ego")
args = p.parse_args()

enc, dev = DinoEncoder(args.tok), "cuda"
files = sorted((args.dataset / "data").glob("chunk-*/*.parquet"))
if not files:
    raise FileNotFoundError(f"no data/chunk-*/*.parquet under {args.dataset}")
Z, IMG, S, A, EP = [], [], [], [], []
for k, f in enumerate(files):
    d = pd.read_parquet(f)
    if d["episode_index"].nunique() != 1:
        raise ValueError(f"{f} holds {d['episode_index'].nunique()} episodes; this script expects one per file")
    video = args.dataset / "videos" / f"observation.images.{args.camera}" / f.parent.name / f.name.replace(".parquet", ".mp4")
    frames = [fr.to_ndarray(format="rgb24") for fr in av.open(str(video)).decode(video=0)]
    st, ac = np.stack(d["observation.state"]), np.stack(d["action"])
    n = min(len(frames), len(d)) // STRIDE * STRIDE
    x = torch.from_numpy(np.stack(frames[:n:STRIDE])).to(dev).permute(0, 3, 1, 2).float() / 255
    Z.append(enc(x).cpu())
    IMG.append((F.interpolate(x, (96, 128), mode="bilinear", antialias=True) * 255).byte().cpu())
    S.append(torch.from_numpy(st[:n:STRIDE]).float())
    A.append(torch.from_numpy(ac[:n].reshape(-1, STRIDE * ACT_DIM)).float())  # actions over the next 0.2 s
    EP.append(torch.full((n // STRIDE,), k))
    if k % 20 == 0:
        print(f"{k + 1}/{len(files)} episodes", flush=True)
torch.save({"z": torch.cat(Z), "img": torch.cat(IMG), "state": torch.cat(S), "act": torch.cat(A), "ep": torch.cat(EP),
            "tok": args.tok}, args.out)
print("saved", args.out, tuple(torch.cat(Z).shape))
