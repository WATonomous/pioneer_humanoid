"""Does an RGB-D encoder use both inputs? Centred cosine between its features on the real input and on the input
with one modality wiped (depth = 0 m / RGB = black). ~1.0 = that input is ignored. Also the same stability numbers
as pioneer_viz (clean vs shifted). Writes results/modality_<name>.json.

    .venv/bin/python modality_check.py pioneernet_v2
"""
import glob
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

import common as C
import encoders as E
from pioneer_viz import feats

name = sys.argv[1]
z = np.load(sorted(glob.glob(str(C.DATA / "test" / "ep_*.npz")))[0])
enc = E.load(name)
out = {}
for cam in C.CAMS:
    r = torch.from_numpy(z[f"clean_{cam}_rgb"][:96]).cuda()
    d = torch.from_numpy(z[f"{cam}_depth"][:96].astype(np.float32)).cuda() / 1000
    with torch.no_grad():
        a = E.run(enc, r, d)[1].float()
        nod = E.run(enc, r, torch.zeros_like(d))[1].float()
        norgb = E.run(enc, torch.zeros_like(r), d)[1].float()
    mu = a.mean((0, 2, 3), keepdim=True)
    cc = lambda b: float(F.cosine_similarity(a - mu, b - mu, dim=1).mean())
    out[cam] = {"depth_zeroed": cc(nod), "rgb_zeroed": cc(norgb)}
    clean = feats(enc, z, cam, "clean")
    mu = clean.mean((0, 2, 3), keepdim=True)
    for cond in ("light", "color", "noise", "light+noise"):
        out[cam][cond] = float(F.cosine_similarity(clean - mu, feats(enc, z, cam, cond) - mu, dim=1).mean())
json.dump(out, open(C.OUT / f"modality_{name}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
