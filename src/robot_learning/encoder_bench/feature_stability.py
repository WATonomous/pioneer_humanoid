"""Feature stability: how much an encoder's tokens move when the scene changes. For held-out layouts, render the
clean scene and each scenario (scenarios.py) from both cameras, encode both, and report the mean per-token cosine
similarity between the clean and the shifted tokens (1 = unchanged). No policy involved.

    BENCH_ROBOT=8dof MUJOCO_GL=egl .venv/bin/python feature_stability.py --encoders pioneernet_v5 pioneernet_v6
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch, torch.nn.functional as F
import common as C, encoders as E, scenarios as S
from scenario_examples import render_all


def tokens(enc, rgb, dep):
    r = torch.from_numpy(np.stack(rgb)).cuda()
    d = torch.from_numpy(np.stack(dep).astype(np.float32)).cuda()
    if enc.modality == "rgbd":
        x = enc.prep(r, d)
    elif enc.modality == "depth":
        x = enc.prep(d)
    else:
        x = enc.prep(r)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, t = enc(x)
    return t.float().flatten(2)        # (N, D, tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--seeds", default="7000:7020")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s0, s1 = map(int, a.seeds.split(":"))
    shots = [render_all(s) for s in range(s0, s1)]
    res = {}
    for name in a.encoders:
        enc = E.load(name)
        res[name] = {}
        for sc in S.SCENARIOS:
            sims = []
            for cam in C.CAMS:
                cr = [o["clean", cam][0] for o in shots]; cd = [o["clean", cam][1] for o in shots]
                sr = [o[sc, cam][0] for o in shots]; sd = [o[sc, cam][1] for o in shots]
                tc, ts = tokens(enc, cr, cd), tokens(enc, sr, sd)
                sims.append(F.cosine_similarity(tc, ts, dim=1).mean().item())
            res[name][sc] = float(np.mean(sims))
        print(name, " ".join(f"{k} {v:.3f}" for k, v in res[name].items()), flush=True)
        del enc; torch.cuda.empty_cache()
    out = a.out or str(C.OUT / "feature_stability.json")
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
