"""Frozen-encoder probe benchmark.

For every config (a set of (encoder, camera) streams, e.g. DINOv2-RGB on ego + DeFM-depth on wrist), we
  1. run the frozen encoder(s) once over all train/test images and cache the patch-token grids (fp16, CPU),
  2. train an *attentive probe* (learned queries cross-attending over all streams' tokens, like ACT's
     transformer reads its ResNet feature map) to regress the scene geometry + classify the task phase,
  3. also fit a ridge *linear probe* on the global vectors (CLS / avg-pool),
  4. evaluate on held-out episodes: clean, and under shift (RGB: light / colour; depth: D455 / stereo noise).

    .venv/bin/python probe.py --configs r18_imnet@ego+wrist_left defm_s14@ego+wrist_left ...
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
import dataset as D
import encoders as E

TARGET_KEYS = ("tcp", "peg", "hole_xy", "peg_minus_tcp", "peg_minus_hole_xy")
TRAIN_EVERY = 2   # train probes on every 2nd image (6.25 Hz): neighbouring frames are near-duplicates


def parse_config(s: str):
    """'dinov2_s14@ego+wrist_left,defm_s14@ego' -> [(enc, cam), ...]"""
    streams = []
    if s == "none":
        return streams
    for part in s.split(","):
        enc, cams = part.split("@")
        streams += [(enc, c) for c in cams.split("+")]
    return streams


@torch.no_grad()
def extract(enc, rgb: np.ndarray | None, depth: np.ndarray | None, noise_cam: str | None = None, seed: int = 0, bs: int = 64):
    """rgb uint8 (N,H,W,3) and/or depth uint16 mm (N,H,W), as the encoder needs.
    Returns (global (N, Cg) fp16 tensor, tokens (N, C, h, w) as an int8 E.Q8 cache) on CPU."""
    gs = ts = None   # preallocated on the first batch (torch.cat of chunks would double peak RAM)
    if getattr(enc, "hw", (224,))[0] > 300:
        bs = 16          # 448 px ViT: 4x the tokens, attention memory ~16x
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n = len(rgb) if rgb is not None else len(depth)
    for i in range(0, n, bs):
        r = torch.from_numpy(rgb[i:i + bs]).cuda() if rgb is not None else None
        d = None
        if depth is not None:
            d = torch.from_numpy(depth[i:i + bs].astype(np.float32)).cuda() / 1000.0
            if noise_cam is not None:
                d = E.NOISE[noise_cam](d, gen)
        g, t = E.run(enc, r, d)
        if gs is None:
            gs = torch.empty((n, *g.shape[1:]), dtype=torch.float16)
            ts = E.Q8(n, tuple(t.shape[1:]))   # int8 tokens: DeFM ViT-L's fp16 grids alone were ~34 GB
        gs[i:i + len(g)] = g.float().half().cpu()
        ts.put(i, t)
    return gs, ts


class AttentiveProbe(nn.Module):
    """Per-stream linear projection + learned 2D position embedding -> K learned queries cross-attend
    (2 blocks) -> MLP heads. Small on purpose: it measures what is linearly-accessible-after-pooling."""

    def __init__(self, stream_shapes, n_out, n_cls, d=256, q=8, heads=8, blocks=2):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(c), nn.Linear(c, d)) for c, h, w in stream_shapes])
        self.pos = nn.ParameterList([nn.Parameter(torch.randn(1, h * w, d) * 0.02) for c, h, w in stream_shapes])
        self.queries = nn.Parameter(torch.randn(1, q, d) * 0.02)
        self.blocks = nn.ModuleList([nn.TransformerDecoderLayer(d, heads, 4 * d, 0.0, batch_first=True, norm_first=True)
                                     for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(q * d), nn.Linear(q * d, 512), nn.GELU(), nn.Linear(512, n_out + n_cls))
        self.n_out = n_out

    def forward(self, toks):
        mem = torch.cat([p(t.flatten(2).transpose(1, 2)) + pe for t, p, pe in zip(toks, self.proj, self.pos)], 1)
        x = self.queries.expand(len(mem), -1, -1)
        for b in self.blocks:
            x = b(x, mem)
        y = self.head(x.flatten(1))
        return y[:, :self.n_out], y[:, self.n_out:]


def stack_targets(d):
    t = D.targets(d)
    return np.concatenate([t[k] for k in TARGET_KEYS], 1).astype(np.float32)


SLICES, _o = {}, 0
for _k, _n in zip(TARGET_KEYS, (3, 3, 2, 3, 2)):
    SLICES[_k] = slice(_o, _o + _n)
    _o += _n


def metrics(pred_y, pred_c, y, phase):
    out = {}
    for k, s in SLICES.items():
        err = np.linalg.norm(pred_y[:, s] - y[:, s], axis=1) * 1000
        out[f"{k}_mm"] = float(np.sqrt((err ** 2).mean()))           # RMS Euclidean error, mm
    # insertion-critical: peg-vs-hole offset only where the peg is lifted / over the hole (phase 2-3)
    m = (phase >= 2) & (phase <= 3)
    if m.any():
        err = np.linalg.norm(pred_y[m, SLICES["peg_minus_hole_xy"]] - y[m, SLICES["peg_minus_hole_xy"]], axis=1) * 1000
        out["align_mm"] = float(np.sqrt((err ** 2).mean()))
    out["phase_acc"] = float((pred_c.argmax(1) == phase).mean())
    s = SLICES["peg_minus_hole_xy"]
    for ph in range(5):   # where does the peg-to-hole error come from?
        m = phase == ph
        if m.sum() > 20:
            out[f"peg_hole_ph{ph}_mm"] = float(np.sqrt((np.linalg.norm(pred_y[m, s] - y[m, s], axis=1) ** 2).mean()) * 1000)
    err = np.linalg.norm(pred_y[:, s] - y[:, s], axis=1) * 1000
    out["peg_hole_p50_mm"], out["peg_hole_p95_mm"], out["peg_hole_max_mm"] = \
        float(np.median(err)), float(np.percentile(err, 95)), float(err.max())
    return out


PROBE_KW = {}   # --width/--queries/--blocks


def train_probe(tr_toks, ytr, ctr, epochs=40, bs=128, lr=3e-4, seed=0, groups=None, moddrop=0.0):
    """groups: per stream, which input it reads ("rgb" / "depth"); with ``moddrop`` > 0 each training sample loses
    all streams of one input (prob moddrop/2 each), so the probe cannot lean on one modality alone."""
    torch.manual_seed(seed)
    shapes = [tuple(t.shape[1:]) for t in tr_toks]
    mu, sd = ytr.mean(0), ytr.std(0) + 1e-6
    model = AttentiveProbe(shapes, ytr.shape[1], 5, **PROBE_KW).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    n = len(ytr)
    steps = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    yt = torch.from_numpy((ytr - mu) / sd)
    ct = torch.from_numpy(ctr)
    free = torch.cuda.mem_get_info()[0]   # the GPU is shared (other people's jobs): only cache what fits
    gtoks = [t.gpu_copy() for t in tr_toks] if sum(t.nbytes() for t in tr_toks) < min(6e9, 0.5 * free) else tr_toks
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            toks = [t.get(idx) for t in gtoks]
            if moddrop > 0 and groups and len(set(groups)) > 1:
                u = torch.rand(len(idx), device="cuda")
                drop = {"rgb": u < moddrop / 2, "depth": (u >= moddrop / 2) & (u < moddrop)}
                toks = [t * (~drop[g]).float().view(-1, 1, 1, 1) if g in drop else t for t, g in zip(toks, groups)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                py, pc = model(toks)
            loss = F.smooth_l1_loss(py.float(), yt[idx].cuda(), beta=0.05) + 0.2 * F.cross_entropy(pc.float(), ct[idx].cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    del gtoks
    model.eval()
    return model, mu, sd


@torch.no_grad()
def predict(model, mu, sd, toks, bs=256):
    ys, cs = [], []
    for i in range(0, len(toks[0]), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            py, pc = model([t.get(torch.arange(i, min(i + bs, len(t)))) for t in toks])
        ys.append(py.float().cpu().numpy())
        cs.append(pc.float().cpu().numpy())
    return np.concatenate(ys) * sd + mu, np.concatenate(cs)


def ridge(gtr, ytr, gte_list, lam=1.0):
    """Closed-form ridge on standardised global features (GPU, fp32)."""
    X = gtr.cuda().float()
    m, s = X.mean(0), X.std(0) + 1e-6
    X = (X - m) / s
    Y = torch.from_numpy(ytr).cuda()
    ym = Y.mean(0)
    A = X.T @ X + lam * len(X) * 1e-3 * torch.eye(X.shape[1], device="cuda")
    Wt = torch.linalg.solve(A, X.T @ (Y - ym))
    return [(((g.cuda().float() - m) / s) @ Wt + ym).cpu().numpy() for g in gte_list]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--train_noise", action="store_true", help="train on noisy depth too (sensor-matched training)")
    ap.add_argument("--max_eps", type=int, default=None, help="limit episodes (smoke tests)")
    ap.add_argument("--moddrop", type=float, default=0.0, help="modality dropout prob for multi-input configs")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--lazy_test", action="store_true", help="extract/evaluate/free one test condition at a time (big encoders)")
    args = ap.parse_args()
    PROBE_KW.update(d=args.width, q=args.queries, blocks=args.blocks)

    t0 = time.time()
    tr = D.load_split("train", every=TRAIN_EVERY, max_eps=args.max_eps)
    te = {v: D.load_split("test", every=2, variant=v, depth=(v == "clean"), max_eps=args.max_eps)
          for v in ("clean", "light", "color")}
    if D.episode_files("test_cam"):   # held-out camera-pose shift (own RGB + depth; same seeds and labels)
        te["camera"] = D.load_split("test_cam", every=2, variant="camera", max_eps=args.max_eps)
        assert np.allclose(stack_targets(te["camera"]), stack_targets(te["clean"])), "test_cam must match test seeds"
    ytr, ctr = stack_targets(tr), tr["phase"]
    yte, cte = stack_targets(te["clean"]), te["clean"]["phase"]
    print(f"data: train {len(ytr)} imgs / {len(tr['streams'])} eps, test {len(yte)} imgs / {len(te['clean']['streams'])} eps "
          f"[{time.time()-t0:.0f}s]", flush=True)

    cache: dict = {}

    def feats(enc_name, cam, split, cond):
        """cond: clean | light | color (RGB appearance shift) | noise (depth sensor noise). A condition that
        does not apply to an encoder's input falls back to the clean input."""
        enc = encs[enc_name]
        need = E.needs(enc)
        rgb_var = cond if (cond in ("light", "color", "dark", "camera") and "rgb" in need) else "clean"
        noisy = "depth" in need and (cond == "noise" or (split == "train" and args.train_noise))
        key = (enc_name, cam, split, rgb_var if "rgb" in need else "-", noisy, cond == "camera")
        if key not in cache:
            src_rgb = tr if split == "train" else te["clean" if rgb_var == "dark" else rgb_var]
            src_d = tr if split == "train" else te["camera" if cond == "camera" else "clean"]
            rgb_in = src_rgb[f"{cam}_rgb"] if "rgb" in need else None
            if rgb_in is not None and rgb_var == "dark":
                rgb_in = np.zeros_like(rgb_in)
            cache[key] = extract(enc, rgb_in,
                                 src_d[f"{cam}_depth"] if "depth" in need else None,
                                 noise_cam=cam if noisy else None, seed=1 if split == "train" else 2)
        return cache[key]

    results = []
    for cfg in args.configs:
        streams = parse_config(cfg)
        encs = {}
        for e, _ in streams:
            if e not in encs:
                encs[e] = E.load(e)
        mods = set().union(*[E.needs(encs[e]) for e, _ in streams])
        conds = ["clean"] + (["light", "color"] if "rgb" in mods else []) + (["noise"] if "depth" in mods else []) + \
            (["dark"] if {"rgb", "depth"} <= mods else []) + \
            (["camera"] if "camera" in te else [])   # dark: colour camera dead (all-black RGB), depth still there
        t1 = time.time()
        trf = [feats(e, c, "train", "clean") for e, c in streams]
        rec = {"config": cfg, "train_noise": args.train_noise, "moddrop": args.moddrop, "probe": dict(PROBE_KW, epochs=args.epochs),
               "streams": streams,
               "desc": " + ".join(sorted({encs[e].desc for e, _ in streams}))}
        # linear probe on concatenated global vectors
        gtr = torch.cat([f[0] for f in trf], 1)
        for cond in conds:
            rec[f"attn/{cond}"] = []
        groups = [E.needs(encs[e]).pop() if len(E.needs(encs[e])) == 1 else "rgbd" for e, _ in streams]
        if args.lazy_test:
            # 2026-11-16: big-token encoders (C-RADIOv4-SO400M, 1152-d) OOM'd at 56 GB caching every test condition up
            # front. Same numbers, lower peak: train every seed's probe first, then extract -> evaluate -> free one
            # test condition at a time.
            probes = [train_probe([f[1] for f in trf], ytr, ctr, epochs=args.epochs, seed=s, groups=groups,
                                  moddrop=args.moddrop) for s in range(args.seeds)]
            for cond in conds:
                tf = [feats(e, c, "test", cond) for e, c in streams]
                p = ridge(gtr, ytr, [torch.cat([f[0] for f in tf], 1)])[0]
                rec[f"linear/{cond}"] = metrics(p, np.zeros((len(p), 5)), yte, cte)
                for model, mu, sd in probes:
                    py, pc = predict(model, mu, sd, [f[1] for f in tf])
                    rec[f"attn/{cond}"].append(metrics(py, pc, yte, cte))
                del tf
                for k in [k for k in cache if k[2] == "test"]:
                    del cache[k]
            del probes
        else:
            tef = {cond: [feats(e, c, "test", cond) for e, c in streams] for cond in conds}
            lin = ridge(gtr, ytr, [torch.cat([f[0] for f in tef[cond]], 1) for cond in conds])
            for cond, p in zip(conds, lin):
                rec[f"linear/{cond}"] = metrics(p, np.zeros((len(p), 5)), yte, cte)
            # attentive probe, several seeds
            for s in range(args.seeds):
                model, mu, sd = train_probe([f[1] for f in trf], ytr, ctr, epochs=args.epochs, seed=s,
                                            groups=groups, moddrop=args.moddrop)
                for cond in conds:
                    py, pc = predict(model, mu, sd, [f[1] for f in tef[cond]])
                    rec[f"attn/{cond}"].append(metrics(py, pc, yte, cte))
                del model
        for cond in conds:
            runs = rec[f"attn/{cond}"]
            rec[f"attn/{cond}"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]} | \
                                  {k + "_std": float(np.std([r[k] for r in runs])) for k in runs[0]}
        rec["secs"] = time.time() - t1
        results.append(rec)
        a = rec["attn/clean"]
        print(f"{cfg:55s} attn: tcp {a['tcp_mm']:.1f} peg {a['peg_mm']:.1f} hole {a['hole_xy_mm']:.1f} "
              f"peg-tcp {a['peg_minus_tcp_mm']:.1f} peg-hole {a['peg_minus_hole_xy_mm']:.1f} align {a.get('align_mm', 0):.1f} "
              f"phase {a['phase_acc']:.3f} | " +
              " ".join(f"{c}: peg-hole {rec[f'attn/{c}']['peg_minus_hole_xy_mm']:.1f}" for c in conds[1:]) +
              f" | lin peg-tcp {rec['linear/clean']['peg_minus_tcp_mm']:.1f} [{rec['secs']:.0f}s]", flush=True)
        with open(C.OUT / f"{args.tag}.json", "w") as f:
            json.dump(results, f, indent=1)
        # free features of encoders no later config needs
        later = {e for c2 in args.configs[args.configs.index(cfg) + 1:] for e, _ in parse_config(c2)}
        for k in [k for k in cache if k[0] not in later]:
            del cache[k]
        del encs
        torch.cuda.empty_cache()
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
