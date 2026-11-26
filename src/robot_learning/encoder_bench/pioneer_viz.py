"""PioneerNet vs its teachers under shift: figure + video + feature-stability numbers.

Each encoder's PCA colour basis is fitted on its CLEAN-input features and then reused on the shifted inputs,
so a region that keeps its colour = a feature that did not move under the shift. Stability is also measured
directly: mean cosine similarity between each patch feature on clean vs shifted input (1.0 = unchanged).

    .venv/bin/python pioneer_viz.py   # figures/pioneer_<cam>.png, figures/pioneer_video.mp4, results/stability.json
"""
from __future__ import annotations

import glob
import json
import subprocess

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as C  # noqa: E402
import encoders as E  # noqa: E402

PHASES = ["reach", "grasp", "carry", "align/insert", "inserted"]
# condition -> (rgb variant, noisy depth?)
CONDS = {"clean": ("clean", False), "light": ("light", False), "color": ("color", False), "noise": ("clean", True),
         "light+noise": ("light", True)}


@torch.no_grad()
def feats(enc, z, cam, cond, seed=0):
    var, noisy = CONDS[cond]
    need = E.needs(enc)
    rgb, dep = z[f"{var}_{cam}_rgb"], z[f"{cam}_depth"]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for i in range(0, len(rgb), 32):
        r = torch.from_numpy(rgb[i:i + 32]).cuda() if "rgb" in need else None
        d = None
        if "depth" in need:
            d = torch.from_numpy(dep[i:i + 32].astype(np.float32)).cuda() / 1000
            if noisy:
                d = E.NOISE[cam](d, gen)
        out.append(E.run(enc, r, d)[1].float())
    return torch.cat(out)


def pca_basis(ref):
    X = ref.permute(0, 2, 3, 1).reshape(-1, ref.shape[1])
    mu = X.mean(0)
    _, _, V = torch.pca_lowrank(X - mu, q=3, center=False)
    P = (X - mu) @ V
    return mu, V, P.quantile(0.01, 0), P.quantile(0.99, 0)


def to_rgb(t, basis):
    mu, V, lo, hi = basis
    X = t.permute(0, 2, 3, 1).reshape(-1, t.shape[1])
    P = (((X - mu) @ V - lo) / (hi - lo)).clamp(0, 1)
    return P.reshape(t.shape[0], t.shape[2], t.shape[3], 3).cpu().numpy()


def stability(clean, shifted):
    return float(F.cosine_similarity(clean, shifted, dim=1).mean())


def noisy_depth(z, cam):
    gen = torch.Generator(device="cuda").manual_seed(0)
    d = z[f"{cam}_depth"]
    return np.concatenate([E.NOISE[cam](torch.from_numpy(d[i:i + 32].astype(np.float32)).cuda() / 1000, gen).cpu().numpy()
                           for i in range(0, len(d), 32)])


def main():
    import sys
    pn = sys.argv[1] if len(sys.argv) > 1 else "pioneernet"   # checkpoint name (results/<pn>.pt)
    f = sorted(glob.glob(str(C.DATA / "test" / "ep_*.npz")))[0]
    z = np.load(f)
    phase = z["phase"]
    n = len(phase)
    keyf = [int(np.flatnonzero(phase == p)[len(np.flatnonzero(phase == p)) // 2]) for p in (0, 2, 3, 4) if (phase == p).any()]

    # (encoder, label, conditions shown)
    rows = [("dinov2_s14", "DINOv2-S (RGB teacher)", ["clean", "light", "color"]),
            ("defm_s14", "DeFM-S (depth teacher)", ["clean", "noise"]),
            ("r18_imnet", "ResNet-18 ImageNet (ACT today)", ["clean", "light", "color"]),
            (pn, f"PioneerNet {pn.split('_')[1] if '_' in pn else 'v1'} (ours, RGB-D)",
             ["clean", "light", "color", "noise", "light+noise"])]
    encs = {name: E.load(name) for name, _, _ in rows}
    F_ = {}
    stab = {}
    for name, _, conds in rows:
        for cam in C.CAMS:
            clean = feats(encs[name], z, cam, "clean")
            basis = pca_basis(clean)
            for cond in conds:
                t = clean if cond == "clean" else feats(encs[name], z, cam, cond)
                F_[(name, cam, cond)] = to_rgb(t, basis)
                if cond != "clean":
                    stab.setdefault(name, {}).setdefault(cond, []).append(stability(clean, t))
    stab = {k: {c: float(np.mean(v)) for c, v in d.items()} for k, d in stab.items()}
    json.dump(stab, open(C.OUT / "stability.json", "w"), indent=1)
    print(json.dumps(stab, indent=1))
    nd = {cam: noisy_depth(z, cam) for cam in C.CAMS}

    # ---- figure per camera: rows = encoder x condition, columns = key frames -------------------------
    for cam in C.CAMS:
        panels = [("RGB (clean)", lambda k: z[f"clean_{cam}_rgb"][k], None),
                  ("RGB, lighting changed", lambda k: z[f"light_{cam}_rgb"][k], None),
                  ("RGB, colours changed", lambda k: z[f"color_{cam}_rgb"][k], None),
                  ("depth + sensor noise [m]", lambda k: nd[cam][k], "turbo")]
        for name, label, conds in rows:
            for cond in conds:
                s = "" if cond == "clean" else f"\nstability {stab[name][cond]:.2f}"
                panels.append((f"{label}\n{cond}{s}", (lambda k, a=(name, cam, cond): F_[a][k]), "pca"))
        H = 2.3 * len(panels)
        fig, ax = plt.subplots(len(panels), len(keyf), figsize=(3.0 * len(keyf), H))
        # fixed margins in inches: the default top margin is a fraction of the height, ~5 in of blank on a tall figure
        fig.subplots_adjust(top=1 - 0.75 / H, bottom=0.1 / H, hspace=0.12, wspace=0.04)
        dmax = np.percentile(z[f"{cam}_depth"], 99) / 1000
        for i, (lab, fn, kind) in enumerate(panels):
            for j, k in enumerate(keyf):
                a = ax[i, j]
                if kind == "turbo":
                    a.imshow(fn(k), cmap="turbo", vmin=0, vmax=dmax)
                elif kind == "pca":
                    a.imshow(fn(k), interpolation="nearest", extent=(0, 320, 240, 0))
                else:
                    a.imshow(fn(k))
                a.set_xticks([])
                a.set_yticks([])
                if i == 0:
                    a.set_title(f"t={k / 12.5:.1f}s  {PHASES[phase[k]]}", fontsize=10)
            ax[i, 0].set_ylabel(lab, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=8)
        cam_t = "Chest camera (D455)" if cam == "ego" else "Left wrist camera"
        fig.suptitle(f"{cam_t}: features under shift. PCA colours fitted on each encoder's CLEAN features and reused,\n"
                     f"so unchanged colours = features unchanged by the shift. Stability = mean cosine(clean, shifted) "
                     f"per patch (1 = identical).", fontsize=11, y=1 - 0.08 / H, va="top")
        plt.savefig(C.FIG / f"pioneer_{cam}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("saved", C.FIG / f"pioneer_{cam}.png")

    # ---- video: worst case (lighting changed + noisy depth), chest + wrist -------------------------
    cols = [("input RGB (lighting changed)", lambda cam, k: z[f"light_{cam}_rgb"][k], None),
            ("input depth (sensor noise)", lambda cam, k: nd[cam][k], "turbo"),
            ("DINOv2-S clean", lambda cam, k: F_[("dinov2_s14", cam, "clean")][k], "pca"),
            ("DINOv2-S lighting changed", lambda cam, k: F_[("dinov2_s14", cam, "light")][k], "pca"),
            ("DeFM-S noisy depth", lambda cam, k: F_[("defm_s14", cam, "noise")][k], "pca"),
            ("PioneerNet clean", lambda cam, k, pn=pn: F_[(pn, cam, "clean")][k], "pca"),
            ("PioneerNet light + noise", lambda cam, k, pn=pn: F_[(pn, cam, "light+noise")][k], "pca")]
    frames = C.FIG / "_pframes"
    frames.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, len(cols), figsize=(3.0 * len(cols), 4.6))
    for k in range(n):
        for r, cam in enumerate(C.CAMS):
            dmax = np.percentile(z[f"{cam}_depth"], 99) / 1000
            for c, (title, fn, kind) in enumerate(cols):
                a = ax[r, c]
                a.clear()
                if kind == "turbo":
                    a.imshow(fn(cam, k), cmap="turbo", vmin=0, vmax=dmax)
                else:
                    a.imshow(fn(cam, k), interpolation="nearest")
                a.set_xticks([])
                a.set_yticks([])
                if r == 0:
                    a.set_title(title, fontsize=10)
            ax[r, 0].set_ylabel("chest (D455)" if cam == "ego" else "wrist", fontsize=10)
        fig.suptitle(f"PioneerNet vs teachers, worst case (lighting changed + noisy depth)   t = {k / 12.5:5.2f} s   "
                     f"phase: {PHASES[phase[k]]}", fontsize=12)
        fig.savefig(frames / f"{k:05d}.png", dpi=70)
    plt.close(fig)
    subprocess.run([C.ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", "12.5", "-i", str(frames / "%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    str(C.FIG / "pioneer_video.mp4")], check=True)
    for p in frames.glob("*.png"):
        p.unlink()
    frames.rmdir()
    print("saved", C.FIG / "pioneer_video.mp4")


if __name__ == "__main__":
    main()
