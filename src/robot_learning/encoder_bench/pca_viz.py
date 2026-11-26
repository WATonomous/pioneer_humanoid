"""PCA feature visualisations (figure + video): the first 3 principal components of each encoder's patch
tokens mapped to R, G, B. Clean, object-aligned colour regions = the encoder separates the scene's parts.

    .venv/bin/python pca_viz.py            # figures/pca_<cam>.png, figures/pca_video.mp4
"""
from __future__ import annotations

import subprocess

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as C  # noqa: E402
import encoders as E  # noqa: E402

FFMPEG = C.ffmpeg_exe()
ROWS = [  # (encoder, input, row label)
    ("r18_imnet", "rgb", "ResNet-18 ImageNet\n(RGB, ACT today)"),
    ("clip_b16", "rgb", "CLIP ViT-B/16\n(RGB, our DiT)"),
    ("siglip_b16", "rgb", "SigLIP ViT-B/16\n(RGB, pi0.5/SmolVLA)"),
    ("dinov2_s14", "rgb", "DINOv2 ViT-S/14\n(RGB)"),
    ("dinov2_s14_d", "depth", "DINOv2 ViT-S/14\n(min-max depth)"),
    ("defm_r18", "depth", "DeFM ResNet-18\n(depth)"),
    ("defm_s14", "depth", "DeFM ViT-S/14\n(depth)"),
    ("defm_s14", "noisy", "DeFM ViT-S/14\n(D455/stereo-noised depth)"),
]
PHASES = ["reach", "grasp", "carry", "align/insert", "inserted"]


def pca_rgb(tok: torch.Tensor, ref: torch.Tensor | None = None):
    """tok (N, C, h, w) -> (N, h, w, 3) in 0..1 using PCA fitted on ``ref`` tokens (default: tok)."""
    N, Cc, h, w = tok.shape
    X = tok.permute(0, 2, 3, 1).reshape(-1, Cc).float()
    R = X if ref is None else ref.permute(0, 2, 3, 1).reshape(-1, Cc).float()
    mu = R.mean(0)
    _, _, V = torch.pca_lowrank(R - mu, q=3, center=False)
    P = (X - mu) @ V
    Pr = (R - mu) @ V
    lo, hi = Pr.quantile(0.01, 0), Pr.quantile(0.99, 0)
    P = ((P - lo) / (hi - lo)).clamp(0, 1)
    return P.reshape(N, h, w, 3)


@torch.no_grad()
def tokens(enc, raw, noisy_cam=None):
    out = []
    gen = torch.Generator(device="cuda").manual_seed(0)
    for i in range(0, len(raw), 32):
        x = torch.from_numpy(raw[i:i + 32]).cuda()
        if enc.modality == "depth":
            x = x.float() / 1000
            if noisy_cam:
                x = E.NOISE[noisy_cam](x, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(enc(enc.prep(x))[1].float())
    return torch.cat(out)


def main():
    import glob
    f = sorted(glob.glob(str(C.DATA / "test" / "ep_*.npz")))[0]
    z = np.load(f)
    phase = z["phase"]
    keyf = [int(np.flatnonzero(phase == p)[len(np.flatnonzero(phase == p)) // 2]) for p in range(5) if (phase == p).any()]
    n = len(phase)
    encs = {}
    feats = {}
    for name, inp, _ in ROWS:
        if name not in encs:
            encs[name] = E.load(name)
        enc = encs[name]
        for cam in C.CAMS:
            raw = z[f"clean_{cam}_rgb"] if inp == "rgb" else z[f"{cam}_depth"]
            t = tokens(enc, raw, cam if inp == "noisy" else None)
            feats[(name, inp, cam)] = pca_rgb(t).cpu().numpy()   # PCA fit over the whole episode (temporally stable)
    noisy_depth = {}
    gen = torch.Generator(device="cuda").manual_seed(0)
    for cam in C.CAMS:
        noisy_depth[cam] = np.concatenate([E.NOISE[cam](torch.from_numpy(z[f"{cam}_depth"][i:i+32]).cuda().float() / 1000, gen).cpu().numpy()
                                           for i in range(0, n, 32)])

    # ---- static figure per camera -------------------------------------------------------------
    for cam in C.CAMS:
        rows = 3 + len(ROWS)
        fig, ax = plt.subplots(rows, len(keyf), figsize=(3.2 * len(keyf), 2.55 * rows))
        dmax = np.percentile(z[f"{cam}_depth"], 99) / 1000
        for j, k in enumerate(keyf):
            ax[0, j].imshow(z[f"clean_{cam}_rgb"][k])
            ax[0, j].set_title(f"t={k / 12.5:.1f}s  phase: {PHASES[phase[k]]}", fontsize=10)
            im = ax[1, j].imshow(z[f"{cam}_depth"][k] / 1000, cmap="turbo", vmin=0, vmax=dmax)
            im2 = ax[2, j].imshow(noisy_depth[cam][k], cmap="turbo", vmin=0, vmax=dmax)
            for i, (name, inp, _) in enumerate(ROWS):
                ax[3 + i, j].imshow(feats[(name, inp, cam)][k], interpolation="nearest", extent=(0, 320, 240, 0))
        labels = ["RGB", "depth (clean sim) [m]", "depth + sensor noise [m]\n(0 = invalid)"] + [r[2] for r in ROWS]
        for i, lab in enumerate(labels):
            ax[i, 0].set_ylabel(lab, fontsize=9, rotation=0, ha="right", va="center", labelpad=10)
        for a in ax.flat:
            a.set_xticks([])
            a.set_yticks([])
        fig.colorbar(im, ax=ax[1, :].tolist(), fraction=0.02, pad=0.01, label="m")
        fig.colorbar(im2, ax=ax[2, :].tolist(), fraction=0.02, pad=0.01, label="m")
        cam_t = "Ego camera (chest RealSense D455)" if cam == "ego" else "Left wrist camera (depth via stereo)"
        fig.suptitle(f"{cam_t}: first 3 PCA components of each encoder's patch features -> RGB\n"
                     f"(PCA fit per encoder over the whole test episode; same colour = similar feature)", fontsize=12)
        plt.savefig(C.FIG / f"pca_{cam}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        print("saved", C.FIG / f"pca_{cam}.png")

    # ---- video ------------------------------------------------------------------------------------
    vid_rows = [("dinov2_s14", "rgb"), ("defm_s14", "depth"), ("defm_s14", "noisy"), ("r18_imnet", "rgb")]
    titles = ["RGB", "depth (sim)", "depth + noise", "DINOv2-S RGB PCA", "DeFM-S depth PCA", "DeFM-S noisy PCA",
              "ResNet18-IN RGB PCA"]
    frames_dir = C.FIG / "_frames"
    frames_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 7, figsize=(21, 4.6))
    for k in range(n):
        for r, cam in enumerate(C.CAMS):
            dmax = np.percentile(z[f"{cam}_depth"], 99) / 1000
            panels = [z[f"clean_{cam}_rgb"][k], z[f"{cam}_depth"][k] / 1000, noisy_depth[cam][k]] + \
                     [feats[(nm, inp, cam)][k] for nm, inp in vid_rows]
            for c, p in enumerate(panels):
                a = ax[r, c]
                a.clear()
                if c in (1, 2):
                    a.imshow(p, cmap="turbo", vmin=0, vmax=dmax)
                else:
                    a.imshow(p, interpolation="nearest")
                a.set_xticks([])
                a.set_yticks([])
                if r == 0:
                    a.set_title(titles[c], fontsize=11)
            ax[r, 0].set_ylabel("ego (D455)" if cam == "ego" else "wrist", fontsize=11)
        fig.suptitle(f"peg_insert test episode  t = {k / 12.5:5.2f} s   phase: {PHASES[phase[k]]}", fontsize=13)
        fig.savefig(frames_dir / f"{k:05d}.png", dpi=70)
    plt.close(fig)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", "12.5", "-i", str(frames_dir / "%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    str(C.FIG / "pca_video.mp4")], check=True)
    for p in frames_dir.glob("*.png"):
        p.unlink()
    frames_dir.rmdir()
    print("saved", C.FIG / "pca_video.mp4")


if __name__ == "__main__":
    main()
