"""PCA of grasp postures -> hand synergies (eigengrasps): q = mean + sum_k a_k * PC_k.

    python -m hand_synergies.synergies fit  --grasps out/grasps.npz --out out/
    python -m hand_synergies.synergies eval --grasps out/grasps.npz --out out/

fit:  synergies.npz (mean, components, stds, variance ratio), variance.png, loadings.png,
      pc<k>.gif (the hand swept along each PC, -2 sd .. +2 sd) and synergies.png (a still grid).
eval: re-holds each grasp with its posture rebuilt from the first k PCs (plus the usual squeeze)
      and re-runs the six-direction shake test -> reconstruction.png / reconstruction.csv.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np
from pioneer_humanoid.mujoco_hand import JOINT_NAMES

BLUE, RED, NEUTRAL, INK, MUTED = "#2a78d6", "#e34948", "#f0efec", "#2b2b29", "#8a8984"
SURFACE = "#fcfcfb"


# --- PCA --------------------------------------------------------------------------------------
def fit_pca(q: np.ndarray) -> dict:
    mean = q.mean(axis=0)
    _, s, vt = np.linalg.svd(q - mean, full_matrices=False)
    var = s**2 / (len(q) - 1)
    # PCA signs are arbitrary; point each PC the way that curls the fingers more.
    from pioneer_humanoid.mujoco_hand import close_direction

    closing = close_direction(np.stack([q.min(0), q.max(0)], axis=1))
    vt *= np.where(vt @ closing < 0, -1.0, 1.0)[:, None]
    return dict(mean=mean, components=vt, stds=np.sqrt(var), variance_ratio=var / var.sum())


def project(q: np.ndarray, pca: dict, k: int) -> np.ndarray:
    """Rebuild postures from their first k PC scores."""
    w = pca["components"][:k]
    return pca["mean"] + (q - pca["mean"]) @ w.T @ w


# --- plots ------------------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_variance(pca: dict, path: Path, n_grasps: int) -> None:
    import matplotlib.pyplot as plt

    ratio = pca["variance_ratio"] * 100
    cum = np.cumsum(ratio)
    k = np.arange(1, len(ratio) + 1)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    _style(ax)
    ax.bar(k, ratio, width=0.6, color=BLUE, label="per component")
    ax.plot(k, cum, color=INK, linewidth=2, marker="o", markersize=5, label="cumulative")
    for target in (80, 90):
        n = int(np.argmax(cum >= target)) + 1
        ax.axhline(target, color=MUTED, linewidth=0.8, linestyle="--")
        ax.annotate(f"{target}% at {n} PCs", (n, cum[n - 1]), textcoords="offset points", xytext=(8, -14),
                    fontsize=9, color=INK)
    ax.set_xticks(k)
    ax.set_ylim(0, 102)
    ax.set_xlabel("principal component", color=INK)
    ax.set_ylabel("variance explained (%)", color=INK)
    ax.set_title(f"Pioneer hand grasp synergies ({n_grasps} stable grasps)", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loadings(pca: dict, path: Path, k: int = 6) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    w = pca["components"][:k] * pca["stds"][:k, None]  # rad per 1 sd of each PC
    lim = np.abs(w).max()
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    fig, ax = plt.subplots(figsize=(8, 5.6), facecolor=SURFACE)
    im = ax.imshow(w.T, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(k), [f"PC{i + 1}\n{pca['variance_ratio'][i] * 100:.0f}%" for i in range(k)], fontsize=9)
    ax.set_yticks(range(len(JOINT_NAMES)), JOINT_NAMES, fontsize=8)
    for y in (3.5, 7.5, 11.5, 15.5):  # digit boundaries
        ax.axhline(y, color=SURFACE, linewidth=2)
    for i in range(k):
        for j in range(len(JOINT_NAMES)):
            if abs(w[i, j]) > 0.5 * lim:
                ax.text(i, j, f"{w[i, j]:+.2f}", ha="center", va="center", fontsize=7, color=SURFACE)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("joint change per +1 sd (rad)", fontsize=9)
    ax.set_title("Joint loadings (thumb, index, middle, ring, pinky)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- rendering --------------------------------------------------------------------------------
def _renderer(width=360, height=300):
    from .scene import make_model

    model = make_model(width, height)
    for gid in range(model.ngeom):  # hide the object
        if model.geom_bodyid[gid] == model.body("object").id:
            model.geom_rgba[gid, 3] = 0
    return model, mujoco.MjData(model), mujoco.Renderer(model, height, width)


_VIEWS = (  # (azimuth, elevation): from the thumb side, and looking up into the palm
    (200, -10),
    (90, 80),
)


def _render_pose(model, data, renderer, q):
    idx = [model.joint(n).qposadr[0] for n in JOINT_NAMES]
    data.qpos[idx] = q
    mujoco.mj_forward(model, data)
    frames = []
    for az, el in _VIEWS:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.02, 0.1, -0.02]
        cam.distance, cam.azimuth, cam.elevation = 0.36, az, el
        renderer.update_scene(data, cam)
        frames.append(renderer.render())
    return np.hstack(frames)


def render_synergies(pca: dict, out: Path, k: int = 4, n_frames: int = 24) -> None:
    from PIL import Image, ImageDraw

    from pioneer_humanoid.mujoco_hand import joint_ranges

    model, data, renderer = _renderer()
    lo, hi = joint_ranges(model).T
    grid = []
    for i in range(k):
        pc, sd = pca["components"][i], pca["stds"][i]
        amps = 2 * sd * np.sin(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))
        frames = []
        for a in amps:
            img = Image.fromarray(_render_pose(model, data, renderer, np.clip(pca["mean"] + a * pc, lo, hi)))
            ImageDraw.Draw(img).text((8, 8), f"PC{i + 1}  {a / sd:+.1f} sd", fill=(255, 255, 255))
            frames.append(img)
        frames[0].save(out / f"pc{i + 1}.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)
        row = []
        for a in (-2, 0, 2):
            img = Image.fromarray(_render_pose(model, data, renderer, np.clip(pca["mean"] + a * sd * pc, lo, hi)))
            ImageDraw.Draw(img).text((8, 8), f"PC{i + 1}  {a:+d} sd", fill=(255, 255, 255))
            row.append(np.asarray(img))
        grid.append(np.hstack(row))
    Image.fromarray(np.vstack(grid)).save(out / "synergies.png")
    renderer.close()


# --- reconstruction test ----------------------------------------------------------------------
_EVAL = {}


def _eval_one(args):
    g, k, rec = args
    from .grasp_gen import SETTLE_S, GraspGen, set_object

    gen = _EVAL.get("gen") or _EVAL.setdefault("gen", GraspGen())
    m, d, idx = gen.model, gen.data, gen.idx
    mujoco.mj_resetData(m, d)
    kind = str(rec["kind"])
    set_object(m, idx, kind, rec["size"][: {"sphere": 1, "cylinder": 2, "box": 3}[kind]])
    m.opt.gravity[:] = 0.0
    d.qpos[idx.qpos] = rec["q"]
    d.qpos[idx.obj_qpos:idx.obj_qpos + 3] = rec["obj_pos"]
    d.qpos[idx.obj_qpos + 3:idx.obj_qpos + 7] = rec["obj_quat"]
    # Same per-joint squeeze as the original grasp, on top of the rebuilt posture.
    ctrl = np.clip(rec["q_k"] + (rec["ctrl"] - rec["q"]), *gen.ranges.T)
    d.ctrl[idx.act] = ctrl
    mujoco.mj_step(m, d, nstep=int(SETTLE_S / m.opt.timestep))
    return g, k, gen.hold_test(ctrl)


def evaluate(grasps: dict, pca: dict, ks, n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(grasps["q"]), size=min(n, len(grasps["q"])), replace=False)
    jobs = []
    for k in ks:
        q_k = project(grasps["q"][pick], pca, k)
        for row, g in enumerate(pick):
            rec = {f: grasps[f][g] for f in ("q", "ctrl", "kind", "size", "obj_pos", "obj_quat")}
            rec["q_k"] = q_k[row]
            jobs.append((int(g), k, rec))
    held = {k: [] for k in ks}
    with mp.Pool() as pool:
        for _, k, ok in pool.imap_unordered(_eval_one, jobs, chunksize=8):
            held[k].append(ok)
    return {k: float(np.mean(v)) for k, v in held.items()}


def plot_reconstruction(rates: dict, pca: dict, path: Path, n: int) -> None:
    import matplotlib.pyplot as plt

    ks = sorted(rates)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    _style(ax)
    ax.plot(ks, [rates[k] * 100 for k in ks], color=BLUE, linewidth=2, marker="o", markersize=6,
            label="grasps still held")
    cum = np.cumsum(pca["variance_ratio"]) * 100
    ax.plot(ks, [cum[k - 1] for k in ks], color=MUTED, linewidth=2, linestyle="--", label="variance explained")
    for k in ks:
        ax.annotate(f"{rates[k] * 100:.0f}%", (k, rates[k] * 100), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(ks)
    ax.set_ylim(0, 105)
    ax.set_xlabel("synergies used (k PCs)", color=INK)
    ax.set_ylabel("%", color=INK)
    ax.set_title(f"Grasps that survive the shake test with a k-PC posture (n={n})", loc="left", color=INK,
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=("fit", "eval"))
    p.add_argument("--grasps", default="out/grasps.npz")
    p.add_argument("--out", default="out")
    p.add_argument("--n-eval", type=int, default=300, help="grasps re-tested per k (eval)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grasps = dict(np.load(args.grasps))
    q = grasps["q"]

    if args.cmd == "fit":
        pca = fit_pca(q)
        np.savez(out / "synergies.npz", joint_names=np.array(JOINT_NAMES), **pca)
        cum = np.cumsum(pca["variance_ratio"])
        print(f"{len(q)} grasps; cumulative variance: " + ", ".join(f"{i + 1}:{c:.0%}" for i, c in enumerate(cum[:10])))
        plot_variance(pca, out / "variance.png", len(q))
        plot_loadings(pca, out / "loadings.png")
        render_synergies(pca, out)
        print(f"wrote {out}/synergies.npz, variance.png, loadings.png, synergies.png, pc1-4.gif")
    else:
        pca = dict(np.load(out / "synergies.npz"))
        ks = [1, 2, 3, 4, 5, 6, 8, 10, 20]
        rates = evaluate(grasps, pca, ks, args.n_eval)
        n = min(args.n_eval, len(q))
        with open(out / "reconstruction.csv", "w") as f:
            f.write("k,variance_explained,held\n")
            for k in ks:
                f.write(f"{k},{np.cumsum(pca['variance_ratio'])[k - 1]:.4f},{rates[k]:.4f}\n")
        plot_reconstruction(rates, pca, out / "reconstruction.png", n)
        print("held: " + ", ".join(f"k={k}:{rates[k]:.0%}" for k in ks))


if __name__ == "__main__":
    main()
