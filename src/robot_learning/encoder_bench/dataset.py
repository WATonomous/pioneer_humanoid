"""Load the rendered benchmark episodes into RAM."""
from __future__ import annotations

import glob

import numpy as np

import common as C

LABEL_KEYS = ("tcp", "peg", "hole", "tilt", "phase")
LABEL_SHAPES = {"tcp": (3,), "peg": (3,), "hole": (3,), "tilt": (), "phase": ()}


def episode_files(split: str):
    return sorted(glob.glob(str(C.DATA / split / "ep_*.npz")))


def load_split(split, every: int = 1, variant: str = "clean", max_eps: int | None = None, depth: bool = True,
               rgb: bool = True):
    """Images and labels of all episodes in ``split``, keeping every ``every``-th image.

    ``split`` may also be a list of (split, variant) or (split, variant, every) sources, loaded one after another
    into the same arrays (``variant`` is then ignored; ``max_eps`` applies per source). A split may point at another
    scene's data folder, e.g. "../data_8dof_zip_tie/train". Labels a source lacks (peg / hole / tilt outside
    peg_insert) are NaN, so every array stays aligned.

    Returns dict: {cam}_rgb uint8 (N,H,W,3), {cam}_depth uint16 mm (N,H,W), labels, ep (N,), t (N,) image index,
    and per-episode action/state streams + img_idx (lists).

    Image arrays are preallocated and filled episode by episode, so peak RAM is the final size (a list +
    np.concatenate needs twice that: 34.7k frames of RGB-D for both cameras is ~27 GB)."""
    sources = [(split, variant)] if isinstance(split, str) else list(split)
    files = [(f, src[1], src[2] if len(src) > 2 else every) for src in sources for f in episode_files(src[0])[:max_eps]]
    sels = []
    for f, _, ev in files:
        with np.load(f) as z:
            sels.append(np.arange(0, len(z["img_idx"]), ev))
    N = sum(len(s) for s in sels)
    res = {}
    for c in C.CAMS:
        if rgb:
            res[f"{c}_rgb"] = np.empty((N, C.H, C.W, 3), np.uint8)
        if depth:
            res[f"{c}_depth"] = np.empty((N, C.H, C.W), np.uint16)
    lab = {k: [] for k in LABEL_KEYS}
    ep, tt, streams = [], [], []
    o = 0
    for i, ((f, variant, _), sel) in enumerate(zip(files, sels)):
        with np.load(f) as z:
            for c in C.CAMS:
                if rgb:
                    res[f"{c}_rgb"][o:o + len(sel)] = z[f"{variant}_{c}_rgb"][sel]
                if depth:   # geometric variants (camera) carry their own depth
                    dk = f"{variant}_{c}_depth" if f"{variant}_{c}_depth" in z.files else f"{c}_depth"
                    res[f"{c}_depth"][o:o + len(sel)] = z[dk][sel]
            for k in LABEL_KEYS:   # peg / hole / tilt exist for peg_insert only; tcp and phase for every scene
                lab[k].append(z[k][sel] if k in z.files else np.full((len(sel),) + LABEL_SHAPES[k], np.nan, np.float32))
            streams.append(dict(action=z["action"], state=z["state"], img_idx=z["img_idx"], file=f))
        ep.append(np.full(len(sel), i))
        tt.append(sel)
        o += len(sel)
    res |= {k: np.concatenate(v) for k, v in lab.items() if v and not all(np.isnan(x).all() for x in v)}
    res["ep"], res["t"] = np.concatenate(ep), np.concatenate(tt)
    res["streams"] = streams
    return res


def targets(d) -> dict[str, np.ndarray]:
    """Regression targets in metres."""
    return {
        "tcp": d["tcp"],
        "peg": d["peg"],
        "hole_xy": d["hole"][:, :2],
        "peg_minus_tcp": d["peg"] - d["tcp"],
        "peg_minus_hole_xy": d["peg"][:, :2] - d["hole"][:, :2],
    }
