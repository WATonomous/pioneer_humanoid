"""Figure: every test shift, seen by both robot cameras at the same moment of the same held-out episode.

    .venv/bin/python shift_examples.py   # figures/shift_examples.png
"""
from __future__ import annotations

import glob

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as C  # noqa: E402
import encoders as E  # noqa: E402


def main():
    f = sorted(glob.glob(str(C.DATA / "test" / "ep_*.npz")))[0]
    fc = C.DATA / "test_cam" / f.split("/")[-1]
    z, zc = np.load(f), np.load(fc)
    ph = z["phase"]
    k = int(np.flatnonzero(ph == 3)[len(np.flatnonzero(ph == 3)) // 2])     # peg over the hole, about to insert
    gen = torch.Generator().manual_seed(0)
    cols = [("Clean", lambda c: z[f"clean_{c}_rgb"][k], lambda c: z[f"{c}_depth"][k] / 1000),
            ("Lighting changed", lambda c: z[f"light_{c}_rgb"][k], None),
            ("Colours changed", lambda c: z[f"color_{c}_rgb"][k], None),
            ("Sensor-noised depth", None,
             lambda c: E.NOISE[c](torch.from_numpy(z[f"{c}_depth"][k:k + 1].astype(np.float32) / 1000), gen)[0].numpy()),
            ("Colour camera dead", lambda c: np.zeros_like(z[f"clean_{c}_rgb"][k]), lambda c: z[f"{c}_depth"][k] / 1000),
            ("Cameras moved\n(held-out)", lambda c: zc[f"camera_{c}_rgb"][k], lambda c: zc[f"camera_{c}_depth"][k] / 1000)]
    rows = [("ego", "rgb", "Chest D455\ncolour"), ("ego", "depth", "Chest D455\ndepth"),
            ("wrist_left", "rgb", "Wrist\ncolour"), ("wrist_left", "depth", "Wrist\ndepth")]
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(2.6 * len(cols), 2.1 * len(rows)))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.01, wspace=0.03, hspace=0.04)
    for j, (title, frgb, fdep) in enumerate(cols):
        ax[0, j].set_title(title, fontsize=10)
        for i, (cam, mod, lab) in enumerate(rows):
            a = ax[i, j]
            a.set_xticks([]), a.set_yticks([])
            fn = frgb if mod == "rgb" else fdep
            if fn is None:   # this shift does not touch this input
                a.imshow(np.full((C.H, C.W, 3), 245, np.uint8))
                a.text(C.W / 2, C.H / 2, "unchanged", ha="center", va="center", fontsize=9, color="#888")
            elif mod == "rgb":
                a.imshow(fn(cam))
            else:
                d = fn(cam)
                dmax = 1.4 if cam == "ego" else 0.6
                im = a.imshow(np.where(d > 0, d, np.nan), cmap="turbo", vmin=0, vmax=dmax)
                im.cmap.set_bad("black")
            if j == 0:
                a.set_ylabel(lab, fontsize=9.5, rotation=0, ha="right", va="center", labelpad=6)
    fig.suptitle("Test shifts, same held-out episode and moment (peg over the hole). Depth in turbo; black = no reading.",
                 fontsize=11)
    out = C.FIG / "shift_examples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
