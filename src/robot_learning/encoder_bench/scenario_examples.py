"""Figure: every scenario shift (scenarios.py) on the same held-out layout, both cameras, colour and depth.

    MUJOCO_GL=egl .venv/bin/python scenario_examples.py [--seed 7000]   # figures_8dof/scenario_examples.png
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as C  # noqa: E402
import scenarios as S  # noqa: E402

NICE = {"clean": "Clean", "fog": "Fog / haze", "glare": "Sun glare", "lens": "Wet / dirty lens",
        "shadow": "Hard side shadows", "clutter": "Clutter\n(distractors)", "background": "Changed\nbackground", "table": "New tablecloth\n(HELD OUT)"}


def render_all(seed):
    import mujoco
    from synthetic_teleop import TeleopSim
    sim = TeleopSim(C.SCENE, seed, cameras={c: (C.H, C.W) for c in C.CAMS}, robot=C.ROBOT)
    m, d = sim.model, sim.data
    mujoco.mj_forward(m, d)
    base = {k: getattr(m, k).copy() for k in S.LIGHT_FIELDS}
    head = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
    r = mujoco.Renderer(m, C.H, C.W)
    out = {}
    for name in ("clean",) + S.SCENARIOS:
        for k, v in base.items():
            getattr(m, k)[:] = v
        m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = head
        look = None
        if name != "clean":
            look = S.make(name, m, d, np.random.default_rng([seed, 7]))
            S.apply_model(m, look)
        for c in C.CAMS:
            r.update_scene(d, c)
            if look:
                S.add_geoms(r.scene, look)
            rgb = r.render().copy()
            r.enable_depth_rendering()
            r.update_scene(d, c)
            if look:
                S.add_geoms(r.scene, look)
            dep = r.render().copy()
            r.disable_depth_rendering()
            if look:
                rgb, dep = S.post(look, c, rgb, dep, np.random.default_rng([seed, 13]))
            out[name, c] = (rgb, dep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7000)
    args = ap.parse_args()
    out = render_all(args.seed)
    names = ("clean",) + S.SCENARIOS
    rows = [("ego", "rgb", "Chest D455\ncolour"), ("ego", "depth", "Chest D455\ndepth"),
            ("wrist_left", "rgb", "Wrist\ncolour"), ("wrist_left", "depth", "Wrist\ndepth")]
    fig, ax = plt.subplots(len(rows), len(names), figsize=(2.6 * len(names), 2.1 * len(rows)))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.9, bottom=0.01, wspace=0.03, hspace=0.04)
    for j, n in enumerate(names):
        hit = "colour + depth" if n in S.DEPTH_HIT else ("colour only" if n != "clean" else "")
        ax[0, j].set_title(NICE[n] + (f"\n[{hit}]" if hit else "\n"), fontsize=10)
        for i, (cam, mod, lab) in enumerate(rows):
            a = ax[i, j]
            a.set_xticks([]), a.set_yticks([])
            rgb, dep = out[n, cam]
            if mod == "rgb":
                a.imshow(rgb)
            else:
                dmax = 1.4 if cam == "ego" else 0.6
                im = a.imshow(np.where(dep > 0, dep, np.nan), cmap="turbo", vmin=0, vmax=dmax)
                im.cmap.set_bad("black")
            if j == 0:
                a.set_ylabel(lab, fontsize=9.5, rotation=0, ha="right", va="center", labelpad=6)
    fig.suptitle(f"Scenario shifts, held-out layout {args.seed}. Depth in turbo; black = no reading.", fontsize=11)
    path = C.FIG / "scenario_examples.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("saved", path)


if __name__ == "__main__":
    main()
