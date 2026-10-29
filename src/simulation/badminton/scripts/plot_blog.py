"""Blog figures from a bank eval (runs/eval_pstar_hits*.npy written by eval_rl.py).

  MUJOCO_GL=egl uv run scripts/plot_blog.py --input runs/eval_pstar_hits_holdout_seed1.npy \
      --out runs/figs_blog --bank "Evaluation bank, seed 1, 8192 episodes"

Writes, in one editorial style (off-white surface, title top-left, direct
labels, footer rule with the data source, WATonomous blue accent):

  fig_layout.png     rendered scene with the launch volume and intercept
                     region drawn in, above a dimensioned top view
  fig_scorecard.png  headline numbers and the outcome of every serve per arm
  fig_hit_rate.png   smoothed hit rate over the intercept point, plus the
                     rate per lateral and height band
  fig_arm_pick.png   which arm the chooser takes, where the 50/50 line runs,
                     and the hit rate when it agrees or disagrees with the
                     strike-plane rule
  fig_returns.png    where the returns cross the net and where they land
                     relative to the serve's origin
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm  # noqa: E402
from matplotlib.patches import Rectangle, Ellipse  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aero  # noqa: E402

# -- style -----------------------------------------------------------------
BG, INK, INK2, INK3 = "#f4f3ef", "#1b1b1b", "#66655f", "#9b9a94"
GRID, RULE = "#dddcd6", "#c6c5bf"
BLUE = "#1f5d96"                              # WATonomous brand blue
RIGHT, LEFT, MISS = "#1b1b1b", BLUE, "#c0392b"  # right arm, left arm, miss
BLUES = ["#e9eff6", "#c9d9ea", "#a3bfdc", "#7aa3cb", "#5386b7", "#2f6ba3", "#1f5d96", "#153f66"]
SEQ = LinearSegmentedColormap.from_list("seq", BLUES)
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#8fb0d2", "#e9e8e3", "#8f8e89", "#2b2b2b"])
DPI = 170

plt.rcParams.update({
    "font.family": ["Noto Sans", "Cantarell", "DejaVu Sans"],
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "font.size": 9.5, "legend.frameon": False,
    "axes.titlesize": 10.5, "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
})

# margins in inches, shared by every figure
M_LEFT, M_RIGHT = 0.55, 0.55
M_TITLE, M_SUB, M_LEGEND = 0.55, 0.92, 1.27      # from the top edge
M_PANEL_TOP = 1.62                               # panel area starts here
M_PANEL_BOTTOM, M_RULE, M_FOOT = 1.3, 0.72, 0.42  # from the bottom edge
MIN_N = 8


def load(path):
    r = np.load(path)
    p = aero.load_params()
    d = dict(
        front=r[:, 0], z=r[:, 1], x=r[:, 2], hit=r[:, 3].astype(bool),
        land=r[:, 7], lok=r[:, 8].astype(bool), locked=r[:, 10].astype(bool),
        chosen=r[:, 11], rule=r[:, 12], t_lock=r[:, 13],
        land_xy=r[:, 14:16], p0_xy=r[:, 16:18], cross_xz=r[:, 18:20],
    )
    d["y"] = d["front"] + p["arm"]["base_y"]
    return d, p


def frame(w, h, title, subtitle, foot_left, foot_right, legend=False):
    fig = plt.figure(figsize=(w, h), dpi=DPI)
    L, R = M_LEFT / w, 1 - M_RIGHT / w
    fig.text(L, 1 - M_TITLE / h, title, fontsize=19, color=INK, va="center", ha="left")
    fig.text(L, 1 - M_SUB / h, subtitle, fontsize=10.5, color=INK2, va="center")
    fig.add_artist(plt.Line2D([L, R], [M_RULE / h] * 2, color=RULE, lw=0.8))
    fig.text(L, M_FOOT / h, foot_left, fontsize=8, color=INK3, va="center")
    fig.text(R, M_FOOT / h, foot_right, fontsize=8, color=INK3, va="center", ha="right")
    top = 1 - (M_PANEL_TOP + (0.32 if legend else 0)) / h
    fig.subplots_adjust(left=L + 0.03, right=R, top=top, bottom=M_PANEL_BOTTOM / h)
    fig._frame = dict(L=L, R=R, legend_y=1 - M_LEGEND / h - (0.05 if legend else 0), h=h, w=w)
    return fig


def swatch_row(fig, x0, items):
    """Small colour swatches with labels on the legend row."""
    y = fig._frame["legend_y"]
    for i, (lab, col) in enumerate(items):
        fig.patches.append(Rectangle((x0, y - 0.011), 0.012, 0.022, fc=col, ec="none",
                                     transform=fig.transFigure, figure=fig))
        fig.text(x0 + 0.017, y, lab, fontsize=8, color=INK2, va="center")
        x0 += 0.017 + 0.0062 * len(lab) + 0.022


def dot_row(fig, x0, items):
    y = fig._frame["legend_y"]
    for lab, col, marker in items:
        fig.text(x0, y, marker, color=col, fontsize=9, va="center", ha="center")
        fig.text(x0 + 0.012, y, lab, fontsize=8, color=INK2, va="center")
        x0 += 0.012 + 0.0062 * len(lab) + 0.025


def hgrid(ax, ys):
    for yv in ys:
        ax.axhline(yv, color=GRID, lw=0.6, zorder=0)


def smooth_rate(x, z, ok, gx, gz, bw=0.12, min_w=6.0):
    """Gaussian-kernel hit rate on a grid; cells with little support are NaN."""
    X, Z = np.meshgrid(gx, gz)
    w = np.exp(-((X[..., None] - x) ** 2 + (Z[..., None] - z) ** 2) / (2 * bw ** 2))
    tot = w.sum(-1)
    rate = (w * ok).sum(-1) / np.maximum(tot, 1e-9)
    rate[tot < min_w] = np.nan
    return X, Z, rate


def shoulders(ax, color=INK3):
    for s in (-1, 1):
        ax.axvline(s * 0.245, color=color, lw=0.7, ls=(0, (2, 3)), zorder=0)


def band_rates(d, key, edges, mask=None):
    v = d[key]; hit = d["hit"]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo) & (v < hi)
        if mask is not None:
            m &= mask
        out.append((lo, hi, m.sum(), hit[m].mean() if m.sum() >= MIN_N else np.nan))
    return out


# -- figure 1: layout --------------------------------------------------------
def render_overview(p, out_png):
    """Scene render with the launch volume and the intercept region added as
    translucent geoms; returns the image and a world->pixel projector."""
    import mujoco
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = mujoco.MjModel.from_xml_path(os.path.join(here, "scene", "badminton.xml"))
    W, H = 2400, 1350
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    m.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 1.0, 0.8]
    cam.distance, cam.azimuth, cam.elevation = 7.6, 124, -20
    r.update_scene(d, cam)
    scn = r.scene
    scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    lp = p["launcher"]
    box_c = [0.0, (lp["launch_y_range"][0] + lp["launch_y_range"][1]) / 2,
             (lp["launch_z_range"][0] + lp["launch_z_range"][1]) / 2]
    box_s = [lp["launch_x_max"], (lp["launch_y_range"][1] - lp["launch_y_range"][0]) / 2,
             (lp["launch_z_range"][1] - lp["launch_z_range"][0]) / 2]
    eye = np.eye(3).flatten()
    for gtype, size, pos, rgba in (
        (mujoco.mjtGeom.mjGEOM_BOX, box_s, box_c, [0.12, 0.36, 0.59, 0.25]),
        (mujoco.mjtGeom.mjGEOM_ELLIPSOID, [1.5 * s for s in lp["sweet_sigma"]],
         lp["sweet_point"], [0.11, 0.11, 0.11, 0.22]),
    ):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.array(size, float), np.array(pos, float),
                            eye, np.array(rgba, np.float32))
        scn.ngeom += 1
    img = r.render().copy()
    img[img.max(axis=-1) < 6] = np.array([0xf4, 0xf3, 0xef], np.uint8)
    import PIL.Image as I
    I.fromarray(img).save(out_png)

    gl = scn.camera
    pos = (np.array(gl[0].pos) + np.array(gl[1].pos)) / 2
    fwd = np.array(gl[0].forward); up = np.array(gl[0].up)
    right = np.cross(fwd, up)
    near, top, bot = gl[0].frustum_near, gl[0].frustum_top, gl[0].frustum_bottom
    halfh = (top - bot) / 2
    halfw = halfh * W / H

    def project(pt):
        v = np.array(pt, float) - pos
        zc = v @ fwd
        xn = (v @ right) / zc * near
        yn = (v @ up) / zc * near
        return (xn / halfw + 1) / 2 * W, (1 - (yn - (top + bot) / 2) / halfh) / 2 * H
    return img, project


def fig_layout(p, out, bank):
    img, proj = render_overview(p, os.path.join(out, "render_overview.png"))
    bgpx = np.array([0xf4, 0xf3, 0xef], np.uint8)
    rows = np.where((np.abs(img.astype(int) - bgpx).sum(-1) > 12).any(axis=1))[0]
    r0, r1 = max(rows[0] - 40, 0), min(rows[-1] + 20, img.shape[0])
    W_fig = 12.0
    render_h = (r1 - r0) / img.shape[1] * (W_fig - M_LEFT - M_RIGHT - 0.36)
    top_h = 2.7
    fig = frame(W_fig, M_PANEL_TOP + render_h + 0.55 + top_h + M_PANEL_BOTTOM, "Simulated court",
                "Robot position, serve launch volume (blue) and intercept region (grey)",
                "Scene rendered from the MuJoCo model; dimensions from scene/params.yaml",
                "WATonomous humanoid, badminton receive")
    gs = fig.add_gridspec(2, 1, height_ratios=[render_h, top_h], hspace=0.55 / ((render_h + top_h) / 2))
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])

    a.imshow(img[r0:r1]); a.set_xticks([]); a.set_yticks([])
    lp = p["launcher"]
    call = [  # world point, label, pixel offset of the label from the point
        ((0.245, -2.0, 1.2), "both arms at rest,\nshoulders 0.49 m apart", (-330, -250)),
        ((0.0, 0.0, 1.55), "net, 1.55 m high", (-60, -150)),
        ((-lp["launch_x_max"], lp["launch_y_range"][0], lp["launch_z_range"][1]),
         "serves launched inside this volume\ny 3.0 to 6.2 m, z 1.0 to 2.5 m, |x| ≤ 2.5 m", (40, -120)),
        ((lp["sweet_point"][0], lp["sweet_point"][1], lp["sweet_point"][2] - 0.5),
         "intercept region, 1.5σ\ncentred 0.4 m in front, chest height", (300, 210)),
    ]
    for pt, text, (dx, dy) in call:
        px, py = proj(pt); py -= r0
        a.annotate(text, xy=(px, py), xytext=(px + dx, py + dy), fontsize=8.5, color=INK,
                   ha="left" if dx > 0 else "right", va="center",
                   arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=2, shrinkB=2))
        a.plot(px, py, "o", ms=3, color=INK, mec=BG, mew=0.6)
    o = proj((0, 0, 0)); o = (o[0], o[1] - r0)
    for tip, lab in (((0.8, 0, 0), "+x"), ((0, 0.8, 0), "+y"), ((0, 0, 0.8), "+z")):
        t = proj(tip); t = (t[0], t[1] - r0)
        a.annotate("", xy=t, xytext=o, zorder=6,
                   arrowprops=dict(arrowstyle="->", color="#9cc4ea", lw=1.3, shrinkA=0, shrinkB=0))
        a.text(t[0] + 0.3 * (t[0] - o[0]) + 6, t[1] + 0.3 * (t[1] - o[1]), lab, fontsize=8.5,
               color="#9cc4ea", ha="left", va="center", zorder=6)
    a.text(o[0] - 14, o[1] + 10, "origin,\nnet centre", fontsize=8, color="#9cc4ea", ha="right", va="top", zorder=6)
    a.set_title("Rendered scene from behind the robot's right shoulder")

    c = p["court"]; L, Wc = c["length"], c["width"]
    b.set_aspect("equal")
    b.add_patch(Rectangle((-L / 2, -Wc / 2), L, Wc, fc="#eceeea", ec=INK3, lw=0.8))
    b.plot([0, 0], [-Wc / 2 - 0.15, Wc / 2 + 0.15], color=INK, lw=2)
    y0, y1 = lp["launch_y_range"]; xm = lp["launch_x_max"]
    b.add_patch(Rectangle((y0, -xm), y1 - y0, 2 * xm, fc=BLUE, alpha=0.18, ec=BLUE, lw=1))
    sp, sg = lp["sweet_point"], lp["sweet_sigma"]
    b.add_patch(Ellipse((sp[1], sp[0]), 3 * sg[1], 3 * sg[0], fc="#8f8e89", alpha=0.35, ec="none"))
    b.plot([-2.1, -2.1], [-0.35, 0.35], color=INK3, lw=4, solid_capstyle="butt", zorder=4)
    for sgn, col, lab in ((1, RIGHT, "R"), (-1, LEFT, "L")):
        b.plot(-2.0, sgn * 0.245, "o", ms=5, color=col, mec=BG, mew=0.8, zorder=5)
        b.text(-2.4, sgn * 0.245, lab, ha="right", va="center", fontsize=7.5, color=col)
    b.set_xlim(-L / 2 - 0.6, L / 2 + 0.9); b.set_ylim(-Wc / 2 - 2.0, Wc / 2 + 0.9)
    b.set_xticks([]); b.set_yticks([])

    def dim(x0, x1, y, text):
        b.annotate("", xy=(x0, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
        b.text((x0 + x1) / 2, y - 0.18, text, ha="center", va="top", fontsize=8, color=INK2)
    yd = -Wc / 2 - 0.45
    dim(-2.0, 0.0, yd, "2.0 m"); dim(0.0, y0, yd, "3.0 m"); dim(y0, y1, yd, "3.2 m")
    dim(-L / 2, L / 2, yd - 1.1, f"court {L:g} m")
    b.annotate("", xy=(L / 2 + 0.3, -Wc / 2), xytext=(L / 2 + 0.3, Wc / 2),
               arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
    b.text(L / 2 + 0.45, 0, f"{Wc:g} m", rotation=-90, va="center", ha="left", fontsize=8, color=INK2)
    for xx, lab in ((0, "net"), (-2.0, "robot"), ((y0 + y1) / 2, "launch volume")):
        b.text(xx, Wc / 2 + 0.25, lab, ha="center", va="bottom", fontsize=8.5, color=INK)
    b.text(sp[1], -0.75, "intercepts", ha="center", va="top", fontsize=8, color=INK2)
    for (dx, dy), lab, ha, va in (((1.3, 0), "+y", "left", "center"), ((0, 1.3), "+x, robot's right", "left", "bottom")):
        b.annotate("", xy=(dx, dy), xytext=(0, 0), zorder=6,
                   arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0, shrinkA=0, shrinkB=0))
        b.text(dx + 0.12, dy + 0.1, lab, fontsize=8, color=BLUE, ha=ha, va=va, zorder=6)
    b.plot(0, 0, "o", ms=3.5, color=BLUE, mec=BG, mew=0.6, zorder=7)
    b.text(0.15, -0.35, "origin", fontsize=7.5, color=BLUE, ha="left", va="top", zorder=6)
    b.set_title("Top view, to scale")
    fig.savefig(os.path.join(out, "fig_layout.png"))
    plt.close(fig)


# -- figure 2: scorecard -----------------------------------------------------
def fig_scorecard(d, p, out, bank):
    hit, lk = d["hit"], d["locked"]
    n = len(hit)
    right = d["chosen"] == 0
    clear = hit & d["lok"]
    rel = np.linalg.norm(d["land_xy"] - d["p0_xy"], axis=1)
    med = np.nanmedian(rel[clear])
    t_med = np.median(d["t_lock"][lk])

    h = 5.5
    fig = frame(12, h, "Receive performance on the evaluation bank",
                f"{n} serves, none seen in training", "Policy model_11996 with chooser "
                "model_11997, margin 1.0", bank)
    L, R = fig._frame["L"], fig._frame["R"]
    tiles = [
        (f"{hit.mean():.1%}", "serves hit", f"{hit.sum()} of {n}"),
        (f"{right.mean():.0%} / {1 - right.mean():.0%}", "right / left arm picked",
         f"locked within {t_med:.2f} s median"),
        (f"{clear.sum() / hit.sum():.0%}", "returns clear the net", "by 0.5 m, landing on the far court"),
        (f"{med:.2f} m", "median landing error", "cleared returns, to the serve origin"),
    ]
    y_big = 1 - (M_PANEL_TOP - 0.3) / h
    for i, (big, lab, sub) in enumerate(tiles):
        x = L + i * (R - L) / 4
        fig.text(x, y_big, big, fontsize=26, color=INK, va="top")
        fig.text(x, y_big - 0.62 / h, lab, fontsize=10, color=INK, va="top")
        fig.text(x, y_big - 0.86 / h, sub, fontsize=8, color=INK3, va="top")

    ax = fig.add_axes([L + 0.07, M_PANEL_BOTTOM / h, R - L - 0.07, 1.1 / h])
    cats = [("hit, return clears the net", hit & d["lok"], BLUES[6]),
            ("hit, return does not clear", hit & ~d["lok"], BLUES[2]),
            ("miss", ~hit, "#8f8e89")]
    for row, (name, mask) in enumerate((("right arm", right), ("left arm", ~right))):
        tot = mask.sum(); left = 0.0
        for cname, cm, cc in cats:
            k = (mask & cm).sum(); w = k / tot
            ax.barh(1 - row, w, left=left, height=0.6, color=cc, edgecolor=BG, linewidth=1.5)
            if w > 0.06:
                ax.text(left + w / 2, 1 - row, f"{w:.1%}", ha="center", va="center", fontsize=8.5,
                        color=BG if cc == BLUES[6] else INK)
            left += w
        ax.text(-0.01, 1 - row, f"{name}\n{tot} serves", ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(-0.55, 1.55); ax.set_yticks([]); ax.set_xticks([])
    fig.text(L + 0.07, (M_PANEL_BOTTOM + 1.1 + 0.5) / h, "Serve outcomes by arm",
             fontsize=10.5, color=INK, va="center")
    y_leg = (M_PANEL_BOTTOM + 1.1 + 0.22) / h
    x = L + 0.07
    for cname, _, cc in cats:
        fig.text(x, y_leg, "●", color=cc, fontsize=9, va="center")
        fig.text(x + 0.013, y_leg, cname, color=INK2, fontsize=8, va="center")
        x += 0.013 + 0.0085 * len(cname) + 0.03
    fig.savefig(os.path.join(out, "fig_scorecard.png"))
    plt.close(fig)


# -- figure 3: hit rate ------------------------------------------------------
def fig_hit_rate(d, p, out, bank):
    hit, x, z = d["hit"], d["x"], d["z"]
    fig = frame(12, 5.6, "Hit rate by intercept position",
                f"Smoothed hit rate over the intercept point, from the net; "
                f"overall {hit.mean():.1%}, {(~hit).sum()} misses",
                "12 cm Gaussian kernel; blank where fewer than ~6 serves fall nearby",
                bank, legend=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.28)
    a = fig.add_subplot(gs[0])
    gs_r = gs[1].subgridspec(2, 1, hspace=0.75)
    b = fig.add_subplot(gs_r[0]); c = fig.add_subplot(gs_r[1])

    levels = [0.0, 0.80, 0.90, 0.95, 0.98, 1.0001]
    norm = BoundaryNorm(levels, SEQ.N)
    gx, gz = np.linspace(-1.4, 1.4, 141), np.linspace(0.45, 2.05, 81)
    X, Z, rate = smooth_rate(x, z, hit, gx, gz)
    a.contourf(X, Z, rate, levels=levels, cmap=SEQ, norm=norm)
    a.contour(X, Z, rate, levels=[0.95, 0.98], colors=[INK], linewidths=0.5, alpha=0.5)
    a.scatter(x[~hit], z[~hit], s=14, marker="x", c=MISS, lw=0.9, zorder=5)
    shoulders(a)
    nt = p["net"]["height_top"]
    a.axhline(nt, color=INK3, lw=0.7, ls=":")
    a.text(1.38, nt + 0.015, "net top", fontsize=7.5, color=INK2, ha="right", va="bottom",
           bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none", alpha=0.85))
    a.set_xlim(-1.4, 1.4); a.set_ylim(0.45, 2.05)
    a.set_xticks(np.arange(-1.0, 1.01, 0.5)); a.set_yticks(np.arange(0.5, 2.01, 0.5))
    a.set_xlabel("x across the court (m), +x = robot's right"); a.set_ylabel("height (m)")
    a.set_title("Hit rate map with misses")
    swatch_row(fig, a.get_position().x0, [("below 80%", SEQ(norm(0.5))), ("80 to 90%", SEQ(norm(0.85))),
                                         ("90 to 95%", SEQ(norm(0.92))), ("95 to 98%", SEQ(norm(0.96))),
                                         ("98 to 100%", SEQ(norm(0.99)))])
    dot_row(fig, b.get_position().x0, [("miss", MISS, "×")])

    def dots(ax, rows, fmt, title):
        n = len(rows)
        for k, (lo, hi, cnt, r) in enumerate(rows):
            y = n - 1 - k
            if np.isfinite(r):
                ax.plot([0.85, r], [y, y], color=GRID, lw=2.5, zorder=1)
                ax.plot(r, y, "o", ms=6.5, color=BLUE, mec=BG, mew=1, zorder=3)
                ax.text(r + 0.004, y, f"{r:.1%}", fontsize=7.5, color=INK, va="center")
            ax.text(0.845, y, fmt(lo, hi), fontsize=8, color=INK, ha="right", va="center")
            ax.text(1.062, y, f"n {cnt}", fontsize=7, color=INK3, ha="right", va="center")
        ax.set_xlim(0.85, 1.065); ax.set_ylim(-0.6, n - 0.4); ax.set_yticks([])
        ax.set_xticks([0.9, 0.95, 1.0]); ax.set_xticklabels(["90%", "95%", "100%"])
        for xv in (0.9, 0.95, 1.0):
            ax.axvline(xv, color=GRID, lw=0.6, zorder=0)
        ax.set_title(title)

    dots(b, band_rates(d, "x", [-1.4, -1.0, -0.4, 0.0, 0.4, 1.0, 1.4]),
         lambda lo, hi: f"{lo:+.1f} to {hi:+.1f}", "Hit rate by x band (m)")
    dots(c, band_rates(d, "z", [0.5, 0.9, 1.1, 1.7, 2.0]),
         lambda lo, hi: f"{lo:.1f} to {hi:.1f}", "Hit rate by height band (m)")
    fig.savefig(os.path.join(out, "fig_hit_rate.png"))
    plt.close(fig)


# -- figure 4: arm pick ------------------------------------------------------
def fig_arm_pick(d, p, out, bank):
    hit, lk, x, z = d["hit"], d["locked"], d["x"], d["z"]
    right = d["chosen"] == 0
    agree = lk & (d["chosen"] == d["rule"])
    fig = frame(12, 5.6, "Arm selection by the critic",
                f"Arm locked per serve, and hit rate for same-side vs cross-body picks "
                f"(same-side on {agree.mean():.0%} of serves)",
                "Same-side arm: the arm on the side where the serve crosses y = -1.6 m; "
                "cross-body: the other one. Dashed line = the split at x = 0",
                bank, legend=True)
    fig.subplots_adjust(right=fig._frame["R"] - 0.075)   # room for the value column
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.3, 1], wspace=0.55)
    a, b, c = (fig.add_subplot(gs[i]) for i in range(3))

    for arm, col in ((right, RIGHT), (~right, LEFT)):
        a.scatter(x[arm & hit], z[arm & hit], s=4, c=col, alpha=0.35, lw=0, zorder=2)
    a.scatter(x[~hit], z[~hit], s=14, marker="x", c=MISS, lw=0.9, zorder=5)
    shoulders(a)
    a.set_xlim(-1.4, 1.4); a.set_ylim(0.45, 2.05)
    a.set_xticks(np.arange(-1.0, 1.01, 0.5)); a.set_yticks(np.arange(0.5, 2.01, 0.5))
    a.set_xlabel("x across the court (m), +x = robot's right"); a.set_ylabel("height (m)")
    a.set_title("Intercepts by arm")
    dot_row(fig, a.get_position().x0, [("right arm", RIGHT, "●"), ("left arm", LEFT, "●"), ("miss", MISS, "×")])

    gx, gz = np.linspace(-1.4, 1.4, 141), np.linspace(0.45, 2.05, 81)
    X, Z, share = smooth_rate(x, z, right, gx, gz, bw=0.14)
    b.contourf(X, Z, share, levels=np.linspace(0, 1, 11), cmap=DIV)
    b.contour(X, Z, share, levels=[0.5], colors=[INK], linewidths=1.0)
    b.axvline(0, color=INK2, lw=0.8, ls=(0, (4, 3)))
    b.text(0.70, 0.52, "50/50 line", fontsize=7.5, color=INK, ha="left", va="bottom")
    b.set_xlim(-1.4, 1.4); b.set_ylim(0.45, 2.05)
    b.set_xticks(np.arange(-1.0, 1.01, 0.5)); b.set_yticks([])
    b.set_xlabel("x across the court (m)")
    b.set_title("Right-arm share, with the 50/50 line")
    swatch_row(fig, b.get_position().x0, [("all left", DIV(0.0)), ("50 / 50", DIV(0.5)), ("all right", DIV(1.0))])

    bands = [(-1.4, -0.7), (-0.7, -0.4), (-0.4, 0.0), (0.0, 0.4), (0.4, 0.7), (0.7, 1.4)]
    labels = []
    for k, (lo, hi) in enumerate(bands):
        y = len(bands) - 1 - k
        m = lk & (x >= lo) & (x < hi)
        ag, dg = m & agree, m & ~agree
        ha = hit[ag].mean() if ag.sum() >= MIN_N else np.nan
        hd = hit[dg].mean() if dg.sum() >= MIN_N else np.nan
        if np.isfinite(ha) and np.isfinite(hd):
            c.plot([ha, hd], [y, y], color=GRID, lw=2.5, zorder=1)
        if np.isfinite(ha):
            c.plot(ha, y, "o", ms=6.5, color=RIGHT, mec=BG, mew=1, zorder=3)
        if np.isfinite(hd):
            c.plot(hd, y, "o", ms=6.5, color=LEFT, mec=BG, mew=1, zorder=3)
        labels.append(f"{lo:+.1f} to {hi:+.1f}")
        txt = (f"{ha:.1%}" if np.isfinite(ha) else "–") + " · " + (f"{hd:.1%}" if np.isfinite(hd) else "–")
        c.text(1.05, y, txt, fontsize=7.5, color=INK, ha="left", va="center",
               transform=c.get_yaxis_transform())
    c.set_xlim(0.955, 1.0); c.set_ylim(-0.6, len(bands) - 0.4)
    c.set_yticks(range(len(bands) - 1, -1, -1)); c.set_yticklabels(labels)
    c.set_xticks([0.96, 0.98, 1.0]); c.set_xticklabels(["96%", "98%", "100%"])
    for xv in (0.96, 0.98, 1.0):
        c.axvline(xv, color=GRID, lw=0.6, zorder=0)
    c.set_xlabel("hit rate")
    c.set_title("Hit rate by x band (m), same-side vs cross-body")
    dot_row(fig, c.get_position().x0, [("same-side arm", RIGHT, "●"), ("cross-body arm", LEFT, "●")])
    fig.savefig(os.path.join(out, "fig_arm_pick.png"))
    plt.close(fig)


# -- figure 5: returns -------------------------------------------------------
def fig_returns(d, p, out, bank):
    hit = d["hit"]
    nt, margin = p["net"]["height_top"], p["control"]["return_net_clearance"]
    cx, cz = d["cross_xz"][:, 0], d["cross_xz"][:, 1]
    crossed = hit & np.isfinite(cx)
    short = hit & ~np.isfinite(cx)
    into = crossed & (cz <= nt)
    low = crossed & (cz > nt) & (cz <= nt + margin)
    clear = crossed & (cz > nt + margin)
    cl_n = int((hit & d["lok"]).sum())
    fig = frame(12, 5.6, "Return quality",
                f"{hit.sum()} hits: {clear.sum() / hit.sum():.1%} over the net by 0.5 m or more, "
                f"{low.sum() / hit.sum():.1%} by less, {into.sum() / hit.sum():.1%} into the net, "
                f"{short.sum() / hit.sum():.1%} short of it; {cl_n} cleared returns mapped",
                "Return flight predicted from the shuttle state at contact through the drag model",
                bank, legend=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.22)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])

    W = p["court"]["width"]
    a.add_patch(Rectangle((-W / 2, p["net"]["height_bottom"]), W, nt - p["net"]["height_bottom"],
                          fc="#e4e3de", ec="none"))
    a.plot([-W / 2, W / 2], [nt, nt], color=INK, lw=1.5)
    a.axhline(nt + margin, color=INK2, lw=0.7, ls=":")
    a.text(W / 2 - 0.05, nt + margin + 0.03, "required clearance (+0.5 m)", fontsize=7.5, color=INK2,
           ha="right", va="bottom")
    a.text(W / 2 - 0.05, nt + 0.03, "net top, 1.55 m", fontsize=7.5, color=INK2, ha="right", va="bottom")
    a.scatter(cx[clear], cz[clear], s=5, c=BLUE, alpha=0.25, lw=0, zorder=3)
    a.scatter(cx[low], cz[low], s=9, c=INK, alpha=0.8, lw=0, zorder=4)
    a.scatter(cx[into], cz[into], s=14, marker="x", c=MISS, lw=0.8, zorder=5)
    a.set_xlim(-W / 2 - 0.3, W / 2 + 0.3); a.set_ylim(-0.2, 4.0)
    a.set_xlabel("x across the court at the net line (m), +x = robot's right"); a.set_ylabel("height (m)")
    hgrid(a, (1, 2, 3, 4))
    a.set_title("Net-line crossings, seen from the robot")
    dot_row(fig, a.get_position().x0, [(f"clears by the margin ({clear.sum()})", BLUE, "●"),
                                          (f"over, inside the margin ({low.sum()})", INK, "●"),
                                          (f"into the net ({into.sum()})", MISS, "×")])

    rel = d["land_xy"] - d["p0_xy"]
    cl = hit & d["lok"]
    e = np.arange(-3.0, 3.01, 0.25)
    Hc, _, _ = np.histogram2d(rel[cl, 0], rel[cl, 1], bins=[e, e])
    Hc = np.ma.masked_equal(Hc / cl.sum() * 100, 0)
    cm = SEQ.copy(); cm.set_bad(BG)
    b.pcolormesh(e, e, Hc.T, cmap=cm, vmin=0, vmax=np.percentile(Hc.compressed(), 99.5), edgecolors="none")
    dist = np.linalg.norm(rel[cl], axis=1)
    for rr in (0.5, 1.0, 2.0):
        b.add_patch(plt.Circle((0, 0), rr, fc="none", ec=INK, lw=0.8, ls=(0, (3, 2))))
        b.text(rr * 0.72, rr * 0.72, f"{(dist <= rr).mean():.0%} within {rr:g} m", fontsize=7.5,
               color=INK, ha="left", va="bottom")
    b.plot(0, 0, "+", color=INK, ms=9, mew=1.2)
    b.set_aspect("equal"); b.set_xlim(-3, 3); b.set_ylim(-3, 3)
    b.set_xlabel("across (m), +x = robot's right"); b.set_ylabel("along the court (m), + = past the serve origin")
    b.set_title("Landing point relative to the serve's origin")
    fig.savefig(os.path.join(out, "fig_returns.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="runs/eval_pstar_hits_holdout_seed1.npy")
    ap.add_argument("--out", default="runs/figs_blog")
    ap.add_argument("--bank", default="Evaluation bank, 8192 episodes")
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d, p = load(args.input)
    if not args.skip_render:
        fig_layout(p, args.out, args.bank)
    fig_scorecard(d, p, args.out, args.bank)
    fig_hit_rate(d, p, args.out, args.bank)
    fig_arm_pick(d, p, args.out, args.bank)
    fig_returns(d, p, args.out, args.bank)
    print("wrote", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
