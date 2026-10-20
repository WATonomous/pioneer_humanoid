"""Pictorial figures from a bank eval: the rally drawn on the court, a smooth
strike-zone map on a robot silhouette, and a scorecard with the flow of
serves to outcomes.

  uv run scripts/plot_eval_story.py [--input runs/eval_pstar_hits.npy]
                                    [--out runs/figs] [--serves 40] [--seed 0]

Serve arcs come from the launcher (sampled locally, CPU) and the return arcs
from the ideal shooting solution aero.solve_u_out at each sampled intercept.
The arm drawn for a sampled serve is the chooser's majority pick among the
15 nearest eval intercepts, and its hit chance their hit rate.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, FancyBboxPatch, PathPatch, Ellipse  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aero  # noqa: E402
import launcher  # noqa: E402
from plot_eval import (RIGHT, LEFT, MISS, INK, INK2, GRID, SURF, SEQ_BLUE,  # noqa: E402
                       DIV_ARM, load)

NEUTRAL = "#9a9890"


def robot_front(ax, p, scale=1.0):
    """Stand + shoulder bar + two hanging rackets, seen from the net."""
    h = p["arm"]["mount_height"]
    bx = p["arm"]["base_x"]
    ax.add_patch(Rectangle((bx - 0.06, 0), 0.12, h, color=INK2, alpha=0.5, lw=0))
    ax.add_patch(Rectangle((bx - 0.32, h - 0.04), 0.64, 0.08, color=INK2,
                           alpha=0.6, lw=0))
    for sx, col in ((0.245, RIGHT), (-0.245, LEFT)):
        ax.plot([bx + sx, bx + sx], [h, h - 0.55], color=col, lw=3,
                solid_capstyle="round", alpha=0.9)
        ax.add_patch(Ellipse((bx + sx, h - 0.55 - 0.08), 0.12, 0.15,
                             fc=SURF, ec=col, lw=2))


def robot_side(ax, p):
    """Stand seen from the side (x into the page)."""
    h = p["arm"]["mount_height"]
    by = p["arm"]["base_y"]
    ax.add_patch(Rectangle((by - 0.06, 0), 0.12, h, color=INK2, alpha=0.5, lw=0))
    ax.plot([by - 0.1, by + 0.1], [h, h], color=INK2, lw=4, alpha=0.6,
            solid_capstyle="round")
    ax.plot([by, by], [h, h - 0.55], color=INK, lw=3, solid_capstyle="round",
            alpha=0.6)
    ax.add_patch(Ellipse((by, h - 0.63), 0.03, 0.15, fc=SURF, ec=INK, lw=2))


def net_side(ax, p):
    nb, nt = p["net"]["height_bottom"], p["net"]["height_top"]
    ax.plot([0, 0], [0, nt], color=INK, lw=1.2)
    ax.add_patch(Rectangle((-0.015, nb), 0.03, nt - nb, color=INK, alpha=0.85, lw=0))
    ax.text(0.08, nt + 0.03, "net", color=INK2, fontsize=9)


def sample_rally(d, p, n_serves, seed):
    """Serve specs with their ideal return arcs and the empirical pick/hit."""
    specs = launcher.sample_specs(n_serves, seed=seed, params=p)
    pts = np.stack([d["x"], d["y"], d["z"]], axis=1)
    tree = cKDTree(pts)
    k = p["shuttle"]["k"]
    out = []
    for s in specs:
        _, idx = tree.query(s.p_star, k=15)
        right_share = (d["chosen"][idx] == 0).mean()
        hit_p = d["hit"][idx].mean()
        # the policy is rewarded for landing the return where the serve was
        # launched from (mdp.return_landing), so that is the target drawn
        u, speed, v_out = aero.solve_u_out(s.p_star, s.p0[:2], p)
        t, P, _ = aero.simulate(s.p_star, v_out, k, dt=2e-3, t_max=4.0)
        out.append(dict(spec=s, right=right_share >= 0.5, hit_p=hit_p, ret=P))
    return out


def fig_rally(d, p, out, label, rallies):
    L, W = p["court"]["length"], p["court"]["width"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(15, 12),
                               gridspec_kw={"height_ratios": [1.15, 0.85]})
    # side view
    a.add_patch(Rectangle((-L / 2, -0.03), L, 0.03, color=INK2, lw=0))
    net_side(a, p)
    robot_side(a, p)
    for r in rallies:
        P = r["spec"].ref_traj[1]
        a.plot(P[:, 1], P[:, 2], color=NEUTRAL, lw=0.9, alpha=0.55)
        col = RIGHT if r["right"] else LEFT
        R = r["ret"]
        a.plot(R[:, 1], R[:, 2], color=col, lw=1.1, alpha=0.6)
        a.plot(P[-1, 1], P[-1, 2], "o", color=col, ms=5, mec=SURF, mew=0.6, zorder=5)
    for r in rallies:
        p0 = r["spec"].p0
        a.plot([p0[1], p0[1]], [0, p0[2]], color=NEUTRAL, lw=0.6, ls=":", alpha=0.6)
        a.plot(p0[1], 0, "v", color=NEUTRAL, ms=5, zorder=6)
    zmax = max(r["ret"][:, 2].max() for r in rallies) + 0.3
    a.text(-3.1, zmax - 0.1, "serves launched from the far\ncourt (grey arcs, from the\n"
           "grey squares) and intercepted\nin front of the robot (dots)",
           fontsize=9, color=INK2, va="top")
    a.text(-3.1, zmax * 0.55, "ideal returns from each\nintercept back to where\n"
           "that serve was launched\nfrom (grey triangles),\ncoloured by the arm\n"
           "the chooser picks", fontsize=9, color=INK2, va="top")
    a.set_xlim(-3.2, L / 2 + 0.3)
    a.set_ylim(-0.4, zmax)
    a.set_aspect("equal")
    a.set_xlabel("y along the court (m); robot at y = -2, net at 0")
    a.set_ylabel("height (m)")
    a.set_title("Side view: a sample of the serve bank and the ideal return from each intercept "
                "to the serve's origin")
    a.grid(False)
    a.spines["left"].set_visible(False)
    # top view
    b.add_patch(Rectangle((-L / 2, -W / 2), L, W, fill=False, ec=INK2, lw=1.4))
    b.plot([0, 0], [-W / 2, W / 2], color=INK, lw=2.5)
    b.plot([-L / 2, L / 2], [-W / 2 + 0.46, -W / 2 + 0.46], color=GRID, lw=1)
    b.plot([-L / 2, L / 2], [W / 2 - 0.46, W / 2 - 0.46], color=GRID, lw=1)
    by = p["arm"]["base_y"]
    b.add_patch(Rectangle((by - 0.1, -0.35), 0.2, 0.7, color=INK2, alpha=0.6, lw=0))
    b.plot([by, by], [0.245, 0.245], "o", color=RIGHT, ms=7, zorder=5)
    b.plot([by, by], [-0.245, -0.245], "o", color=LEFT, ms=7, zorder=5)
    for r in rallies:
        P = r["spec"].ref_traj[1]
        b.plot(P[:, 1], P[:, 0], color=NEUTRAL, lw=0.9, alpha=0.55)
        col = RIGHT if r["right"] else LEFT
        R = r["ret"]
        b.plot(R[:, 1], R[:, 0], color=col, lw=1.1, alpha=0.6)
        b.plot(P[-1, 1], P[-1, 0], "o", color=col, ms=5, mec=SURF, mew=0.6, zorder=5)
        b.plot(P[0, 1], P[0, 0], "s", color=NEUTRAL, ms=3, zorder=4)
    b.text(3.2, 2.75, "each return aims back at its own serve's launch point "
           "(the square it came from)", fontsize=9, color=INK2, ha="center")
    b.set_xlim(-3.2, L / 2 + 0.3)
    b.set_ylim(-W / 2 - 0.25, W / 2 + 0.25)
    b.set_aspect("equal")
    b.set_xlabel("y along the court (m)")
    b.set_ylabel("x across the court (m)\n+x = robot's right")
    b.set_title("Top view: squares = where the serve was launched, dots = intercept "
                "(coloured by the arm picked), return arcs go back to the square")
    b.grid(False)
    fig.suptitle(f"The rally, {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rally.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def smooth_rate(x, z, ok, gx, gz, bw=0.12, min_w=6.0):
    """Gaussian-kernel (Nadaraya-Watson) rate on a grid; nan where too few
    episodes contribute (sum of kernel weights < min_w)."""
    X, Z = np.meshgrid(gx, gz)
    rate = np.full(X.shape, np.nan)
    wsum = np.zeros(X.shape)
    for i in range(0, len(x), 2048):
        dx = (X[..., None] - x[i:i + 2048]) / bw
        dz = (Z[..., None] - z[i:i + 2048]) / bw
        w = np.exp(-0.5 * (dx * dx + dz * dz))
        wsum += w.sum(-1)
        rate = np.nan_to_num(rate) + (w * ok[i:i + 2048]).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(wsum >= min_w, rate / wsum, np.nan)
    return rate


def fig_strike_zone(d, p, out, label):
    gx = np.linspace(-1.6, 1.6, 161)
    gz = np.linspace(0.3, 2.1, 91)
    hit = smooth_rate(d["x"], d["z"], d["hit"].astype(float), gx, gz)
    right = smooth_rate(d["x"], d["z"], (d["chosen"] == 0).astype(float), gx, gz)
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6.2))
    vmin = 0.8
    im = a.contourf(gx, gz, np.clip(hit, vmin, 1), levels=np.linspace(vmin, 1, 11),
                    cmap=SEQ_BLUE, vmin=vmin, vmax=1)
    a.contour(gx, gz, hit, levels=[0.9, 0.95], colors=[INK], linewidths=0.6,
              linestyles=["--", "-"])
    m = ~d["hit"]
    a.scatter(d["x"][m], d["z"][m], s=22, marker="x", c=MISS, lw=1,
              label=f"miss (n={m.sum()})", zorder=5)
    robot_front(a, p)
    nb, nt = p["net"]["height_bottom"], p["net"]["height_top"]
    a.axhline(nt, color=INK2, lw=0.8, ls=":")
    a.text(-1.55, nt + 0.02, "net top", fontsize=8, color=INK2)
    cb = fig.colorbar(im, ax=a, shrink=0.85, pad=0.02, ticks=[0.8, 0.9, 0.95, 1.0])
    cb.set_label("hit rate (smoothed, 12 cm kernel)")
    cb.outline.set_visible(False)
    a.set_title(f"Hit chance seen from the net (overall {d['hit'].mean():.1%})")
    a.legend(loc="upper left", fontsize=9)

    im2 = b.contourf(gx, gz, right, levels=np.linspace(0, 1, 11), cmap=DIV_ARM,
                     vmin=0, vmax=1)
    b.contour(gx, gz, right, levels=[0.5], colors=[INK], linewidths=0.8)
    robot_front(b, p)
    cb2 = fig.colorbar(im2, ax=b, shrink=0.85, pad=0.02, ticks=[0, 0.5, 1])
    cb2.set_ticklabels(["left arm", "50/50", "right arm"])
    cb2.set_label("which arm the chooser picks")
    cb2.outline.set_visible(False)
    b.set_title("Which arm takes the shuttle (black line = 50/50)")
    for ax in (a, b):
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(0, 2.1)
        ax.set_xlabel("x across the court (m), +x = robot's right")
        ax.set_ylabel("height (m)")
        ax.grid(False)
        ax.axhline(0, color=INK2, lw=1)
    fig.suptitle(f"Strike zone, {label}  (white = fewer than ~6 episodes nearby)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "strike_zone.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def band(ax, x0, y0, h0, x1, y1, h1, color, alpha=0.55):
    """Smooth ribbon from a segment at x0 to a segment at x1."""
    verts = [(x0, y0), ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1), (x1, y1),
             (x1, y1 + h1), ((x0 + x1) / 2, y1 + h1), ((x0 + x1) / 2, y0 + h0),
             (x0, y0 + h0), (x0, y0)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=color, ec="none", alpha=alpha))


def fig_scorecard(d, p, out, label):
    n = len(d["hit"])
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.42, 1], hspace=0.35, wspace=0.25)
    # tiles
    hit = d["hit"].mean()
    clear = d["lok"][d["hit"]].mean()
    has = np.isfinite(d["land"]) & d["lok"]
    med = np.median(d["land"][has])
    tiles = [(f"{hit:.1%}", "serves hit", f"{d['hit'].sum()} of {n}"),
             (f"{(d['chosen'] == 0).mean():.0%} / {(d['chosen'] == 1).mean():.0%}",
              "right / left arm picked", "critic decides within 0.22 s"),
             (f"{clear:.0%}", "returns clear the net", "by 0.5 m, landing on the far court"),
             (f"{med:.2f} m", "median landing error", "cleared returns, to the serve origin")]
    for i, (big, name, sub) in enumerate(tiles):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02",
                                    fc="#f1f0ec", ec="none"))
        ax.text(0.5, 0.62, big, ha="center", va="center", fontsize=26,
                fontweight="bold", color=INK)
        ax.text(0.5, 0.3, name, ha="center", va="center", fontsize=11, color=INK)
        ax.text(0.5, 0.13, sub, ha="center", va="center", fontsize=8.5, color=INK2)
    # flow
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.08)
    cols = {0: RIGHT, 1: LEFT}
    gap = 0.03
    # column 1: all serves; column 2: arm; column 3: outcome per arm
    x0, x1, x2 = 0.5, 3.8, 7.2
    w = 0.7
    ax.add_patch(Rectangle((x0, 0), w, 1, color=NEUTRAL, lw=0))
    ax.text(x0 + w / 2, 1.03, f"{n} serves", ha="center", fontsize=10)
    y_src = 0.0
    y_arm = 0.0
    y_out = 0.0
    outs = []
    for arm, name in ((0, "right arm"), (1, "left arm")):
        m = d["chosen"] == arm
        f = m.mean()
        ax.add_patch(Rectangle((x1, y_arm), w, f - gap / 2, color=cols[arm], lw=0))
        band(ax, x0 + w, y_src, f - gap / 2, x1, y_arm, f - gap / 2, cols[arm], 0.35)
        ax.text(x1 + w / 2, y_arm + (f - gap / 2) / 2, f"{name}\n{m.sum()}",
                ha="center", va="center", fontsize=9, color=SURF)
        parts = [("hit, return clears net", m & d["hit"] & d["lok"], cols[arm], 1.0),
                 ("hit, return does not clear", m & d["hit"] & ~d["lok"], cols[arm], 0.45),
                 ("miss", m & ~d["hit"], MISS, 1.0)]
        yy = y_arm
        label_y = None
        for pname, pm, pc, pa in reversed(parts):
            pf = pm.mean()
            hh = max(pf - gap / 4, 0.004)
            ax.add_patch(Rectangle((x2, y_out), w, hh, color=pc, alpha=pa, lw=0))
            band(ax, x1 + w, yy, pf, x2, y_out, hh, pc, 0.3 * pa + 0.08)
            ly = y_out + hh / 2
            if label_y is not None:
                ly = max(ly, label_y + 0.05)
            label_y = ly
            ax.plot([x2 + w, x2 + w + 0.1], [y_out + hh / 2, ly], color=INK2,
                    lw=0.5)
            ax.text(x2 + w + 0.12, ly,
                    f"{pname}: {pm.sum()} ({pm.sum() / max(m.sum(), 1):.1%} of {name})",
                    va="center", fontsize=9, color=INK)
            yy += pf
            y_out += hh + gap / 4
        y_src += f
        y_arm += f + gap / 2
        y_out += gap
    fig.suptitle(f"Scorecard, {label}", fontsize=13, y=0.98)
    fig.savefig(os.path.join(out, "scorecard.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def court_top(ax, p, far_only=True):
    L, W = p["court"]["length"], p["court"]["width"]
    y0 = 0 if far_only else -L / 2
    ax.add_patch(Rectangle((-W / 2, y0), W, L / 2 if far_only else L, fill=False,
                           ec=INK2, lw=1.4, zorder=3))
    ax.plot([-W / 2, W / 2], [0, 0], color=INK, lw=3, zorder=4)
    for sx in (-W / 2 + 0.46, W / 2 - 0.46):
        ax.plot([sx, sx], [y0, y0 + (L / 2 if far_only else L)], color=GRID, lw=1, zorder=3)
    ax.plot([-W / 2, W / 2], [1.98, 1.98], color=GRID, lw=1, zorder=3)  # short service line
    ax.plot([-W / 2, W / 2], [L / 2 - 0.76, L / 2 - 0.76], color=GRID, lw=1, zorder=3)
    ax.set_aspect("equal")


def fig_landing_map(d, p, out, label):
    if "land_xy" not in d:
        print("landing_map: eval rows have no landing columns, skipped")
        return
    L, W = p["court"]["length"], p["court"]["width"]
    hit = d["hit"] & np.isfinite(d["land_xy"][:, 0])
    lx, ly = d["land_xy"][hit, 0], d["land_xy"][hit, 1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 7.2),
                               gridspec_kw={"width_ratios": [1.25, 1]})
    # where the returns come down, full width so the net-side landings show
    xe = np.arange(-W / 2 - 0.5, W / 2 + 0.51, 0.25)
    ye = np.arange(-2.5, L / 2 + 0.76, 0.25)
    H, _, _ = np.histogram2d(lx, ly, bins=[xe, ye])
    H = H / hit.sum() * 100
    im = a.pcolormesh(xe, ye, np.where(H.T > 0, H.T, np.nan), cmap=SEQ_BLUE,
                      vmin=0, vmax=np.percentile(H[H > 0], 98), zorder=2)
    court_top(a, p, far_only=False)
    lp = p["launcher"]
    a.add_patch(Rectangle((-lp["launch_x_max"], lp["launch_y_range"][0]),
                          2 * lp["launch_x_max"],
                          lp["launch_y_range"][1] - lp["launch_y_range"][0],
                          fill=False, ec=LEFT, lw=1.5, ls="--", zorder=5))
    a.text(-lp["launch_x_max"] + 0.1, lp["launch_y_range"][1] - 0.25,
           "serves are launched\nfrom inside this box", fontsize=8.5, color=LEFT,
           va="top", zorder=6, bbox=dict(fc=SURF, ec="none", alpha=0.8, pad=1))
    by = p["arm"]["base_y"]
    a.plot([-0.245, 0.245], [by, by], "s", color=INK, ms=6, zorder=6)
    a.text(0.4, by, "robot", va="center", fontsize=9, color=INK2)
    short = (ly > 0) & (ly < 1.98)
    own = ly <= 0
    a.text(-W / 2 - 0.45, -0.3, f"{own.mean():.1%} come down on the robot's side",
           fontsize=8.5, color=INK2, va="top")
    a.set_xlim(-W / 2 - 0.5, W / 2 + 0.5)
    a.set_ylim(-2.5, L / 2 + 0.75)
    a.set_xlabel("x across the court (m), +x = robot's right")
    a.set_ylabel("y along the court (m), net at 0")
    a.set_title(f"Where the {hit.sum()} returns come down (% of returns per 25 cm cell)")
    a.grid(False)
    cb = fig.colorbar(im, ax=a, shrink=0.8, pad=0.02)
    cb.set_label("% of returns")
    cb.outline.set_visible(False)
    # relative to the serve origin: the accuracy picture
    rel = d["land_xy"][hit] - d["p0_xy"][hit]
    cl = d["lok"][hit]
    re = np.arange(-3.0, 3.01, 0.2)
    H2, _, _ = np.histogram2d(rel[cl, 0], rel[cl, 1], bins=[re, re])
    H2 = H2 / cl.sum() * 100
    im2 = b.pcolormesh(re, re, np.where(H2.T > 0, H2.T, np.nan), cmap=SEQ_BLUE,
                       vmin=0, vmax=np.percentile(H2[H2 > 0], 98), zorder=2)
    dist = np.linalg.norm(rel[cl], axis=1)
    for rr in (0.5, 1.0, 2.0):
        b.add_patch(plt.Circle((0, 0), rr, fill=False, ec=INK, lw=0.9, ls="--", zorder=4))
        b.text(rr * 0.72, rr * 0.72, f"{(dist <= rr).mean():.0%} within {rr:g} m",
               fontsize=8.5, color=INK, zorder=5,
               bbox=dict(fc=SURF, ec="none", alpha=0.7, pad=1))
    b.plot(0, 0, "+", color=INK, ms=12, mew=1.5, zorder=5)
    b.text(0.12, -0.3, "serve origin", fontsize=8.5, color=INK, zorder=5,
           bbox=dict(fc=SURF, ec="none", alpha=0.7, pad=1))
    b.set_aspect("equal")
    b.set_xlim(-3, 3)
    b.set_ylim(-3, 3)
    b.set_xlabel("across (m), + = robot's right")
    b.set_ylabel("along the court (m), + = deeper than the serve origin")
    b.set_title(f"Landing point relative to the serve's origin ({cl.sum()} returns that cleared the net)")
    b.grid(False)
    cb2 = fig.colorbar(im2, ax=b, shrink=0.8, pad=0.02)
    cb2.set_label("% of cleared returns")
    cb2.outline.set_visible(False)
    fig.suptitle(f"Return accuracy, {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "landing_map.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_net_view(d, p, out, label):
    if "cross_xz" not in d:
        print("net_view: eval rows have no net-crossing columns, skipped")
        return
    W = p["court"]["width"]
    nb, nt = p["net"]["height_bottom"], p["net"]["height_top"]
    margin = p["control"]["return_net_clearance"]
    hit = d["hit"]
    cx, cz = d["cross_xz"][:, 0], d["cross_xz"][:, 1]
    crossed = hit & np.isfinite(cx)
    never = hit & ~np.isfinite(cx)
    into_net = crossed & (cz <= nt)
    low = crossed & (cz > nt) & (cz <= nt + margin)
    clear = crossed & (cz > nt + margin)
    fig, ax = plt.subplots(figsize=(15, 7))
    # posts, band, mesh
    ax.add_patch(Rectangle((-W / 2, nb), W, nt - nb, fc="#eceae4", ec="none", zorder=1))
    for gz in np.arange(nb, nt, 0.08):
        ax.plot([-W / 2, W / 2], [gz, gz], color=GRID, lw=0.5, zorder=1)
    for gx in np.arange(-W / 2, W / 2 + 0.01, 0.08):
        ax.plot([gx, gx], [nb, nt], color=GRID, lw=0.5, zorder=1)
    ax.plot([-W / 2, W / 2], [nt, nt], color=INK, lw=3, zorder=3)
    for px in (-W / 2, W / 2):
        ax.plot([px, px], [0, nt], color=INK, lw=4, zorder=3)
    ax.axhline(nt + margin, color=INK2, lw=0.8, ls=":", zorder=3)
    ax.text(W / 2 + 0.1, nt + margin, f"required clearance\n(+{margin:g} m)", fontsize=8.5,
            color=INK2, va="center")
    ax.text(W / 2 + 0.1, nt, "net top", fontsize=8.5, color=INK2, va="center")
    ax.axhline(0, color=INK2, lw=1)
    # crossings
    ax.scatter(cx[clear], cz[clear], s=7, c=RIGHT, alpha=0.35, lw=0,
               label=f"clears by the margin ({clear.sum()}, {clear.sum() / hit.sum():.1%})", zorder=4)
    ax.scatter(cx[low], cz[low], s=16, c="#eda100", alpha=0.8, lw=0,
               label=f"over the net but inside the margin ({low.sum()}, {low.sum() / hit.sum():.1%})", zorder=5)
    ax.scatter(cx[into_net], cz[into_net], s=30, marker="x", c=MISS, lw=1.2,
               label=f"into the net ({into_net.sum()}, {into_net.sum() / hit.sum():.1%})", zorder=6)
    ax.legend(loc="upper left", fontsize=9, markerscale=1.5)
    # robot position for reference
    for sx, col in ((0.245, RIGHT), (-0.245, LEFT)):
        ax.plot(sx, 0.05, "^", color=col, ms=8, zorder=6)
    ax.text(0, -0.22, "robot's arms, 2 m behind the net", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlim(-W / 2 - 0.3, W / 2 + 1.4)
    ax.set_ylim(-0.35, max(4.0, np.nanpercentile(cz[crossed], 99.5) + 0.2))
    ax.set_aspect("equal")
    ax.set_xlabel("x across the court (m), +x = robot's right")
    ax.set_ylabel("height at the net line (m)")
    ax.grid(False)
    extra = f"; {never.sum()} hit returns never reach the net line" if never.any() else ""
    ax.set_title(f"Where the returns cross the net, seen from the robot's side "
                 f"({hit.sum()} hits{extra})")
    fig.suptitle(f"Net view, {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "net_view.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="runs/eval_pstar_hits.npy")
    ap.add_argument("--out", default="runs/figs")
    ap.add_argument("--label", default="")
    ap.add_argument("--serves", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None,
                    help="pickle of the sampled rallies (written if missing)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d, p = load(args.input)
    label = args.label or os.path.basename(args.input)
    fig_scorecard(d, p, args.out, label)
    fig_strike_zone(d, p, args.out, label)
    fig_landing_map(d, p, args.out, label)
    fig_net_view(d, p, args.out, label)
    import pickle
    if args.cache and os.path.exists(args.cache):
        with open(args.cache, "rb") as f:
            rallies = pickle.load(f)
    else:
        rallies = sample_rally(d, p, args.serves, args.seed)
        if args.cache:
            with open(args.cache, "wb") as f:
                pickle.dump(rallies, f)
    fig_rally(d, p, args.out, label, rallies)
    print(f"figures in {args.out}/: scorecard.png strike_zone.png landing_map.png "
          "net_view.png rally.png")


if __name__ == "__main__":
    main()
