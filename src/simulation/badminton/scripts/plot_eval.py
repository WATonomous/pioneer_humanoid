"""Figures from a bank eval (runs/eval_pstar_hits.npy written by eval_rl.py).

  uv run scripts/plot_eval.py [--input runs/eval_pstar_hits.npy] [--out runs/figs]
                              [--label "model_11996 + chooser 11997"]

Writes
  picks.png          top-down court and front view of every intercept point
                     p*, coloured by the arm the chooser locked, hits filled,
                     misses hollow; rule disagreements ringed
  hit_chance.png     hit rate binned over p* (front view x/z, top view x/y)
  returns.png        outcome per lateral and height bin (miss / hit that
                     does not clear the net / hit that clears and lands on
                     the far court) and the landing error of cleared returns
  face_map.png       cork position at closest approach in each racket's face
                     frame, hits vs misses
  arm_pick.png       lock time and rule agreement of the arm pick
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aero  # noqa: E402

# categorical slots 1/2 (right/left arm), status red for misses, sequential blue
RIGHT, LEFT, MISS = "#2a78d6", "#eb6834", "#e34948"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGES = ["#fde3d6", "#f9bda0", "#f39a70", "#eb6834", "#c94f1f", "#9c3b13", "#6e280a"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seqblue", BLUES)
SEQ_ORANGE = LinearSegmentedColormap.from_list("seqorange", ORANGES)
DIV_ARM = LinearSegmentedColormap.from_list(
    "divarm", ["#c94f1f", "#f39a70", "#e8e7e2", "#86b6ef", "#1c5cab"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def load(path):
    r = np.load(path)
    p = aero.load_params()
    base_y = p["arm"]["base_y"]
    d = dict(
        front=r[:, 0], z=r[:, 1], x=r[:, 2], hit=r[:, 3].astype(bool),
        u=r[:, 4], v=r[:, 5], dmin=r[:, 6], land=r[:, 7],
        lok=r[:, 8].astype(bool), side=r[:, 9], locked=r[:, 10].astype(bool),
        chosen=r[:, 11], rule=r[:, 12], t_lock=r[:, 13],
    )
    d["y"] = d["front"] + base_y
    if r.shape[1] >= 20:  # eval_rl.py from 2026-10-20 on
        d["land_xy"], d["p0_xy"], d["cross_xz"] = r[:, 14:16], r[:, 16:18], r[:, 18:20]
    return d, p


def court(ax, p):
    """Top-down court outline: x across, y along, net at y = 0."""
    L, W = p["court"]["length"], p["court"]["width"]
    ax.add_patch(Rectangle((-W / 2, -L / 2), W, L, fill=False, lw=1.2,
                           ec=INK2, zorder=1))
    ax.axhline(0, color=INK, lw=2, zorder=2)
    ax.text(-1.8, 0.07, "net (y = 0)", va="bottom", ha="left", color=INK2)
    by, bx = p["arm"]["base_y"], p["arm"]["base_x"]
    ax.plot([bx - 0.245, bx + 0.245], [by, by], "s", color=INK, ms=6, zorder=3)
    ax.text(bx + 0.35, by, "arm bases", va="center", ha="left", color=INK2)
    ax.set_aspect("equal")
    ax.set_xlabel("x lateral (m), +x = robot's right")
    ax.set_ylabel("y along court (m)")


def scatter_arm(ax, xk, yk, d):
    """Hits filled in the arm colour, misses red crosses on top."""
    big = len(d["hit"]) > 2000
    for arm, col, name in ((0, RIGHT, "right arm"), (1, LEFT, "left arm")):
        m = d["chosen"] == arm
        h = m & d["hit"]
        ax.scatter(d[xk][h], d[yk][h], s=5 if big else 14, c=col, lw=0,
                   alpha=0.35 if big else 0.75,
                   label=f"{name}, hit (n={h.sum()})", zorder=4)
        mm = m & ~d["hit"]
        ax.scatter(d[xk][mm], d[yk][mm], s=34, marker="x", c=MISS, lw=1.2,
                   label=f"{name}, miss (n={mm.sum()})" if arm == 0 else None,
                   zorder=5)
        if arm == 1 and mm.any():
            ax.scatter(d[xk][mm], d[yk][mm], s=34, marker="x", c=MISS, lw=1.2,
                       zorder=5)


PICK_XE = np.array([-1.4, -1.0, -0.7, -0.4, -0.2, 0.0, 0.2, 0.4, 0.7, 1.0, 1.4])
PICK_ZE = np.array([0.5, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0])


def fig_picks(d, p, out, label):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(18, 5.4),
                                  gridspec_kw={"width_ratios": [1, 1, 1.15]})
    court(a, p)
    scatter_arm(a, "x", "y", d)
    a.set_xlim(-1.9, 1.9)
    a.set_ylim(-2.4, 0.35)
    a.set_title("Intercept points p*, top-down (court cropped to |x| < 1.9 m)")
    a.legend(loc="lower left", fontsize=8, markerscale=1.2)

    scatter_arm(b, "x", "z", d)
    nb, nt = p["net"]["height_bottom"], p["net"]["height_top"]
    b.axhspan(nb, nt, color=GRID, alpha=0.5, lw=0, zorder=0)
    b.text(-1.85, (nb + nt) / 2, "net band\nheight", va="center", fontsize=8,
           color=INK2)
    b.axvline(0.245, color=RIGHT, lw=0.8, ls=":", zorder=1)
    b.axvline(-0.245, color=LEFT, lw=0.8, ls=":", zorder=1)
    b.set_aspect("equal")
    b.set_xlim(-1.9, 1.9)
    b.set_ylim(0.3, 2.3)
    b.set_xlabel("x lateral (m)")
    b.set_ylabel("z height (m)")
    b.set_title("Intercept points p*, seen from the net")

    right = d["chosen"] == 0
    rate, n = binned_rate(d["x"], d["z"], right, PICK_XE, PICK_ZE)
    im = c.pcolormesh(PICK_XE, PICK_ZE, rate.T, cmap=DIV_ARM, vmin=0, vmax=1,
                      edgecolors=SURF, lw=2)
    for i in range(len(PICK_XE) - 1):
        for j in range(len(PICK_ZE) - 1):
            if n[i, j] < 8:
                continue
            c.text((PICK_XE[i] + PICK_XE[i + 1]) / 2,
                   (PICK_ZE[j] + PICK_ZE[j + 1]) / 2,
                   f"{rate[i, j]:.0%}", ha="center", va="center", fontsize=7,
                   color=SURF if abs(rate[i, j] - 0.5) > 0.3 else INK)
    c.axvline(0.245, color=RIGHT, lw=0.8, ls=":")
    c.axvline(-0.245, color=LEFT, lw=0.8, ls=":")
    c.set_xlim(PICK_XE[0], PICK_XE[-1])
    c.set_ylim(PICK_ZE[0], PICK_ZE[-1])
    c.set_xticks(PICK_XE)
    c.tick_params(axis="x", labelsize=8)
    c.grid(False)
    c.set_xlabel("x lateral (m)")
    c.set_ylabel("z height (m)")
    c.set_title("Share of episodes where the right arm was picked")
    cb = fig.colorbar(im, ax=c, shrink=0.9, pad=0.02)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["all left", "50/50", "all right"])
    cb.outline.set_visible(False)
    fig.suptitle(f"Arm picks over the serve bank, {label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "picks.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def binned_rate(xv, yv, ok, xe, ye):
    n, _, _ = np.histogram2d(xv, yv, bins=[xe, ye])
    k, _, _ = np.histogram2d(xv[ok], yv[ok], bins=[xe, ye])
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(n > 0, k / n, np.nan)
    return rate, n


def heat(ax, xv, yv, ok, xe, ye, cmap, vmin, title, xl, yl, min_n=8):
    rate, n = binned_rate(xv, yv, ok, xe, ye)
    im = ax.pcolormesh(xe, ye, rate.T, cmap=cmap, vmin=vmin, vmax=1.0,
                       edgecolors=SURF, lw=2)
    for i in range(len(xe) - 1):
        for j in range(len(ye) - 1):
            if n[i, j] == 0:
                continue
            cx, cy = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
            if n[i, j] < min_n:
                ax.add_patch(Rectangle((xe[i], ye[j]), xe[i + 1] - xe[i],
                                       ye[j + 1] - ye[j], fill=False,
                                       hatch="///", ec=INK2, lw=0))
            dark = rate[i, j] > 0.5 * (1 + vmin)
            ax.text(cx, cy, f"{rate[i, j]:.0%}\n{int(n[i, j])}",
                    ha="center", va="center", fontsize=6.5,
                    color=SURF if dark else INK)
    ax.set_title(title)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.set_xticks(xe)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(False)
    return im


def fig_hit_chance(d, p, out, label):
    xe = np.array([-1.4, -1.0, -0.7, -0.4, -0.2, 0.0, 0.2, 0.4, 0.7, 1.0, 1.4])
    ze = np.array([0.5, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0])
    ye = np.array([-1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.6))
    vmin = max(0.0, np.floor(np.nanmin(binned_rate(d["x"], d["z"], d["hit"], xe, ze)[0]) * 10) / 10)
    vmin = min(vmin, 0.8)
    im = heat(a, d["x"], d["z"], d["hit"], xe, ze, SEQ_BLUE, vmin,
              "Hit rate by p* lateral x and height z (from the net)",
              "x lateral (m)", "z height (m)")
    heat(b, d["x"], d["y"], d["hit"], xe, ye, SEQ_BLUE, vmin,
         "Hit rate by p* lateral x and y along court (top-down)",
         "x lateral (m)", "y along court (m), net at 0, bases at -2")
    cb = fig.colorbar(im, ax=[a, b], shrink=0.85, pad=0.02)
    cb.set_label("hit rate")
    cb.outline.set_visible(False)
    fig.suptitle(f"Hit chance over the serve bank, {label}  "
                 f"(overall {d['hit'].mean():.1%}, n={len(d['hit'])}; "
                 "cell = hit rate over episode count; hatched = fewer than 8 episodes)", fontsize=12)
    fig.savefig(os.path.join(out, "hit_chance.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def outcome_bars(ax, v, d, edges, xl):
    hit, lok = d["hit"], d["lok"]
    cats = [("miss", ~hit, MISS), ("hit, return does not clear the net",
                                   hit & ~lok, "#9ec5f4"),
            ("hit, return clears and lands on far court", hit & lok, "#1c5cab")]
    centres = (edges[:-1] + edges[1:]) / 2
    widths = (edges[1:] - edges[:-1]) * 0.9
    bottom = np.zeros(len(centres))
    for name, m, col in cats:
        n_all = np.histogram(v, edges)[0].astype(float)
        n_c = np.histogram(v[m], edges)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(n_all > 0, n_c / n_all, 0.0)
        ax.bar(centres, frac, widths, bottom=bottom, color=col, label=name,
               lw=0)
        bottom += frac
    for k, (c, n) in enumerate(zip(centres, np.histogram(v, edges)[0])):
        ax.text(c, 1.01 + 0.045 * (k % 2), f"n={n}", ha="center",
                va="bottom", fontsize=6.5, color=INK2)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(xl)
    ax.set_ylabel("fraction of episodes")
    ax.set_xticks(edges)
    ax.grid(axis="x", visible=False)


def fig_returns(d, p, out, label):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8),
                            gridspec_kw={"width_ratios": [1.2, 1, 1]})
    outcome_bars(axs[0], d["x"], d, np.array([-1.4, -1.0, -0.7, -0.4, -0.2, 0.0,
                                               0.2, 0.4, 0.7, 1.0, 1.4]),
                 "p* lateral x (m)")
    axs[0].set_title("Outcome by lateral position")
    axs[0].tick_params(axis="x", labelsize=8)
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
                  fontsize=8)
    outcome_bars(axs[1], d["z"], d, np.array([0.5, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]),
                 "p* height z (m)")
    axs[1].set_title("Outcome by height")
    has = np.isfinite(d["land"])
    cl = has & d["lok"]
    nc = has & ~d["lok"]
    bins = np.linspace(0, 6, 31)
    axs[2].hist(d["land"][nc], bins, color="#9ec5f4", lw=0,
                label=f"did not clear (n={nc.sum()})")
    axs[2].hist(d["land"][cl], bins, color="#1c5cab", lw=0, alpha=0.9,
                label=f"cleared (n={cl.sum()})")
    if cl.any():
        med = np.median(d["land"][cl])
        axs[2].axvline(med, color=INK, lw=1, ls="--")
        axs[2].text(med + 0.05, axs[2].get_ylim()[1] * 0.95,
                    f"median {med:.2f} m", fontsize=8, va="top")
    axs[2].set_xlabel("predicted landing error to the launch origin (m)")
    axs[2].set_ylabel("hits")
    axs[2].set_title("Where the return lands")
    axs[2].legend(fontsize=8)
    clear = d["lok"][d["hit"]].mean() if d["hit"].any() else float("nan")
    fig.suptitle(f"Return quality, {label}  ({clear:.1%} of hits clear the "
                 f"net by {p['control']['return_net_clearance']:.1f} m "
                 "and land on the far court)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "returns.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_face(d, p, out, label):
    fw, fh = p["racket"]["face_size"][:2]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    for ax, arm, col, name in ((axs[0], 0, RIGHT, "right"),
                               (axs[1], 1, LEFT, "left")):
        m = (d["side"] == arm) & np.isfinite(d["u"])
        h, mm = m & d["hit"], m & ~d["hit"]
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(fw / 2 * 100 * np.cos(th), fh / 2 * 100 * np.sin(th),
                color=INK2, lw=1.2)
        ax.scatter(d["u"][h] * 100, d["v"][h] * 100, s=12, c=col, alpha=0.6,
                   lw=0, label=f"hit (n={h.sum()})", zorder=3)
        ax.scatter(d["u"][mm] * 100, d["v"][mm] * 100, s=36, marker="x",
                   c=MISS, lw=1.2, label=f"miss, closest approach (n={mm.sum()})",
                   zorder=4)
        ax.set_aspect("equal")
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_xlabel("u across the face (cm)")
        ax.set_title(f"{name} racket face")
        ax.legend(fontsize=8, loc="lower left")
    axs[0].set_ylabel("v along the face (cm), + = toward the tip")
    fig.suptitle(f"Cork centre in the face frame at contact or closest approach, "
                 f"{label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "face_map.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_pick(d, p, out, label):
    lk = d["locked"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.6),
                                  gridspec_kw={"width_ratios": [1, 0.8, 1.15]})
    bins = np.linspace(0, max(0.6, np.nanmax(d["t_lock"][lk]) if lk.any() else 0.6), 25)
    for arm, col, name in ((0, RIGHT, "right"), (1, LEFT, "left")):
        m = lk & (d["chosen"] == arm)
        a.hist(d["t_lock"][m], bins, color=col, alpha=0.8, lw=0,
               label=f"{name} (n={m.sum()})")
    a.set_xlabel("time after serve when the chooser locked the arm (s)")
    a.set_ylabel("episodes")
    a.set_title("Lock time")
    a.legend(fontsize=8)
    agree = lk & (d["chosen"] == d["rule"])
    dis = lk & (d["chosen"] != d["rule"])
    groups = [("agrees with\nstrike-plane rule", agree),
              ("differs from\nrule", dis)]
    if (~lk).any():
        groups.append(("never locked", ~lk))
    rates = [d["hit"][m].mean() if m.any() else 0 for _, m in groups]
    cols = ["#1c5cab", "#6da7ec", "#9ec5f4"][: len(groups)]
    b.bar(range(len(groups)), rates, 0.6, color=cols, lw=0)
    for i, (g, m) in enumerate(groups):
        b.text(i, rates[i] + 0.015, f"{rates[i]:.1%}\nn={m.sum()}",
               ha="center", fontsize=8)
    b.set_xticks(range(len(groups)), [g for g, _ in groups])
    b.set_ylim(0, 1.12)
    b.set_ylabel("hit rate")
    b.set_title("Hit rate by how the arm was picked")
    b.grid(axis="x", visible=False)
    im = heat(c, d["x"][lk], d["z"][lk], agree[lk], PICK_XE, PICK_ZE, SEQ_BLUE,
              0.0, "Share of picks that agree with the strike-plane rule",
              "x lateral (m)", "z height (m)")
    c.axvline(0.245, color=RIGHT, lw=0.8, ls=":")
    c.axvline(-0.245, color=LEFT, lw=0.8, ls=":")
    cb = fig.colorbar(im, ax=c, shrink=0.9, pad=0.02)
    cb.outline.set_visible(False)
    fig.suptitle(f"Arm pick, {label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "arm_pick.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="runs/eval_pstar_hits.npy")
    ap.add_argument("--out", default="runs/figs")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d, p = load(args.input)
    label = args.label or os.path.basename(args.input)
    fig_picks(d, p, args.out, label)
    fig_hit_chance(d, p, args.out, label)
    fig_returns(d, p, args.out, label)
    fig_face(d, p, args.out, label)
    fig_pick(d, p, args.out, label)
    print(f"{len(d['hit'])} episodes, hit {d['hit'].mean():.3f}; figures in {args.out}/")


if __name__ == "__main__":
    main()
