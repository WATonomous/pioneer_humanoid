"""Run the traversability layers on a saved elevation grid (.npy) and report / plot what they say.

    python eval_traversability_npy.py elevation.npy --resolution 0.05 --out layers.png

The grid must be a 2D array of heights in metres, NaN for unknown cells. If the file is 3D (several layers, e.g. an
elevation_mapping dump), choose the elevation layer with --layer. Resolution is the cell size in metres: ask whoever
made the file, it is NOT stored in a plain .npy.
"""
import argparse

import numpy as np

from terrain_core.traversability import RobotLimits, TraversabilityParams, compute_traversability
from terrain_core.traversability._window import window_radius
from terrain_core.traversability.slope import estimate_noise_sigma


def load_grid(path, layer):
    arr = np.load(path)
    print(f"loaded {path}: shape {arr.shape}, dtype {arr.dtype}")
    if arr.ndim == 3:
        if layer is None:
            raise SystemExit("file is 3D: pass --layer N to pick the elevation layer "
                             f"(axis 0 has {arr.shape[0]} entries); check which one is height with the file's author")
        arr = arr[layer]
    if arr.ndim != 2:
        raise SystemExit(f"expected a 2D grid, got shape {arr.shape}")
    return arr.astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npy")
    ap.add_argument("--resolution", type=float, required=True, help="cell size in metres, e.g. 0.05")
    ap.add_argument("--layer", type=int, default=None, help="layer index if the file is 3D")
    ap.add_argument("--plane-window-m", type=float, default=None)
    ap.add_argument("--max-slope-deg", type=float, default=7.0)
    ap.add_argument("--max-step-m", type=float, default=0.08)
    ap.add_argument("--max-roughness-m", type=float, default=0.03)
    ap.add_argument("--out", default="layers.png")
    args = ap.parse_args()

    z = load_grid(args.npy, args.layer)
    finite = np.isfinite(z)
    print(f"valid cells: {finite.mean() * 100:.1f}%  height min/median/max: "
          f"{np.nanmin(z):.3f} / {np.nanmedian(z):.3f} / {np.nanmax(z):.3f} m")
    if not finite.any():
        raise SystemExit("no valid cells")

    limits = RobotLimits(args.max_slope_deg, args.max_step_m, args.max_roughness_m)
    params = TraversabilityParams() if args.plane_window_m is None else TraversabilityParams(plane_window_m=args.plane_window_m)
    out = compute_traversability(z, args.resolution, limits, params=params)

    cells = (2 * window_radius(params.plane_window_m, args.resolution) + 1) ** 2
    print(f"estimated per-cell noise: {estimate_noise_sigma(out['roughness'], cells) * 100:.2f} cm "
          f"(window {params.plane_window_m} m)")

    known = np.isfinite(out["traversability"])
    t = out["traversable"]
    print(f"traversable: {t.sum() / max(known.sum(), 1) * 100:.1f}% of cells with an estimate")
    with np.errstate(invalid="ignore"):
        over_slope = known & (out["slope"] > np.deg2rad(limits.max_slope_deg))
        over_step = known & (out["step"] > limits.max_step_m)
        over_rough = known & (out["roughness"] > limits.max_roughness_m)
    for name, m in (("slope over limit", over_slope), ("step over limit", over_step), ("roughness over limit", over_rough)):
        print(f"  {name}: {m.sum() / max(known.sum(), 1) * 100:.1f}%")
    print("note: slope-over-limit cells on step edges are still enforced by the step limit; see combined.py")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping the plot (pip install matplotlib)")
        return
    panels = [("elevation (m)", z), ("slope (deg)", np.rad2deg(out["slope"])), ("step (m)", out["step"]),
              ("roughness (m)", out["roughness"]), ("traversability", out["traversability"]),
              ("traversable", np.where(known, t.astype(float), np.nan))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (title, data) in zip(axes.ravel(), panels):
        im = ax.imshow(data, origin="lower")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
