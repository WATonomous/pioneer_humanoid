"""Evaluate TR-7 traversability layers on a saved elevation grid (.npy).

Example from ``src/terrain_mapping/terrain_core``:
    python terrain_tools/eval_traversability_npy.py /path/to/elevation.npy --resolution 0.04
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from terrain_core.traversability import TraversabilityConfig, compute_traversability
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
    ap.add_argument(
        "--config",
        default=Path(__file__).resolve().parents[1] / "config" / "terrain.yaml",
        help="terrain.yaml containing robot limits and window sizes",
    )
    ap.add_argument("--plane-window-m", type=float, default=None)
    ap.add_argument("--max-slope-deg", type=float, default=None)
    ap.add_argument("--max-step-m", type=float, default=None)
    ap.add_argument("--max-roughness-m", type=float, default=None)
    ap.add_argument("--out", default="layers.png")
    args = ap.parse_args()

    z = load_grid(args.npy, args.layer)
    finite = np.isfinite(z)
    print(f"valid cells: {finite.mean() * 100:.1f}%  height min/median/max: "
          f"{np.nanmin(z):.3f} / {np.nanmedian(z):.3f} / {np.nanmax(z):.3f} m")
    if not finite.any():
        raise SystemExit("no valid cells")

    config = TraversabilityConfig.from_yaml(args.config)
    limits = replace(
        config.limits,
        max_slope_deg=config.limits.max_slope_deg if args.max_slope_deg is None else args.max_slope_deg,
        max_step_m=config.limits.max_step_m if args.max_step_m is None else args.max_step_m,
        max_roughness_m=(
            config.limits.max_roughness_m if args.max_roughness_m is None else args.max_roughness_m
        ),
    )
    params = config.params if args.plane_window_m is None else replace(
        config.params, plane_window_m=args.plane_window_m
    )
    print(
        "limits: "
        f"slope <= {limits.max_slope_deg:g} deg, step <= {limits.max_step_m * 100:g} cm, "
        f"roughness <= {limits.max_roughness_m * 100:g} cm"
    )
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
