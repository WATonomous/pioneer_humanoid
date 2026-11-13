#!/usr/bin/env python3
"""Test-course evaluation script for the humanoid terrain mapping pipeline (TR-11).

Compares a 2.5D elevation map against the physical course ground truth in
config/test_course.yaml and reports the TR-11 evaluation metrics:
  1. Height & slope error (flat floor, 14 cm step, 9 deg ramp, 39 cm box)
  2. False obstacles on the flat floor (cells raised above threshold)
  3. Under-feet coverage (fraction of cells under the robot with valid heights)

Usage:
    # Run built-in synthetic benchmark suite (perfect map + 3 known-mistake maps):
    python3 terrain_tools/evaluate.py

    # Render top-down 2.5D height plots into src/terrain_mapping/terrain_core/data/:
    python3 terrain_tools/evaluate.py --png

    # Evaluate one or more saved map files (.npz or .npy):
    python3 evaluate.py path/to/map.npz [--json report.json] [--strict]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from synthetic_maps import EDGE_MARGIN_M, GridMap, build_synthetic_suite

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "test_course.yaml"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _next_available_path(target_path):
    """If target_path already exists, return target_1.png, target_2.png, etc."""
    p = Path(target_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    idx = 1
    while True:
        cand = p.parent / f"{stem}_{idx}{suffix}"
        if not cand.exists():
            return cand
        idx += 1


# ------------------------------------------------------------------------------
# Metric Calculations (Height Error, False Obstacles, Under-Feet Coverage)
# ------------------------------------------------------------------------------

def _error_stats(err_m):
    """Compute signed mean bias, RMSE, median, and max absolute error in cm."""
    if err_m.size == 0:
        return {"n": 0, "mean_cm": None, "median_error_cm": None, "rmse_cm": None, "max_abs_cm": None}
    cm = err_m * 100.0
    return {
        "n": int(err_m.size),
        "mean_cm": float(np.mean(cm)),
        "median_error_cm": float(np.median(cm)),
        "rmse_cm": float(np.sqrt(np.mean(cm**2))),
        "max_abs_cm": float(np.max(np.abs(cm))),
    }


def _coverage(gm, mask):
    n = int(mask.sum())
    return float(np.isfinite(gm.z[mask]).sum() / n) if n else 0.0


def eval_under_feet(gm, cfg):
    """Fraction of cells under the robot's feet that have a valid height estimate."""
    mask = gm.region_mask(cfg["bounds"])
    cov = _coverage(gm, mask)
    if "min_coverage_pct" in cfg:
        min_cov = float(cfg["min_coverage_pct"]) / 100.0
    else:
        min_cov = float(cfg.get("min_coverage", 0.90))
    return {
        "total_cells": int(mask.sum()),
        "valid_cells": int(np.isfinite(gm.z[mask]).sum()),
        "coverage": cov,
        "min_coverage": min_cov,
        "pass": bool(cov >= min_cov),
    }


def eval_flat_floor(gm, cfg, obstacle_bounds, obstacle_thresh_m):
    """Height error and false obstacle count on the flat floor (excluding known obstacles)."""
    mask = gm.region_mask(cfg["bounds"], EDGE_MARGIN_M)
    for b in obstacle_bounds:
        pad = [b[0] - 0.18, b[1] + 0.18, b[2] - 0.18, b[3] + 0.52]
        mask &= ~gm.region_mask(pad)

    valid = mask & np.isfinite(gm.z)
    err = gm.z[valid] - float(cfg["expected_height_m"])
    tol_cm = float(cfg["tolerance_cm"])
    res = _error_stats(err)

    if gm.occupied is not None:
        false_mask = gm.occupied[mask]
    else:
        z_floor = gm.z[mask]
        false_mask = np.isfinite(z_floor) & (np.abs(z_floor - float(cfg["expected_height_m"])) > obstacle_thresh_m)

    n_cells = int(valid.sum())
    false_cells = int(false_mask.sum())
    false_rate = float(false_cells / n_cells) if n_cells else 0.0
    max_false_rate = float(cfg.get("max_false_obstacle_rate", 0.01))
    within_2cm = float(np.mean(np.abs(err) <= 0.02)) if err.size else 0.0

    res.update(
        coverage=_coverage(gm, mask),
        within_2cm=within_2cm,
        area_m2=float(n_cells * (gm.res**2)),
        false_obstacle_cells=false_cells,
        false_obstacle_rate=false_rate,
        tolerance_cm=tol_cm,
    )
    res["pass"] = bool(
        res["coverage"] >= 0.50
        and res["rmse_cm"] is not None
        and res["rmse_cm"] <= tol_cm
        and false_rate <= max_false_rate
    )
    return res


def eval_flat_top_obstacle(gm, cfg, truth_h_m, tol_cm):
    """Evaluate a flat-top obstacle (Step or Box) against its known height."""
    mask = gm.region_mask(cfg["bounds"], EDGE_MARGIN_M)
    valid = mask & np.isfinite(gm.z)
    err = gm.z[valid] - truth_h_m
    res = _error_stats(err)
    res["coverage"] = _coverage(gm, mask)
    res["measured_height_cm"] = float(np.median(gm.z[valid]) * 100.0) if err.size else None
    res["expected_height_cm"] = float(truth_h_m * 100.0)
    res["tolerance_cm"] = tol_cm
    res["pass"] = bool(
        res["coverage"] >= 0.50
        and res["rmse_cm"] is not None
        and res["rmse_cm"] <= tol_cm
    )
    return res


def eval_ramp(gm, cfg):
    """Evaluate the inclined ramp for surface height error and slope angle error."""
    x0, x1, y0, y1 = cfg["bounds"]
    mask = gm.region_mask(cfg["bounds"], EDGE_MARGIN_M)
    valid = mask & np.isfinite(gm.z)
    X, Y = gm.xy_grids()

    h_start = float(cfg["start_height_m"])
    h_end = float(cfg["end_height_m"])
    truth = h_start + (h_end - h_start) * (Y - y0) / (y1 - y0)
    err = gm.z[valid] - truth[valid]

    tol_cm = float(cfg.get("tolerance_cm", 2.5))
    slope_tol_deg = float(cfg.get("slope_tolerance_deg", 2.5))
    res = _error_stats(err)
    res["coverage"] = _coverage(gm, mask)
    res["tolerance_cm"] = tol_cm

    slope_deg = None
    if err.size >= 10:
        A = np.c_[X[valid], Y[valid], np.ones(err.size)]
        a, b, _ = np.linalg.lstsq(A, gm.z[valid], rcond=None)[0]
        slope_deg = float(np.degrees(np.arctan(np.hypot(a, b))))

    expected_deg = float(cfg["expected_slope_deg"])
    res["measured_slope_deg"] = slope_deg
    res["expected_slope_deg"] = expected_deg
    res["slope_error_deg"] = None if slope_deg is None else slope_deg - expected_deg
    res["pass"] = bool(
        res["coverage"] >= 0.50
        and res["rmse_cm"] is not None
        and res["rmse_cm"] <= tol_cm
        and slope_deg is not None
        and abs(res["slope_error_deg"]) <= slope_tol_deg
    )
    return res


def evaluate(gm, course, obstacle_thresh=None):
    """Run all TR-11 checks on a GridMap and return the structured results dict."""
    meta = course["course_metadata"]
    r = course["regions"]
    fl_thresh = r["flat_floor"].get("false_obstacle_threshold_m", meta.get("obstacle_threshold_m", 0.04))
    thresh = float(fl_thresh) if obstacle_thresh is None else float(obstacle_thresh)
    obst_bounds = [r["step"]["bounds"], r["ramp"]["bounds"], r["obstacle_box"]["bounds"]]

    step_cfg = r["step"]
    box_cfg = r["obstacle_box"]
    uf_cfg = r.get("under_feet") or course.get("under_feet") or {"bounds": [-0.4, 0.4, -0.4, 0.4], "min_coverage": 0.90}

    return {
        "under_feet": eval_under_feet(gm, uf_cfg),
        "flat_floor": eval_flat_floor(gm, r["flat_floor"], obst_bounds, thresh),
        "step": eval_flat_top_obstacle(gm, step_cfg, float(step_cfg["expected_height_m"]), float(step_cfg["tolerance_cm"])),
        "ramp": eval_ramp(gm, r["ramp"]),
        "obstacle_box": eval_flat_top_obstacle(
            gm, box_cfg, float(box_cfg["height_m"]), float(box_cfg.get("tolerance_cm", step_cfg["tolerance_cm"]))
        ),
    }


# ------------------------------------------------------------------------------
# Reporting, Visualization & CLI
# ------------------------------------------------------------------------------

def _fmt(v, nd=2, signed=False):
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def print_report(results, label):
    uf = results["under_feet"]
    fl = results["flat_floor"]
    st = results["step"]
    rp = results["ramp"]
    bx = results["obstacle_box"]
    overall = "PASS" if all(m["pass"] for m in results.values()) else "FAIL"

    print(f"\n=== Terrain Evaluation: {label} [{overall}] ===")
    print(f"  Under-feet coverage : {_fmt(uf['coverage'] * 100, 1)}% ({uf['valid_cells']}/{uf['total_cells']} cells, min {uf['min_coverage'] * 100:.0f}%) -> {'PASS' if uf['pass'] else 'FAIL'}")
    print(f"  Flat floor (0.0 cm) : mean err {_fmt(fl['mean_cm']):>6} cm | RMSE {_fmt(fl['rmse_cm']):>5} cm | False obstacles: {fl['false_obstacle_cells']} cells ({_fmt(fl['false_obstacle_rate'] * 100, 2)}%) -> {'PASS' if fl['pass'] else 'FAIL'}")
    print(f"  Step      (14.0 cm) : measured {_fmt(st['measured_height_cm']):>6} cm | mean err {_fmt(st['mean_cm']):>6} cm | RMSE {_fmt(st['rmse_cm']):>5} cm -> {'PASS' if st['pass'] else 'FAIL'}")
    print(f"  Ramp      ( 9.0 deg): measured {_fmt(rp['measured_slope_deg'], 1):>6} deg | mean err {_fmt(rp['mean_cm']):>6} cm | RMSE {_fmt(rp['rmse_cm']):>5} cm -> {'PASS' if rp['pass'] else 'FAIL'}")
    print(f"  Box       (39.0 cm) : measured {_fmt(bx['measured_height_cm']):>6} cm | mean err {_fmt(bx['mean_cm']):>6} cm | RMSE {_fmt(bx['rmse_cm']):>5} cm -> {'PASS' if bx['pass'] else 'FAIL'}")


def format_markdown_summary(all_runs):
    headers = [
        "Map",
        "Under-Feet Cov",
        "Floor Mean / RMSE",
        "False Obstacles",
        "Step Err",
        "Ramp Err / Slope",
        "Box Err",
        "Overall",
    ]
    rows = []
    for name, res in all_runs.items():
        uf = res["under_feet"]
        fl = res["flat_floor"]
        st = res["step"]
        rp = res["ramp"]
        bx = res["obstacle_box"]
        overall = "PASS" if all(m["pass"] for m in res.values()) else "FAIL"

        rows.append([
            name,
            f"{uf['coverage'] * 100:.1f}%",
            f"{_fmt(fl['mean_cm'], 1, signed=True)} / {_fmt(fl['rmse_cm'], 1)} cm",
            f"{fl['false_obstacle_cells']} ({fl['false_obstacle_rate'] * 100:.2f}%)",
            f"{_fmt(st['mean_cm'], 1, signed=True)} cm",
            f"{_fmt(rp['mean_cm'], 1, signed=True)} cm / {_fmt(rp['measured_slope_deg'], 1)} deg",
            f"{_fmt(bx['mean_cm'], 1, signed=True)} cm",
            overall,
        ])

    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def _fmt_row(cols):
        cells = [
            cols[i].ljust(widths[i]) if i in (0, len(cols) - 1) else cols[i].rjust(widths[i])
            for i in range(len(cols))
        ]
        return "| " + " | ".join(cells) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"

    lines = [
        "=== Evaluation Summary ===",
        _fmt_row(headers),
        sep,
        *(_fmt_row(r) for r in rows),
        "",
    ]
    return "\n".join(lines)


def _ideal_ground_truth_grid(gm, course):
    """Build the ideal expected height grid (0 cm floor + step + ramp + box) to compute signed error."""
    r = course["regions"]
    floor_h = float(r["flat_floor"].get("expected_height_m", 0.0))
    truth = np.full(gm.z.shape, floor_h, dtype=float)
    _, Y = gm.xy_grids()

    # Step expected height
    sb = r["step"]["bounds"]
    truth[gm.region_mask(sb)] = float(r["step"]["expected_height_m"])

    # Ramp expected slope surface
    rb = r["ramp"]["bounds"]
    rmask = gm.region_mask(rb)
    h0 = float(r["ramp"]["start_height_m"])
    h1 = float(r["ramp"]["end_height_m"])
    truth[rmask] = h0 + (h1 - h0) * (Y[rmask] - rb[2]) / (rb[3] - rb[2])

    # Obstacle box expected height
    bb = r["obstacle_box"]["bounds"]
    box_h = r["obstacle_box"].get("height_m", r["obstacle_box"].get("expected_height_m", 0.39))
    truth[gm.region_mask(bb)] = float(box_h)
    return truth


def plot_maps(maps_dict, all_runs, course, out_path=None, show=False):
    """Render 2 rows: (1) 2.5D Elevation Map and (2) 2.5D Baseline Deviation Map."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n = len(maps_dict)
    fig = plt.figure(figsize=(3.9 * n, 9.6), dpi=140, facecolor="#18181b")
    gs = fig.add_gridspec(4, n, height_ratios=[1.0, 0.042, 1.0, 0.042], hspace=0.14, wspace=0.08)

    cmap_elev = plt.cm.turbo.copy()
    cmap_elev.set_bad(color="#28282c")

    # Symmetrical baseline-deviation colormap anchored cleanly at 0 cm = Green:
    # -6 cm (Blue) -> -2 cm (Cyan) -> 0 cm (Green) -> +2 cm (Amber) -> +5/+6 cm (Red)
    cmap_err = LinearSegmentedColormap.from_list(
        "baseline_dev",
        [
            (0.00, "#1d4ed8"),  # -6 cm
            (0.33, "#06b6d4"),  # -2 cm
            (0.50, "#16a34a"),  #  0 cm (on baseline)
            (0.67, "#facc15"),  # +2 cm (floor tolerance limit)
            (0.90, "#ef4444"),  # +5 cm (shifted above baseline)
            (1.00, "#991b1b"),  # +6 cm+
        ],
    )
    cmap_err.set_bad(color="#28282c")

    faults = course.get("synthetic_faults") or course.get("regions", {}).get("synthetic_faults") or {}
    shift_cm = float(faults.get("floor_shift_m", 0.05)) * 100.0
    titles = {
        "perfect_course": "1. Ground Truth Course",
        "floor_shifted_up": f"2. Floor Shifted ({shift_cm:+.0f} cm)",
        "fake_blob_on_floor": "3. Fake Obstacles on Floor",
        "hole_under_feet": "4. Missing Hole Under Feet",
    }
    err_titles = {
        "perfect_course": "1. Baseline Deviation (On 0 cm Plane)",
        "floor_shifted_up": f"2. Baseline Deviation (Plane {shift_cm:+.0f} cm)",
        "fake_blob_on_floor": "3. Baseline Deviation (Fake Obstacles)",
        "hole_under_feet": "4. Baseline Deviation (Missing Hole)",
    }

    im_top = None
    im_bot = None
    for i, (name, gm) in enumerate(maps_dict.items()):
        res = all_runs[name]
        fl = res["flat_floor"]
        uf = res["under_feet"]
        overall = "PASS" if all(m["pass"] for m in res.values()) else "FAIL"
        badge_col = "#4ade80" if overall == "PASS" else "#f87171"
        ext = [
            gm.origin[0],
            gm.origin[0] + gm.z.shape[1] * gm.res,
            gm.origin[1],
            gm.origin[1] + gm.z.shape[0] * gm.res,
        ]

        # Row 1: 2.5D Elevation Map (-10 cm to +50 cm)
        ax1 = fig.add_subplot(gs[0, i], facecolor="#28282c")
        im_top = ax1.imshow(gm.z * 100.0, origin="lower", extent=ext, cmap=cmap_elev, vmin=-10.0, vmax=50.0, interpolation="nearest")
        # Camera / robot start origin dot at world (0, 0)
        ax1.plot(
            0.0, 0.0, marker="o", markersize=6.5,
            color="#ffffff", markeredgecolor="#18181b", markeredgewidth=1.4, zorder=6,
        )
        if i == 0:
            ax1.annotate(
                "Camera (0, 0)", xy=(0.0, 0.0), xytext=(0.14, 0.08),
                color="#ffffff", fontsize=7.5, fontweight="bold", zorder=7,
                bbox={"facecolor": "#18181bcc", "edgecolor": "none", "pad": 1.5},
            )
        hud1 = (
            f"{titles.get(name, name)}  [{overall}]\n"
            f"Floor RMSE: {_fmt(fl['rmse_cm'])} cm  |  False Obst: {fl['false_obstacle_cells']} cells\n"
            f"Under-Feet Coverage: {uf['coverage']*100:.0f}%"
        )
        ax1.text(
            0.03, 0.97, hud1, transform=ax1.transAxes, va="top", ha="left",
            color="white", fontsize=8.0, fontweight="bold",
            bbox={"facecolor": "#18181bdd", "edgecolor": badge_col, "linewidth": 1.2, "pad": 3.5},
        )
        ax1.set_xticks([])
        ax1.set_yticks([])
        for sp in ax1.spines.values():
            sp.set_color("#3f3f46")

        # Row 2: 2.5D Signed Baseline Deviation Grid (measured - expected ground truth, in cm)
        # Green = 0 cm (on baseline), Yellow = +2 cm (tol), Red = +5 cm+ (above baseline!)
        truth_z = _ideal_ground_truth_grid(gm, course)
        err_grid_cm = (gm.z - truth_z) * 100.0

        ax2 = fig.add_subplot(gs[2, i], facecolor="#28282c")
        im_bot = ax2.imshow(err_grid_cm, origin="lower", extent=ext, cmap=cmap_err, vmin=-6.0, vmax=6.0, interpolation="nearest")
        ax2.plot(
            0.0, 0.0, marker="o", markersize=5.5,
            color="#ffffff", markeredgecolor="#18181b", markeredgewidth=1.2, zorder=6,
        )
        hud2 = (
            f"{err_titles.get(name, name)}\n"
            f"Floor: {_fmt(fl['mean_cm'], 1, signed=True)} cm | Step: {_fmt(res['step']['mean_cm'], 1, signed=True)} cm | Ramp: {_fmt(res['ramp']['mean_cm'], 1, signed=True)} cm"
        )
        ax2.text(
            0.03, 0.97, hud2, transform=ax2.transAxes, va="top", ha="left",
            color="white", fontsize=7.6, fontweight="bold",
            bbox={"facecolor": "#18181bdd", "edgecolor": badge_col, "linewidth": 1.0, "pad": 3.0},
        )
        ax2.set_xticks([])
        ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_color("#3f3f46")

    # Colorbar 1: Raw Elevation (cm above floor)
    cax1 = fig.add_subplot(gs[1, :])
    cb1 = fig.colorbar(
        im_top, cax=cax1, orientation="horizontal",
        ticks=[0.0, 14.0, 25.0, 39.0, 50.0],
    )
    cb1.ax.set_xticklabels(
        ["0 cm (Flat Floor)", "14 cm (Step / Ramp Top)", "25 cm (Fake Blob)", "39 cm (Obstacle Box)", "50 cm"],
        color="#e4e4e7", fontsize=8.5, fontweight="bold",
    )
    cb1.ax.tick_params(colors="#e4e4e7", labelsize=8.5, pad=3)

    # Colorbar 2: Height Error / Plane Deviation from Baseline (cm)
    cax2 = fig.add_subplot(gs[3, :])
    cb2 = fig.colorbar(
        im_bot, cax=cax2, orientation="horizontal",
        ticks=[-5.0, -2.0, 0.0, 2.0, 5.0],
    )
    cb2.ax.set_xticklabels(
        ["-5 cm (Below)", "-2 cm (Min Tol)", "0 cm (On Baseline - PASS)", "+2 cm (Max Tol)", "+5 cm (Above Baseline - FAIL)"],
        color="#e4e4e7", fontsize=8.5, fontweight="bold",
    )
    cb2.ax.tick_params(colors="#e4e4e7", labelsize=8.5, pad=3)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.05)
    if out_path:
        final_p = _next_available_path(out_path)
        plt.savefig(final_p, facecolor=fig.get_facecolor())
        abs_p = final_p.resolve()
        print(f"Saved map plot to {abs_p}")
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(abs_p))
            elif shutil.which("wslview"):
                subprocess.Popen(["wslview", str(abs_p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif shutil.which("explorer.exe"):
                win_p = subprocess.check_output(["wslpath", "-w", str(abs_p)], text=True).strip()
                subprocess.Popen(["explorer.exe", win_p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("maps", nargs="*", help="optional .npz or .npy map file(s) to evaluate; if omitted, runs the synthetic test suite")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to test course config YAML")
    ap.add_argument("--shift", nargs=2, type=float, metavar=("DX", "DY"), default=(0.0, 0.0), help="course origin in map coordinates")
    ap.add_argument("--obstacle-thresh", type=float, default=None, help="override false-obstacle height threshold (m)")
    ap.add_argument("--save-synthetic", metavar="DIR", help="save the 4 synthetic .npz maps to this directory")
    ap.add_argument(
        "--plot", "--png", nargs="?", const="course_maps.png", metavar="PNG",
        help="save plot PNG into src/terrain_mapping/terrain_core/data/ (auto-increments course_maps_1.png, etc.)",
    )
    ap.add_argument("--show", action="store_true", help="open an interactive matplotlib window showing the maps")
    ap.add_argument("--json", help="write detailed metrics JSON to this path")
    ap.add_argument("--md", help="write Markdown benchmark table to this path")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any evaluated map fails")
    args = ap.parse_args(argv)

    course = yaml.safe_load(Path(args.config).read_text())
    all_runs = {}
    loaded_maps = {}

    if not args.maps:
        suite = build_synthetic_suite(course)
        if args.save_synthetic:
            out_dir = Path(args.save_synthetic)
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, gm in suite.items():
                gm.save(out_dir / f"{name}.npz")
        for name, gm in suite.items():
            loaded_maps[name] = gm
            res = evaluate(gm, course, args.obstacle_thresh)
            all_runs[name] = res
            print_report(res, name)
    else:
        for map_path in args.maps:
            p = Path(map_path)
            files = sorted(p.glob("*.npz")) + sorted(p.glob("*.npy")) if p.is_dir() else [p]
            for f in files:
                gm = GridMap.load(f)
                gm.set_origin(gm.origin - np.asarray(args.shift))
                loaded_maps[f.stem] = gm
                res = evaluate(gm, course, args.obstacle_thresh)
                all_runs[f.stem] = res
                print_report(res, f.name)

    if args.plot or args.show:
        plot_target = None
        if args.plot:
            p = Path(args.plot)
            plot_target = p if p.parent != Path(".") else (DEFAULT_DATA_DIR / p.name)
        plot_maps(loaded_maps, all_runs, course, out_path=plot_target, show=args.show)

    if len(all_runs) > 1:
        print("\n" + format_markdown_summary(all_runs))

    if args.json:
        payload = all_runs if len(all_runs) > 1 else next(iter(all_runs.values()))
        Path(args.json).write_text(json.dumps({"results": payload}, indent=2))
    if args.md:
        Path(args.md).write_text(format_markdown_summary(all_runs))

    if args.strict and not all(all(m["pass"] for m in r.values()) for r in all_runs.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
