"""Exercise TR-8 with CPU-rendered depth images of flat ground, ramps and a step."""
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from slam_core.types.frame import Frame
from terrain_core.preprocess.pipeline import PreprocessConfig, preprocess
from terrain_core.traversability import TraversabilityConfig, compute_traversability

RES = 0.04
RAMP_END_X = 1.60  # keep the landing in view even for long, shallow ramps
CONFIG = Path(__file__).resolve().parents[1] / "config" / "terrain.yaml"


def simulate_depth(slope_deg, rise_m, noise_m):
    """Render a plane, a vertical step, or a finite ramp joining two level surfaces."""
    v, u = np.indices((240, 320))
    K = np.array([[220, 0, 159.5], [0, 220, 119.5], [0, 0, 1]])
    pitch = np.deg2rad(40)
    c, s = np.cos(pitch), np.sin(pitch)
    pose = np.array([[0, -s, c, -0.5], [-1, 0, 0, 0], [0, -c, -s, 1.4], [0, 0, 0, 1]])
    # Optical z is one along each unscaled ray, so intersection distance in this
    # parameterization gives axial depth directly.
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    dx, _, dz = (rays @ pose[:3, :3].T).transpose(2, 0, 1)
    x0, _, z0 = pose[:3, 3]
    grade = np.tan(np.deg2rad(slope_deg))
    floor_depth = -z0 / dz
    depth = (grade * x0 - z0) / (dz - grade * dx)
    if rise_m:
        landing_depth = (rise_m - z0) / dz
        if slope_deg:
            # A ramp with this grade needs exactly rise / grade metres of run.
            end = RAMP_END_X
            start = end - rise_m / grade
            incline_depth = (grade * (x0 - start) - z0) / (dz - grade * dx)
            incline_x = x0 + incline_depth * dx
            middle = np.where((incline_x >= start) & (incline_x <= end), incline_depth, np.inf)
        else:
            start = end = 1.0
            riser_depth = (start - x0) / dx
            riser_z = z0 + riser_depth * dz
            middle = np.where((riser_z >= 0) & (riser_z <= rise_m), riser_depth, np.inf)
        depth = np.minimum.reduce([
            np.where(x0 + floor_depth * dx <= start, floor_depth, np.inf),
            middle,
            np.where(x0 + landing_depth * dx >= end, landing_depth, np.inf),
        ])
    depth += np.random.default_rng(8).normal(0, noise_m, depth.shape)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
    return Frame(0.0, np.zeros(depth.shape, dtype=np.uint8), depth.astype(np.float32), K), pose


def elevation_grid(points):
    """Highest observed point per 4 cm cell; empty cells remain unknown."""
    grid = np.full((40, 64), -np.inf)
    col = np.floor(points[:, 0] / RES).astype(int)
    row = np.floor((points[:, 1] + 0.8) / RES).astype(int)
    inside = (col >= 0) & (col < grid.shape[1]) & (row >= 0) & (row < grid.shape[0])
    np.maximum.at(grid, (row[inside], col[inside]), points[inside, 2])
    grid[~np.isfinite(grid)] = np.nan
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-mm", type=float, default=2, help="axial depth noise, standard deviation")
    parser.add_argument("--out", type=Path, help="directory for each scene's elevation and expected-label .npy grids")
    parser.add_argument("--report", type=Path, help="save one PDF with diagrams and results for every simulated scene")
    args = parser.parse_args()
    if not np.isfinite(args.noise_mm) or args.noise_mm < 0:
        parser.error("--noise-mm must be finite and nonnegative")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    pdf = nullcontext()
    if args.report:
        from matplotlib.backends.backend_pdf import PdfPages

        args.report.parent.mkdir(parents=True, exist_ok=True)
        pdf = PdfPages(args.report)
    with pdf as report:
        simulate_course(args.noise_mm, args.out, report)
    if args.report:
        print(f"Saved {args.report.resolve()}")


def simulate_course(noise_mm, output_dir, report):
    config = TraversabilityConfig.from_yaml(CONFIG)
    preprocess_config = PreprocessConfig.from_yaml(CONFIG)
    for name, slope, rise, expected in [
        ("floor", 0, 0, True), ("gentle_ramp", 5, 0, True),
        ("steep_ramp", 9, 0, False), ("step_14cm", 0, 0.14, False),
        ("step_14cm_ramp", 5, 0.14, True), ("step_14cm_gentle_ramp", 3, 0.14, True),
        ("step_14cm_steep_ramp", 9, 0.14, False), ("step_20cm", 0, 0.20, False),
    ]:
        frame, pose = simulate_depth(slope, rise, noise_mm / 1000)
        grid = elevation_grid(preprocess(frame, pose, preprocess_config).points)
        layers = compute_traversability(grid, RES, config.limits, params=config.params)
        if slope and rise:
            # Sample the middle of the incline and the level landing. Leave room
            # for the plane-fit window on either side of the change in slope.
            run = rise / np.tan(np.deg2rad(slope))
            midpoint = RAMP_END_X - run / 2
            col = round(midpoint / RES)
            landing = round((RAMP_END_X + 0.32) / RES)
            regions = [("incline", slice(col - 2, col + 2), expected),
                       ("landing", slice(landing, landing + 4), True)]
        elif rise:
            regions = [("vertical edge", slice(24, 26), expected)]
        else:
            regions = [("surface", slice(20, 40), expected)]

        labels = np.full(grid.shape, np.nan)
        summary = []
        for surface, cols, target in regions:
            region = np.s_[10:30, cols]
            labels[region] = target
            known = np.isfinite(layers["traversability"][region])
            passed = layers["traversable"][region]
            correct = np.count_nonzero(known & (passed == target))
            result = (f"{surface}: {correct}/{known.size} correct  "
                      f"({np.sum(known & passed)} traversable, {np.sum(known & ~passed)} blocked, "
                      f"{np.sum(~known)} unknown)")
            summary.append(result)
            print(f"{name} {result}")
        if report is not None:
            from terrain_core.traversability.plot import plot_layers

            geometry = f"{slope}° incline" if slope else "Level ground"
            if rise:
                geometry += f" with {rise * 100:g} cm rise"
            caption = f"Simulated terrain: {name}  |  {geometry}, {noise_mm:g} mm depth noise"
            report.savefig(plot_layers(grid, layers, "\n".join([caption, *summary])))
        if output_dir:
            np.save(output_dir / f"{name}.npy", grid)
            np.save(output_dir / f"{name}_labels.npy", labels)


if __name__ == "__main__":
    main()
