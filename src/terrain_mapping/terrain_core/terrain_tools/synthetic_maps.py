#!/usr/bin/env python3
"""Synthetic 2.5D grid map generator for the humanoid terrain test course.

Builds a realistic 2.5D GridMap from the physical course specifications in
config/test_course.yaml, plus the 4-map benchmark suite:
  1. perfect_course     : matches the course ground truth
  2. floor_shifted_up   : entire map shifted vertically (configured via floor_shift_m)
  3. fake_blob_on_floor : deliberately placed fake obstacle(s) on the flat floor
  4. hole_under_feet    : missing (NaN) hole under the robot's feet
"""
from pathlib import Path

import numpy as np

EDGE_MARGIN_M = 0.04  # 1-cell border margin around each obstacle ROI


class GridMap:
    """2.5D elevation grid with world (x, y) <-> cell index conversion."""

    def __init__(self, elevation, resolution, origin, occupied=None):
        self.z = np.asarray(elevation, dtype=float)
        self.res = float(resolution)
        self.occupied = None if occupied is None else np.asarray(occupied, dtype=bool)
        self.set_origin(origin)

    def set_origin(self, origin):
        self.origin = np.asarray(origin, dtype=float).reshape(2)
        h, w = self.z.shape
        self.xs = self.origin[0] + (np.arange(w) + 0.5) * self.res
        self.ys = self.origin[1] + (np.arange(h) + 0.5) * self.res

    def copy(self):
        occ = None if self.occupied is None else self.occupied.copy()
        return GridMap(self.z.copy(), self.res, self.origin.copy(), occ)

    def save(self, path):
        arrays = {"elevation": self.z, "resolution": self.res, "origin": self.origin}
        if self.occupied is not None:
            arrays["occupied"] = self.occupied
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path, default_res=0.04, default_origin=(-2.0, -0.8)):
        p = Path(path)
        if p.suffix == ".npy":
            z = np.load(p)
            return cls(z, default_res, default_origin)
        d = np.load(p)
        missing = [k for k in ("elevation", "resolution", "origin") if k not in d]
        if missing:
            raise SystemExit(f"{path}: missing required arrays {missing}")
        occ = d["occupied"] if "occupied" in d else None
        return cls(d["elevation"], d["resolution"], d["origin"], occ)

    def region_mask(self, bounds, margin=0.0):
        """Boolean mask of grid cells whose centre lies inside [x_min, x_max, y_min, y_max]."""
        x0, x1, y0, y1 = bounds
        mx = (self.xs >= x0 + margin) & (self.xs <= x1 - margin)
        my = (self.ys >= y0 + margin) & (self.ys <= y1 - margin)
        return my[:, None] & mx[None, :]

    def xy_grids(self):
        return np.meshgrid(self.xs, self.ys)


def _smooth_2d(arr, passes=3):
    out = arr.copy()
    for _ in range(passes):
        out = 0.20 * (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
        )
    return out


def _inject_fake_obstacles(gm, fake_list):
    """Stamp one or more user-configured fake obstacles (circles or boxes) onto a GridMap."""
    X, Y = gm.xy_grids()
    for item in fake_list:
        h = float(item.get("height_m", 0.24))
        shape = str(item.get("shape", "circle")).lower()
        if "bounds" in item or shape == "box":
            b = item["bounds"]
            mask = gm.region_mask(b)
        else:
            cx, cy = item.get("center", [-0.55, 1.65])
            rad = float(item.get("radius_m", 0.22))
            mask = (X - float(cx)) ** 2 + (Y - float(cy)) ** 2 <= rad**2
        gm.z[mask] = h


def build_course_map(course, realistic_hallway=True):
    """Build a 2.5D GridMap of the Nov 6 hallway course from config/test_course.yaml.

    Includes:
      - Step (bottom-right: 73 x 28 cm, 14 cm tall)
      - Ramp (~5 steps ahead on right: 81 x 90 cm, 9 deg incline, 0 -> 14.2 cm)
      - Box (further ahead on left: 39 x 36 cm, 39 cm tall) + attached occlusion shadow
      - Hallway corridor boundaries (unobserved NaN outside walls + noisy left wall strip)
      - Realistic multi-scale D455 floor texture and 1-cell obstacle edge smear
    """
    meta = course["course_metadata"]
    res = float(meta["grid_resolution_m"])
    x0, x1, y0, y1 = meta.get("map_bounds_m", [-2.0, 2.0, -0.8, 5.2])
    nx = int(round((x1 - x0) / res))
    ny = int(round((y1 - y0) / res))

    gm = GridMap(np.zeros((ny, nx), dtype=float), res, origin=[x0, y0])
    X, Y = gm.xy_grids()
    r = course["regions"]

    # 1. Step (bottom-right: 73 x 28 cm, 14 cm tall)
    sb = r["step"]["bounds"]
    gm.z[gm.region_mask(sb)] = float(r["step"]["expected_height_m"])

    # 2. Ramp (~5 steps ahead of step: 81 x 90 cm, 9 deg incline along +y)
    rb = r["ramp"]["bounds"]
    rmask = gm.region_mask(rb)
    h_start = float(r["ramp"]["start_height_m"])
    h_end = float(r["ramp"]["end_height_m"])
    gm.z[rmask] = h_start + (h_end - h_start) * (Y[rmask] - rb[2]) / (rb[3] - rb[2])

    # 3. Box (further up on left: 39 x 36 cm, 39 cm tall)
    bb = r["obstacle_box"]["bounds"]
    gm.z[gm.region_mask(bb)] = float(r["obstacle_box"]["height_m"])

    if realistic_hallway:
        rng = np.random.default_rng(42)
        # Soft 1-cell edge bevel around obstacles (like real Kalman cell fusion) + crisp interior
        beveled = _smooth_2d(gm.z, passes=1)
        for b in (sb, rb, bb):
            interior = gm.region_mask(b, margin=EDGE_MARGIN_M)
            beveled[interior] = gm.z[interior]
        gm.z = beveled

        # Multi-scale D455 floor texture (~0.5 cm RMSE)
        coarse_wave = _smooth_2d(rng.normal(0.0, 0.032, gm.z.shape), passes=18)
        med_tex = _smooth_2d(rng.normal(0.0, 0.014, gm.z.shape), passes=4)
        fine_noise = rng.normal(0.0, 0.0025, gm.z.shape)
        # Mild edge taper near the right wall (outside the [-1.25, 1.25] flat_floor ROI)
        edge_roll = -0.010 * np.clip((X - 1.25) / 0.25, 0.0, 1.0) ** 2
        gm.z += coarse_wave + med_tex + fine_noise + edge_roll

        # Keep under-feet patch well within tolerance
        uf_cfg = r.get("under_feet") or course.get("under_feet") or {"bounds": [-0.4, 0.4, -0.4, 0.4]}
        uf_mask = gm.region_mask(uf_cfg["bounds"])
        gm.z[uf_mask] *= 0.5

        # Camera occlusion shadow (NaN hole) attached directly behind/upper-left of the 39 cm box
        bx_cx = 0.5 * (bb[0] + bb[1])
        dx_s = X - (bx_cx - 0.09)
        dy_s = Y - (bb[3] + 0.10)
        rot_u = 0.85 * dx_s + 0.52 * dy_s
        rot_v = -0.52 * dx_s + 0.85 * dy_s
        shadow = ((rot_u / 0.19) ** 2 + (rot_v / 0.25) ** 2 <= 1.0) & (~gm.region_mask(bb))
        gm.z[shadow] = np.nan

        # Hallway left and right walls with organic ragged scan edges
        hw_l, hw_r = meta.get("hallway_width_m", [-1.48, 1.48])
        row_jitter_l = _smooth_2d(rng.normal(0.0, 0.05, (ny, 1)), passes=2)
        row_jitter_r = _smooth_2d(rng.normal(0.0, 0.06, (ny, 1)), passes=3)
        x_left_wall = hw_l + 0.02 * Y + row_jitter_l
        x_right_wall = hw_r + 0.02 * Y + row_jitter_r

        # Light sparse wall-base returns along the left wall edge (outside flat_floor ROI)
        left_band = (X >= x_left_wall) & (X <= x_left_wall + 0.08) & (Y >= -0.65) & (Y <= 4.95)
        left_strip = left_band & (rng.random(gm.z.shape) < 0.45)
        gm.z[left_strip] = rng.uniform(0.02, 0.22, int(left_strip.sum()))

        # Unobserved (NaN) outside the hallway walls and beyond the far depth cap
        top_jitter = _smooth_2d(rng.normal(0.0, 0.08, (1, nx)), passes=3)
        outside = (
            (X < x_left_wall)
            | (X > x_right_wall)
            | (Y > 4.92 - 0.06 * np.abs(X) + top_jitter)
            | (Y < -0.68 + 0.05 * np.abs(X))
        )
        gm.z[outside] = np.nan

    return gm


def build_synthetic_suite(course):
    """Build the 4 benchmark maps: 1 exact course match + 3 with known errors from the course YAML."""
    base = build_course_map(course, realistic_hallway=True)
    faults = course.get("synthetic_faults") or course.get("regions", {}).get("synthetic_faults") or {}

    # 1. Exact course map (ground truth hallway run)
    perfect = base.copy()
    if faults.get("inject_into_perfect_map", False):
        _inject_fake_obstacles(perfect, faults.get("fake_obstacles", []))

    # 2. Known mistake #1: Floor (and map) shifted vertically (configured via floor_shift_m)
    shift_m = float(faults.get("floor_shift_m", 0.05))
    shifted_up = base.copy()
    shifted_up.z += shift_m

    # 3. Known mistake #2: Fake obstacle(s) placed on the flat hallway floor (configured in YAML)
    fake_blob = base.copy()
    default_fakes = [{"shape": "circle", "center": [-0.55, 1.65], "radius_m": 0.22, "height_m": 0.24}]
    fake_list = faults.get("fake_obstacles", default_fakes)
    _inject_fake_obstacles(fake_blob, fake_list)

    # 4. Known mistake #3: Hole (missing NaN data) under the robot's feet
    hole_under_feet = base.copy()
    uf_cfg = course["regions"].get("under_feet") or course.get("under_feet") or {"bounds": [-0.4, 0.4, -0.4, 0.4]}
    uf_bounds = faults.get("hole_under_feet_bounds", uf_cfg["bounds"])
    hole_under_feet.z[hole_under_feet.region_mask(uf_bounds)] = np.nan

    return {
        "perfect_course": perfect,
        "floor_shifted_up": shifted_up,
        "fake_blob_on_floor": fake_blob,
        "hole_under_feet": hole_under_feet,
    }
