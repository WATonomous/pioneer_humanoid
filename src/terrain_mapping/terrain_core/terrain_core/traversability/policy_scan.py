"""Resample terrain-map layers into the locomotion policy's 33 x 21 height-scan grid.

Map convention (document at every call site): elevation[row, col], col <-> +x, row <-> +y,
origin_xy = world (x, y) of the CENTRE of cell [0, 0]. NaN = unknown.
Typical use:
    cfg = TraversabilityConfig.from_yaml("config/terrain.yaml")
    trav = compute_traversability(elev, res, limits=cfg.limits, params=cfg.params)
    out = make_policy_scan(elev, res, origin_xy, base_xy, base_z, yaw, trav)
Flatten with out["height_scan"].reshape(-1) to get the policy's ray order (iy*33 + ix).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

SCAN_RES = 0.05
SCAN_NX, SCAN_NY = 33, 21                 # x forward (fast), y left (rows)
SCAN_ORIGIN = (-0.8, -0.5)                # base-frame position of sample (0, 0)
HEIGHT_OFFSET = 0.5                       # policy: height = base_z - terrain_z - 0.5
PATCH_R = 3                               # 7 x 7 foothold patch


def foothold_layer(traversable, resolution, traversability=None,
                   foot_radius_m=0.10, edge_margin_m=0.05, full_clear_m=0.30):
    """Return (bool mask, float32 score) on the map grid; run on the FULL map, never a crop."""
    ok = np.asarray(traversable, dtype=bool)
    padded = np.pad(ok, 1, constant_values=False)          # map border counts as an edge
    clearance = distance_transform_edt(padded)[1:-1, 1:-1] * resolution
    good = ok & (clearance >= foot_radius_m + edge_margin_m - 1e-6)
    score = np.clip(clearance / full_clear_m, 0.0, 1.0)
    if traversability is not None:
        score = np.minimum(score, np.nan_to_num(np.asarray(traversability, float), nan=0.0))
    score[~ok] = 0.0
    return good, score.astype(np.float32)


def scan_grid_xy():
    """Base-frame x and y of every scan sample, each shaped (21, 33)."""
    xs = SCAN_ORIGIN[0] + SCAN_RES * np.arange(SCAN_NX)
    ys = SCAN_ORIGIN[1] + SCAN_RES * np.arange(SCAN_NY)
    return np.meshgrid(xs, ys)


def sample_to_scan(layer, resolution, origin_xy, base_xy, yaw, fill):
    """Nearest-cell lookup of a map layer at the yaw-aligned scan points; off-map -> fill."""
    layer = np.asarray(layer)
    xs, ys = scan_grid_xy()
    c, s = np.cos(yaw), np.sin(yaw)
    wx = base_xy[0] + c * xs - s * ys                      # world = base + R(yaw) @ [x, y]
    wy = base_xy[1] + s * xs + c * ys
    col = np.rint((wx - origin_xy[0]) / resolution).astype(np.int64)
    row = np.rint((wy - origin_xy[1]) / resolution).astype(np.int64)
    inside = (row >= 0) & (row < layer.shape[0]) & (col >= 0) & (col < layer.shape[1])
    out = np.full(xs.shape, fill, dtype=layer.dtype)
    out[inside] = layer[row[inside], col[inside]]
    return out


def make_policy_scan(elevation, resolution, origin_xy, base_xy, base_z, yaw, trav,
                     foot_radius_m=0.10, edge_margin_m=0.05):
    """Height scan + foothold layer on the policy grid. `trav` is compute_traversability()'s dict."""
    elev = np.asarray(elevation, dtype=float)
    if np.shape(trav["traversable"]) != elev.shape:
        raise ValueError("traversable must have the same shape as elevation")
    ok = np.asarray(trav["traversable"], dtype=bool) & np.isfinite(elev)
    mask, score = foothold_layer(ok, resolution, trav.get("traversability"),
                                 foot_radius_m, edge_margin_m)
    args = (resolution, origin_xy, base_xy, yaw)
    elev_s = sample_to_scan(elev, *args, np.nan)
    height = (base_z - elev_s - HEIGHT_OFFSET).astype(np.float32)
    return {
        "height_scan": height,                              # NaN = unknown / off-map
        "known": np.isfinite(height),
        "foothold": sample_to_scan(mask, *args, False),
        "foothold_score": sample_to_scan(score, *args, 0.0).astype(np.float32),
    }


def foothold_obs(height_scan, feet_xy, fallback=0.0):
    """102-value policy foothold observation + per-foot NaN fraction.

    feet_xy: two base-frame (x, y) positions. Per foot: [x, y, 49 heights], x slowest in patch.
    NaN fill: patch median, else scan median, else `fallback`. No training noise/drift applied.
    """
    scan = np.asarray(height_scan, dtype=np.float32)
    if scan.shape != (SCAN_NY, SCAN_NX):
        raise ValueError(f"height_scan must be {(SCAN_NY, SCAN_NX)}, got {scan.shape}")
    flat = scan.reshape(-1)
    k = np.arange(-PATCH_R, PATCH_R + 1)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    scan_fill = np.median(flat[np.isfinite(flat)]) if np.isfinite(flat).any() else fallback
    obs, nan_frac = [], []
    for fx, fy in np.asarray(feet_xy, dtype=float).reshape(2, 2):
        ax = int(np.rint((fx - SCAN_ORIGIN[0]) / SCAN_RES))   # half-to-even, like torch.round
        ay = int(np.rint((fy - SCAN_ORIGIN[1]) / SCAN_RES))
        ix = np.clip(ax + kx, 0, SCAN_NX - 1).reshape(-1)
        iy = np.clip(ay + ky, 0, SCAN_NY - 1).reshape(-1)
        patch = flat[iy * SCAN_NX + ix].copy()
        bad = ~np.isfinite(patch)
        nan_frac.append(bad.mean())
        if bad.any():
            patch[bad] = np.median(patch[~bad]) if (~bad).any() else scan_fill
        obs.append(np.concatenate([[fx, fy], patch]))
    return np.concatenate(obs).astype(np.float32), np.array(nan_frac, dtype=np.float32)