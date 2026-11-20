"""Slope and roughness from a least-squares plane fit over a window, plus the noise-aware planar test."""
from __future__ import annotations

import numpy as np

from ._window import shifted_stack, window_offsets, window_radius


def plane_fit(
    elevation: np.ndarray,
    resolution: float,
    window_m: float = 0.40,
    min_valid_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (slope_rad, roughness_m); NaN where the cell or too much of its window is unknown."""
    elevation = np.asarray(elevation, dtype=float)
    radius = window_radius(window_m, resolution)
    min_valid = max(3, int(np.ceil(min_valid_frac * (2 * radius + 1) ** 2)))

    z = shifted_stack(elevation, radius)                  # (K, H, W)
    offs = window_offsets(radius) * resolution            # (K, 2) metres
    dy = offs[:, 0][:, None, None]
    dx = offs[:, 1][:, None, None]

    z = z - elevation[None]                               # fit relative to the cell's own height
    valid = np.isfinite(z)
    w = valid.astype(float)
    zz = np.where(valid, z, 0.0)

    n = w.sum(0)
    sx = (w * dx).sum(0)
    sy = (w * dy).sum(0)
    sxx = (w * dx * dx).sum(0)
    sxy = (w * dx * dy).sum(0)
    syy = (w * dy * dy).sum(0)
    sz = zz.sum(0)
    sxz = (zz * dx).sum(0)
    syz = (zz * dy).sum(0)
    szz = (zz * zz).sum(0)

    h, wd = elevation.shape
    A = np.empty((h, wd, 3, 3))
    A[..., 0, :] = np.stack([sxx, sxy, sx], axis=-1)
    A[..., 1, :] = np.stack([sxy, syy, sy], axis=-1)
    A[..., 2, :] = np.stack([sx, sy, n], axis=-1)
    rhs = np.stack([sxz, syz, sz], axis=-1)               # (H, W, 3)

    ok = (n >= min_valid) & np.isfinite(elevation)
    det = np.linalg.det(A)
    ok &= np.abs(det) > 1e-18
    A[~ok] = np.eye(3)                                    # keep the batched solve safe on unused cells
    rhs[~ok] = 0.0

    params = np.linalg.solve(A, rhs[..., None])[..., 0]   # (H, W, 3): a, b, c

    slope = np.arctan(np.hypot(params[..., 0], params[..., 1]))
    ssr = szz - (params * rhs).sum(-1)                    # sum of squared residuals
    rough = np.sqrt(np.maximum(ssr, 0.0) / np.maximum(n, 1.0))

    slope[~ok] = np.nan
    rough[~ok] = np.nan
    return slope, rough


def estimate_noise_sigma(roughness: np.ndarray, window_cells: int) -> float:
    """Per-cell elevation noise (m) from the map's median roughness; 0.0 if there is no valid roughness."""
    r = np.asarray(roughness, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    shrink = np.sqrt(max(window_cells - 3, 1) / window_cells)
    return float(np.median(r) / shrink)


def planar_tolerance(sigma: float, floor_m: float = 0.01, noise_factor: float = 1.5) -> float:
    """Roughness below which a cell counts as planar: max(floor, noise_factor * sigma)."""
    return max(floor_m, noise_factor * sigma)


def compute_slope(elevation, resolution, window_m=0.40, min_valid_frac=0.5) -> np.ndarray:
    return plane_fit(elevation, resolution, window_m, min_valid_frac)[0]


def compute_roughness(elevation, resolution, window_m=0.40, min_valid_frac=0.5) -> np.ndarray:
    return plane_fit(elevation, resolution, window_m, min_valid_frac)[1]
