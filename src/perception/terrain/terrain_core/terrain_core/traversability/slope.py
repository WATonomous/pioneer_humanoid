"""Slope and roughness from a local least-squares plane fit.

For each cell, fit z = a*x + b*y + c to the valid cells in a (2r+1)^2 window.
  slope     = atan(sqrt(a^2 + b^2))            [radians]
  roughness = RMS of the fit residuals          [metres]

Both come from the same fit, so they are computed together. Cells with fewer
than `min_valid` valid neighbours (including themselves) are NaN, and so are
cells whose own height is unknown, so the layers never claim more than the
elevation map knows.
"""
from __future__ import annotations

import numpy as np

from ._window import shifted_stack, window_offsets


def plane_fit(
    elevation: np.ndarray,
    resolution: float,
    radius: int = 1,
    min_valid: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (slope_rad, roughness_m), both shaped like `elevation`."""
    elevation = np.asarray(elevation, dtype=float)
    if min_valid is None:
        min_valid = max(3, ((2 * radius + 1) ** 2 + 1) // 2)

    z = shifted_stack(elevation, radius)                  # (K, H, W)
    offs = window_offsets(radius) * resolution            # (K, 2) metres
    dy = offs[:, 0][:, None, None]                        # row offset
    dx = offs[:, 1][:, None, None]                        # col offset

    # Fit relative to each cell's own height. The plane is invariant to a constant offset, and
    # small numbers keep the residual sum-of-squares free of catastrophic cancellation.
    z = z - elevation[None]
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
    # Replace singular / unused systems with identity so batched solve is safe.
    det = np.linalg.det(A)
    ok &= np.abs(det) > 1e-18
    A[~ok] = np.eye(3)
    rhs[~ok] = 0.0

    params = np.linalg.solve(A, rhs[..., None])[..., 0]   # (H, W, 3) a, b, c

    slope = np.arctan(np.hypot(params[..., 0], params[..., 1]))

    # Sum of squared residuals = szz - params . rhs (normal-equation identity).
    ssr = szz - (params * rhs).sum(-1)
    rough = np.sqrt(np.maximum(ssr, 0.0) / np.maximum(n, 1.0))

    slope[~ok] = np.nan
    rough[~ok] = np.nan
    return slope, rough


def compute_slope(elevation, resolution, radius=1, min_valid=None) -> np.ndarray:
    return plane_fit(elevation, resolution, radius, min_valid)[0]


def compute_roughness(elevation, resolution, radius=1, min_valid=None) -> np.ndarray:
    return plane_fit(elevation, resolution, radius, min_valid)[1]
