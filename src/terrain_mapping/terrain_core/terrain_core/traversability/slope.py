"""Slope and roughness from a local least-squares plane fit, plus the noise-aware "planar" test.

For each cell, fit z = a*x + b*y + c to the valid cells in a window about `window_m` wide.
  slope     = atan(sqrt(a^2 + b^2))            [radians]
  roughness = RMS of the fit residuals          [metres]

Window size matters because elevation maps are noisy. With per-cell noise sigma, the slope error of a plane fit is
about sigma / sqrt(sum(dx^2)), which shrinks fast as the window grows: at 1-2 cm noise a 3x3 window (10 cm) gives
several degrees of slope noise on flat floor, a ~40 cm window gives about 1 degree. The default is 0.40 m.

Cells with too few valid neighbours, and cells whose own height is unknown, are NaN, so the layers never claim more
than the elevation map knows.

Planar test: a step edge or box wall inside the window is not a slope (the plane fit reports a steep tilt that is really
the step), and the step layer already covers it. Cells are therefore "planar" (slope enforced) only if their
residual RMS is at most `planar_tol`, which should sit just above the map's own noise level. See planar_tolerance().
"""
from __future__ import annotations

import numpy as np

from ._window import shifted_stack, window_offsets, window_radius


def plane_fit(
    elevation: np.ndarray,
    resolution: float,
    window_m: float = 0.40,
    min_valid_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (slope_rad, roughness_m), both shaped like `elevation`.

    `min_valid_frac` is the fraction of the window that must hold valid cells for a fit to be trusted.
    """
    elevation = np.asarray(elevation, dtype=float)
    radius = window_radius(window_m, resolution)
    min_valid = max(3, int(np.ceil(min_valid_frac * (2 * radius + 1) ** 2)))

    z = shifted_stack(elevation, radius)                  # (K, H, W)
    offs = window_offsets(radius) * resolution            # (K, 2) metres
    dy = offs[:, 0][:, None, None]                        # row offset
    dx = offs[:, 1][:, None, None]                        # col offset

    # Fit relative to each cell's own height. The plane is invariant to a constant offset, and small numbers keep the
    # residual sum-of-squares free of catastrophic cancellation.
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
    # Replace singular / unused systems with identity so the batched solve is safe.
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


def estimate_noise_sigma(roughness: np.ndarray, window_cells: int) -> float:
    """Estimate the map's per-cell elevation noise (metres) from its plane-fit roughness.

    On smooth terrain the residual RMS of a plane fit over n cells is about sigma * sqrt((n - 3) / n). The median over
    the whole map is robust to the minority of cells that sit on steps or obstacles. On a map that is mostly rough
    (rubble) this overestimates sigma, which only makes the planar test stricter about what counts as flat.
    Returns 0.0 if the map has no valid roughness.
    """
    r = np.asarray(roughness, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    shrink = np.sqrt(max(window_cells - 3, 1) / window_cells)
    return float(np.median(r) / shrink)


def planar_tolerance(sigma: float, floor_m: float = 0.01, noise_factor: float = 1.5) -> float:
    """Residual RMS below which a cell counts as a planar surface: max(floor, noise_factor * sigma).

    Residual RMS over a window has a tight spread (about sigma / sqrt(2 * n)), so 1.5 x sigma is many standard
    deviations above what noise alone produces, while a step of height h adds about 0.25 * h of residual and so
    still separates from noise once h is more than a few times sigma.
    """
    return max(floor_m, noise_factor * sigma)


def compute_slope(elevation, resolution, window_m=0.40, min_valid_frac=0.5) -> np.ndarray:
    return plane_fit(elevation, resolution, window_m, min_valid_frac)[0]


def compute_roughness(elevation, resolution, window_m=0.40, min_valid_frac=0.5) -> np.ndarray:
    return plane_fit(elevation, resolution, window_m, min_valid_frac)[1]
