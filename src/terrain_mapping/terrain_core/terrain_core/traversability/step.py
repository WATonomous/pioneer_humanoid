"""Step height from neighbouring cell differences on a small window, with ramp correction on planar cells."""
from __future__ import annotations

import numpy as np

from ._window import shifted_stack, window_offsets


def step_radius_cells(radius_m: float, resolution: float) -> int:
    """Neighbour reach in cells for a radius in metres (at least 1)."""
    if radius_m <= 0 or resolution <= 0:
        raise ValueError("radius_m and resolution must be positive")
    return max(1, int(np.floor(radius_m / resolution + 0.5 + 1e-9)))


def compute_step(
    elevation: np.ndarray,
    resolution: float,
    radius_m: float = 0.05,
    slope: np.ndarray | None = None,
    planar: np.ndarray | None = None,
) -> np.ndarray:
    """Max height difference to neighbours within `radius_m`, NaN where the cell is unknown.

    With `slope` (rad), the rise a tilt explains is subtracted, only on `planar` cells if given.
    """
    elevation = np.asarray(elevation, dtype=float)
    radius = step_radius_cells(radius_m, resolution)
    z = shifted_stack(elevation, radius)                       # (K, H, W)
    offs = window_offsets(radius)
    dist = np.hypot(offs[:, 0], offs[:, 1]) * resolution       # (K,)

    diff = np.abs(z - elevation[None])                         # NaN if either side is NaN
    if slope is not None:
        use = np.isfinite(slope)
        if planar is not None:
            use &= np.asarray(planar, dtype=bool)
        tilt = np.where(use, np.tan(np.where(np.isfinite(slope), slope, 0.0)), 0.0)
        diff = np.maximum(diff - tilt[None] * dist[:, None, None], 0.0)

    diff[dist == 0.0] = np.nan                                 # skip the cell itself
    has_neighbour = np.isfinite(diff).any(0)
    step = np.where(has_neighbour, np.max(np.where(np.isfinite(diff), diff, -np.inf), axis=0), np.nan)
    step[~np.isfinite(elevation)] = np.nan
    return step
