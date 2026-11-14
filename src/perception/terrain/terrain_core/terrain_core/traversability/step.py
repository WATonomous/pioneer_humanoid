"""Step height from neighbouring cell differences.

step(cell) = max |z(cell) - z(neighbour)| over valid neighbours within `radius` cells.

Ramp correction: on a ramp, cells a few centimetres apart differ in height only because the ground is
tilted, which is not a step. Where the local surface is planar (plane-fit residual <= `planar_tol`) we
subtract tan(slope) * distance, so ramps do not read as stairs. Where it is NOT planar (a step edge, a
box wall) the plane fit reports a meaningless steep slope, so we do not correct and the full height shows.
"""
from __future__ import annotations

import numpy as np

from ._window import shifted_stack, window_offsets


def compute_step(
    elevation: np.ndarray,
    resolution: float,
    radius: int = 1,
    slope: np.ndarray | None = None,
    roughness: np.ndarray | None = None,
    planar_tol: float = 0.01,
) -> np.ndarray:
    """Return step height in metres, NaN where the cell itself is unknown.

    Pass `slope` and `roughness` (from slope.plane_fit) to enable the ramp correction. With only
    `slope`, the correction is applied everywhere; with neither, raw neighbour differences are returned.
    """
    elevation = np.asarray(elevation, dtype=float)
    z = shifted_stack(elevation, radius)                       # (K, H, W)
    offs = window_offsets(radius)
    dist = np.hypot(offs[:, 0], offs[:, 1]) * resolution       # (K,)

    diff = np.abs(z - elevation[None])                         # NaN if either side is NaN
    if slope is not None:
        planar = np.isfinite(slope)
        if roughness is not None:
            with np.errstate(invalid="ignore"):
                planar &= roughness <= planar_tol
        expected = np.where(planar, np.tan(np.where(np.isfinite(slope), slope, 0.0)), 0.0)[None] \
            * dist[:, None, None]
        diff = np.maximum(diff - expected, 0.0)

    diff[dist == 0.0] = np.nan                                 # skip the cell itself
    has_neighbour = np.isfinite(diff).any(0)
    step = np.where(has_neighbour, np.max(np.where(np.isfinite(diff), diff, -np.inf), axis=0), np.nan)
    step[~np.isfinite(elevation)] = np.nan
    return step
