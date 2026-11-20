"""Step height from neighbouring cell differences, on a small window.

step(cell) = max |z(cell) - z(neighbour)| over valid neighbours within `radius_m`, minus the height change a pure
slope would explain.

The step window is deliberately small (default 5 cm, i.e. the 8 immediate neighbours at 4-5 cm cells): a step is a
local event, and a bigger window would smear it across more cells and mix in real tilt. Slope uses its own, larger
window (see slope.py).

Ramp correction: on a ramp, cells a few centimetres apart differ in height only because the ground is tilted, which
is not a step. On planar cells we subtract tan(slope) * distance, so ramps do not read as stairs. On non-planar cells
(a step edge, a box wall) the plane fit reports a meaningless steep slope, so we do not correct and the full height shows.
"""
from __future__ import annotations

import numpy as np

from ._window import shifted_stack, window_offsets


def step_radius_cells(radius_m: float, resolution: float) -> int:
    """Neighbour reach in cells for a step radius in metres (at least 1: the 8 immediate neighbours)."""
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
    """Return step height in metres, NaN where the cell itself is unknown.

    `slope` (radians) enables the ramp correction. `planar` (bool, same shape) limits it to planar cells; if omitted
    the correction is applied everywhere. With neither, raw neighbour differences are returned.
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
