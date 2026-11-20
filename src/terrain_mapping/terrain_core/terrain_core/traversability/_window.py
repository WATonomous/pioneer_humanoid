"""Shared helpers for sliding-window operations on a 2.5D grid.

Grids are 2D float arrays indexed [row, col] with NaN meaning "no estimate".
Window sizes are given in metres by callers and converted to cell radii here, so the same config works
for any grid resolution (5 cm, ETH's 4 cm, ...).
"""
from __future__ import annotations

import numpy as np


def window_radius(window_m: float, resolution: float) -> int:
    """Cell radius r of the (2r+1) x (2r+1) window whose width is closest to `window_m`.

    Always at least 1 (a 3x3 window). Ties round to the larger window, e.g. 0.40 m at 5 cm cells is
    r=4 (9 cells, 45 cm) and at 4 cm cells is r=5 (11 cells, 44 cm).
    """
    if window_m <= 0 or resolution <= 0:
        raise ValueError("window_m and resolution must be positive")
    return max(1, int(np.floor((window_m / resolution - 1.0) / 2.0 + 0.5 + 1e-9)))


def shifted_stack(grid: np.ndarray, radius: int) -> np.ndarray:
    """Return an array of shape (K, H, W) holding `grid` shifted by every
    (dr, dc) offset in [-radius, radius]^2. Out-of-bounds entries are NaN.
    """
    h, w = grid.shape
    padded = np.full((h + 2 * radius, w + 2 * radius), np.nan, dtype=float)
    padded[radius:radius + h, radius:radius + w] = grid
    layers = []
    for dr in range(2 * radius + 1):
        for dc in range(2 * radius + 1):
            layers.append(padded[dr:dr + h, dc:dc + w])
    return np.stack(layers, axis=0)


def window_offsets(radius: int) -> np.ndarray:
    """(K, 2) array of (dr, dc) offsets matching the order of shifted_stack."""
    offs = [(dr - radius, dc - radius)
            for dr in range(2 * radius + 1)
            for dc in range(2 * radius + 1)]
    return np.asarray(offs, dtype=float)
