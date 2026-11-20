"""Sliding-window helpers for 2D height grids (NaN = unknown). Window sizes are given in metres."""
from __future__ import annotations

import numpy as np


def window_radius(window_m: float, resolution: float) -> int:
    """Cell radius r of the (2r+1) x (2r+1) window closest to `window_m` wide (at least 1; ties go larger)."""
    if window_m <= 0 or resolution <= 0:
        raise ValueError("window_m and resolution must be positive")
    return max(1, int(np.floor((window_m / resolution - 1.0) / 2.0 + 0.5 + 1e-9)))


def shifted_stack(grid: np.ndarray, radius: int) -> np.ndarray:
    """(K, H, W) array of `grid` shifted by every offset in [-radius, radius]^2; out-of-bounds is NaN."""
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
