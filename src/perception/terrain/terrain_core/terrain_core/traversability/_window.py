"""Shared helpers for sliding-window operations on a 2.5D grid.

Grids are 2D float arrays indexed [row, col] with NaN meaning "no estimate".
"""
from __future__ import annotations

import numpy as np


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
