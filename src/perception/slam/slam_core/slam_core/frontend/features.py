"""Feature detection and description."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Features:
    """Keypoints for one image. All arrays share the same row order."""

    keypoints: np.ndarray    # (N, 2) float32, pixel (x, y)
    scores: np.ndarray       # (N,)
    descriptors: np.ndarray  # (N, 64) float32


def grid_select(kps, scores, h, w, cell=40, per_cell=6):
    """Indices of the top `per_cell` keypoints (by score) in each cell x cell block."""
    cx = (kps[:, 0] // cell).astype(int)
    cy = (kps[:, 1] // cell).astype(int)
    cell_id = cy * (w // cell + 1) + cx
    keep, counts = [], {}
    for i in np.argsort(-scores):
        c = cell_id[i]
        if counts.get(c, 0) < per_cell:
            counts[c] = counts.get(c, 0) + 1
            keep.append(i)
    return np.array(keep, dtype=np.int64)
