"""Feature detection and description."""
from dataclasses import dataclass

import numpy as np
import time
import torch


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


class FeatureExtractor:
    """ XFeat, then a score filter and grid selection. 
        extract() takes a uint8 image and returns the features
        last_ms is the netowrk time of the last call
        last_post_ms is filter + grid
    """

    def __init__(self, top_k=4000, min_score=0.01, cell=40, per_cell=6, device=None):
        self.top_k = top_k
        self.min_score = min_score
        self.cell = cell
        self.per_cell = per_cell
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=top_k
        ).to(self.device)
        self.last_ms = 0.0
        self.last_post_ms = 0.0

    def extract(self, image):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        h, w = image.shape[:2]

        self._sync()
        t0 = time.perf_counter()
        out = self.model.detectAndCompute(image, top_k=self.top_k)[0]
        self._sync()
        t1 = time.perf_counter()

        kps = out["keypoints"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        desc = out["descriptors"].cpu().numpy()

        good = np.where(scores >= self.min_score)[0]
        keep = good[grid_select(kps[good], scores[good], h, w, self.cell, self.per_cell)]
        t2 = time.perf_counter()

        self.last_ms = (t1 - t0) * 1000
        self.last_post_ms = (t2 - t1) * 1000
        return Features(kps[keep], scores[keep], desc[keep])

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
