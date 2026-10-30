"""A verified relative-pose measurement between two existing keyframes."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LoopClosure:
    old_id: int
    new_id: int
    stamp: float
    T_old_new: np.ndarray
    covariance: np.ndarray
    n_inliers: int = 0

    def __post_init__(self):
        self.T_old_new = np.asarray(self.T_old_new, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)

