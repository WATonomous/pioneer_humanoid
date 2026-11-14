"""A verified relative-pose measurement between two existing keyframes."""

from dataclasses import dataclass

import numpy as np

from .pose import is_pose


@dataclass
class LoopClosure:
    old_id: int
    new_id: int
    stamp: float
    T_old_new: np.ndarray
    covariance: np.ndarray
    n_inliers: int = 0

    # Validate on construction, like Keyframe. A bad pose or covariance here would
    # otherwise only show up later as a GTSAM error or a warped map.
    def __post_init__(self):
        self.T_old_new = np.asarray(self.T_old_new, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)

        name = f"loop {self.old_id}->{self.new_id}"
        if self.old_id < 0 or self.old_id >= self.new_id:
            raise ValueError(f"{name}: need 0 <= old_id < new_id")
        if not np.isfinite(self.stamp):
            raise ValueError(f"{name}: stamp must be finite")
        if not is_pose(self.T_old_new):
            raise ValueError(f"{name}: T_old_new is not a 4x4 rigid transform")

        c = self.covariance
        if c.shape != (6, 6) or not np.all(np.isfinite(c)):
            raise ValueError(f"{name}: covariance must be a finite 6x6 matrix")
        if not np.allclose(c, c.T) or np.any(np.diag(c) <= 0):
            raise ValueError(f"{name}: covariance must be symmetric with a positive diagonal")

        if self.n_inliers < 0:
            raise ValueError(f"{name}: n_inliers must be >= 0")
