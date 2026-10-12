"""Keyframe is what VO sends to the backend and for is loop closure 

The backend makes one pose variable per keyframe and one odometry (between) factor
per T_prev_kf. Whatever produces keyframes (real VO, fake VO, dataset replay) just
has to fill these fields correctly.

Pose conventions are in slam_core.types.pose (T_a_b = pose of b in a, 4x4, OpenCV
optical frame).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pose import is_pose


# One keyframe from the frontend. Holds the odometry measurement to the previous
# keyframe, VO's own pose estimate, and optional features for loop closure.
@dataclass
class Keyframe:
    id: int                   # unique, increasing
    prev_id: int              # keyframe T_prev_kf is measured from, -1 for the first one
    stamp: float              # seconds

    # measurement used for the between factor
    T_prev_kf: np.ndarray     # 4x4, this keyframe in the previous keyframe's frame
    covariance: np.ndarray    # 6x6 for T_prev_kf, GTSAM order (rx ry rz tx ty tz)

    # VO's estimate, only used as the initial guess
    T_odom_kf: np.ndarray     # 4x4, this keyframe in the odom frame

    # set when tracking was lost since prev; T_prev_kf is then just a guess
    # (usually identity) and the covariance should be large
    tracking_gap: bool = False
    n_inliers: int = 0        # PnP inliers, 0 if unknown or a gap

    # loop closure / local map only, backend ignores these. None if not available.
    keypoints: Optional[np.ndarray] = None    # (N, 2) float32, pixels
    descriptors: Optional[np.ndarray] = None  # (N, D) uint8, D = 32 for ORB
    points_3d: Optional[np.ndarray] = None    # (N, 3) float32, m, camera frame

    # Validate everything on construction. Bad poses or covariances here would
    # otherwise only show up later as a warped map.
    def __post_init__(self):
        self.T_prev_kf = np.asarray(self.T_prev_kf, dtype=np.float64)
        self.T_odom_kf = np.asarray(self.T_odom_kf, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)

        if self.prev_id >= self.id:
            raise ValueError(f"keyframe {self.id}: prev_id {self.prev_id} must be smaller than id")
        if self.prev_id < -1:
            raise ValueError(f"keyframe {self.id}: prev_id must be -1 (first keyframe) or >= 0")

        for name in ("T_prev_kf", "T_odom_kf"):
            if not is_pose(getattr(self, name)):
                raise ValueError(f"keyframe {self.id}: {name} is not a 4x4 rigid transform")

        c = self.covariance
        if c.shape != (6, 6) or not np.all(np.isfinite(c)):
            raise ValueError(f"keyframe {self.id}: covariance must be a finite 6x6 matrix")
        if not np.allclose(c, c.T) or np.any(np.diag(c) <= 0):
            raise ValueError(f"keyframe {self.id}: covariance must be symmetric with a positive diagonal")

        # feature arrays must be 2D, have the right width, and all share the same N
        n = None
        for name, cols in (("keypoints", 2), ("points_3d", 3), ("descriptors", None)):
            a = getattr(self, name)
            if a is None:
                continue
            if a.ndim != 2 or (cols is not None and a.shape[1] != cols):
                raise ValueError(f"keyframe {self.id}: {name} has shape {a.shape}")
            if n is not None and len(a) != n:
                raise ValueError(f"keyframe {self.id}: keypoints, descriptors, points_3d lengths differ")
            n = len(a)

    # True for the first keyframe, which has no odometry factor.
    @property
    def is_first(self):
        return self.prev_id == -1
