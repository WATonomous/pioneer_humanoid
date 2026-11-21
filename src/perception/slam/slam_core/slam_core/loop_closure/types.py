"""Data the loop closure module hands to the backend."""
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# One loop closure edge: keyframe `new_id` sits at T_old_new in keyframe `old_id`'s frame.
# The backend turns this into a between factor; `id` lets it retract the same factor later.
@dataclass
class LoopConstraint:
    id: int
    old_id: int
    new_id: int
    T_old_new: np.ndarray       # 4x4
    covariance: np.ndarray      # 6x6, GTSAM order (rx ry rz tx ty tz)
    n_inliers: int
    score: float                # place recognition similarity that found it
    stamp: float = 0.0


# Why a candidate was thrown out, kept for debugging and for the demo plots
@dataclass
class Rejection:
    old_id: int
    new_id: int
    reason: str                 # "verify" | "gate" | "unconfirmed" | "inconsistent"
    value: float = 0.0          # inlier count or Mahalanobis distance, depending on the reason


# What one process() call changes: loops to add, loop ids to take back out, and the rejections
@dataclass
class LoopUpdate:
    added: List[LoopConstraint] = field(default_factory=list)
    retracted: List[int] = field(default_factory=list)
    rejected: List[Rejection] = field(default_factory=list)


# What a verifier returns for one (old, new) keyframe pair
@dataclass
class Verification:
    T_old_new: np.ndarray
    covariance: np.ndarray
    n_inliers: int
