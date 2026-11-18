"""Use one physical yaw reference for locomotion without changing body state or task history."""

import math

import torch

from .strike_history import rotate_history


# Express every historical world vector in the same frame, retaining real body feedback and timing.
def in_navigation_frame(observations, reference):
    if reference is None:
        return observations
    if not math.isfinite(reference):
        raise ValueError("Navigation yaw reference must be finite")
    values = observations["actor"]
    if values.shape[1] != 207:
        raise ValueError("Navigation reference requires the shared 207-value history")
    angle = reference - torch.atan2(values[:, 175], values[:, 176])
    result = observations.clone()
    result["actor"] = rotate_history(values, angle)
    return result
