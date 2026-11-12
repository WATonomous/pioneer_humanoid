"""Batched lift -> transfer -> settle metrics, usable by your own Isaac policy runner.

Success requires a previous lift, the whole bag bounding box inside the target,
upright orientation, its bottom at tray-floor height, and low speed for one second.
It measures settled placement, not gripper release/contact or learned-policy quality.
"""
import math

import torch

from .scene import TABLE_TOP_Z, TARGET_HALF_XY, TARGET_XY


def placement_metrics(root_state, env_origins, target_xy=TARGET_XY):
    """Accept (N,13) Isaac root state (xyz, wxyz, linear vel, angular vel)."""
    pos = root_state[:, :3] - env_origins
    w, x, y, z = root_state[:, 3:7].unbind(-1)
    matrix = torch.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    # Conservative bounds of NASA's body and handle, after source metre conversion.
    lower = pos.new_tensor([-0.2032, -0.11451235, -0.26679093])
    upper = pos.new_tensor([0.2032, 0.11451235, 0.31119004])
    centre = pos + matrix @ ((lower + upper) / 2)
    half = matrix.abs() @ ((upper - lower) / 2)
    bottom = centre[:, 2] - half[:, 2]
    inside = ((centre[:, :2] - torch.as_tensor(target_xy, dtype=pos.dtype, device=pos.device)).abs() + half[:, :2]
              <= pos.new_tensor(TARGET_HALF_XY)).all(-1)
    upright = matrix[:, 2, 2] >= math.cos(math.radians(15.0))
    supported = (bottom - TABLE_TOP_Z).abs() < 0.015
    slow = (root_state[:, 7:10].norm(dim=-1) < 0.03) & (root_state[:, 10:13].norm(dim=-1) < 0.15)
    return {"lifted": bottom > TABLE_TOP_Z + 0.06, "inside_target": inside,
            "upright": upright, "settled": inside & upright & supported & slow,
            "bottom_height": bottom}


class CargoTask:
    def __init__(self, scene, moving_target=False):
        self.scene = scene
        self.moving_target = moving_target
        self.lifted = torch.zeros(scene.num_envs, dtype=torch.bool, device=scene.device)
        self.settled_s = torch.zeros(scene.num_envs, device=scene.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.lifted[ids] = False
        self.settled_s[ids] = 0.0

    def update(self, dt):
        target = (self.scene["target_station"].data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
                  if self.moving_target else TARGET_XY)
        metrics = placement_metrics(self.scene["cargo"].data.root_state_w, self.scene.env_origins, target)
        self.lifted |= metrics["lifted"]
        self.settled_s = torch.where(metrics["settled"] & self.lifted, self.settled_s + dt, 0.0)
        return {**metrics, "ever_lifted": self.lifted.clone(), "success": self.settled_s >= 1.0}
