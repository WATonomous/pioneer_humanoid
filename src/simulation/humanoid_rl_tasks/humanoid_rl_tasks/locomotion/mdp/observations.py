"""Locomotion observation terms (shadows Isaac Lab's ``height_scan`` with a NaN-safe version)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.observations import height_scan as _isaac_height_scan

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg

__all__ = ["height_scan"]


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan with NaN ray hits replaced by zero (the observation clip passes NaN through)."""
    scan = _isaac_height_scan(env, sensor_cfg, offset)
    return torch.where(torch.isnan(scan), torch.zeros_like(scan), scan)
