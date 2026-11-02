"""Locomotion observation terms.

This module intentionally shadows Isaac Lab's ``height_scan`` term: every
locomotion config in this package imports ``mdp`` from here, so defining a
NaN-safe version once keeps a malformed ray hit out of the policy input.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.observations import height_scan as _isaac_height_scan

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg

__all__ = ["height_scan"]


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan with NaN ray hits replaced by zero.

    The stock term returns ``sensor_height - hit_z - offset``. A failed ray can
    come back as NaN instead of +/-inf, and the observation clip passes NaN
    through, poisoning every policy input that follows it. Treat an
    unparseable hit as zero so one bad ray cannot corrupt the observation.
    """
    scan = _isaac_height_scan(env, sensor_cfg, offset)
    return torch.where(torch.isnan(scan), torch.zeros_like(scan), scan)
