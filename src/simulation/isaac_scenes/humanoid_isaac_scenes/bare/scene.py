"""Bare scenes: the empty lightbox workcell + arm.

  ``bare``      SIMPLE (the default teleop scene): arm on its floor stand, 30.5-inch table, CAD lightbox and its tube light.
  ``bare_lab``  FULL LAB: the same plus the lab room and the overhead lighting.

Both come from ``humanoid_rl_tasks.workcell``; ``bare_lab`` is built on ``bare``, so changes to the rig reach both.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from humanoid_isaac_scenes import scene
from humanoid_rl_tasks.workcell import (
    LAB_WORKCELL_CAMERA,
    ROBOT_BASE_POS,
    WORKCELL_CAMERA,
    LabRoomCfg,
    LightboxWorkcellCfg,
)


@scene("bare", robot_pos=ROBOT_BASE_POS, camera=WORKCELL_CAMERA)
@configclass
class BareSceneCfg(LightboxWorkcellCfg):
    pass


@scene("bare_lab", robot_pos=ROBOT_BASE_POS, camera=LAB_WORKCELL_CAMERA)
@configclass
class BareLabSceneCfg(BareSceneCfg, LabRoomCfg):
    pass
