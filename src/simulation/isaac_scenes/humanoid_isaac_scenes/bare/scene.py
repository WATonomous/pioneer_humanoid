"""Bare scene: the empty lightbox workcell + arm. The default teleop scene.

The lightbox (``humanoid_rl_tasks.workcell``) is the default manipulation
setup, so even "nothing on the table" is the real rig: arm on its floor stand,
30.5-inch table, CAD lightbox.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from humanoid_isaac_scenes import scene
from humanoid_rl_tasks.workcell import ROBOT_BASE_POS, WORKCELL_CAMERA, LightboxWorkcellCfg


@scene("bare", robot_pos=ROBOT_BASE_POS, camera=WORKCELL_CAMERA)
@configclass
class BareSceneCfg(LightboxWorkcellCfg):
    pass
