"""Bare scene: the shared lightbox workcell and arm. The default scene."""
from __future__ import annotations

from humanoid_mujoco_scenes import ROBOT_BASE_POS, WORKCELL_CAMERA, add_lightbox_workcell, scene


@scene("bare", robot_pos=ROBOT_BASE_POS, camera=WORKCELL_CAMERA)
def build(spec):
    add_lightbox_workcell(spec)
