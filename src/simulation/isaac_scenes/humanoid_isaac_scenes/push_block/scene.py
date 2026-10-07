"""Teleop registration for the RL push-block scene.

The scene geometry lives with the RL task (`humanoid_rl_tasks.push_block.scene`)
— it's the single source of truth, shared between the RL env cfg and teleop.
Here we just register it under the ``push`` name so ``keyboard_teleop --scene
push`` (and the other teleop scripts) can pull it in with the arm plugged into
its ``MISSING`` robot slot.
"""
from __future__ import annotations

from humanoid_isaac_scenes import scene
from humanoid_rl_tasks.push_block.scene import PushBlockSceneCfg, ROBOT_BASE_POS
from humanoid_rl_tasks.workcell import WORKCELL_CAMERA

# Arm on its floor stand in the lightbox workcell; camera framed on the table top.
scene("push", robot_pos=ROBOT_BASE_POS, camera=WORKCELL_CAMERA)(PushBlockSceneCfg)
