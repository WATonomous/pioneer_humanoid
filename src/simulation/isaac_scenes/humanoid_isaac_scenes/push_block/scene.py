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

# Arm lifted onto its floor stand; camera framed on the table + ramp box.
scene(
    "push",
    robot_pos=ROBOT_BASE_POS,
    camera=([1.4, -1.0, 0.9], [0.35, -0.2, 0.05]),
)(PushBlockSceneCfg)
