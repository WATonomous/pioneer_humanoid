"""Rocker (decora) switch: a wide flat paddle on a wall plate. Press the top half in to turn the light on.

The paddle hinges about Y at its middle. Its top half pressed in (toward the wall) turns it on, its bottom half
pressed in turns it off; a detent holds it at either end. The paddle is 32 x 57 mm and only tips about 9 degrees,
so it is a finer target than the toggle. The gripper points forward and a finger presses the top half.
See ``humanoid_mujoco_scenes/_wall_switch.py`` for the wall, plate, hinge and detent shared with ``toggle_switch``.
``success``: the top is in past ON_THRESHOLD.
"""
from __future__ import annotations

import mujoco

from humanoid_mujoco_scenes import add_workcell, scene
from humanoid_mujoco_scenes._wall_switch import CAMERA, add_hinge, add_wall, make_hooks

LIMIT = 0.15                    # rad (about 8.6 degrees) each way
ON_THRESHOLD = 0.11
DETENT_TORQUE = 0.04            # N m: about 1.4 N at the paddle's top edge
PADDLE = (0.012, 0.032, 0.057)  # x (depth), y, z
PADDLE_MASS = 0.02

reset, step, angle, is_on = make_hooks(LIMIT, ON_THRESHOLD, DETENT_TORQUE)


@scene("rocker_switch", camera=CAMERA, reset=reset, step=step, success=is_on)
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    plate = add_wall(spec)
    paddle = add_hinge(spec, plate, LIMIT)
    px, py, pz = PADDLE
    # Hinge on the plate's front face; the paddle's back face sits on it and it stands out toward the robot.
    paddle.add_geom(name="paddle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[px / 2, py / 2, pz / 2], pos=[-px / 2, 0, 0],
                    mass=PADDLE_MASS, friction=[1.0, 0.02, 0.001], rgba=[0.97, 0.97, 0.95, 1])
