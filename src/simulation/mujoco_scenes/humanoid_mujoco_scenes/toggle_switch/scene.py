"""Toggle switch: the classic lever light switch on a wall plate. Flip the lever up to turn the light on.

The lever sticks out of the plate toward the robot and hinges about Y. Down (-LIMIT) is off, up (+LIMIT) is on,
and a detent holds it at either end. The robot's gripper points forward and the fingers push the lever up from
below. See ``humanoid_mujoco_scenes/_wall_switch.py`` for the wall, plate, hinge and detent shared with
``rocker_switch``. ``success``: the lever is up past ON_THRESHOLD.
"""
from __future__ import annotations

import mujoco

from humanoid_mujoco_scenes import add_workcell, scene
from humanoid_mujoco_scenes._wall_switch import CAMERA, PLATE_SIZE, add_hinge, add_wall, make_hooks

LIMIT = 0.5                     # rad (about 29 degrees) each way
ON_THRESHOLD = 0.35
DETENT_TORQUE = 0.04            # N m: about 1 N at the lever tip
LEVER_LENGTH = 0.04             # plate to tip, along -X (toward the robot)
LEVER_WIDTH = 0.012             # y
LEVER_THICK = 0.008             # z
LEVER_MASS = 0.01

reset, step, angle, is_on = make_hooks(LIMIT, ON_THRESHOLD, DETENT_TORQUE)


@scene("toggle_switch", camera=CAMERA, reset=reset, step=step, success=is_on)
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    plate = add_wall(spec)
    lever = add_hinge(spec, plate, LIMIT)
    # Hinge at the plate's front face (x = 0); the lever runs toward the robot (-x) from there.
    lever.add_geom(name="lever", type=mujoco.mjtGeom.mjGEOM_BOX, size=[LEVER_LENGTH / 2, LEVER_WIDTH / 2, LEVER_THICK / 2],
                   pos=[-LEVER_LENGTH / 2, 0, 0], mass=LEVER_MASS, friction=[1.0, 0.02, 0.001], rgba=[0.97, 0.97, 0.95, 1])
