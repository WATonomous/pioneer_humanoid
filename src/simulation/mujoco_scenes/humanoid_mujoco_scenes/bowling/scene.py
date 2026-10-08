"""Upright bottle: a cylinder lying horizontally on the table in front of the left arm.

Task is for the robot to have the cylinder stand upright, with the cap up.
"""
from __future__ import annotations

import math
import mujoco

from humanoid_mujoco_scenes import add_floor, scene

TABLE_TOP_Z = 0.705             # same table height as the Isaac push scene
TABLE_X = (0.15, 0.85)          # front/back edge, robot frame
TABLE_HALF_Y = 0.6

BOTTLE_SIZE = 0.08              # square cross-section (m); the gripper's collision hulls close past 0
BOTTLE_HEIGHT = 0.21
CAP_SIZE = 0.04
CAP_HEIGHT = 0.03
BOTTLE_MASS = 0.15              # in kg
BOTTLE1_POS = (0.42, 0.26)       # xy on the table; both inside the left arm's gripper-down reach
BOTTLE2_POS = (0.22, 0.40)
BOTTLE3_POS = (0.65, 0.10)

# ---- hooks

# ---- scene
@scene("bowling", camera=dict(lookat=[0.38, 0.29, 0.75], distance=2.0, azimuth=200, elevation=-35))
def build(spec: mujoco.MjSpec) -> None:
    add_floor(spec)
    world = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX     # should make it cylinder (?)

    x0, x1 = TABLE_X
    world.add_geom(
        name="table", type=box, size=[(x1 - x0) / 2, TABLE_HALF_Y, TABLE_TOP_Z / 2],
        pos=[(x0 + x1) / 2, 0.0, TABLE_TOP_Z / 2], rgba=[0.55, 0.42, 0.3, 1],
    )

    bx1, by1 = BOTTLE1_POS
    lie = [0.683, 0.683, 0.183, 0.183]
    bottle = world.add_body(name="bottle1", pos=[bx1, by1, TABLE_TOP_Z + BOTTLE_SIZE / 2 + 0.001], quat = lie)
    bottle.add_freejoint(name="bottle1_free")
    bottle.add_geom(
        name="bottle1", type=box, size=[BOTTLE_SIZE / 2, BOTTLE_SIZE / 2, BOTTLE_HEIGHT / 2],
        mass=BOTTLE_MASS, friction=[1.0, 0.02, 0.001], condim=4, rgba=[0.85, 0.3, 0.2, 1],
    )

    bx2, by2 = BOTTLE2_POS
    lie = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
    bottle = world.add_body(name="bottle2", pos=[bx2, by2, TABLE_TOP_Z + BOTTLE_SIZE / 2 + 0.001], quat = lie)
    bottle.add_freejoint(name="bottle2_free")
    bottle.add_geom(
        name="bottle2", type=box, size=[BOTTLE_SIZE / 2, BOTTLE_SIZE / 2, BOTTLE_HEIGHT / 2],
        mass=BOTTLE_MASS, friction=[1.0, 0.02, 0.001], condim=4, rgba=[0.85, 0.3, 0.2, 1],
    )

    bx3, by3 = BOTTLE3_POS
    lie = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
    bottle = world.add_body(name="bottle3", pos=[bx3, by3, TABLE_TOP_Z + BOTTLE_SIZE / 2 + 0.001], quat = lie)
    bottle.add_freejoint(name="bottle3_free")
    bottle.add_geom(
        name="bottle3", type=box, size=[BOTTLE_SIZE / 2, BOTTLE_SIZE / 2, BOTTLE_HEIGHT / 2],
        mass=BOTTLE_MASS, friction=[1.0, 0.02, 0.001], condim=4, rgba=[0.85, 0.3, 0.2, 1],
    )
