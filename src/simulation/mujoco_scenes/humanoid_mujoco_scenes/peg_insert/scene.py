"""Peg insertion: a square peg on the table and a block with a matching square hole, in front of the LEFT arm.

All primitives (boxes), so the fit is exact: the hole is ``PEG_SIZE + CLEARANCE`` wide and runs
down to the table top. Change CLEARANCE to make it harder or easier. Every reset shifts the peg and
the block a little. ``is_inserted``: the peg is in the hole, at least INSERT_DEPTH deep.
"""
from __future__ import annotations

import mujoco
import numpy as np

from humanoid_mujoco_scenes import add_floor, scene

TABLE_TOP_Z = 0.705             # same table height as the Isaac push scene
TABLE_X = (0.15, 0.85)          # front/back edge, robot frame
TABLE_HALF_Y = 0.6

PEG_SIZE = 0.04                 # square cross-section (m); the gripper's collision hulls close past 0
PEG_HEIGHT = 0.10
PEG_MASS = 0.05
PEG_POS = (0.32, 0.26)          # xy on the table; both inside the left arm's gripper-down reach

CLEARANCE = 0.004               # hole width minus peg width (m): 2 mm a side, about what a person manages by teleop
BLOCK_SIZE = 0.12
BLOCK_HEIGHT = 0.06
BLOCK_POS = (0.43, 0.32)
PEG_JITTER = 0.015              # m, random shift of the peg's start per reset (each axis)...
BLOCK_JITTER = 0.01             # ...and of the block; the two never overlap
INSERT_DEPTH = 0.02             # peg bottom this far below the block top counts as inserted


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """New episode: shift the peg (upright, square) and the block. Call after mj_resetData."""
    adr = model.joint("peg").qposadr[0]
    px, py = np.add(PEG_POS, rng.uniform(-PEG_JITTER, PEG_JITTER, 2))
    data.qpos[adr:adr + 7] = [px, py, TABLE_TOP_Z + PEG_HEIGHT / 2, 1, 0, 0, 0]
    model.body_pos[model.body("hole_block").id, :2] = np.add(BLOCK_POS, rng.uniform(-BLOCK_JITTER, BLOCK_JITTER, 2))


def is_inserted(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    peg, block = data.xpos[model.body("peg").id], data.xpos[model.body("hole_block").id]
    bottom = peg[2] - PEG_HEIGHT / 2
    inside = np.all(np.abs(peg[:2] - block[:2]) < (PEG_SIZE + CLEARANCE) / 2)
    return bool(inside and bottom < TABLE_TOP_Z + BLOCK_HEIGHT - INSERT_DEPTH)


@scene("peg_insert", camera=dict(lookat=[0.38, 0.29, 0.75], distance=1.0, azimuth=200, elevation=-35), reset=reset)
def build(spec: mujoco.MjSpec) -> None:
    add_floor(spec)
    world = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX

    x0, x1 = TABLE_X
    world.add_geom(
        name="table", type=box, size=[(x1 - x0) / 2, TABLE_HALF_Y, TABLE_TOP_Z / 2],
        pos=[(x0 + x1) / 2, 0.0, TABLE_TOP_Z / 2], rgba=[0.55, 0.42, 0.3, 1],
    )

    # Block = four walls around the hole; the table top is the hole's floor.
    hole = PEG_SIZE + CLEARANCE
    wall = (BLOCK_SIZE - hole) / 2
    bx, by = BLOCK_POS
    hz = BLOCK_HEIGHT / 2
    block = world.add_body(name="hole_block", pos=[bx, by, TABLE_TOP_Z + hz])
    for name, pos, size in (
        ("wall_px", [(hole + wall) / 2, 0, 0], [wall / 2, BLOCK_SIZE / 2, hz]),
        ("wall_nx", [-(hole + wall) / 2, 0, 0], [wall / 2, BLOCK_SIZE / 2, hz]),
        ("wall_py", [0, (hole + wall) / 2, 0], [hole / 2, wall / 2, hz]),
        ("wall_ny", [0, -(hole + wall) / 2, 0], [hole / 2, wall / 2, hz]),
    ):
        block.add_geom(name=name, type=box, pos=pos, size=size, rgba=[0.2, 0.45, 0.75, 1])

    px, py = PEG_POS
    peg = world.add_body(name="peg", pos=[px, py, TABLE_TOP_Z + PEG_HEIGHT / 2])
    peg.add_freejoint(name="peg")
    peg.add_geom(
        name="peg", type=box, size=[PEG_SIZE / 2, PEG_SIZE / 2, PEG_HEIGHT / 2],
        mass=PEG_MASS, friction=[1.0, 0.02, 0.001], condim=4, rgba=[0.85, 0.3, 0.2, 1],
    )
