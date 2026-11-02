"""Rubik's cube: one 5.6 cm cube on the table, straight in front of the LEFT arm. The only object.

The cube sits in line with the left gripper's home pose (elbow bent 90 deg, forearm forward): centred
between the open jaws (~9.5 cm apart there) and just past their tips, so reaching it is lowering the
forearm ~15 cm and reaching ~5 cm further. One collision box (a solved cube; the layers don't turn); the 54 stickers are visual only.

Each episode moves the cube a little around that spot and turns it; the task is done once it is lifted and steady.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from humanoid_mujoco_scenes import TABLE_TOP_Z, add_workcell, scene

CUBE_SIZE = 0.056               # m, a standard 3x3
CUBE_MASS = 0.09                # kg, a speed cube is 70-100 g
CUBE_POS = (0.45, 0.29)         # xy on the table: in line with the left jaws at home (x 0.36-0.45, centred y 0.289)
CUBE_JITTER = (0.03, 0.03)      # m, +- around CUBE_POS per episode
CUBE_YAW_JITTER = math.radians(20.0)  # rad, +-
LIFT_HEIGHT = 0.05              # m above its resting height, for success
HELD_SPEED = 0.2                # m/s: slower than this, it is held rather than thrown or falling

STICKER_FRACTION = 0.85        # sticker width / cubie pitch; the rest shows the black body between them
STICKER_THICKNESS = 0.0005      # m, stands proud of the body so it never z-fights
# Face normal -> colour, Western scheme: white up, green toward the robot (-x), red on its left (+y).
FACE_COLORS = {
    (0, 0, 1): (1.0, 1.0, 1.0),
    (0, 0, -1): (1.0, 0.85, 0.0),
    (-1, 0, 0): (0.0, 0.62, 0.28),
    (1, 0, 0): (0.0, 0.27, 0.68),
    (0, 1, 0): (0.72, 0.07, 0.2),
    (0, -1, 0): (1.0, 0.35, 0.0),
}


def add_stickers(body: mujoco.MjsBody) -> None:
    """3x3 thin visual boxes on each face, centred on the cubies."""
    half, pitch = CUBE_SIZE / 2, CUBE_SIZE / 3
    sticker_half = STICKER_FRACTION * pitch / 2
    for normal, rgb in FACE_COLORS.items():
        axis = [abs(n) for n in normal].index(1)
        in_plane = [a for a in range(3) if a != axis]
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                pos, size = [0.0] * 3, [sticker_half] * 3
                pos[axis] = normal[axis] * (half + STICKER_THICKNESS / 2)
                pos[in_plane[0]], pos[in_plane[1]] = i * pitch, j * pitch
                size[axis] = STICKER_THICKNESS / 2
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=[*rgb, 1],
                              contype=0, conaffinity=0, mass=0, group=1)


def randomise(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """Move the cube to a new spot on the table, at rest."""
    joint = model.joint("rubiks_cube")
    x, y = np.asarray(CUBE_POS) + rng.uniform(-1.0, 1.0, 2) * CUBE_JITTER
    yaw = rng.uniform(-CUBE_YAW_JITTER, CUBE_YAW_JITTER)
    data.qpos[joint.qposadr[0]:joint.qposadr[0] + 7] = [x, y, TABLE_TOP_Z + CUBE_SIZE / 2,
                                                        math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    data.qvel[joint.dofadr[0]:joint.dofadr[0] + 6] = 0.0


def lifted(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """The cube is off the table by LIFT_HEIGHT and not moving fast."""
    joint = model.joint("rubiks_cube")
    height = data.qpos[joint.qposadr[0] + 2] - (TABLE_TOP_Z + CUBE_SIZE / 2)
    return bool(height >= LIFT_HEIGHT and np.linalg.norm(data.qvel[joint.dofadr[0]:joint.dofadr[0] + 3]) < HELD_SPEED)


@scene("rubiks_cube", camera=dict(lookat=[0.38, 0.29, TABLE_TOP_Z + 0.08], distance=0.5, azimuth=180, elevation=-40),
       reset=randomise, success=lifted)
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    x, y = CUBE_POS
    cube = spec.worldbody.add_body(name="rubiks_cube", pos=[x, y, TABLE_TOP_Z + CUBE_SIZE / 2])
    cube.add_freejoint(name="rubiks_cube")
    cube.add_geom(name="rubiks_cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_SIZE / 2] * 3, mass=CUBE_MASS,
                  friction=[1.0, 0.02, 0.001], condim=4, rgba=[0.05, 0.05, 0.05, 1])
    add_stickers(cube)
