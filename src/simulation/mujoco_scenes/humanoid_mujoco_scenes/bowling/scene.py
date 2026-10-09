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

# Reset: every new episode, scatter the bottles to random spots on the table, each lying on its
# side and pointing in a random direction, so the policy can't just memorise one layout.
BOTTLE_NAMES = ["bottle1", "bottle2", "bottle3"]

# Where bottle centres may land (m, robot frame), on the left arm's side of the table. The near
# edges keep every bottle clear of the robot's stand, whose collision shape bulges out over the
# table near the centre line (checked over 2000 random resets: no bottle spawns touching it).
# TODO: check the far edges against what the left arm can actually reach.
SPAWN_X = (0.30, 0.55)
SPAWN_Y = (0.18, 0.50)

# Overlap check: each bottle is marked by dots along its length, and two bottles are too close
# if any of their dots are nearer than MIN_DOT_GAP (about one bottle width plus 2 cm of air).
DOTS_PER_BOTTLE = 7
MIN_DOT_GAP = BOTTLE_SIZE + 0.02
MAX_TRIES = 20000              # ~1% of random layouts fit, so this many tries never runs out


def lying_quat(yaw: float) -> list[float]:
    """Orientation of a bottle lying on its side, then turned by ``yaw`` (rad) about the vertical.

    MuJoCo stores orientations as quaternions [w, x, y, z]. This is two rotations combined:
    tip over 90 degrees about the x axis (long axis now horizontal), then spin by yaw about z.
    """
    tip = math.pi / 2
    return [
        math.cos(yaw / 2) * math.cos(tip / 2),
        math.cos(yaw / 2) * math.sin(tip / 2),
        math.sin(yaw / 2) * math.sin(tip / 2),
        math.sin(yaw / 2) * math.cos(tip / 2),
    ]


def bottle_dots(x: float, y: float, yaw: float) -> list[tuple[float, float]]:
    """Points evenly spaced from one end of a lying bottle to the other, seen from above.
    """
    half = BOTTLE_HEIGHT / 2
    dots = []
    for k in range(DOTS_PER_BOTTLE):
        t = -half + k * (2 * half) / (DOTS_PER_BOTTLE - 1)   # -half ... +half
        dots.append((x - t * math.sin(yaw), y + t * math.cos(yaw)))
    return dots


def too_close(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    return any(math.dist(p, q) < MIN_DOT_GAP for p in a for q in b)


def pick_layout(rng, count: int) -> list[tuple[float, float, float]]:
    """``count`` random (x, y, yaw) bottle poses in the spawn area that don't overlap.

    Draws a whole layout at once and simply redraws it if any two bottles are too close.
    """
    for _ in range(MAX_TRIES):
        layout = [
            (rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y), rng.uniform(0, 2 * math.pi))
            for _ in range(count)
        ]
        dots = [bottle_dots(x, y, yaw) for x, y, yaw in layout]
        overlap = any(too_close(dots[i], dots[j]) for i in range(count) for j in range(i + 1, count))
        if not overlap:
            return layout
    raise RuntimeError("bowling: couldn't fit the bottles apart; widen SPAWN_X/Y")


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng) -> None:
    z = TABLE_TOP_Z + BOTTLE_SIZE / 2 + 0.001   # lying on its side, 1 mm above the table
    for name, (x, y, yaw) in zip(BOTTLE_NAMES, pick_layout(rng, len(BOTTLE_NAMES))):
        adr = model.joint(f"{name}_free").qposadr[0]   # where this bottle's 7 numbers start in qpos
        data.qpos[adr:adr + 7] = [x, y, z, *lying_quat(yaw)]

# ---- scene
@scene("bowling", camera=dict(lookat=[0.38, 0.29, 0.75], distance=2.0, azimuth=200, elevation=-35),
       reset=reset)
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