"""Colour sort: put the red, blue and yellow pins on their matching pads, in that order, right arm.

Three pins (5 x 5 x 13 cm boxes, 150 g) lie on their sides. Every reset scatters them to random,
non-overlapping spots and directions inside SPAWN_X / SPAWN_Y, a strip near the robot. Three flat
pads of the same colours sit in a row behind them (PADS), inside the right arm's gripper-down reach.

Long horizon, 3 steps in order: red, then blue, then yellow. Step k is done once pin k rests on its
pad, latched in data.userdata[0], so moving a pin off its pad after its step doesn't undo the step.

Checks: ``on_pad(model, data, color)`` (centre over its pad, resting on the table lying or standing,
nearly still, not touching the arm) and ``all_sorted(model, data)``.
"""
from __future__ import annotations

import math
import mujoco
import numpy as np

from humanoid_mujoco_scenes import TABLE_TOP_Z, add_workcell, scene
from pioneer_humanoid.mujoco_bimanual_arm import ARM_CONTYPE

COLORS = {                      # sorting order
    "red": [0.85, 0.2, 0.15, 1],
    "blue": [0.15, 0.35, 0.85, 1],
    "yellow": [0.95, 0.8, 0.1, 1],
}
PIN_SIZE = 0.05                 # square cross-section (m)
PIN_LENGTH = 0.13
PIN_MASS = 0.15                 # in kg

# Pads: flat squares on the table (visual only, so pins rest on the table itself), in a row at the
# back of the right arm's reach (gripper down: x 0.25-0.55, y -0.60 to -0.10).
PAD_HALF = 0.05
PADS = {"red": (0.48, -0.20), "blue": (0.48, -0.32), "yellow": (0.48, -0.44)}

# Where pin centres may land (m, robot frame): in front of the pads, so a pin never spawns on one.
# A pin reaches PIN_LENGTH / 2 past its centre, which still clears the pads' near edge (x 0.43).
SPAWN_X = (0.26, 0.36)
SPAWN_Y = (-0.48, -0.18)

# Overlap check, as in bowling: dots along each pin, too close if any are nearer than MIN_DOT_GAP.
DOTS_PER_PIN = 7
MIN_DOT_GAP = PIN_SIZE + 0.02
MAX_TRIES = 20000

# Settled on the pad: centre no higher than standing on the table (plus slack), and nearly still.
REST_Z_SLACK = 0.01
STILL_SPEED = 0.05

# ---- hooks


def lying_quat(yaw: float) -> list[float]:
    tip = math.pi / 2
    return [
        math.cos(yaw / 2) * math.cos(tip / 2),
        math.cos(yaw / 2) * math.sin(tip / 2),
        math.sin(yaw / 2) * math.sin(tip / 2),
        math.sin(yaw / 2) * math.cos(tip / 2),
    ]


def pin_dots(x: float, y: float, yaw: float) -> list[tuple[float, float]]:
    half = PIN_LENGTH / 2
    dots = []
    for k in range(DOTS_PER_PIN):
        t = -half + k * (2 * half) / (DOTS_PER_PIN - 1)   # -half ... +half
        dots.append((x - t * math.sin(yaw), y + t * math.cos(yaw)))
    return dots


def too_close(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    return any(math.dist(p, q) < MIN_DOT_GAP for p in a for q in b)


def pick_layout(rng, count: int) -> list[tuple[float, float, float]]:
    for _ in range(MAX_TRIES):
        layout = [
            (rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y), rng.uniform(0, 2 * math.pi))
            for _ in range(count)
        ]
        dots = [pin_dots(x, y, yaw) for x, y, yaw in layout]
        overlap = any(too_close(dots[i], dots[j]) for i in range(count) for j in range(i + 1, count))
        if not overlap:
            return layout
    raise RuntimeError("color_sort: couldn't fit the pins apart; widen SPAWN_X/Y")


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng) -> None:
    z = TABLE_TOP_Z + PIN_SIZE / 2 + 0.001   # lying on its side, 1 mm above the table
    for color, (x, y, yaw) in zip(COLORS, pick_layout(rng, len(COLORS))):
        adr = model.joint(f"{color}_pin_free").qposadr[0]   # where this pin's 7 numbers start in qpos
        data.qpos[adr:adr + 7] = [x, y, z, *lying_quat(yaw)]


def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    k = int(data.userdata[0])
    if k < len(COLORS) and on_pad(model, data, list(COLORS)[k]):
        data.userdata[0] = k + 1


def progress(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, str]:
    k = int(data.userdata[0])
    color = list(COLORS)[min(k, len(COLORS) - 1)]
    return k, len(COLORS), f"put the {color} pin on the {color} pad"

# ---- scene
@scene("color_sort", camera=dict(lookat=[0.40, -0.32, TABLE_TOP_Z], distance=0.8, azimuth=160, elevation=-55),
       step=step, reset=reset, progress=progress)
def build(spec: mujoco.MjSpec) -> None:
    add_workcell(spec)
    spec.nuserdata = 1                  # [latched step]
    world = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX

    for color, rgba in COLORS.items():
        px, py = PADS[color]
        world.add_geom(
            name=f"{color}_pad", type=box, size=[PAD_HALF, PAD_HALF, 0.0005],
            pos=[px, py, TABLE_TOP_Z + 0.0005], rgba=rgba, contype=0, conaffinity=0,
        )

    # Build-time spots only; reset() scatters them before every episode.
    for i, (color, rgba) in enumerate(COLORS.items()):
        pin = world.add_body(name=f"{color}_pin", pos=[0.31, -0.22 - 0.12 * i, TABLE_TOP_Z + PIN_SIZE / 2 + 0.001],
                             quat=lying_quat(math.pi / 2))
        pin.add_freejoint(name=f"{color}_pin_free")
        pin.add_geom(
            name=f"{color}_pin", type=box, size=[PIN_SIZE / 2, PIN_SIZE / 2, PIN_LENGTH / 2],
            mass=PIN_MASS, friction=[1.0, 0.02, 0.001], condim=4, rgba=rgba,
        )


# ---- check
def on_pad(model: mujoco.MjModel, data: mujoco.MjData, color: str) -> bool:
    b = model.body(f"{color}_pin").id
    x, y, z = data.xpos[b]
    px, py = PADS[color]
    over = abs(x - px) <= PAD_HALF and abs(y - py) <= PAD_HALF
    resting = z < TABLE_TOP_Z + PIN_LENGTH / 2 + REST_Z_SLACK      # lying or standing, not held up
    v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
    geom = model.geom(f"{color}_pin").id
    released = not any(
        geom in (c.geom1, c.geom2)
        and model.geom_contype[c.geom2 if c.geom1 == geom else c.geom1] & ARM_CONTYPE
        for c in data.contact[:data.ncon]
    )
    return bool(over and resting and np.linalg.norm(v) < STILL_SPEED and released)


def all_sorted(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return all(on_pad(model, data, color) for color in COLORS)
