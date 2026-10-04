"""Matcha (long horizon): scoop matcha into the cup, pour in the water, whisk it, serve it on the tray.

Everything in front of the LEFT arm, laid out for the gripper pointing down: every tool has a square
handle sticking up (square, so a held tool doesn't swing about the jaws when tipped). Powder and water
are small balls: green in the ladle (a pre-measured scoop), blue in the pitcher.

  ladle   holds a scoop of matcha, standing in a ring, handle post on the side away from the cup
  pitcher free, full of water, handle post on the side away from the cup; tip it (wrist pitch) to pour
  whisk   stands in its holder
  cup     free, in the middle; tray beside it

Steps, each checked automatically and latched in order (``progress``):
  1. spoon the matcha into the cup    at least POWDER_NEEDED green balls in the cup
  2. pour the water into the cup      at least WATER_NEEDED blue balls in the cup
  3. whisk the matcha                 whisk tines moved WHISK_PATH inside the cup (any pattern)
  4. serve the cup on the tray        cup upright on the tray, still holding most of the drink
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from humanoid_mujoco_scenes import add_floor, scene

T = TABLE_TOP_Z = 0.705
TABLE_X = (0.15, 0.85)
TABLE_HALF_Y = 0.6

CUP_POS = (0.30, 0.34)
CUP = dict(r=0.030, h=0.065, wall=0.003, n=14, mass=0.2)    # ceramic: heavy enough not to topple at a brush
LADLE_POS = (0.26, 0.43)                       # ladle (already holding the matcha) stands in its holder
# Scooping a handful of 5 mm balls out of a jar isn't reliable with one arm in this space (tipped steeply
# enough to scoop, the ladle's hand end hits its neighbours), so the ladle starts filled: carry it level, tip it.
PITCHER_POS = (0.385, 0.43)
PITCHER = dict(r=0.030, h=0.06, wall=0.003, n=14, mass=0.05)
PITCHER_HANDLE = 0.07                          # post above the rim (jaws, 44 mm half-length, clear the rim)
WHISK_POS = (0.24, 0.265)
TRAY = ((0.355, 0.455), (0.215, 0.29))         # x, y extent on the table
RANDOM_SHIFT = 0.012                           # cup and pitcher placed within +-this per episode

# 40 balls of 5 mm: physics ~0.3 s per simulated second (84 balls of 4 mm took 3 s: they touch each other).
BALL_R = 0.005
POWDER = dict(count=8, mass=0.0005, friction=0.6, rgba=[0.45, 0.7, 0.25, 1])
WATER = dict(count=24, mass=0.0008, friction=0.15, rgba=[0.45, 0.7, 0.95, 1])
POWDER_NEEDED = 5
WATER_NEEDED = 12
WHISK_PATH = 0.20                              # m the tines travel inside the cup (circles or a zig-zag)
SERVE_KEEP = 0.6                               # of the drink's balls still in the cup when served

# Square tool handles, chunky: the sim gripper squeezes by position error, ~3 N/finger on a 12 mm handle
# (a held tool slides along the 61 mm jaws at the first knock), ~6 N on 25 mm. Grippy, like rubber tape.
HANDLE = 0.025
HANDLE_FRICTION = [1.5, 0.02, 0.001]
LADLE_R = 0.020
LADLE_H = 0.022                                # two layers of matcha
LADLE_TOP = 0.13                               # handle top above the bowl bottom
STEPS = ("spoon the matcha into the cup", "pour the water into the cup", "whisk the matcha", "serve the cup on the tray")
# data.userdata: [latched step, whisking path so far (m), last tine position in the cup (x, y), position valid]
_U_STEP, _U_PATH, _U_X, _U_Y, _U_VALID = range(5)


# ----------------------------------------------------------------------------- hooks
def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Track whisking and advance the latched step when the current one's check passes."""
    _track_whisk(model, data)
    k = int(data.userdata[_U_STEP])
    if k < len(STEPS) and _step_done(model, data, k):
        data.userdata[_U_STEP] = k + 1


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """New episode: shift the cup and the pitcher (with its water) a little."""
    for body, extra in (("cup", ()), ("pitcher", [f"water{i}" for i in range(WATER["count"])])):
        dx, dy = rng.uniform(-RANDOM_SHIFT, RANDOM_SHIFT, 2)
        for name in (body, *extra):
            adr = model.joint(name).qposadr[0]
            data.qpos[adr] += dx
            data.qpos[adr + 1] += dy


def progress(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, str]:
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


# ----------------------------------------------------------------------------- scene
@scene("matcha", camera=dict(lookat=[0.31, 0.35, 0.76], distance=0.7, azimuth=200, elevation=-40),
       step=step, reset=reset, progress=progress)
def build(spec: mujoco.MjSpec) -> None:
    add_floor(spec)
    spec.nuserdata = 5
    world = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX
    x0, x1 = TABLE_X
    world.add_geom(name="table", type=box, size=[(x1 - x0) / 2, TABLE_HALF_Y, T / 2], pos=[(x0 + x1) / 2, 0, T / 2],
                   rgba=[0.55, 0.42, 0.3, 1])
    (tx0, tx1), (ty0, ty1) = TRAY
    world.add_geom(name="tray", type=box, size=[(tx1 - tx0) / 2, (ty1 - ty0) / 2, 0.003],
                   pos=[(tx0 + tx1) / 2, (ty0 + ty1) / 2, T + 0.003], rgba=[0.15, 0.15, 0.17, 1])

    ceramic, wood = [0.95, 0.94, 0.9, 1], [0.6, 0.45, 0.3, 1]
    # Ladle, holding the matcha, standing in a low ring. Its handle is a post on the bowl's +X rim (like
    # the pitcher): held at the top and tipped toward the cup, the bowl empties over its far lip.
    ladle_rest = world.add_body(name="ladle_rest", pos=[*LADLE_POS, T])
    _container(ladle_rest, "ladle_rest", r=LADLE_R + 0.006, h=0.012, wall=0.003, n=14, rgba=[0.3, 0.32, 0.35, 1])
    ladle = world.add_body(name="ladle", pos=[LADLE_POS[0], LADLE_POS[1], T + 0.004])
    ladle.add_freejoint(name="ladle")
    _container(ladle, "ladle_bowl", r=LADLE_R, h=LADLE_H, wall=0.002, n=12, rgba=wood, mass=0.03)
    rim = LADLE_H + 0.002
    post_x = LADLE_R + 0.002 + HANDLE / 2
    ladle.add_geom(name="ladle_bridge", type=box, size=[HANDLE / 2 + 0.002, HANDLE / 2, 0.003],
                   pos=[post_x - 0.002, 0, rim - 0.003], rgba=wood, mass=0.002)
    ladle.add_geom(name="ladle_handle", type=box, size=[HANDLE / 2, HANDLE / 2, (LADLE_TOP - rim) / 2 + 0.003],
                   pos=[post_x, 0, (LADLE_TOP + rim) / 2 - 0.003], rgba=wood, mass=0.015, friction=HANDLE_FRICTION, condim=4)
    _balls(world, "powder", LADLE_POS, LADLE_R, T + 0.004 + 0.002, **POWDER)

    # Pitcher of water. Its handle is a tall square post on the +X rim (the side away from the cup): held
    # by the post top and tipped toward the cup, it pours from the far lip, clear of the fingers.
    pitcher = world.add_body(name="pitcher", pos=[*PITCHER_POS, T])
    pitcher.add_freejoint(name="pitcher")
    _container(pitcher, "pitcher", **PITCHER, rgba=[0.75, 0.85, 0.95, 0.5])
    rim = PITCHER["wall"] + PITCHER["h"]
    post_x = PITCHER["r"] + PITCHER["wall"] + HANDLE / 2
    pitcher.add_geom(name="pitcher_bridge", type=box, size=[HANDLE / 2 + 0.002, HANDLE / 2, 0.004],
                     pos=[post_x - 0.002, 0, rim - 0.004], rgba=[0.3, 0.3, 0.32, 1], mass=0.004)
    pitcher.add_geom(name="pitcher_handle", type=box, size=[HANDLE / 2, HANDLE / 2, PITCHER_HANDLE / 2],
                     pos=[post_x, 0, rim + PITCHER_HANDLE / 2 - 0.008], rgba=[0.3, 0.3, 0.32, 1], mass=0.012,
                     friction=HANDLE_FRICTION, condim=4)
    _balls(world, "water", PITCHER_POS, PITCHER["r"], T + PITCHER["wall"], **WATER)

    # Whisk standing in its holder (fixed).
    holder = world.add_body(name="whisk_holder", pos=[*WHISK_POS, T])
    # Deep enough to keep it upright, wider than the square handle's corners (17.7 mm) so it can't wedge.
    _container(holder, "holder", r=0.022, h=0.05, wall=0.003, n=14, rgba=[0.3, 0.32, 0.35, 1])
    whisk = world.add_body(name="whisk", pos=[WHISK_POS[0], WHISK_POS[1], T + 0.003 + 0.05])
    whisk.add_freejoint(name="whisk")
    _handle(whisk, "whisk_handle", 0.0, 0.10, wood, mass=0.02)
    for i in range(6):   # tines: a cage of bowed wires below the handle, meeting at its foot
        a = i * math.pi / 3
        mid = [0.013 * math.cos(a), 0.013 * math.sin(a), -0.025]
        for p0, p1 in (([0, 0, 0.0], mid), (mid, [0, 0, -0.048])):
            whisk.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*p0, *p1], size=[0.0012, 0, 0],
                           rgba=[0.75, 0.68, 0.5, 1], mass=0.0005)
    whisk.add_site(name="whisk_tip", pos=[0, 0, -0.03])

    # Cup in the middle.
    cup = world.add_body(name="cup", pos=[*CUP_POS, T])
    cup.add_freejoint(name="cup")
    _container(cup, "cup", **CUP, rgba=ceramic)


def _container(body, name, r, h, wall, n, rgba, mass=None):
    """Open-top cylinder: a disc floor and n wall boxes in a ring (inner radius r)."""
    kw = {} if mass is None else {"mass": mass / (n + 1)}
    body.add_geom(name=f"{name}_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r + wall, wall / 2, 0],
                  pos=[0, 0, wall / 2], rgba=rgba, **kw)
    half_t = (r + wall) * math.tan(math.pi / n) + 0.0005
    for i in range(n):
        a = 2 * math.pi * i / n
        body.add_geom(name=f"{name}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[wall / 2, half_t, h / 2],
                      pos=[(r + wall / 2) * math.cos(a), (r + wall / 2) * math.sin(a), wall + h / 2],
                      quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)], rgba=rgba, **kw)


def _handle(body, name, z0, length, rgba, mass):
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[HANDLE / 2, HANDLE / 2, length / 2],
                  pos=[0, 0, z0 + length / 2], rgba=rgba, mass=mass, friction=HANDLE_FRICTION, condim=4)


def _balls(world, prefix, xy, r_in, z_floor, count, mass, friction, rgba):
    """`count` free balls stacked in a hexagonal-ish pile inside a container of inner radius r_in."""
    pts, z, layer = [], z_floor + BALL_R + 0.0005, 0
    while len(pts) < count:
        ring_r, k = [0.0], 0
        rr = 2.2 * BALL_R
        while rr < r_in - BALL_R - 0.001:
            ring_r.append(rr)
            rr += 2.2 * BALL_R
        for rr in ring_r:
            m = 1 if rr == 0 else int(2 * math.pi * rr / (2.2 * BALL_R))
            for j in range(m):
                a = 2 * math.pi * j / m + layer * 0.4
                pts.append((xy[0] + rr * math.cos(a), xy[1] + rr * math.sin(a), z))
                k += 1
        z += 2.1 * BALL_R
        layer += 1
    for i, p in enumerate(pts[:count]):
        b = world.add_body(name=f"{prefix}{i}", pos=list(p))
        b.add_freejoint(name=f"{prefix}{i}")
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0], mass=mass, rgba=rgba,
                   friction=[friction, 0.005, 0.0001])


# ----------------------------------------------------------------------------- checks
def balls_in_cup(model: mujoco.MjModel, data: mujoco.MjData, prefix: str) -> int:
    count = POWDER["count"] if prefix == "powder" else WATER["count"]
    ids = [model.body(f"{prefix}{i}").id for i in range(count)]
    cup = model.body("cup").id
    rel = (data.xpos[ids] - data.xpos[cup]) @ data.xmat[cup].reshape(3, 3)
    inside = (np.hypot(rel[:, 0], rel[:, 1]) < CUP["r"]) & (rel[:, 2] > 0) & (rel[:, 2] < CUP["h"] + CUP["wall"])
    return int(inside.sum())


def balls_in_ladle(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    ids = [model.body(f"powder{i}").id for i in range(POWDER["count"])]
    ladle = model.body("ladle").id
    rel = (data.xpos[ids] - data.xpos[ladle]) @ data.xmat[ladle].reshape(3, 3)
    return int(((np.hypot(rel[:, 0], rel[:, 1]) < LADLE_R) & (rel[:, 2] > 0) & (rel[:, 2] < LADLE_H + 0.002)).sum())


def cup_upright(model: mujoco.MjModel, data: mujoco.MjData, max_tilt_deg: float = 20.0) -> bool:
    return data.xmat[model.body("cup").id].reshape(3, 3)[2, 2] > math.cos(math.radians(max_tilt_deg))


def _whisk_tip_in_cup(model, data):
    cup = model.body("cup").id
    rel = (data.site_xpos[model.site("whisk_tip").id] - data.xpos[cup]) @ data.xmat[cup].reshape(3, 3)
    return rel, math.hypot(rel[0], rel[1]) < CUP["r"] and 0 < rel[2] < CUP["h"]


def _track_whisk(model, data):
    """Accumulate how far the whisk's tines move sideways in the cup (cup frame) while inside it."""
    rel, inside = _whisk_tip_in_cup(model, data)
    if not inside:
        data.userdata[_U_VALID] = 0
        return
    if data.userdata[_U_VALID]:
        data.userdata[_U_PATH] += math.hypot(rel[0] - data.userdata[_U_X], rel[1] - data.userdata[_U_Y])
    data.userdata[_U_X], data.userdata[_U_Y], data.userdata[_U_VALID] = rel[0], rel[1], 1


def whisk_path(data: mujoco.MjData) -> float:
    return float(data.userdata[_U_PATH])


def _step_done(model, data, k):
    if k == 0:
        return balls_in_cup(model, data, "powder") >= POWDER_NEEDED
    if k == 1:
        return balls_in_cup(model, data, "water") >= WATER_NEEDED
    if k == 2:
        return whisk_path(data) >= WHISK_PATH
    cup = data.xpos[model.body("cup").id]
    (tx0, tx1), (ty0, ty1) = TRAY
    on_tray = tx0 < cup[0] < tx1 and ty0 < cup[1] < ty1 and cup[2] < T + 0.02
    kept = balls_in_cup(model, data, "powder") + balls_in_cup(model, data, "water")
    return on_tray and cup_upright(model, data) and kept >= SERVE_KEEP * (POWDER_NEEDED + WATER_NEEDED)
