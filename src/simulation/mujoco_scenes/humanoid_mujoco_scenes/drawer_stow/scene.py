"""Drawer stow (long horizon): open the drawer, put three blocks in it one by one, close it.

A cabinet stands on the table in front of the LEFT arm; its drawer slides out toward the robot. A tab
on top of the drawer front is the handle (pinch it from above or from the front, pull). Three 4 cm
blocks (red, green, blue) stand in a strip beside the cabinet. Every reset shuffles the blocks'
places and the order they must go in.

Steps, each checked automatically and latched in order (``progress``):
  1. open the drawer                  drawer out at least OPEN_TRAVEL
  2-4. put the <colour> block in the drawer   that block resting inside the drawer tray
  5. close the drawer                 drawer in, all three blocks inside
"""
from __future__ import annotations

import mujoco
import numpy as np

from humanoid_mujoco_scenes import add_floor, scene

T = TABLE_TOP_Z = 0.705         # same table as peg_insert
TABLE_X = (0.15, 0.85)
TABLE_HALF_Y = 0.6

# Cabinet (fixed): front face at CAB_X, beyond the arm's reach except for its front.
CAB_X = 0.38
CAB_Y = (0.33, 0.47)            # toward the arm's outer side: the open gripper is ~16 cm wide
CAB_DEPTH = 0.18
CAB_HEIGHT = 0.10
WALL = 0.01

# Drawer: slides out toward the robot (-X) on a joint; its tray is open on top.
DRAWER_TRAVEL = 0.13            # fully out
OPEN_TRAVEL = 0.10              # "open" from here (the tray is then reachable)
CLOSED_TRAVEL = 0.01            # "closed" below this
DRAWER_FRICTION = 1.0           # N, slide friction (pulling it stays well inside the wrist's torque)
TRAY_FLOOR = 0.012              # tray floor bottom above the table
TRAY_WALL_H = 0.058             # tray walls above the floor
TRAY_INNER = (0.155, 0.108)     # inner length (X) x width (Y)
FRONT_T = 0.015                 # drawer front panel thickness
TAB = (0.06, 0.016, 0.03)       # handle tab: protrudes toward the robot (jaws are 6 cm long), thick in Y

BLOCK = 0.04                    # cube edge
BLOCK_MASS = 0.05
COLOURS = {"red": [0.85, 0.2, 0.2, 1], "green": [0.2, 0.7, 0.3, 1], "blue": [0.2, 0.4, 0.85, 1]}
BLOCK_AREA = ((0.21, 0.37), (0.222, 0.250))   # x, y ranges the blocks are shuffled in (beside the drawer)
# Centre-to-centre: the jaws are 61 mm long, so a closer neighbour gets clipped grasping its partner.
BLOCK_SPACING = 0.07
BLOCK_YAW = 0.2                 # rad, random turn of each block

STEPS_TOTAL = 2 + len(COLOURS)


def _box(body, name, lo, hi, rgba, **kw):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list((hi - lo) / 2),
                         pos=list((hi + lo) / 2), rgba=rgba, **kw)


def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Advance the latched step counter (data.userdata[0]) when the current step's check passes."""
    k = int(data.userdata[0])
    if k < STEPS_TOTAL and _step_done(model, data, k):
        data.userdata[0] = k + 1


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """New episode: shuffle where the blocks stand and the order they go in. Call after mj_resetData."""
    data.userdata[1:1 + len(COLOURS)] = rng.permutation(len(COLOURS))
    (x0, x1), (y0, y1) = BLOCK_AREA
    while True:
        xy = np.column_stack([rng.uniform(x0, x1, len(COLOURS)), rng.uniform(y0, y1, len(COLOURS))])
        if min(np.linalg.norm(a - b) for i, a in enumerate(xy) for b in xy[i + 1:]) >= BLOCK_SPACING:
            break
    for (name, _), (x, y) in zip(COLOURS.items(), xy):
        adr = model.joint(f"block_{name}").qposadr[0]
        yaw = rng.uniform(-BLOCK_YAW, BLOCK_YAW)
        data.qpos[adr:adr + 7] = [x, y, T + BLOCK / 2, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def progress(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, str]:
    """(current step index, number of steps, its instruction); index == total when finished, with the
    last step's instruction (frames after it, e.g. backing off, still belong to it)."""
    k = int(data.userdata[0])
    return k, STEPS_TOTAL, _step_text(data, min(k, STEPS_TOTAL - 1))


@scene("drawer_stow", camera=dict(lookat=[0.33, 0.33, 0.76], distance=0.9, azimuth=210, elevation=-35),
       step=step, reset=reset, progress=progress)
def build(spec: mujoco.MjSpec) -> None:
    add_floor(spec)
    spec.nuserdata = 1 + len(COLOURS)   # [latched step, block order...]
    world = spec.worldbody
    x0, x1 = TABLE_X
    _box(world, "table", [x0, -TABLE_HALF_Y, 0], [x1, TABLE_HALF_Y, T], [0.55, 0.42, 0.3, 1])

    # Cabinet: bottom, top, sides, back (world geoms).
    wood = [0.82, 0.78, 0.7, 1]
    cx0, cx1 = CAB_X, CAB_X + CAB_DEPTH
    (cy0, cy1), z1 = CAB_Y, T + CAB_HEIGHT
    _box(world, "cab_bottom", [cx0, cy0, T], [cx1, cy1, T + WALL], wood)
    _box(world, "cab_top", [cx0, cy0, z1 - WALL], [cx1, cy1, z1], wood)
    _box(world, "cab_side_l", [cx0, cy1 - WALL, T + WALL], [cx1, cy1, z1 - WALL], wood)
    _box(world, "cab_side_r", [cx0, cy0, T + WALL], [cx1, cy0 + WALL, z1 - WALL], wood)
    _box(world, "cab_back", [cx1 - WALL, cy0 + WALL, T + WALL], [cx1, cy1 - WALL, z1 - WALL], wood)

    # Drawer, in its own frame: origin at the closed front face, centred in Y, on the table.
    cy = (cy0 + cy1) / 2
    drawer = world.add_body(name="drawer", pos=[cx0, cy, T])
    drawer.add_joint(name="drawer", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0], range=[0, DRAWER_TRAVEL],
                     limited=mujoco.mjtLimited.mjLIMITED_TRUE, frictionloss=DRAWER_FRICTION, damping=5.0, armature=0.05)
    tray = [0.9, 0.86, 0.78, 1]
    L, W = TRAY_INNER
    w = 0.005
    zf0, zf1 = TRAY_FLOOR, TRAY_FLOOR + w
    zw = zf1 + TRAY_WALL_H
    _box(drawer, "tray_floor", [0, -W / 2 - w, zf0], [L + w, W / 2 + w, zf1], tray, mass=0.15)
    _box(drawer, "tray_side_l", [0, W / 2, zf1], [L + w, W / 2 + w, zw], tray, mass=0.03)
    _box(drawer, "tray_side_r", [0, -W / 2 - w, zf1], [L + w, -W / 2, zw], tray, mass=0.03)
    _box(drawer, "tray_back", [L, -W / 2, zf1], [L + w, W / 2, zw], tray, mass=0.03)
    hy, hz = (cy1 - cy0) / 2 - 0.002, CAB_HEIGHT - 0.006
    _box(drawer, "drawer_front", [-FRONT_T, -hy, 0.004], [0, hy, hz], wood, mass=0.08)
    tx, ty, tz = TAB
    _box(drawer, "drawer_tab", [-FRONT_T - tx, -ty / 2, hz - tz], [-FRONT_T, ty / 2, hz], [0.3, 0.3, 0.32, 1], mass=0.02)
    spec.add_exclude(bodyname1="world", bodyname2="drawer")   # runs on its joint, not on the cabinet

    # Blocks, in a default row (reset() shuffles them).
    (bx0, bx1), (by0, by1) = BLOCK_AREA
    for i, (name, rgba) in enumerate(COLOURS.items()):
        b = world.add_body(name=f"block_{name}", pos=[bx0 + i * (bx1 - bx0) / 2, (by0 + by1) / 2, T + BLOCK / 2])
        b.add_freejoint(name=f"block_{name}")
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BLOCK / 2] * 3, mass=BLOCK_MASS, rgba=rgba,
                   friction=[1.0, 0.02, 0.001], condim=4)


def drawer_travel(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return float(data.qpos[model.joint("drawer").qposadr[0]])


def in_drawer(model: mujoco.MjModel, data: mujoco.MjData, colour: str) -> bool:
    """Block resting inside the tray: centre within the tray's inner box and nearly still."""
    body = model.body(f"block_{colour}").id
    d = model.body("drawer").id
    rel = data.xmat[d].reshape(3, 3).T @ (data.xpos[body] - data.xpos[d])
    L, W = TRAY_INNER
    floor = TRAY_FLOOR + 0.005
    inside = 0.0 < rel[0] < L and abs(rel[1]) < W / 2 and floor < rel[2] < floor + TRAY_WALL_H
    v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, v, 0)
    drawer_v = data.qvel[model.joint("drawer").dofadr[0]]
    return bool(inside and np.linalg.norm(v[3:]) < 0.05 + abs(drawer_v))


def _order(data: mujoco.MjData) -> list[str]:
    names = list(COLOURS)
    idx = [int(i) for i in data.userdata[1:1 + len(names)]]
    return [names[i] for i in idx] if sorted(idx) == list(range(len(names))) else names


def _step_text(data: mujoco.MjData, k: int) -> str:
    if k == 0:
        return "open the drawer"
    if k <= len(COLOURS):
        return f"put the {_order(data)[k - 1]} block in the drawer"
    return "close the drawer"


def _step_done(model: mujoco.MjModel, data: mujoco.MjData, k: int) -> bool:
    if k == 0:
        return drawer_travel(model, data) >= OPEN_TRAVEL
    if k <= len(COLOURS):
        return in_drawer(model, data, _order(data)[k - 1])
    return drawer_travel(model, data) <= CLOSED_TRAVEL and all(in_drawer(model, data, c) for c in COLOURS)
