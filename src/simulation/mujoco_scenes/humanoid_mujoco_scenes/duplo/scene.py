"""Duplo tower (long horizon): stack three Duplo bricks on the baseplate in the colour order announced.

Duplo-size 2x4 bricks (64 x 32 x 19.2 mm, studs 16 mm apart): big enough for this gripper and arm
(regular LEGO needs ~1 mm placement). Studs on top and walls + tubes underneath are real collision
geometry, so a brick only seats when it is square on the stud grid; otherwise it rests on the studs.

Snapping (``step``): MuJoCo can't do the slight interference fit that holds real bricks, so a brick that
sits seated and square on a support (the baseplate or another brick) is welded to it, pulled exactly onto
the grid. Pulling it off harder than SNAP_BREAK_FORCE (or twisting past SNAP_BREAK_TORQUE) releases it.
Bricks lie in a row beside the baseplate, long side along X so the jaws pinch their 32 mm width.

Steps, checked automatically and latched in order (``progress``):
  1. put the <c1> brick on the baseplate      c1 snapped onto the baseplate
  2. put the <c2> brick on the <c1> brick     c2 snapped onto c1
  3. put the <c3> brick on the <c2> brick     c3 snapped onto c2
Every reset shuffles the colour order and where the bricks lie.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from humanoid_mujoco_scenes import add_floor, scene

T = TABLE_TOP_Z = 0.705
TABLE_X = (0.15, 0.85)
TABLE_HALF_Y = 0.6

PITCH = 0.016                     # stud pitch
BRICK = (0.0638, 0.0318, 0.0192)  # 2x4 brick body (x, y, z)
STUD_R, STUD_H = 0.0047, 0.0045
STUD_TIP_R = 0.0027               # studs taper to this at the top: the lead-in that centres a brick
WALL = 0.0015
TUBE_R = 0.0065                   # tubes under the brick, between its studs
TOP = 0.002                       # brick top plate thickness
BRICK_MASS = 0.02
# ABS on ABS. With friction 1 a brick put down 2 mm off sticks on the stud tapers; at 0.35 it slides
# down onto the grid from up to 4 mm off. A grasp uses the jaws' higher friction (MuJoCo takes the max).
BRICK_FRICTION = [0.35, 0.005, 0.0001]
COLOURS = {"red": [0.85, 0.12, 0.1, 1], "blue": [0.1, 0.35, 0.85, 1], "yellow": [0.98, 0.8, 0.1, 1]}

PLATE_POS = (0.30, 0.395)         # baseplate centre; 6 x 4 studs, fixed
PLATE_STUDS = (6, 4)
PLATE_T = 0.004
BRICK_ROW_Y = 0.262               # loose bricks lie in a row here, long side along X
BRICK_ROW_X = (0.22, 0.30, 0.38)
JITTER = 0.004                    # m, and BRICK_YAW rad, random per episode
BRICK_YAW = 0.08

SNAP_POS_TOL = 0.0015             # seated within this of the grid (m)...
SNAP_YAW_TOL = math.radians(5)    # ...and square to it
SNAP_BREAK_FORCE = 6.0            # N pulling a snapped brick off
SNAP_BREAK_TORQUE = 0.25          # N m twisting it off
SNAP_SOLREF = [0.005, 1.0]

_U_STEP = 0                       # data.userdata: [latched step, colour order...]


# ----------------------------------------------------------------------------- hooks
def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Snap seated bricks, release over-pulled ones, advance the latched step."""
    _release_overloaded(model, data)
    _snap_seated(model, data)
    k = int(data.userdata[_U_STEP])
    if k < len(COLOURS) and _step_done(model, data, k):
        data.userdata[_U_STEP] = k + 1


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    order = rng.permutation(len(COLOURS))
    data.userdata[1:1 + len(COLOURS)] = order
    for (name, _), x in zip(COLOURS.items(), rng.permutation(BRICK_ROW_X)):
        adr = model.joint(f"brick_{name}").qposadr[0]
        dx, dy = rng.uniform(-JITTER, JITTER, 2)
        yaw = rng.uniform(-BRICK_YAW, BRICK_YAW)
        data.qpos[adr:adr + 7] = [x + dx, BRICK_ROW_Y + dy, T, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def progress(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, str]:
    k = int(data.userdata[_U_STEP])
    return k, len(COLOURS), _step_text(data, min(k, len(COLOURS) - 1))


# ----------------------------------------------------------------------------- scene
@scene("duplo", camera=dict(lookat=[0.30, 0.34, 0.75], distance=0.6, azimuth=200, elevation=-40),
       step=step, reset=reset, progress=progress)
def build(spec: mujoco.MjSpec) -> None:
    add_floor(spec)
    spec.nuserdata = 1 + len(COLOURS)
    world = spec.worldbody
    box, cyl, mesh = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_MESH
    # Stud: a frustum from its base (z=0) to its top, so a brick placed a little off slides onto the grid.
    ring = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    spec.add_mesh(name="stud", uservert=[v for c, s_ in ring for v in (STUD_R * c, STUD_R * s_, 0.0)]
                  + [v for c, s_ in ring for v in (STUD_TIP_R * c, STUD_TIP_R * s_, STUD_H)])
    x0, x1 = TABLE_X
    world.add_geom(name="table", type=box, size=[(x1 - x0) / 2, TABLE_HALF_Y, T / 2], pos=[(x0 + x1) / 2, 0, T / 2],
                   rgba=[0.55, 0.42, 0.3, 1])

    # Baseplate (fixed): a thin slab with a grid of studs; its frame is the slab's top centre.
    green = [0.2, 0.6, 0.25, 1]
    nx, ny = PLATE_STUDS
    plate = world.add_body(name="baseplate", pos=[*PLATE_POS, T + PLATE_T])
    plate.add_geom(type=box, size=[nx * PITCH / 2, ny * PITCH / 2, PLATE_T / 2], pos=[0, 0, -PLATE_T / 2], rgba=green)
    for i in range(nx):
        for j in range(ny):
            plate.add_geom(type=mesh, meshname="stud", rgba=green,
                           pos=[(i - (nx - 1) / 2) * PITCH, (j - (ny - 1) / 2) * PITCH, 0])

    # Bricks: frame at the bottom centre of the body (so a seated brick's frame is on its support's top).
    bx, by, bz = BRICK
    for name, rgba in COLOURS.items():
        b = world.add_body(name=f"brick_{name}", pos=[BRICK_ROW_X[0], BRICK_ROW_Y, T])
        b.add_freejoint(name=f"brick_{name}")
        b.add_geom(type=box, size=[bx / 2, by / 2, TOP / 2], pos=[0, 0, bz - TOP / 2], rgba=rgba, mass=BRICK_MASS * 0.4)
        h = (bz - TOP) / 2
        for sy in (-1, 1):
            b.add_geom(type=box, size=[bx / 2, WALL / 2, h], pos=[0, sy * (by - WALL) / 2, h], rgba=rgba, mass=BRICK_MASS * 0.1)
        for sx in (-1, 1):
            b.add_geom(type=box, size=[WALL / 2, by / 2 - WALL, h], pos=[sx * (bx - WALL) / 2, 0, h], rgba=rgba, mass=BRICK_MASS * 0.05)
        for x in (-PITCH, 0, PITCH):
            b.add_geom(type=cyl, size=[TUBE_R, h, 0], pos=[x, 0, h], rgba=rgba, mass=BRICK_MASS * 0.02)
        for i in range(4):
            for sy in (-1, 1):
                b.add_geom(type=mesh, meshname="stud", rgba=rgba, mass=BRICK_MASS * 0.0075,
                           pos=[(i - 1.5) * PITCH, sy * PITCH / 2, bz])

    for g in spec.geoms:
        if g.parent.name == "baseplate" or g.parent.name.startswith("brick_"):
            g.friction = BRICK_FRICTION

    # One (inactive) weld per brick per possible support; step() turns them on and off.
    for name in COLOURS:
        for support in ["baseplate"] + [f"brick_{o}" for o in COLOURS if o != name]:
            eq = spec.add_equality(name=f"snap_{name}_on_{support}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                   name1=support, name2=f"brick_{name}", active=False)
            eq.solref = SNAP_SOLREF


# ----------------------------------------------------------------------------- snapping
def _support_top(model, support):
    return 0.0 if support == "baseplate" else BRICK[2]


def _seat(model, data, brick: str, support: str):
    """If `brick` sits seated and square on `support`: (grid dx, dy in the support frame, yaw step); else None."""
    sb, bb = model.body(support).id, model.body(f"brick_{brick}").id
    Rs = data.xmat[sb].reshape(3, 3)
    rel = Rs.T @ (data.xpos[bb] - data.xpos[sb])
    if abs(rel[2] - _support_top(model, support)) > SNAP_POS_TOL:
        return None
    Rrel = Rs.T @ data.xmat[bb].reshape(3, 3)
    if Rrel[2, 2] < math.cos(SNAP_YAW_TOL):
        return None
    yaw = math.atan2(Rrel[1, 0], Rrel[0, 0])
    quarter = round(yaw / (math.pi / 2))
    if abs(yaw - quarter * math.pi / 2) > SNAP_YAW_TOL:
        return None
    # Studs (on bricks and the plate) and a brick's sockets all sit at half-pitch offsets from their centres,
    # in either orientation, so a seated brick is a whole number of pitches from its support's centre.
    gx, gy = round(rel[0] / PITCH) * PITCH, round(rel[1] / PITCH) * PITCH
    if math.hypot(rel[0] - gx, rel[1] - gy) > SNAP_POS_TOL:
        return None
    return gx, gy, quarter


def _snap_seated(model, data):
    for name in COLOURS:
        if _snapped_to(model, data, name) is not None:
            continue
        for support in ["baseplate"] + [f"brick_{o}" for o in COLOURS if o != name]:
            if support != "baseplate" and _snapped_to(model, data, support[len("brick_"):]) == f"brick_{name}":
                continue
            seat = _seat(model, data, name, support)
            if seat is None:
                continue
            gx, gy, quarter = seat
            eq = model.equality(f"snap_{name}_on_{support}").id
            a = quarter * math.pi / 2
            # relpose of the brick in the support frame: exactly on the grid, square
            model.eq_data[eq, :11] = [0, 0, 0, gx, gy, _support_top(model, support), math.cos(a / 2), 0, 0, math.sin(a / 2), 1]
            data.eq_active[eq] = 1
            break


def _release_overloaded(model, data):
    """Turn off a snap whose weld is pulled/twisted harder than the break limits (pulled apart)."""
    eq_ids = [model.equality(f"snap_{n}_on_{s}").id for n in COLOURS for s in ["baseplate"] + [f"brick_{o}" for o in COLOURS if o != n]]
    if data.nefc == 0:
        return
    rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    for eq in eq_ids:
        if not data.eq_active[eq]:
            continue
        r = np.where(rows & (data.efc_id == eq))[0]
        if len(r) < 6:
            continue
        f = data.efc_force[r]
        if np.linalg.norm(f[:3]) > SNAP_BREAK_FORCE or np.linalg.norm(f[3:6]) > SNAP_BREAK_TORQUE:
            data.eq_active[eq] = 0


def _snapped_to(model, data, brick: str):
    """Name of the support `brick` is snapped onto, or None."""
    for support in ["baseplate"] + [f"brick_{o}" for o in COLOURS if o != brick]:
        if data.eq_active[model.equality(f"snap_{brick}_on_{support}").id]:
            return support
    return None


# ----------------------------------------------------------------------------- task
def _order(data):
    names = list(COLOURS)
    idx = [int(i) for i in data.userdata[1:1 + len(names)]]
    return [names[i] for i in idx] if sorted(idx) == list(range(len(names))) else names


def _step_text(data, k):
    o = _order(data)
    return f"put the {o[0]} brick on the baseplate" if k == 0 else f"put the {o[k]} brick on the {o[k - 1]} brick"


def _step_done(model, data, k):
    o = _order(data)
    return _snapped_to(model, data, o[k]) == ("baseplate" if k == 0 else f"brick_{o[k - 1]}")


def snapped_to(model: mujoco.MjModel, data: mujoco.MjData, colour: str):
    return _snapped_to(model, data, colour)
