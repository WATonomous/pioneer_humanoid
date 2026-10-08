"""Tidy table, level 1: put every object on the table into a bin (any of the three), in the order announced,
without dropping or knocking over anything on the way.

3-4 objects per episode, drawn from three categories (box, cylinder, ball), at most two of a category.
Every reset re-draws which objects are out, their sizes, masses, friction, colours, where they lie and
the order they go in; each object in an episode has its own colour, so "the red ball" is unambiguous.
The table is messy: any yaw, some cylinders lying on their side, scattered over the reachable area (never
touching, so each one can be picked without pushing the others first).
The bins are fixed: two on the arm's outer side, one at the far end of the table. They are named after the
shapes (BINS; episode_objects gives each object "its" bin, where the scripted demos put it), but any bin counts:
nothing in the scene shows which is which.

Laid out for the LEFT arm with the gripper pointing down. Its reach that way is small: objects lie in a
~18 x 16 cm zone, the bins just outside it. It has no wrist roll, so pointing down the gripper only turns so
far about vertical, and less near the zone's edges (grasp_yaw_range): boxes and lying cylinders are turned so
one of a box's sides, or a cylinder's axis, lines up with a yaw the gripper reaches where it lies. Objects
are at least a finger width apart (each finger is 33 mm thick).

The overhead RGB camera ``top`` looks down on the zone and the bins from above the table's far outer corner:
from straight above or straight ahead, the gripper at home hides the objects under it.

Steps, checked automatically and latched in order (``progress``):
  k. put the <colour> <shape> in a bin     that object resting in any bin, released
Failures, checked every step and kept (``episode_status``):
  dropped      an object resting on the table (or knocked off it) after being picked up
  toppled      a box or cylinder resting on the table tipped (> TOPPLE_ANGLE) from how it was set out
  order        an object put in a bin before its turn
Releasing an object over a bin is fine; a cylinder that falls over inside a bin is home.

MuJoCo can't change a geom's type at runtime, so the scene holds a pool of objects (MAX_PER_SHAPE per
shape); reset() resizes the ones it uses and hides the rest (invisible, no contacts, floating off-stage).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from humanoid_mujoco_scenes import add_floor, scene

T = TABLE_TOP_Z = 0.705          # same table as peg_insert / drawer_stow / duplo
TABLE_X = (0.15, 0.85)
TABLE_HALF_Y = 0.6

SHAPES = ("box", "cylinder", "ball")
MAX_PER_SHAPE = 2
N_OBJECTS = (3, 4)               # objects per episode, inclusive
COLOURS = {
    "red": [0.85, 0.15, 0.12, 1], "orange": [0.95, 0.5, 0.1, 1], "yellow": [0.95, 0.85, 0.15, 1],
    "green": [0.2, 0.7, 0.25, 1], "blue": [0.15, 0.35, 0.9, 1], "purple": [0.55, 0.25, 0.75, 1],
    "pink": [0.95, 0.5, 0.7, 1], "white": [0.95, 0.95, 0.95, 1],
}

# Sizes (m). Narrowest horizontal side 25-50 mm: the open jaws are ~96 mm apart (JAW_OPEN, measured between the
# finger pads). No side of any object is longer than MAX_SIDE, so it fits the jaws whichever way it ends up.
JAW_OPEN = 0.0958
MAX_SIDE = 0.8 * JAW_OPEN        # 76.6 mm
BOX_LEN = (0.030, 0.055)         # along X (the fingers are 61 mm long)
BOX_WIDTH = (0.025, 0.045)       # along Y, between the jaws
BOX_HEIGHT = (0.025, 0.050)
CYL_RADIUS = (0.0125, 0.022)
CYL_HEIGHT = (0.035, 0.075)      # the tall thin ones tip over if nudged
CYL_LYING = 0.35                 # chance a cylinder starts on its side (it can roll when bumped)
BALL_RADIUS = (0.015, 0.025)
DENSITY = (300.0, 900.0)         # kg/m^3, wood to dense plastic
MASS = (0.01, 0.15)              # kg, clamp
FRICTION = (0.5, 1.0)
ROLL_FRICTION = 0.0005           # balls and lying cylinders stop rolling within a few cm, not across the table

ZONE = ((0.165, 0.345), (0.21, 0.37))  # x, y ranges object centres are drawn in
BIN_CLEAR = 0.01                 # the open jaws around every object stay this far from the bins' outer walls
FINGER_T = 0.033                 # finger thickness; the jaws open to the object's width + JAW_SPARE around it
JAW_SPARE = 0.03                 # (closer to a bin than that, the fingers land on its wall, not the table)
FINGER_GAP = 0.04                # clear space between neighbours: a 33 mm finger plus margin
SPAWN_LIFT = 0.001               # spawned this far above resting: zero clearance gets a PhysX-style kick

# Bins: inner size, wall height above the floor plate, wall / floor thickness. Fixed per shape.
BIN_INNER = 0.10
BIN_WALL_H = 0.03
BIN_T = 0.005
# Inner centre x, y: as far out as the arm still reaches to release over them, leaving room around the objects.
BINS = {"box": (0.19, 0.50), "cylinder": (0.30, 0.50), "ball": (0.44, 0.29)}

# The overhead camera: position, the point it looks at, vertical field of view, image size.
TOP_CAMERA = dict(pos=(0.70, 0.75, 1.40), lookat=(0.26, 0.32, T), fovy=26.0, resolution=(640, 480))

PARK = (-2.0, 2.0, 1.0)          # hidden objects float here and beyond
FINGER_BODIES = ("link7l", "link8l", "link7", "link8")
TOPPLE_ANGLE = math.radians(35)
REST_LIN, REST_ANG = 0.03, 0.6   # m/s, rad/s: below both counts as resting
LIFTED_DZ = 0.01                 # held this far above where it stood = picked up

# data.userdata layout
_U_STEP, _U_N = 0, 1
_U_ORDER = 2                                   # [n] pool index per step
_POOL = [f"{s}_{i}" for s in SHAPES for i in range(MAX_PER_SHAPE)]
_U_COLOUR = _U_ORDER + len(_POOL)              # [pool] colour index, -1 = not out this episode
_U_FLAGS = _U_COLOUR + len(_POOL)              # [pool] bit flags below
_U_STAND = _U_FLAGS + len(_POOL)               # [pool] centre height it stood at
_U_UP = _U_STAND + len(_POOL)                  # [pool x 3] its local z axis as set out (world frame)
_N_USER = _U_UP + 3 * len(_POOL)
_LIFTED, _DROPPED, _TOPPLED, _EARLY = 1, 2, 4, 8


# ----------------------------------------------------------------------------- hooks
def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Track picked up / dropped / toppled / early, and latch the current step when its object is home."""
    touch = _touching(model, data)
    k, n = int(data.userdata[_U_STEP]), int(data.userdata[_U_N])
    order = [int(i) for i in data.userdata[_U_ORDER:_U_ORDER + n]]
    for p in order:
        name = _POOL[p]
        f = int(data.userdata[_U_FLAGS + p])
        pos = data.xpos[model.body(f"obj_{name}").id]
        held = "finger" in touch[p]
        if held and pos[2] > data.userdata[_U_STAND + p] + LIFTED_DZ:
            f |= _LIFTED
        # Judged only while lying on the table: a held object whose finger contact flickers for a step is
        # in mid-air, not dropped.
        if not held and "table" in touch[p] and _resting(model, data, name):
            if f & _LIFTED:
                f |= _DROPPED
            elif _shape(name) != "ball" and _tilt(model, data, p) > TOPPLE_ANGLE:
                f |= _TOPPLED
        if not held and _resting(model, data, name) and _in_bin(model, data, p, touch) is not None \
                and order.index(p) > k:
            f |= _EARLY
        if not held and pos[2] < T - 0.02:   # off the table, resting or not
            f |= _DROPPED
        data.userdata[_U_FLAGS + p] = f
    if k < n and _home(model, data, order[k], touch):
        data.userdata[_U_STEP] = k + 1


def reset(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
    """New episode: which objects are out, their size / mass / friction / colour / place, and the order.

    Changes model fields (sizes, masses, ...) as well as data. Call after mj_resetData.
    """
    while True:
        n = int(rng.integers(N_OBJECTS[0], N_OBJECTS[1] + 1))
        shapes = rng.choice(SHAPES, size=n)
        if max(np.sum(shapes == s) for s in SHAPES) <= MAX_PER_SHAPE:
            break
    used = {s: 0 for s in SHAPES}
    pool = []
    for s in shapes:
        pool.append(_POOL.index(f"{s}_{used[s]}"))
        used[s] += 1

    sizes = [_draw_size(_shape(_POOL[p]), rng) for p in pool]
    lying = [_shape(_POOL[p]) == "cylinder" and rng.random() < CYL_LYING for p in pool]
    xy = _draw_places(sizes, [_shape(_POOL[p]) for p in pool], lying, rng)
    while xy is None:                              # couldn't fit them: drop one and retry with fewer
        if len(pool) == 1:
            raise RuntimeError("tidy_table: no room for even one object -- ZONE / BINS / sizes don't fit")
        pool, sizes, lying = pool[:-1], sizes[:-1], lying[:-1]
        xy = _draw_places(sizes, [_shape(_POOL[p]) for p in pool], lying, rng)
    colours = rng.choice(len(COLOURS), size=len(pool), replace=False)

    data.userdata[_U_COLOUR:_U_COLOUR + len(_POOL)] = -1
    data.userdata[_U_FLAGS:_U_FLAGS + len(_POOL)] = 0
    for p, size, lie, (x, y), c in zip(pool, sizes, lying, xy, colours):
        name = _POOL[p]
        _set_object(model, name, size, rng)
        model.geom_rgba[model.geom(f"obj_{name}").id] = list(COLOURS.values())[c]
        z = _stand_height(_shape(name), size, lie)
        quat = _yaw_quat(_draw_yaw(_shape(name), lie, x, y, rng), lie)
        _place(model, data, name, (x, y, z + SPAWN_LIFT), quat=quat)
        data.userdata[_U_COLOUR + p] = c
        data.userdata[_U_STAND + p] = z
        data.userdata[_U_UP + 3 * p:_U_UP + 3 * p + 3] = _z_axis(quat)
    for p in range(len(_POOL)):
        _show(model, _POOL[p], p in pool)
        if p not in pool:
            _place(model, data, _POOL[p], (PARK[0] - 0.15 * p, PARK[1], PARK[2]))
    mujoco.mj_setConst(model, mujoco.MjData(model))   # invweight0 etc. follow the new masses
    data.userdata[_U_N] = len(pool)
    data.userdata[_U_ORDER:_U_ORDER + len(pool)] = rng.permutation(pool)
    data.userdata[_U_STEP] = 0


CONDITION_NAMES = [f"colour_{c}" for c in COLOURS] + [f"shape_{s}" for s in SHAPES] + ["done"]


def condition(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """The current step's target for a policy without language input: one-hot colour, one-hot shape; all
    zeros but ``done`` once every step is done (go home)."""
    v = np.zeros(len(CONDITION_NAMES), dtype=np.float32)
    k, n = int(data.userdata[_U_STEP]), int(data.userdata[_U_N])
    if k >= n:
        v[-1] = 1.0
        return v
    p = int(data.userdata[_U_ORDER + k])
    v[int(data.userdata[_U_COLOUR + p])] = 1.0
    v[len(COLOURS) + SHAPES.index(_shape(_POOL[p]))] = 1.0
    return v


def progress(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int, str]:
    """(current step index, number of steps, its instruction); index == total when finished, with the
    last step's instruction."""
    k, n = int(data.userdata[_U_STEP]), int(data.userdata[_U_N])
    return k, n, _step_text(data, min(k, n - 1))


# ----------------------------------------------------------------------------- scene
@scene("tidy_table", camera=dict(lookat=[0.30, 0.36, 0.75], distance=0.8, azimuth=200, elevation=-45),
       step=step, reset=reset, progress=progress, condition=condition, condition_names=CONDITION_NAMES)
def build(spec: mujoco.MjSpec) -> None:
    add_floor(spec)
    spec.nuserdata = _N_USER
    world = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX
    x0, x1 = TABLE_X
    world.add_geom(name="table", type=box, size=[(x1 - x0) / 2, TABLE_HALF_Y, T / 2], pos=[(x0 + x1) / 2, 0, T / 2],
                   rgba=[0.55, 0.42, 0.3, 1])

    grey = [0.78, 0.78, 0.8, 1]
    h, w = BIN_INNER / 2, BIN_T / 2
    for shape, (bx, by) in BINS.items():
        z_floor, z_wall = T + BIN_T / 2, T + BIN_T + BIN_WALL_H / 2
        world.add_geom(name=f"bin_{shape}_floor", type=box, size=[h + BIN_T, h + BIN_T, w], pos=[bx, by, z_floor], rgba=grey)
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            size = [w, h + BIN_T, BIN_WALL_H / 2] if sx else [h, w, BIN_WALL_H / 2]
            world.add_geom(name=f"bin_{shape}_wall_{sx}{sy}", type=box, size=size,
                           pos=[bx + sx * (h + w), by + sy * (h + w), z_wall], rgba=grey)

    cam = TOP_CAMERA
    world.add_camera(name="top", pos=list(cam["pos"]), quat=_look_at(cam["pos"], cam["lookat"]), fovy=cam["fovy"],
                     resolution=list(cam["resolution"]))

    # Object pool; reset() sizes and places them. Defaults here are mid-range so the model is valid as built.
    for i, name in enumerate(_POOL):
        shape = _shape(name)
        body = world.add_body(name=f"obj_{name}", pos=[PARK[0] - 0.15 * i, PARK[1], PARK[2]], gravcomp=1)
        body.add_freejoint(name=f"obj_{name}")
        kw = dict(name=f"obj_{name}", rgba=[0.6, 0.6, 0.6, 1], mass=0.05, friction=[0.8, 0.02, ROLL_FRICTION])
        if shape == "box":
            body.add_geom(type=box, size=[0.02, 0.0175, 0.0175], condim=4, **kw)
        elif shape == "cylinder":
            # condim 6: rolling friction, so one set out on its side doesn't roll off at a touch
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.017, 0.027, 0], condim=6, **kw)
        else:
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], condim=6, **kw)


# ----------------------------------------------------------------------------- randomisation
def _draw_size(shape: str, rng: np.random.Generator) -> tuple:
    """Geom size (MuJoCo convention: half-extents / radius, half-height), no side longer than MAX_SIDE."""
    if shape == "box":
        size = tuple(rng.uniform(*r) / 2 for r in (BOX_LEN, BOX_WIDTH, BOX_HEIGHT))
    elif shape == "cylinder":
        size = rng.uniform(*CYL_RADIUS), rng.uniform(*CYL_HEIGHT) / 2
    else:
        size = (rng.uniform(*BALL_RADIUS),)
    return tuple(min(s, MAX_SIDE / 2) for s in size)


def _footprint(shape: str, size: tuple, lying: bool = False) -> float:
    """Radius of the object's footprint (any yaw)."""
    if shape == "box":
        return math.hypot(size[0], size[1])
    if shape == "cylinder" and lying:
        return math.hypot(size[0], size[1])
    return size[0]


def _jaw_reach(shape: str, size: tuple) -> float:
    """How far from the object's centre the open jaws reach, opened to its widest side + JAW_SPARE."""
    width = 2 * max(size[0], size[1]) if shape == "box" else 2 * size[0]
    return (width + JAW_SPARE) / 2 + FINGER_T


def _clear_of_bins(x: float, y: float, radius: float) -> bool:
    """Nothing within `radius` of (x, y) -- the object, or the open gripper around it -- touches a bin."""
    for bx, by in BINS.values():
        half = BIN_INNER / 2 + BIN_T + BIN_CLEAR
        dx, dy = max(abs(x - bx) - half, 0.0), max(abs(y - by) - half, 0.0)
        if math.hypot(dx, dy) < radius:
            return False
    return True


def _draw_places(sizes, shapes, lying, rng, tries: int = 4000):
    (x0, x1), (y0, y1) = ZONE
    radii = [_footprint(s, z, lie) for s, z, lie in zip(shapes, sizes, lying)]
    reach = [max(r, _jaw_reach(s, z)) for r, s, z in zip(radii, shapes, sizes)]
    for _ in range(tries):
        xy = np.column_stack([rng.uniform(x0, x1, len(sizes)), rng.uniform(y0, y1, len(sizes))])
        if all(_clear_of_bins(x, y, r) for (x, y), r in zip(xy, reach)) and \
                all(np.linalg.norm(xy[i] - xy[j]) >= radii[i] + radii[j] + FINGER_GAP
                    for i in range(len(xy)) for j in range(i + 1, len(xy))):
            return xy
    return None


def _set_object(model: mujoco.MjModel, name: str, size: tuple, rng: np.random.Generator) -> None:
    """Resize a pool object and give it a matching mass and inertia, a random density and friction."""
    shape, g, b = _shape(name), model.geom(f"obj_{name}").id, model.body(f"obj_{name}").id
    if shape == "box":
        hx, hy, hz = size
        volume, half = 8 * hx * hy * hz, [hx, hy, hz]
    elif shape == "cylinder":
        r, hz = size
        volume, half = math.pi * r * r * 2 * hz, [r, r, hz]
    else:
        r = size[0]
        volume, half = 4 / 3 * math.pi * r ** 3, [r, r, r]
    mass = float(np.clip(rng.uniform(*DENSITY) * volume, *MASS))
    if shape == "box":
        inertia = [mass / 3 * (hy * hy + hz * hz), mass / 3 * (hx * hx + hz * hz), mass / 3 * (hx * hx + hy * hy)]
    elif shape == "cylinder":
        side = mass * (3 * r * r + (2 * hz) ** 2) / 12
        inertia = [side, side, mass * r * r / 2]
    else:
        inertia = [2 / 5 * mass * r * r] * 3
    model.geom_size[g, :len(size)] = size
    model.geom_rbound[g] = float(np.linalg.norm(half)) if shape == "box" else math.hypot(half[0], half[2]) \
        if shape == "cylinder" else half[0]
    model.geom_aabb[g] = [0, 0, 0, *half]
    # The body's bounding-volume box culls collisions before the geoms are tested: left at its compiled
    # size, a grown object sinks into the table (contacts come and go) and a shrunk one floats.
    model.bvh_aabb[model.body_bvhadr[b]] = [0, 0, 0, *half]
    model.geom_friction[g, 0] = rng.uniform(*FRICTION)
    model.body_mass[b] = mass
    model.body_subtreemass[b] = mass
    model.body_inertia[b] = inertia


def _show(model: mujoco.MjModel, name: str, on: bool) -> None:
    """Out this episode, or hidden: invisible, colliding with nothing, gravity cancelled so it stays put."""
    g, b = model.geom(f"obj_{name}").id, model.body(f"obj_{name}").id
    model.geom_contype[g] = model.geom_conaffinity[g] = int(on)
    model.body_gravcomp[b] = 0.0 if on else 1.0
    if not on:
        model.geom_rgba[g, 3] = 0.0


def _stand_height(shape: str, size: tuple, lying: bool = False) -> float:
    if shape == "box":
        return size[2] + T
    if shape == "cylinder" and not lying:
        return size[1] + T
    return size[0] + T


def grasp_yaw_range(x: float, y: float) -> tuple[float, float]:
    """Gripper yaw (rad) the left arm reaches pointing down at grasp height over (x, y) in ZONE: measured with
    humanoid_il/act/ik.py, conservatively. 0..-20 deg reach everywhere; turning positive needs the object on
    the far side (larger y), turning past -30 fails toward the zone's far outer corner."""
    lo = -20.0 if (x > 0.33 and y > 0.355) else -30.0
    hi = 0.0 if y < 0.242 else min(40.0, 10.0 * (math.floor((y - 0.242) / 0.032) + 1))
    return math.radians(lo), math.radians(hi)


def _draw_yaw(shape: str, lying: bool, x: float, y: float, rng: np.random.Generator) -> float:
    """A random yaw the gripper can grasp from above at (x, y): any side of a box (they repeat every 90 deg) or
    a lying cylinder's axis (every 180 deg) turned within grasp_yaw_range. Balls and standing cylinders: any."""
    if shape == "ball" or (shape == "cylinder" and not lying):
        return rng.uniform(-math.pi, math.pi)
    lo, hi = grasp_yaw_range(x, y)
    period = math.pi / 2 if shape == "box" else math.pi
    grasp = rng.uniform(lo, hi) + period * int(rng.integers(0, round(2 * math.pi / period)))
    # The jaws close along the gripper's Y, so a lying cylinder's axis must run along its X; _yaw_quat lays a
    # cylinder down with its axis along -Y, a quarter turn from that.
    return grasp + (math.pi / 2 if shape == "cylinder" else 0.0)


def _yaw_quat(yaw: float, lying: bool = False) -> tuple:
    """Turned `yaw` about vertical; lying: first tipped 90 deg about X (a cylinder's axis then horizontal)."""
    cz, sz = math.cos(yaw / 2), math.sin(yaw / 2)
    if not lying:
        return (cz, 0.0, 0.0, sz)
    c = s = math.sqrt(0.5)
    return (cz * c, cz * s, sz * s, sz * c)


def _z_axis(quat) -> tuple:
    w, x, y, z = quat
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def _place(model, data, name, pos, yaw: float = 0.0, quat=None):
    adr = model.joint(f"obj_{name}").qposadr[0]
    data.qpos[adr:adr + 7] = [*pos, *(quat if quat is not None else _yaw_quat(yaw))]
    data.qvel[model.joint(f"obj_{name}").dofadr[0]:model.joint(f"obj_{name}").dofadr[0] + 6] = 0


def _look_at(pos, target) -> list:
    """Camera quaternion looking from pos at target, image up toward world +Z (MuJoCo cameras look along -Z)."""
    f = np.subtract(target, pos, dtype=float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([right, up, -f]).flatten())
    return list(quat)


# ----------------------------------------------------------------------------- checks
def _shape(name: str) -> str:
    return name.rsplit("_", 1)[0]


def _touching(model, data) -> dict[int, set]:
    """Per pool index, what the object touches: "finger", "table" (or the floor), "bin_<shape>", or the pool
    index of another object."""
    fingers = {model.body(b).id for b in FINGER_BODIES}
    objs = {model.body(f"obj_{n}").id: p for p, n in enumerate(_POOL)}
    out = {p: set() for p in range(len(_POOL))}

    def what(g):
        b = model.geom_bodyid[g]
        if b in fingers:
            return "finger"
        if b in objs:
            return objs[b]
        name = model.geom(g).name
        if name in ("table", "floor"):
            return "table"
        return "_".join(name.split("_")[:2]) if name.startswith("bin_") else None

    for c in data.contact[:data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if b1 in objs:
            out[objs[b1]].add(what(c.geom2))
        if b2 in objs:
            out[objs[b2]].add(what(c.geom1))
    return out


def _resting(model, data, name) -> bool:
    v = data.qvel[model.joint(f"obj_{name}").dofadr[0]:][:6]
    return np.linalg.norm(v[:3]) < REST_LIN and np.linalg.norm(v[3:]) < REST_ANG


def _tilt(model, data, p: int) -> float:
    """Angle between the object's local z axis now and as it was set out."""
    z_now = data.xmat[model.body(f"obj_{_POOL[p]}").id].reshape(3, 3)[:, 2]
    return math.acos(np.clip(z_now @ data.userdata[_U_UP + 3 * p:_U_UP + 3 * p + 3], -1, 1))


def _over_bin(pos):
    """Shape of the bin `pos` is above the inside of, or None."""
    for shape, (bx, by) in BINS.items():
        if abs(pos[0] - bx) < BIN_INNER / 2 and abs(pos[1] - by) < BIN_INNER / 2:
            return shape
    return None


def _in_bin(model, data, p: int, touch: dict):
    """Shape of the bin object p is in -- over its inside and touching it, or touching an object that is
    -- else None. Touching, not height: a held object hovering over a bin is not in it."""
    shape = _over_bin(data.xpos[model.body(f"obj_{_POOL[p]}").id])
    if shape is None:
        return None
    if f"bin_{shape}" in touch[p]:
        return shape
    for q in touch[p]:
        if isinstance(q, int) and f"bin_{shape}" in touch[q] \
                and _over_bin(data.xpos[model.body(f"obj_{_POOL[q]}").id]) == shape:
            return shape
    return None


def _home(model, data, p: int, touch: dict) -> bool:
    name = _POOL[p]
    return ("finger" not in touch[p] and _resting(model, data, name)
            and _in_bin(model, data, p, touch) is not None)


def _describe(data, p: int) -> str:
    return f"{list(COLOURS)[int(data.userdata[_U_COLOUR + p])]} {_shape(_POOL[p])}"


def _step_text(data, k: int) -> str:
    p = int(data.userdata[_U_ORDER + k])
    return f"put the {_describe(data, p)} in a bin"


# ----------------------------------------------------------------------------- public helpers
def episode_objects(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
    """This episode's objects in the announced order: name (body / geom / joint name), shape, colour, its bin's
    centre, geom size, and whether it was set out lying on its side."""
    n = int(data.userdata[_U_N])
    out = []
    for k in range(n):
        p = int(data.userdata[_U_ORDER + k])
        name = _POOL[p]
        out.append(dict(name=f"obj_{name}", shape=_shape(name), colour=list(COLOURS)[int(data.userdata[_U_COLOUR + p])],
                        bin=BINS[_shape(name)], size=tuple(model.geom_size[model.geom(f"obj_{name}").id]),
                        lying=bool(abs(data.userdata[_U_UP + 3 * p + 2]) < 0.5)))
    return out


def episode_status(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Score the episode so far. ``tidiness``: share of objects resting in a bin now. ``success``: every
    step done with nothing dropped, toppled or put away early."""
    k, n = int(data.userdata[_U_STEP]), int(data.userdata[_U_N])
    touch = _touching(model, data)
    order = [int(i) for i in data.userdata[_U_ORDER:_U_ORDER + n]]
    flags = {p: int(data.userdata[_U_FLAGS + p]) for p in order}
    names = lambda bit: [f"{_describe(data, p)}" for p in order if flags[p] & bit]  # noqa: E731
    homed = sum(_home(model, data, p, touch) for p in order)
    status = dict(step=k, total=n, homed=homed, tidiness=homed / n if n else 1.0,
                  dropped=names(_DROPPED), toppled=names(_TOPPLED), early=names(_EARLY))
    status["success"] = k == n and not (status["dropped"] or status["toppled"] or status["early"])
    return status
