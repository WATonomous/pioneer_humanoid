"""Paint shelf (a workshop scene, by sight): three paint sample tins stand on a rubber bench mat, one each side and one
in the middle; their lids are red, green and blue, which is which changing with every seed. Behind them, a low wooden
riser shelf. The robot reads the lids' colours with its head camera and lifts the tins onto the shelf in a row: red
on the left, green in the middle, blue on the right (as it sees them). Each is taken straight up by its lid with the
hand on the side it goes to; one that has to cross the middle is set down in the middle of the mat by one hand and
taken on by the other.

The props are CAD models (``assets/workshop/make_meshes.py``, build123d) in the workshop room (``workshop.py``).
The robot starts at the bench: no walking.

Steps (``progress``): 1. read the lids, 2. shelve them (n of 3 in place).
"""
from __future__ import annotations

import importlib.util
import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.scenes import socket_rail as SR
from pioneer_whole_body.studio import BENCH_TOP_Z
from pioneer_whole_body.workshop import ASSETS, MESH, add_workshop, mesh, paint, texture

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

_spec = importlib.util.spec_from_file_location("make_meshes", ASSETS / "make_meshes.py")
_mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mm)
PAINT, SHELF = _mm.PAINT, _mm.SHELF

MAT_Z = SR.MAT_Z
TINS = ("tin_a", "tin_b", "tin_c")
SOURCES = ((0.445, 0.27), (0.405, 0.0), (0.445, -0.27))  # where the tins stand (left, middle, right)
MIDDLE_YAW = 1.1                         # the fingers on the middle one: turned well across (short of the shelf)
COLOURS = {"red": [0.82, 0.10, 0.08, 1], "green": [0.12, 0.55, 0.20, 1], "blue": [0.10, 0.25, 0.75, 1]}
SPOT_Y = {"red": 0.13, "green": 0.0, "blue": -0.13}
SHELF_X = 0.53                           # the shelf's front edge
SPOT_X = SHELF_X + 0.065
SHELF_Z = T + SHELF["height"]
HEIGHT = PAINT["h"] + PAINT["lid_h"]
STEPS = ("read the lids", "shelve them")
_U_STEP, _U_SEEN = 0, 1


def _colour(model, tin: str) -> str:
    rgba = model.geom_rgba[model.geom(f"{tin}_lid").id]
    return min(COLOURS, key=lambda c: np.linalg.norm(np.asarray(COLOURS[c][:3]) - rgba[:3]))


def _shelved(model, data) -> int:
    n = 0
    for tin in TINS:
        b = model.body(tin).id
        p = data.xpos[b]
        y = SPOT_Y[_colour(model, tin)]
        if abs(p[0] - SPOT_X) < 0.03 and abs(p[1] - y) < 0.03 and abs(p[2] - SHELF_Z) < 0.008 and data.xmat[b][8] > 0.95:
            n += 1
    return n


def step(model, data) -> None:
    if data.userdata[_U_SEEN] and data.userdata[_U_STEP] == 0:
        data.userdata[_U_STEP] = 1
    if data.userdata[_U_STEP] == 1 and _shelved(model, data) == len(TINS):
        data.userdata[_U_STEP] = 2


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k == 1:
        return 1, 2, f"shelve them ({_shelved(model, data)} of 3 in place)"
    return k, 2, STEPS[min(k, 1)]


def reset(model, data, rng) -> None:
    """Which lid is which colour, and each tin a little off its spot, turned any way."""
    for tin, c in zip(TINS, rng.permutation(list(COLOURS))):
        model.geom_rgba[model.geom(f"{tin}_lid").id] = COLOURS[c]
    for tin, (x, y) in zip(TINS, SOURCES):
        a = model.joint(tin).qposadr[0]
        th = rng.uniform(0, 2 * math.pi)
        data.qpos[a:a + 3] = [x + rng.uniform(-0.004, 0.004), y + rng.uniform(-0.004, 0.004), MAT_Z]
        data.qpos[a + 3:a + 7] = [math.cos(th / 2), 0, 0, math.sin(th / 2)]


@scene("paint_shelf", robot_pos=(0.0, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer  |  paint shelf: three paint tins read by their lids and shelved red, green, blue",
       camera=((0.50, 0.0, 0.97), 1.7, 60, -22))
def build(spec: mujoco.MjSpec) -> None:
    add_workshop(spec)
    spec.nuserdata = 2
    world = spec.worldbody
    vis = dict(contype=0, conaffinity=0, group=1)
    col = dict(rgba=[0, 0, 0, 0], group=3)
    # the rubber mat, in front of the shelf
    (x0, x1), hy = (0.37, SHELF_X - 0.01), 0.41
    mat = texture(spec, "rubber_mat", repeat=(2, 6), specular=0.05, shininess=0.1)
    world.add_geom(type=BOX, size=[(x1 - x0) / 2, hy, SR.MAT["h"] / 2], pos=[(x0 + x1) / 2, 0, T + SR.MAT["h"] / 2],
                   material=mat, friction=[1.0, 0.02, 0.002])
    # the shelf: a pine board on two legs
    pine = paint(spec, "pine", [0.80, 0.62, 0.40, 1], specular=0.15, shininess=0.3)
    sh = world.add_body(name="shelf", pos=[SHELF_X, 0, T])
    mesh(spec, "shelf_board")
    sh.add_geom(type=MESH, meshname="shelf_board", material=pine, **vis)
    L, d, H, b, lg = SHELF["length"], SHELF["depth"], SHELF["height"], SHELF["board"], SHELF["leg"]
    sh.add_geom(type=BOX, size=[d / 2, L / 2, b / 2], pos=[d / 2, 0, H - b / 2], friction=[1.0, 0.02, 0.002], **col)
    for s_ in (-1, 1):
        sh.add_geom(type=BOX, size=[(d - 0.01) / 2, lg / 2, (H - b) / 2], pos=[d / 2, s_ * (L / 2 - lg / 2 - 0.01),
                                                                               (H - b) / 2], **col)
    # the tins (their lids' colours set by ``reset``)
    for m_ in ("paint_tin", "paint_label", "paint_lid"):
        mesh(spec, m_)
    tin_mat = paint(spec, "tin_steel", [0.74, 0.75, 0.77, 1], specular=0.8, shininess=0.7)
    label = paint(spec, "paint_label_m", [0.93, 0.92, 0.88, 1], specular=0.1)
    for tin, (x, y) in zip(TINS, SOURCES):
        bd = world.add_body(name=tin, pos=[x, y, MAT_Z])
        bd.add_freejoint(name=tin)
        bd.add_geom(type=MESH, meshname="paint_tin", material=tin_mat, **vis)
        bd.add_geom(type=MESH, meshname="paint_label", material=label, **vis)
        bd.add_geom(name=f"{tin}_lid", type=MESH, meshname="paint_lid", pos=[0, 0, PAINT["h"]], rgba=COLOURS["red"],
                    **vis)
        grip = dict(rgba=[0, 0, 0, 0], group=3, friction=[1.2, 0.02, 0.002], condim=4)
        bd.add_geom(type=CYL, size=[PAINT["r"], PAINT["h"] / 2], pos=[0, 0, PAINT["h"] / 2], mass=0.12, **grip)
        bd.add_geom(type=CYL, size=[PAINT["lid_r"], PAINT["lid_h"] / 2], pos=[0, 0, PAINT["h"] + PAINT["lid_h"] / 2],
                    mass=0.01, **grip)
        bd.add_site(name=f"{tin}_top", pos=[0, 0, HEIGHT])


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.vision import Eye

    m, d = wb.m, wb.d
    wb.start_ready()
    wb.fingers("l", "relaxed", dur=0.6)
    wb.fingers("r", "relaxed", dur=0.6)
    wb.look([0.50, 0.0, T + 0.04], dur=1.0)              # the mat and the shelf: all in view from here
    wb.wait(0.6)
    # 1. which tin has which lid: the coloured blobs on the plane of the lids, each put to the nearest tin's spot
    eye = Eye(m)
    seen = {}
    for blob in eye.find(d, MAT_Z + HEIGHT):
        if blob.colour in SPOT_Y:
            k = min(range(len(SOURCES)), key=lambda k: np.linalg.norm(np.asarray(SOURCES[k]) - blob.world[:2]))
            if np.linalg.norm(np.asarray(SOURCES[k]) - blob.world[:2]) < 0.04 and \
                    (k not in seen or blob.area > seen[k].area):
                seen[k] = blob
    colour = {k: b.colour for k, b in seen.items()}
    notes["lids"] = {("left", "middle", "right")[k]: c for k, c in sorted(colour.items())}
    if sorted(colour.values()) != sorted(SPOT_Y):
        raise S.TaskFailed(f"read the lids as {notes['lids']}")
    d.userdata[_U_SEEN] = 1
    # 2. the green one first (the middle of the shelf: later ones aren't carried over it), then the one in the middle
    # of the mat (that frees it for a change of hands), then the others
    width = 2 * PAINT["lid_r"]
    first = next(k for k in colour if colour[k] == "green")
    for k in dict.fromkeys((first, 1, 0, 2)):
        tin, y_dst = TINS[k], SPOT_Y[colour[k]]
        if k == 1:
            side = "l" if y_dst > 0 else "r"                                   # (the hand on the side it goes to)
            SR._take(wb, side, tin, width, over=0.06, yaw=MIDDLE_YAW)
        else:
            side = "l" if k == 0 else "r"
            SR._take(wb, side, tin, width, over=0.06)
            if (y_dst > 0 and side == "r") or (y_dst < 0 and side == "l"):     # across the middle: change hands
                SR._put(wb, side, tin, SOURCES[1], MAT_Z, carry=0.07)
                side = "r" if side == "l" else "l"
                SR._take(wb, side, tin, width, over=0.06, yaw=MIDDLE_YAW)
        SR._put(wb, side, tin, (SPOT_X, y_dst), SHELF_Z + 0.001, carry=SHELF["height"] + 0.02, straighten=True,
                turn=True)
                                                         # (up here the hand can't lift far: the fingers are
                                                         # straightened before it backs off over the lid)
    wb.settle()
    wb.wait(0.5)
    notes["shelved"] = _shelved(m, d)
    if _shelved(m, d) != len(TINS):
        raise S.TaskFailed(f"{_shelved(m, d)} of 3 in place")


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["paint_shelf"].plan = plan
