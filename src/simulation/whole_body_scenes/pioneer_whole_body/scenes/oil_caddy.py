"""Oil caddy (a workshop scene): three small screw-lid tins (lithium grease, paste wax, solder flux) stand on a rubber
bench mat; behind them, a red steel tool caddy with three bays along its front. The robot puts each away in its bay:
taken straight up by its lid ("lid" grasp, set by the lid's size) with the hand on its side, carried over the
caddy's wall and set down on its bay's floor (the caddy's low dividers and the tube handle up on its end posts
leave the hand room), let go.

The props are CAD models (``assets/workshop/make_meshes.py``, build123d) in the workshop room (``workshop.py``).
The robot starts at the bench: no walking.

Steps (``progress``): 1. put them away (n of 3 in their bays).
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
CANS, CADDY = _mm.CANS, _mm.CADDY

MAT_Z = SR.MAT_Z
ITEMS = (("grease", 0.445, 0.27, "l", None), ("wax", 0.405, 0.0, "r", 1.1), ("flux", 0.445, -0.27, "r", None))
                                         # what, where on the mat (x, y), which hand, its fingers' turn (None: its
                                         # usual; the middle one's turned further: the fingers stop short of the caddy)
CADDY_X = 0.50                           # the caddy's front edge
BAY_Y = tuple((1 - k) * CADDY["length"] / CADDY["bays"] for k in range(CADDY["bays"]))   # left (+y) to right
BAY_X = CADDY_X + 0.06                 # (back in the bay: the thumb, behind the tin, clears the front wall)
FLOOR_Z = T + CADDY["t"]
MASS = dict(grease=0.14, wax=0.10, flux=0.07)
STEPS = ("put them away",)
_U_STEP = 0


def _height(kind: str) -> float:
    return CANS[kind]["h"] + CANS[kind]["cap_h"]


def _in_bays(model, data) -> int:
    n = 0
    for k, (kind, *_) in enumerate(ITEMS):
        b = model.body(kind).id
        p = data.xpos[b]
        if abs(p[0] - BAY_X) < 0.035 and abs(p[1] - BAY_Y[k]) < 0.035 and p[2] < FLOOR_Z + 0.01 and data.xmat[b][8] > 0.9:
            n += 1
    return n


def step(model, data) -> None:
    if data.userdata[_U_STEP] == 0 and _in_bays(model, data) == len(ITEMS):
        data.userdata[_U_STEP] = 1


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, 1, f"put them away ({_in_bays(model, data)} of 3 in their bays)"


def reset(model, data, rng) -> None:
    for kind, x, y, _, _ in ITEMS:
        a = model.joint(kind).qposadr[0]
        th = rng.uniform(0, 2 * math.pi)
        data.qpos[a:a + 3] = [x + rng.uniform(-0.005, 0.005), y + rng.uniform(-0.005, 0.005), MAT_Z]
        data.qpos[a + 3:a + 7] = [math.cos(th / 2), 0, 0, math.sin(th / 2)]


@scene("oil_caddy", robot_pos=(0.0, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer  |  oil caddy: three tins put away in the caddy's bays",
       camera=((0.50, 0.0, 0.97), 1.7, 60, -22))
def build(spec: mujoco.MjSpec) -> None:
    add_workshop(spec)
    spec.nuserdata = 1
    world = spec.worldbody
    vis = dict(contype=0, conaffinity=0, group=1)
    (x0, x1), hy = (0.37, CADDY_X - 0.01), 0.41      # (the rubber mat, in front of the caddy)
    mat = texture(spec, "rubber_mat", repeat=(2, 6), specular=0.05, shininess=0.1)
    world.add_geom(type=BOX, size=[(x1 - x0) / 2, hy, SR.MAT["h"] / 2], pos=[(x0 + x1) / 2, 0, T + SR.MAT["h"] / 2],
                   material=mat, friction=[1.0, 0.02, 0.002])
    # the caddy: its tray, the bays' dividers, the handle's posts and tube collide
    red = paint(spec, "caddy_red", [0.72, 0.08, 0.06, 1], specular=0.45, shininess=0.5)
    c = CADDY
    d, L, wh, bh, hh, t = c["depth"], c["length"], c["wall_h"], c["bay_h"], c["handle_h"], c["t"]
    cd = world.add_body(name="caddy", pos=[CADDY_X, 0, T])
    mesh(spec, "caddy")
    mesh(spec, "caddy_handle")
    cd.add_geom(type=MESH, meshname="caddy", material=red, **vis)
    cd.add_geom(type=MESH, meshname="caddy_handle", material="chrome", **vis)
    col = dict(rgba=[0, 0, 0, 0], group=3)
    for g in ((d / 2, 0, t / 2, d / 2, L / 2, t / 2), (t / 2, 0, wh / 2, t / 2, L / 2, wh / 2),
              (d - t / 2, 0, wh / 2, t / 2, L / 2, wh / 2), (d / 2, L / 2 - t / 2, wh / 2, d / 2, t / 2, wh / 2),
              (d / 2, -L / 2 + t / 2, wh / 2, d / 2, t / 2, wh / 2), (d / 2, 0, bh / 2, t / 2, L / 2, bh / 2)):
        cd.add_geom(type=BOX, pos=list(g[:3]), size=list(g[3:]), **col)
    for k in range(1, c["bays"]):
        cd.add_geom(type=BOX, pos=[d / 4, -L / 2 + k * L / c["bays"], bh / 2], size=[d / 4, t / 2, bh / 2], **col)
    for s_ in (-1, 1):
        cd.add_geom(type=BOX, pos=[c["handle_x"], s_ * (L / 2 - 0.012), hh / 2], size=[0.015, 0.002, hh / 2], **col)
    cd.add_geom(type=CYL, size=[0.011, L / 2 - 0.01], pos=[c["handle_x"], 0, hh - 0.012], quat=[0.7071, 0.7071, 0, 0],
                **col)
    # the things to put away: body, label, cap meshes; a cylinder each for the body and the cap
    looks = dict(grease=([0.70, 0.71, 0.74, 1], [0.95, 0.75, 0.10, 1], [0.12, 0.30, 0.65, 1], 0.7),     # body, label,
                 wax=([0.70, 0.71, 0.74, 1], [0.15, 0.45, 0.20, 1], [0.85, 0.82, 0.75, 1], 0.7),        # lid, shine
                 flux=([0.70, 0.71, 0.74, 1], [0.85, 0.10, 0.08, 1], [0.15, 0.15, 0.16, 1], 0.7))
    for kind, x, y, _, _ in ITEMS:
        cn = CANS[kind]
        body_rgba, label_rgba, cap_rgba, spec_ = looks[kind]
        b = world.add_body(name=kind, pos=[x, y, MAT_Z])
        b.add_freejoint(name=kind)
        for part, rgba, sp in (("body", body_rgba, spec_), ("label", label_rgba, 0.2), ("cap", cap_rgba, 0.3)):
            mesh(spec, f"{kind}_{part}")
            mt = paint(spec, f"{kind}_{part}_m", rgba, specular=sp, shininess=0.5)
            b.add_geom(type=MESH, meshname=f"{kind}_{part}", material=mt, pos=[0, 0, cn["h"] if part == "cap" else 0],
                       **vis)
        grip = dict(rgba=[0, 0, 0, 0], group=3, friction=[1.2, 0.02, 0.002], condim=4)
        b.add_geom(type=CYL, size=[cn["r"], cn["h"] / 2], pos=[0, 0, cn["h"] / 2], mass=MASS[kind], **grip)
        b.add_geom(type=CYL, size=[cn["cap_r"], cn["cap_h"] / 2], pos=[0, 0, cn["h"] + cn["cap_h"] / 2], mass=0.02, **grip)
        b.add_site(name=f"{kind}_top", pos=[0, 0, _height(kind)])


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S

    m, d = wb.m, wb.d
    wb.start_ready()
    wb.fingers("l", "relaxed", dur=0.6)
    wb.fingers("r", "relaxed", dur=0.6)
    wb.look([0.52, 0.0, T + 0.04], dur=1.0)              # the mat and the caddy: all in view from here
    wb.wait(0.6)
    for k, (kind, _, _, side, yaw) in enumerate(ITEMS):
        SR._take(wb, side, kind, 2 * CANS[kind]["cap_r"], over=0.06, yaw=yaw)    # (by the lid: the grip set by its
                                                                                  # size)
        SR._put(wb, side, kind, (BAY_X, BAY_Y[k]), FLOOR_Z + 0.001, carry=CADDY["wall_h"] + 0.028)
        notes[kind] = _in_bays(m, d)
    wb.settle()
    wb.wait(0.5)
    notes["in_bays"] = _in_bays(m, d)
    if _in_bays(m, d) != len(ITEMS):
        raise S.TaskFailed(f"{_in_bays(m, d)} of 3 in their bays")


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["oil_caddy"].plan = plan
