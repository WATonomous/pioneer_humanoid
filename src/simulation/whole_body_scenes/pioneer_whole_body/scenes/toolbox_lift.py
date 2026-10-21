"""Toolbox lift (a workshop scene): a small red steel toolbox stands on the bench, its bar handle on top; to its
right, a low pine riser with a rubber mat on it. The robot hooks the handle with its right hand (palm down, the
fingers already curled, lowered so the tips go down past the bar's far side, then closed under it), lifts the box,
carries it across and sets it down on the riser's mat, lets go (the fingers eased open, the hand straight up: the
tips come up past the bar) and stands back.

The props are CAD models (``assets/workshop/make_meshes.py``, build123d) in the workshop room (``workshop.py``).
The robot starts at the bench: no walking.

Steps (``progress``): 1. lift it, 2. set it on the riser.
"""
from __future__ import annotations

import importlib.util
import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import SHAPES
from pioneer_whole_body.scenes.socket_rail import RAISED
from pioneer_whole_body.studio import BENCH_TOP_Z
from pioneer_whole_body.workshop import ASSETS, MESH, add_workshop, mesh, paint, texture

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

_spec = importlib.util.spec_from_file_location("make_meshes", ASSETS / "make_meshes.py")
_mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mm)
TOOLBOX, RISER = _mm.TOOLBOX, _mm.RISER

START = (0.56, -0.08)                    # the toolbox's centre on the bench (out from the body: the arm lifts high here)
RISER_X, RISER_Y = 0.46, -0.31           # the riser's front edge, its centre in y
SPOT = (RISER_X + RISER["depth"] / 2, RISER_Y)
RISER_Z = T + RISER["height"] + RISER["mat"]
MASS = 0.20
HOOK_BACK, HOOK_ABOVE = 0.055, 0.015     # the palm from the bar's axis: back along the fingers, up
_FLAT_THUMB = {"th_c": 0.0, "th_f": 0.0, "th_mcp": 0.0, "th_ip": 0.0}
_FINGERS = ("i_", "", "r_")
HOOK_PRE = {**{f"{f}mcp": 0.2 for f in _FINGERS}, **{f"{f}pip": 0.8 for f in _FINGERS}, "l_mcp": 1.5, "l_pip": 1.55,
            **_FLAT_THUMB}               # an open hook: the fingers out and their tips down (past the bar's far side),
HOOK = {**SHAPES["knob"], "l_mcp": 1.5, "l_pip": 1.55, **_FLAT_THUMB}   # closed under it; the little finger curled
                                         # in, the thumb flat (clear of the bar and the lid)
CARRY = 0.03                             # the box's bottom this far over the riser's mat on the way across
STEPS = ("lift it", "set it on the riser")
_U_STEP = 0


def _bottom(model, data) -> float:
    return data.xpos[model.body("toolbox").id][2]


def _on_riser(model, data) -> bool:
    b = model.body("toolbox").id
    p = data.xpos[b]
    return (abs(p[0] - SPOT[0]) < 0.05 and abs(p[1] - SPOT[1]) < 0.06 and abs(p[2] - RISER_Z) < 0.006
            and data.xmat[b][8] > 0.97)


def step(model, data) -> None:
    k = data.userdata[_U_STEP]
    if k == 0 and _bottom(model, data) > T + 0.03:
        data.userdata[_U_STEP] = 1
    if k == 1 and _on_riser(model, data):
        data.userdata[_U_STEP] = 2


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, 2, STEPS[min(k, 1)]


def reset(model, data, rng) -> None:
    a = model.joint("toolbox").qposadr[0]
    th = rng.uniform(-0.08, 0.08)
    data.qpos[a:a + 3] = [START[0] + rng.uniform(-0.01, 0.01), START[1] + rng.uniform(-0.01, 0.01), T]
    data.qpos[a + 3:a + 7] = [math.cos(th / 2), 0, 0, math.sin(th / 2)]


@scene("toolbox_lift", robot_pos=(0.0, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer v2  |  toolbox lift: a steel toolbox hooked by its handle and set on the riser",
       camera=((0.50, -0.12, 0.97), 1.7, 60, -22))
def build(spec: mujoco.MjSpec) -> None:
    add_workshop(spec)
    spec.nuserdata = 1
    world = spec.worldbody
    vis = dict(contype=0, conaffinity=0, group=1)
    col = dict(rgba=[0, 0, 0, 0], group=3)
    # the riser: a pine block, the rubber mat on top
    pine = paint(spec, "pine", [0.80, 0.62, 0.40, 1], specular=0.15, shininess=0.3)
    mat = texture(spec, "rubber_mat", repeat=(1, 2), specular=0.05, shininess=0.1)
    rs = world.add_body(name="riser", pos=[RISER_X, RISER_Y, T])
    mesh(spec, "riser")
    rs.add_geom(type=MESH, meshname="riser", material=pine, **vis)
    d, L, H, mh = RISER["depth"], RISER["length"], RISER["height"], RISER["mat"]
    rs.add_geom(type=BOX, size=[d / 2, L / 2, H / 2], pos=[d / 2, 0, H / 2], **col)
    rs.add_geom(type=BOX, size=[d / 2 - 0.01, L / 2 - 0.01, mh / 2], pos=[d / 2, 0, H + mh / 2], material=mat,
                friction=[1.0, 0.02, 0.002])
    # the toolbox: its box and the handle's bar collide (the ears are too thin to matter)
    red = paint(spec, "toolbox_red", [0.70, 0.07, 0.05, 1], specular=0.5, shininess=0.55)
    black = paint(spec, "handle_black", [0.08, 0.08, 0.09, 1], specular=0.35, shininess=0.4)
    c = TOOLBOX
    tb = world.add_body(name="toolbox", pos=[*START, T])
    tb.add_freejoint(name="toolbox")
    for m_ in ("toolbox_body", "toolbox_handle"):
        mesh(spec, m_)
    tb.add_geom(type=MESH, meshname="toolbox_body", material=red, **vis)
    tb.add_geom(type=MESH, meshname="toolbox_handle", material=black, **vis)
    tb.add_geom(type=BOX, size=[c["depth"] / 2, c["length"] / 2, c["height"] / 2], pos=[0, 0, c["height"] / 2],
                mass=MASS, friction=[0.9, 0.02, 0.002], **col)
    bar_z = c["height"] + c["handle_h"]
    tb.add_geom(type=CYL, size=[c["handle_r"], c["handle_len"] / 2], pos=[0, 0, bar_z], quat=[0.7071, 0.7071, 0, 0],
                mass=0.02, friction=[1.2, 0.02, 0.002], condim=4, **col)
    for s_ in (-1, 1):
        tb.add_geom(type=BOX, size=[0.008, 0.002, (c["handle_h"] + c["handle_r"]) / 2],
                    pos=[0, s_ * (c["handle_len"] / 2 + 0.002), c["height"] + (c["handle_h"] + c["handle_r"]) / 2],
                    mass=0.005, **col)
    tb.add_site(name="toolbox_bar", pos=[0, 0, bar_z])


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.controller import READY, palm_down

    m, d = wb.m, wb.d
    wb.start_ready()
    wb.fingers("l", "relaxed", dur=0.6)
    wb.fingers("r", "relaxed", dur=0.6)
    wb.look([0.52, -0.15, T + 0.04], dur=1.0)            # the box and the riser: both in view from here
    wb.wait(0.6)
    # 1. the hook: palm down, the fingers square across the bar, curled; down past it, closed under it
    R = d.xmat[m.body("toolbox").id].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])                   # (the bar runs along the box's y: the fingers along its x)
    r = palm_down(yaw)
    bar = S.site(wb, "toolbox_bar")
    p = bar + [0, 0, HOOK_ABOVE] - r[:, 0] * HOOK_BACK
    wb.fingers("r", HOOK_PRE, dur=0.5)
    wb.arm_joints_to("r", RAISED, 0.8)                   # (READY has the right shoulder at its limit: out via this)
    wb.settle()
    for target, dur in ((p + [-0.06, 0, 0.10], 0.8), (p + [0, 0, 0.08], 0.6)):
        wb.reach("r", target, r, dur)
        wb.settle()
    wb.reach("r", p + [0, 0, 0.02], r, 0.6)
    wb.settle()
    wb.reach("r", p, r, 0.4)
    wb.settle()
    wb.fingers("r", HOOK, amount=S.TWIST_SQUEEZE, dur=0.5)
    wb.settle()
    wb.wait(0.2)
    # 2. up (its bottom clear of the riser's mat), across, down onto it
    lift = RISER_Z + CARRY - T
    wb.reach("r", p + [0, 0, lift], r, 1.0)
    wb.settle()
    wb.wait(0.3)
    if _bottom(m, d) < T + 0.5 * lift:
        raise S.TaskFailed("not lifted")
    off = S.body_pos(wb, "toolbox") - wb.palm_pose("r", False)[0]
    goal = np.array([SPOT[0], SPOT[1], RISER_Z]) - off
    q, _ = wb.hand_goal["r"]
    wb.reach("r", [goal[0], goal[1], q[2]], r, 1.6)
    wb.settle()
    wb.wait(0.3)
    wb.reach("r", goal + [0, 0, 0.008], r, 0.8)
    wb.settle()
    wb.reach("r", goal - [0, 0, 0.004], r, 0.4)          # (a little under: its weight on the mat, not the hand)
    wb.settle()
    wb.wait(0.2)
    # 3. let go: the fingers eased open, straight up (the tips come up past the bar's far side), back to READY
    wb.fingers("r", HOOK_PRE, dur=0.5)
    wb.settle()
    q, _ = wb.hand_goal["r"]
    wb.reach("r", q + [0, 0, 0.07], r, 0.8)
    wb.settle()
    wb.fingers("r", "relaxed", dur=0.4)
    wb.arm_joints_to("r", RAISED, 0.8)
    wb.settle()
    wb.arm_joints_to("r", READY, 1.0)
    wb.settle()
    wb.wait(0.6)
    notes["on_riser"] = bool(_on_riser(m, d))
    if not _on_riser(m, d):
        p = S.body_pos(wb, "toolbox")
        raise S.TaskFailed(f"not on the riser: at {np.round(p, 3).tolist()}")


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["toolbox_lift"].plan = plan
