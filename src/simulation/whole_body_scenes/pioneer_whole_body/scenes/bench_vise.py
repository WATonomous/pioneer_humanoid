"""Bench vise (a workshop scene modelled on the real thing): a machinist's bench vise bolted to a hardwood block on the
workbench, turned on its swivel base toward the robot's right hand; its screw is driven by a speed crank (the knob
spins on the crank's arm, 20 mm of jaw a turn). A short length of 40 mm aluminium round bar stands on the bench. The
right hand takes the crank's knob and winds the jaws open (three turns), lets go, takes the bar, sets it on the
ways between the jaws, then takes the knob again and winds them shut on it until the screw stalls: clamped.

The props are CAD models (``assets/workshop/make_meshes.py``, build123d) in the workshop room (``workshop.py``).
The robot starts at the bench, its legs and base still: no walking.

Steps (``progress``): 1. open the jaws, 2. set the bar between them, 3. clamp it.
"""
from __future__ import annotations

import importlib.util
import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import READY, SHAPES
from pioneer_whole_body.studio import BENCH_TOP_Z
from pioneer_whole_body.workshop import ASSETS, MESH, add_workshop, mesh, paint

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

_spec = importlib.util.spec_from_file_location("make_meshes", ASSETS / "make_meshes.py")
_mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mm)
VISE, BAR, BLOCK = _mm.VISE, _mm.BAR, _mm.BLOCK

YAW_V = -0.877                         # the vise's screw (from the crank back to the fixed jaw): forward and right
S = np.array([math.cos(YAW_V), math.sin(YAW_V), 0.0])
E1 = np.array([-math.sin(YAW_V), math.cos(YAW_V), 0.0])    # across it, to the left: the crank's arm starts this way
Z_S = 0.965                            # the screw's axis, above the floor
ORIGIN = np.array([0.69, -0.31, Z_S])  # on the screw's axis at the fixed jaw's face
Q0 = 0.02                              # the jaws' opening to start
TURNS_OPEN = 3                         # turns to open (20 mm a turn)
WIND = -1.0                            # the way it opens: the arm going down from across-left (the other way,
                                       # the grip slips where the thumb has to push the knob)
HUB_X = -(VISE["jaw_t"] + 0.055 + VISE["screw_out"])   # the crank's hub, from the moving jaw's face along the screw
KNOB_MID = 0.012 + VISE["knob_len"] * 0.65    # where the hand holds the knob: out from the crank's arm (toward the robot)
BAR_XY = (0.43, -0.37)                 # the bar on the bench
KNOB_YAW = math.atan2(E1[1], E1[0])    # the right hand's fingers on the knob: across it (the arm's easy way)
HOOK_BACK, HOOK_ABOVE = 0.05, 0.02     # the palm from the knob's axis: back along the fingers, up (as on a pull)
BAR_YAW = KNOB_YAW                     # the fingers on the bar: along the jaws (the thumb toward a jaw)
RAISED = {**READY, "shoulder_pitch": -0.45, "shoulder_roll": 0.8}    # the right hand up and out to the side
_FINGERS = ("i_", "", "r_")
KNOB_PRE = {**SHAPES["knob_pre"], **{f"{f}mcp": 0.2 for f in _FINGERS}, **{f"{f}pip": 0.8 for f in _FINGERS}}
                                       # an open hook (the tips out past the knob's far side), the thumb ready
STEPS = ("open the jaws", "set the bar between them", "clamp it")
_U_STEP = 0


def _opening(model, data) -> float:
    return Q0 + float(data.qpos[model.joint("vise_jaw").qposadr[0]])


def _local(model, data, body: str) -> np.ndarray:
    """A body's position in the vise's frame (x along the screw from the fixed jaw's face, z up from its axis)."""
    p = data.xpos[model.body(body).id] - ORIGIN
    return np.array([p @ S, p @ E1, p[2]])


def _between(model, data) -> bool:
    b = _local(model, data, "bar")
    q = _opening(model, data)
    return -q < b[0] < 0 and abs(b[1]) < 0.03 and abs(b[2] - VISE["bed"]) < 0.004


def _clamped(model, data) -> bool:
    return _between(model, data) and _opening(model, data) < 2 * BAR["r"] + 0.0015 and \
        data.xmat[model.body("bar").id][8] > 0.98


def step(model, data) -> None:
    k = int(data.userdata[_U_STEP])
    if k == 0 and _opening(model, data) > 2 * BAR["r"] + 0.025:
        data.userdata[_U_STEP] = 1
    elif k == 1 and _between(model, data):
        data.userdata[_U_STEP] = 2
    elif k == 2 and _clamped(model, data):
        data.userdata[_U_STEP] = 3


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def reset(model, data, rng) -> None:
    a = model.joint("bar").qposadr[0]
    th = rng.uniform(0, 2 * math.pi)
    data.qpos[a] += rng.uniform(-0.004, 0.004)
    data.qpos[a + 1] += rng.uniform(-0.004, 0.004)
    data.qpos[a + 3:a + 7] = [math.cos(th / 2), 0, 0, math.sin(th / 2)]


@scene("bench_vise", robot_pos=(0.0, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer v2  |  bench vise: the jaws wound open, a bar set in them, wound shut on it",
       camera=((0.48, -0.14, 1.08), 1.55, 128, -22))
def build(spec: mujoco.MjSpec) -> None:
    add_workshop(spec)
    spec.nuserdata = 1
    world = spec.worldbody
    vis = dict(contype=0, conaffinity=0, group=1)
    col = dict(rgba=[0, 0, 0, 0], group=3)
    c = VISE
    for m_ in ("vise_body", "vise_jaw", "vise_plates", "vise_crank", "vise_knob", "round_bar", "vise_block"):
        mesh(spec, m_)
    iron = paint(spec, "vise_blue", [0.16, 0.27, 0.40, 1], specular=0.35, shininess=0.4)
    steel = paint(spec, "jaw_steel", [0.55, 0.56, 0.58, 1], specular=0.7, shininess=0.6)
    black = paint(spec, "knob_black", [0.06, 0.06, 0.07, 1], specular=0.5, shininess=0.6)
    oak = paint(spec, "oak", [0.62, 0.45, 0.27, 1], specular=0.15, shininess=0.3)
    alu = paint(spec, "aluminium", [0.80, 0.81, 0.83, 1], specular=0.85, shininess=0.75)
    quat = [math.cos(YAW_V / 2), 0, 0, math.sin(YAW_V / 2)]
    w, bed, jh, jt, ways, rear, fz = c["jaw_w"], c["bed"], c["jaw_h"], c["jaw_t"], c["ways"], c["rear"], -c["foot"]
    base_top = fz + c["base_h"]

    # the hardwood block (under the body: the crank sweeps clear in front of it)
    blk = world.add_body(name="vise_block", pos=[*(ORIGIN[:2] + S[:2] * 0.02), Z_S + fz], quat=quat)
    bh = Z_S + fz - T
    blk.add_geom(type=BOX, size=[0.12, BLOCK["w"] / 2, bh / 2], pos=[0, 0, -bh / 2], material=oak)
    # the vise's fixed part: its frame on the screw's axis at the fixed jaw's face
    vb = world.add_body(name="vise", pos=list(ORIGIN), quat=quat)
    vb.add_geom(type=MESH, meshname="vise_body", material=iron, **vis)
    vb.add_geom(type=MESH, meshname="vise_plates", material=steel, **vis)
    for g in ((-ways / 2, 0, (base_top + bed) / 2, ways / 2, w * 0.31, (bed - base_top) / 2),
              (rear / 2, 0, (base_top + bed) / 2, rear / 2, w * 0.4, (bed - base_top) / 2),
              (jt / 2, 0, bed + jh / 2, jt / 2, w / 2, jh / 2),
              (0.04, 0, fz + c["base_h"] / 2, c["base_r"], c["base_r"], c["base_h"] / 2)):
        vb.add_geom(type=BOX, pos=list(g[:3]), size=list(g[3:]), friction=[0.9, 0.02, 0.002], **col)
    # the moving jaw, on its ways (opening: toward the crank, -x)
    jw = vb.add_body(name="vise_jaw", pos=[-Q0, 0, 0])
    jw.add_joint(name="vise_jaw", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0], range=[-Q0, 0.09 - Q0],
                 limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=40.0, armature=0.5)
    jw.add_geom(type=MESH, meshname="vise_jaw", material=iron, mass=1.5, **vis)
    jw.add_geom(type=MESH, meshname="vise_plates", pos=[0, 0, 0], quat=[0, 0, 0, 1], material=steel, **vis)
    for g in ((-jt / 2, 0, bed + jh / 2, jt / 2, w / 2, jh / 2),
              (-(jt + 0.055) / 2, 0, (bed + 0.6 * jh - 0.022) / 2, (jt + 0.055) / 2, w * 0.36,
               (bed + 0.6 * jh + 0.022) / 2)):
        jw.add_geom(type=BOX, pos=list(g[:3]), size=list(g[3:]), friction=[0.9, 0.02, 0.002], **col)
    # the speed crank, on the screw's end (turning it moves the jaw: 20 mm a turn)
    ck = jw.add_body(name="vise_crank", pos=[HUB_X, 0, 0])
    ck.add_joint(name="vise_crank", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], damping=0.05, armature=0.002,
                 frictionloss=0.2)   # (a square-thread screw: it holds where it's left, its weight won't turn it)
    ck.add_geom(type=MESH, meshname="vise_crank", material=steel, mass=0.2, **vis)
    R = c["crank_r"]
    ck.add_geom(type=BOX, size=[0.006, R / 2, 0.008], pos=[-0.006, R / 2, 0], **col)
    ck.add_site(name="crank_hub", pos=[0, 0, 0])
    kn = ck.add_body(name="vise_knob", pos=[0, R, 0])
    kn.add_joint(name="vise_knob", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], damping=0.002)
    kn.add_geom(type=MESH, meshname="vise_knob", material=black, **vis)
    L, kr = c["knob_len"], c["knob_r"]
    kn.add_geom(name="knob_c", type=CAPSULE, size=[kr, L / 2 - kr], pos=[-0.012 - L / 2, 0, 0],
                quat=[0.7071, 0, 0.7071, 0], friction=[2.0, 0.03, 0.003], condim=4, mass=0.03, **col)
    kn.add_site(name="knob_mid", pos=[-KNOB_MID, 0, 0])
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="vise_jaw", name2="vise_crank")
    eq.data[0:5] = [0, WIND * c["pitch"] / (2 * math.pi), 0, 0, 0]
    eq.solref = [0.004, 1.0]                   # (stiff: the jaw goes only where the screw takes it)
    for a, b in (("vise", "vise_jaw"), ("vise_jaw", "vise_crank"), ("vise_crank", "vise_knob"),
                 ("vise", "vise_crank"), ("vise", "vise_knob"), ("vise_block", "vise")):
        spec.add_exclude(bodyname1=a, bodyname2=b)
    # the bar to clamp (free), standing on the bench
    br = world.add_body(name="bar", pos=[*BAR_XY, T])
    br.add_freejoint(name="bar")
    br.add_geom(type=MESH, meshname="round_bar", material=alu, **vis)
    br.add_geom(type=CYL, size=[BAR["r"], BAR["h"] / 2], pos=[0, 0, BAR["h"] / 2], mass=0.2,
                friction=[1.0, 0.02, 0.002], condim=4, **col)
    br.add_site(name="bar_top", pos=[0, 0, BAR["h"]])


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S_
    from pioneer_whole_body.controller import READY, SHAPES

    m, d = wb.m, wb.d
    wb.start_ready()
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.look([0.52, -0.22, Z_S], dur=1.0)                  # the vise and the bar: all in view from here
    wb.wait(0.6)
    # 1. wind the jaws open
    _take_knob(wb)
    _crank(wb, WIND * 2 * math.pi * TURNS_OPEN)
    _let_go_knob(wb)
    notes["opened"] = round(_opening(m, d), 4)
    if _opening(m, d) < 2 * BAR["r"] + 0.02:
        raise S_.TaskFailed(f"the jaws only opened to {_opening(m, d) * 1000:.0f} mm")
    # 2. the bar between the jaws
    _take_bar(wb)
    _set_bar(wb)
    if not _between(m, d):
        raise S_.TaskFailed(f"the bar isn't between the jaws: {np.round(_local(m, d, 'bar'), 3).tolist()}")
    # 3. wind them shut on it, until the screw stalls
    _take_knob(wb)
    _crank(wb, -WIND * 2 * math.pi * TURNS_OPEN, stall=True)
    _let_go_knob(wb)
    wb.fingers("r", "relaxed", dur=0.4)
    wb.arm_joints_to("r", READY, 1.2)
    wb.settle()
    wb.wait(1.0)
    notes["gap_mm"] = round(_opening(m, d) * 1000, 1)
    if not _clamped(m, d):
        raise S_.TaskFailed(f"not clamped: the jaws at {_opening(m, d) * 1000:.1f} mm")


def _knob_point(wb, phi: float) -> np.ndarray:
    """Where the knob's middle is with the crank's arm at ``phi`` (0: across to the left; up from there)."""
    from pioneer_whole_body import skills as S_
    hub = S_.site(wb, "crank_hub")
    return hub + VISE["crank_r"] * (math.cos(phi) * E1 + math.sin(phi) * np.array([0, 0, 1.0])) - S * KNOB_MID


def _phi(wb) -> float:
    return float(wb.d.qpos[wb.m.joint("vise_crank").qposadr[0]])


def _take_knob(wb) -> None:
    """The knob taken like a pull: palm down, the fingers across it and curled, lowered so the tips go down past
    its far side, then closed round it."""
    from pioneer_whole_body import skills as S_
    from pioneer_whole_body.controller import palm_down
    r = palm_down(KNOB_YAW)
    p = _knob_point(wb, _phi(wb)) + [0, 0, HOOK_ABOVE] - r[:, 0] * HOOK_BACK
    wb.fingers("r", KNOB_PRE, dur=0.5)
    if wb.palm_pose("r", False)[0][2] < p[2] + 0.05:     # (from READY: up and out to the side first, over the vise)
        wb.arm_joints_to("r", RAISED, 0.9)
        wb.settle()
    for target, dur in ((p + [0, 0, 0.10], 1.2), (p + [0, 0, 0.03], 0.6), (p, 0.5)):
        wb.reach("r", target, r, dur)
        wb.settle()
    wb.fingers("r", "knob", amount=S_.TWIST_SQUEEZE, dur=0.5)
    wb.settle()
    wb.wait(0.15)


def _crank(wb, turn: float, stall: bool = False, lead: float = math.radians(45), dt: float = 0.08,
           limit: float = 40.0) -> None:
    """The hand round the crank's circle (the knob spins in it) by ``turn`` radians, always a little ahead of where
    the crank is (so the grip leads it, never runs off it); with ``stall``, until the screw stops (the jaws on the
    work)."""
    from pioneer_whole_body.controller import palm_down
    r = palm_down(KNOB_YAW)
    off = np.array([0, 0, HOOK_ABOVE]) - r[:, 0] * HOOK_BACK
    sign = 1.0 if turn > 0 else -1.0
    goal = _phi(wb) + turn
    t0, best, still = wb.t, _phi(wb), 0
    while sign * (goal - _phi(wb)) > math.radians(3) and wb.t - t0 < limit:
        phi = _phi(wb)
        wb.reach("r", _knob_point(wb, phi + sign * min(lead, sign * (goal - phi))) + off, r, dt)
        wb.wait(dt)
        if sign * (_phi(wb) - best) > math.radians(1):
            best, still = _phi(wb), 0
        else:
            still += 1
        if stall and still >= 8:                         # (it's stopped turning: the jaws are on the bar)
            break
    wb.reach("r", _knob_point(wb, _phi(wb)) + off, r, 0.3)
    wb.settle()


def _let_go_knob(wb) -> None:
    from pioneer_whole_body.controller import palm_down
    wb.wait(0.2)
    q, r = wb.hand_goal["r"]
    wb.fingers("r", "knob_pre", dur=0.4)
    wb.reach("r", q - r[:, 0] * 0.02, r, 0.4)
    wb.settle()
    q, r = wb.hand_goal["r"]
    wb.reach("r", q + [0, 0, 0.07], r, 0.7)
    wb.settle()


def _take_bar(wb) -> None:
    """The bar by its top (the "lid" grasp, as for a socket of its size), lifted clear."""
    from pioneer_whole_body import skills as S_
    from pioneer_whole_body.controller import palm_down
    r = palm_down(BAR_YAW)
    top = S_.site(wb, "bar_top")
    p = top - r[:, 0] * 0.016 + [0, 0, 0.004]
    wb.fingers("r", "fist", amount=0.8, dur=0.4)
    q, _ = wb.hand_goal["r"]
    for target, dur in (([q[0], q[1], top[2] + 0.12], 0.6), ([p[0], p[1], top[2] + 0.12], 1.0), (p + [0, 0, 0.05], 0.5)):
        wb.reach("r", target, r, dur)
        wb.settle()
    wb.fingers("r", "lid_pre", dur=0.4)
    wb.settle()
    wb.reach("r", p, r, 0.6)
    wb.settle()
    wb.fingers("r", "lid", amount=S_.SQUEEZE, dur=0.45)
    wb.settle()
    wb.wait(0.2)
    z0 = S_.body_pos(wb, "bar")[2]
    wb.reach("r", p + [0, 0, 0.08], r, 0.6)
    wb.settle()
    if S_.body_pos(wb, "bar")[2] - z0 < 0.05:
        raise S_.TaskFailed("the bar didn't come up")


def _set_bar(wb) -> None:
    """In from the side, between the open jaws at the height of the ways, down onto them, let go, up and out."""
    from pioneer_whole_body import skills as S_
    m, d = wb.m, wb.d
    q = _opening(m, d)
    spot = ORIGIN + S * (-q / 2) + [0, 0, VISE["bed"] + 0.002]          # the bar's bottom, in the gap
    held = S_.body_pos(wb, "bar") - wb.palm_pose("r", False)[0]
    goal = spot - held
    p, r = wb.hand_goal["r"]
    hi = goal[2] + 0.035                                  # (its bottom over the jaws' tops)
    for target, dur in (([p[0], p[1], hi], 0.6), (goal - E1 * 0.10 + [0, 0, hi - goal[2]], 1.0),
                        (goal + [0, 0, hi - goal[2]], 0.8)):
        wb.reach("r", target, r, dur)
        wb.settle()
    for _ in range(3):                                   # hovering over the gap: where the bar is (felt), nudged on
        wb.wait(0.2)
        err = spot[:2] - S_.body_pos(wb, "bar")[:2]
        if np.linalg.norm(err) < 0.002:
            break
        err *= min(1.0, 0.02 / np.linalg.norm(err))
        goal = goal + [err[0], err[1], 0]
        wb.reach("r", goal + [0, 0, hi - goal[2]], r, 0.4)
        wb.settle()
    wb.reach("r", goal + [0, 0, 0.006], r, 0.7)
    wb.settle()
    wb.fingers("r", "lid_pre", dur=0.5)
    wb.settle()
    q_, r = wb.hand_goal["r"]
    wb.reach("r", q_ + [0, 0, 0.09], r, 0.8)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["bench_vise"].plan = plan
