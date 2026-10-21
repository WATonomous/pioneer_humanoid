"""Tool chest (a workshop scene modelled to look like the real thing): a red benchtop tool chest with two drawers on
slides and black bar pulls stands on a butcher-block workbench; in its bottom drawer, a screwdriver
lies in a foam cut-out near the front. The left hand pulls the drawer open by its pull (the part that comes out is
open from above), the right hand takes the screwdriver out and lays it in the wooden tool tray on the bench, the left
pushes the drawer shut.

The props are CAD models (``assets/workshop/make_meshes.py``, build123d) in the workshop room (``workshop.py``).
The robot starts at the bench: no walking.

Steps (``progress``): 1. open the drawer, 2. take the screwdriver out, 3. lay it in the tray, 4. shut the drawer.
"""
from __future__ import annotations

import importlib.util
import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import SHAPES
from pioneer_whole_body.studio import BENCH_TOP_Z
from pioneer_whole_body.workshop import ASSETS, MESH, add_workshop, mesh, paint, texture

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE

_spec = importlib.util.spec_from_file_location("make_meshes", ASSETS / "make_meshes.py")
_mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mm)
CHEST, SD = _mm.CHEST, _mm.SD
CHEST_H = _mm.height()

CHEST_X, CHEST_Y = 0.60, 0.04          # the chest's front (x) and middle (y) on the bench
TRAY = (0.58, -0.42)
DRAWER = 0                             # the drawer the screwdriver is in (the bottom one: its pull at a good height)
OPEN = 0.21                            # how far it opens (its slides let it out 0.24)
PULL_AT = 0.07                         # ... and that far out along the bar, toward the hand's side
PULL_BACK, PULL_ABOVE = 0.055, 0.018   # the palm from the bar's axis: back along the fingers, up
PARK = np.array([0.36, 0.42, 0.97])    # the left hand while the right works: out past the chest's side
TUCKED = {**SHAPES["fist"], "th_c": 0.0, "th_f": 0.0, "th_mcp": 0.0, "th_ip": 0.0}   # fingers curled, thumb flat
SD_YAW, SD_TILT = 0.6, 0.6             # the right hand on the screwdriver's handle (it lies across the drawer):
                                       # fingers forward, turned in a little, pitched well down (round the handle's
                                       # axis: the same wrap), so the wrist and forearm ride high over the drawer's front
LEAN = 0.18                            # the hips back this far while it pulls
PULL_YAW = -0.25                       # the left hand's fingers on the pull: turned in (the arm's easy pose)
FOAM_BIT = 4                           # the collision bit only the foam and the screwdriver share
STEPS = ("open the drawer", "take the screwdriver out", "lay it in the tray", "shut the drawer")
_U_STEP = 0


def _drawer_z(i: int) -> float:
    c = CHEST
    return c["base_h"] + c["gap"] + sum(c["drawers"][:i]) + c["gap"] * i


def _slide(model, data) -> float:
    return float(data.qpos[model.joint(f"drawer{DRAWER}").qposadr[0]])


def _in_tray(model, data) -> bool:
    p = data.xpos[model.body("screwdriver").id]                  # (its tip: the handle's toward -y)
    mid = p[1] - (SD["shaft_len"] + SD["handle_len"]) / 2
    return abs(p[0] - TRAY[0]) < 0.05 and abs(mid - TRAY[1]) < 0.06 and p[2] < T + 0.03


def step(model, data) -> None:
    k = int(data.userdata[_U_STEP])
    sd = data.xpos[model.body("screwdriver").id]
    if k == 0 and _slide(model, data) > OPEN - 0.02:
        data.userdata[_U_STEP] = 1
    elif k == 1 and sd[2] > T + _drawer_z(DRAWER) + 0.08:
        data.userdata[_U_STEP] = 2
    elif k == 2 and _in_tray(model, data):
        data.userdata[_U_STEP] = 3
    elif k == 3 and _slide(model, data) < 0.004:
        data.userdata[_U_STEP] = 4


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


@scene("tool_chest", robot_pos=(0.0, 0.0, 0.0), step=step, progress=progress,
       title="Pioneer v2  |  tool chest: the drawer pulled open, the screwdriver taken out, the drawer shut",
       camera=((0.5, 0.05, 1.0), 2.1, 65, -20))
def build(spec: mujoco.MjSpec) -> None:
    add_workshop(spec)
    spec.nuserdata = 1
    world = spec.worldbody
    red = paint(spec, "chest_red", [0.70, 0.07, 0.06, 1], specular=0.45, shininess=0.55)
    pull_mat = paint(spec, "pull_black", [0.07, 0.07, 0.08, 1], specular=0.5, shininess=0.6)
    chrome = "chrome"
    rubber = paint(spec, "rubber", [0.09, 0.09, 0.10, 1], specular=0.05)
    inner = paint(spec, "drawer_inner", [0.55, 0.06, 0.05, 1], specular=0.2)
    foam = paint(spec, "foam_grey", [0.28, 0.29, 0.31, 1], specular=0.02)
    vis = dict(contype=0, conaffinity=0)
    c = CHEST
    d, w, t = c["depth"], c["width"], c["wall"]

    # the chest (fixed on the bench): meshes for the look, boxes for contact
    chest = world.add_body(name="chest", pos=[CHEST_X, CHEST_Y, T])
    for m, mat in (("chest_body", red), ("chest_mat", rubber), ("chest_handle", chrome)):
        mesh(spec, m)
        chest.add_geom(type=MESH, meshname=m, material=mat, **vis)
    for g in ((d / 2, 0, CHEST_H - t / 2, d / 2, w / 2, t / 2),            # top
              (d / 2, w / 2 - t / 2, CHEST_H / 2, d / 2, t / 2, CHEST_H / 2),
              (d / 2, -w / 2 + t / 2, CHEST_H / 2, d / 2, t / 2, CHEST_H / 2),
              (d - t / 2, 0, CHEST_H / 2, t / 2, w / 2, CHEST_H / 2),
              (d / 2, 0, c["base_h"] / 2, d / 2, w / 2, c["base_h"] / 2)):
        chest.add_geom(type=BOX, pos=list(g[:3]), size=list(g[3:]), rgba=[0, 0, 0, 0], group=3)
    # the drawers, on slides (pulled out toward the robot, -x)
    for i, h in enumerate(c["drawers"]):
        dr = world.add_body(name=f"drawer{i}", pos=[CHEST_X, CHEST_Y, T + _drawer_z(i)])
        dr.add_joint(name=f"drawer{i}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0], range=[0, 0.24],
                     limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=12.0, frictionloss=5.0, armature=0.05)
        mesh(spec, f"drawer{i}")
        dr.add_geom(type=MESH, meshname=f"drawer{i}", material=red, mass=0.6, **vis)
        kz = h - c["pull_down"]                       # the bar pull, near the drawer's top
        mesh(spec, f"pull{i}", "drawer_pull")
        dr.add_geom(type=MESH, meshname=f"pull{i}", pos=[0, 0, kz], material=pull_mat, **vis)
        pl, pr, po = c["pull_len"], c["pull_r"], c["pull_out"]
        dr.add_geom(name=f"pull{i}_c", type=CAPSULE, size=[pr, pl / 2 - pr], pos=[-po, 0, kz],
                    quat=[0.7071, 0.7071, 0, 0], rgba=[0, 0, 0, 0], group=3,
                    friction=[2.0, 0.03, 0.003], condim=4, mass=0.03)
        for y in (-pl / 2 + 0.014, pl / 2 - 0.014):
            dr.add_geom(type=BOX, size=[po / 2, 0.0065, 0.0065], pos=[-po / 2, y, kz], rgba=[0, 0, 0, 0], group=3)
        bw = w - 2 * t - 0.01
        dd = c["drawer_depth"]
        bh = _mm.box_h(i)
        for g in ((0.006, 0, h / 2, 0.006, bw / 2, h / 2), (0.012 + dd / 2, 0, 0.0028, dd / 2, bw / 2, 0.0008),
                  (0.012 + dd / 2, bw / 2 - 0.002, 0.002 + bh / 2, dd / 2, 0.001, bh / 2),
                  (0.012 + dd / 2, -bw / 2 + 0.002, 0.002 + bh / 2, dd / 2, 0.001, bh / 2),
                  (0.012 + dd - 0.001, 0, 0.002 + bh / 2, 0.001, bw / 2, bh / 2)):
            dr.add_geom(type=BOX, pos=list(g[:3]), size=list(g[3:]), rgba=[0, 0, 0, 0], group=3, mass=0.05)
        dr.add_site(name=f"pull{i}", pos=[-po, 0, kz + pr])            # the top of the bar, mid-way
        spec.add_exclude(bodyname1="chest", bodyname2=f"drawer{i}")
        if i == DRAWER:                                      # the foam insert with the screwdriver's cut-out
            mesh(spec, "foam_insert")
            dr.add_geom(type=MESH, meshname="foam_insert", pos=[0.012, 0, 0.0035], material=foam, **vis)
            floor = 0.0035 + _mm.CUT_Z
            fl, fh, cx = _mm.FOAM_L, _mm.FOAM_H, 0.012 + _mm.CUT_X
            soft = dict(rgba=[0, 0, 0, 0], group=3, contype=FOAM_BIT, conaffinity=FOAM_BIT)   # (the foam: only the
            dr.add_geom(type=BOX, pos=[0.012 + fl / 2, 0, 0.0035 + _mm.CUT_Z / 2],       # screwdriver rests on it,
                        size=[fl / 2, bw / 2 - 0.004, _mm.CUT_Z / 2], **soft)             # fingers press into it)
            for sx in (-1, 1):                               # the cut-out's sides (they carry it in and out)
                dr.add_geom(type=BOX, pos=[cx + sx * 0.031, 0, 0.0035 + fh / 2],
                            size=[0.01, bw / 2 - 0.004, fh / 2], **soft)
    spec.add_exclude(bodyname1="drawer0", bodyname2="drawer1")
    # the screwdriver, in its cut-out (its frame: along +y from the tip; turned round, the handle toward -y)
    L = SD["shaft_len"] + SD["handle_len"]
    sd = world.add_body(name="screwdriver", pos=[CHEST_X + 0.012 + _mm.CUT_X, CHEST_Y + L / 2,
                                                 T + _drawer_z(DRAWER) + floor + 0.0005], quat=[0, 0, 0, 1])
    sd.add_freejoint(name="screwdriver")
    for m, mat in (("sd_handle", paint(spec, "sd_black", [0.06, 0.06, 0.07, 1], specular=0.3, shininess=0.5)),
                   ("sd_grip", paint(spec, "sd_yellow", [0.95, 0.70, 0.05, 1], specular=0.25, shininess=0.3)),
                   ("sd_shaft", chrome)):
        mesh(spec, m)
        sd.add_geom(type=MESH, meshname=m, material=mat, **vis)
    r = SD["handle_r"]
    sd.add_geom(name="sd_handle_c", type=CAPSULE, size=[r - 0.001, SD["handle_len"] / 2 - r], pos=[0, SD["shaft_len"] + SD["handle_len"] / 2, r],
                quat=[0.7071, 0.7071, 0, 0], rgba=[0, 0, 0, 0], group=3, mass=0.09, friction=[1.2, 0.02, 0.002],
                condim=4, contype=1 | FOAM_BIT, conaffinity=1 | FOAM_BIT)
    sd.add_geom(type=CAPSULE, size=[SD["shaft_r"], SD["shaft_len"] / 2], pos=[0, SD["shaft_len"] / 2, r],
                quat=[0.7071, 0.7071, 0, 0], rgba=[0, 0, 0, 0], group=3, mass=0.03, contype=1 | FOAM_BIT,
                conaffinity=1 | FOAM_BIT)
    sd.add_site(name="sd_grip", pos=[0, SD["shaft_len"] + SD["handle_len"] * 0.55, 2 * r])
    # the tool tray
    tray = world.add_body(name="tray", pos=[TRAY[0], TRAY[1], T])
    mesh(spec, "tool_tray")
    tray.add_geom(type=MESH, meshname="tool_tray", material=paint(spec, "tray_wood", [0.66, 0.50, 0.33, 1],
                                                                   specular=0.15), **vis)
    for g in ((0, 0, 0.004, 0.07, 0.15, 0.004), (0.066, 0, 0.02, 0.004, 0.15, 0.02), (-0.066, 0, 0.02, 0.004, 0.15, 0.02),
              (0, 0.146, 0.02, 0.07, 0.004, 0.02), (0, -0.146, 0.02, 0.07, 0.004, 0.02)):
        tray.add_geom(type=BOX, pos=list(g[:3]), size=list(g[3:]), rgba=[0, 0, 0, 0], group=3)


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.controller import READY

    m, d = wb.m, wb.d
    wb.start_ready()                                 # (it starts at the bench, arms up)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.look([CHEST_X - 0.05, CHEST_Y - 0.12, T + 0.08], dur=1.0)      # the chest and the tray: all in view from here
    wb.wait(0.8)
    # 1. the left hand takes the bottom drawer's pull and pulls it open
    _take_pull(wb, "l")
    _move(wb, "l", -(OPEN + 0.03), hips=-LEAN)
    _let_go_pull(wb, "l", park=PARK, hips=LEAN)
    notes["opened"] = round(_slide(m, d), 3)
    if _slide(m, d) < OPEN - 0.02:
        raise S.TaskFailed(f"the drawer's open {_slide(m, d):.3f} m")
    # 2. the right hand takes the screwdriver by its handle and lifts it out; 3. lays it in the tray
    _pick(wb)
    _place(wb)
    notes["in_tray"] = _in_tray(m, d)
    # 4. the left hand takes the pull again and pushes the drawer shut
    wb.pelvis(x=wb.cmd["base_x"] - LEAN, dur=1.0)
    wb.settle()
    _take_pull(wb, "l")
    _move(wb, "l", _slide(m, d) + 0.006, hips=LEAN)
    _let_go_pull(wb, "l")
    wb.wait(1.0)
    notes["shut"] = round(_slide(m, d), 4)
    if not _in_tray(m, d):
        raise S.TaskFailed("the screwdriver isn't in the tray")
    if _slide(m, d) > 0.004:
        raise S.TaskFailed(f"the drawer's still {_slide(m, d):.3f} m open")


def _pick(wb) -> None:
    """The right hand takes the screwdriver by its handle and lifts it out of the open drawer: in over the drawer's
    front and side with the fingers curled and the thumb flat (nothing hanging down or sticking out), down, the
    fingers opened (pointing left across the handle) and closed round it; lifted a little, back out from under the
    top drawer, up."""
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.controller import palm_down
    r = palm_down(SD_YAW, SD_TILT)
    axis = S.site(wb, "sd_grip") - [0, 0, SD["handle_r"]]
    p = axis - r @ [S.GRASP_BACK, 0, 0.004 + SD["handle_r"]]          # (the "lid" grasp, as ``knob_pose``)
    wb.fingers("r", TUCKED, dur=0.5)
    q, _ = wb.palm_pose("r")
    r0 = palm_down(SD_YAW)                               # (level till it's over the drawer, then pitched down,
    pre = p + [0.012, 0, 0.045]                          # the forearm with it (there's no wrist): then down in
    x0 = pre[0] - 0.06                                                                 # round the handle)
    for target, rot, dur in (([q[0], q[1], T + 0.155], r0, 0.7), ([x0, CHEST_Y - CHEST["width"] / 2, T + 0.155], r0, 1.0),
                             ([x0, pre[1], T + 0.155], r0, 0.8), ([x0, pre[1], T + 0.155], r, 0.6), (pre, r, 0.7)):
        wb.reach("r", target, rot, dur)
        wb.settle()
    wb.fingers("r", "lid_pre", dur=0.5)
    wb.settle()
    wb.reach("r", p, r, 0.6)
    wb.settle()
    wb.fingers("r", "lid", amount=S.SQUEEZE, dur=0.5)
    wb.settle()
    wb.wait(0.2)
    z0 = S.body_pos(wb, "screwdriver")[2]
    for target, dur in ((p + [0, 0, 0.02], 0.4), (p + [-0.06, 0, 0.03], 0.6), (p + [-0.06, 0, 0.12], 0.7)):
        wb.reach("r", target, r, dur)                    # (up a little, back from under the top drawer, up)
        wb.settle()
    if S.body_pos(wb, "screwdriver")[2] - z0 < 0.06:
        raise S.TaskFailed("the screwdriver didn't come up")


def _place(wb) -> None:
    """The right hand carries the screwdriver over the tray, lowers it in, lets go, goes back to READY."""
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.controller import READY
    held = S.body_pos(wb, "screwdriver") - wb.palm_pose("r", False)[0]          # (felt: where it hangs in the hand)
    goal = np.array([TRAY[0], TRAY[1] + (SD["shaft_len"] + SD["handle_len"]) / 2, T + 0.0085]) - held
    q, rr = wb.hand_goal["r"]
    for target, dur in ((np.array([*goal[:2], q[2]]), 1.6), (goal + [0, 0, 0.03], 0.8), (goal, 0.6)):
        wb.reach("r", target, rr, dur)
        wb.settle()
    wb.fingers("r", "lid_pre", dur=0.9)
    wb.settle()
    q, rr = wb.hand_goal["r"]
    wb.reach("r", q + [0, 0, 0.10], rr, 0.8)
    wb.settle()
    wb.fingers("r", "relaxed", dur=0.4)
    wb.arm_joints_to("r", READY, 1.0)
    wb.settle()


def _take_pull(wb, side: str):
    """Take the drawer's bar pull from above, the way the arm reaches best: palm down, the fingers forward (turned in
    a little, as the arm hangs) and already curled ("knob_pre": tips 77 mm out along the fingers and 47 mm below the
    palm), lowered so the tips go down between the bar and the drawer's face, then closed ("knob": 55 mm out) onto
    the bar's far side, the thumb on its near side: a hook that pulls, and a thumb that pushes."""
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.controller import palm_down
    r = palm_down(PULL_YAW if side == "l" else -PULL_YAW)
    axis = S.site(wb, f"pull{DRAWER}") + [0, PULL_AT if side == "l" else -PULL_AT, -CHEST["pull_r"]]
    p = axis + [0, 0, PULL_ABOVE] - r[:, 0] * PULL_BACK
    wb.fingers(side, "knob_pre", dur=0.5)
    wb.reach(side, p + [-0.08, 0, 0.025], r, 1.2)        # (in from the front, the tips under the top drawer's face
    wb.settle()                                          # and over the bar, then down behind it)
    wb.reach(side, p + [0, 0, 0.025], r, 0.6)
    wb.settle()
    wb.reach(side, p, r, 0.5)
    wb.settle()
    wb.fingers(side, "knob", amount=S.TWIST_SQUEEZE, dur=0.5)
    wb.settle()
    wb.wait(0.15)


def _move(wb, side: str, dx: float, hips: float = 0.0, step: float = 0.01, dt: float = 0.12) -> None:
    """The hand straight along x by ``dx`` in 1 cm steps (the drawer's slide); the hips go ``hips`` along x with it
    (feet planted: leaning back to pull it out further than the arm reaches)."""
    p, r = wb.hand_goal[side]
    n = max(1, math.ceil(abs(dx) / step))
    if hips:
        wb.pelvis(x=wb.cmd["base_x"] + hips, dur=n * dt)
    for k in range(1, n + 1):
        wb.reach(side, p + [dx * k / n, 0, 0], r, dt)
        wb.settle()


def _let_go_pull(wb, side: str, park=None, hips: float = 0.0) -> None:
    """The fingers eased off ("knob_pre"), the hand straight up (the curled tips come up out from behind the bar),
    a fist (nothing sticks out); then out sideways in front of the open drawer and held at ``park`` (a point, out to
    the side of the chest) while the hips come ``hips`` forward (READY would sweep through the open drawer), or back
    to READY."""
    from pioneer_whole_body.controller import READY
    wb.wait(0.2)
    wb.fingers(side, "knob_pre", dur=0.4)
    wb.settle()
    q, r = wb.hand_goal[side]
    wb.reach(side, q + [0, 0, 0.05], r, 0.7)
    wb.settle()
    if park is None:
        wb.fingers(side, "relaxed", dur=0.4)
        wb.arm_joints_to(side, READY, 1.0)
        wb.settle()
        return
    wb.fingers(side, "fist", amount=0.8, dur=0.5)
    wb.settle()
    q, r = wb.hand_goal[side]
    wb.reach(side, [q[0], park[1], park[2]], r, 0.8)
    wb.settle()
    wb.reach(side, park, r, 0.6)
    if hips:
        wb.pelvis(x=wb.cmd["base_x"] + hips, dur=1.0)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["tool_chest"].plan = plan
