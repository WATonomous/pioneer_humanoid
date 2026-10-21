"""Parts cabinet (a workshop scene modelled on the real thing): a steel small-parts cabinet of six clear plastic
drawers (two rows of three, each with a moulded pull and a printed label: M12 nuts, 1" couplings, M16 washers ...)
stands on the workbench, the hardware visible through the drawers; a stainless parts dish beside it. The left hand
hooks the pull of the 1" coupling drawer, draws it out to its stop and holds out of the way; the right hand takes a
brass pipe coupling out of it by its rim and drops it in the dish; the left hand pushes the drawer shut.

The props are CAD models (``assets/workshop/make_meshes.py``, build123d; the labels ``make_textures.py``) in the
workshop room (``workshop.py``). The robot starts at the bench, its legs and base still: no walking.

Steps (``progress``): 1. open the drawer, 2. take a coupling out, 3. put it in the dish, 4. shut the drawer.
"""
from __future__ import annotations

import importlib.util
import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import READY, SHAPES
from pioneer_whole_body.studio import BENCH_TOP_Z
from pioneer_whole_body.workshop import ASSETS, MESH, add_workshop, mesh, paint, texture

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

_spec = importlib.util.spec_from_file_location("make_meshes", ASSETS / "make_meshes.py")
_mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mm)
CAB, COUPLING, DISH = _mm.CABINET, _mm.COUPLING, _mm.DISH
CAB_W, CAB_H = _mm.cabinet_size()
_tspec = importlib.util.spec_from_file_location("make_textures", ASSETS / "make_textures.py")
_mt = importlib.util.module_from_spec(_tspec)
_tspec.loader.exec_module(_mt)
LABELS = _mt.LABELS

CAB_X, CAB_Y = 0.73, 0.05              # the cabinet's front (x) and middle (y) on the bench
TARGET = (1, 1)                        # the drawer (column, row): 1" couplings, left of middle, the top row (open above)
OPEN = 0.22                            # how far it's drawn out (to its stop: the arm alone, the base stays put)
OPENED = 0.12                          # ... and open at least this far once the hand lets go
RELEASE_BACK = 0.04                    # the hand backs off this far as the fingers ease off the pull
TAKE_AT = (0.055, -0.06)               # the coupling taken: back from the drawer's front, across (toward the right
                                       # hand: the left one's on the pull's other end)
DISH_XY = (0.54, -0.36)
PULL_YAW = -0.25                       # the left hand's fingers on the pull: turned in (the arm's easy pose)
PULL_AT = 0.05                         # ... that far along the bar toward the hand's side
PULL_BACK, PULL_ABOVE = 0.055, 0.018   # the palm from the bar's axis: back along the fingers, up
PARK = np.array([0.36, 0.42, 0.97])    # the left hand while the right works: out to its side
GRIP_YAW, GRIP_TILT = 0.1, 0.25       # the right hand on the coupling: the fingers forward (the thumb comes in from
                                       # the side, clear of the drawer's front), pitched a little (not down to its floor)
GRIP_AT = np.array([0.02, 0.0, 0.015])  # the coupling's top in the palm's frame: the "lid" grasp, pitched
TUCKED = {**SHAPES["fist"], "th_c": 0.0, "th_f": 0.0, "th_mcp": 0.0, "th_ip": 0.0}   # fingers curled, thumb flat
RAISED = {**READY, "shoulder_pitch": -0.45, "shoulder_roll": 0.8}    # the right hand up and out to the side
STEPS = ("open the drawer", "take a coupling out", "put it in the dish", "shut the drawer")
_U_STEP = 0


def _drawer(k: int, j: int) -> str:
    return f"drawer_{k}{j}"


def _slide(model, data) -> float:
    return float(data.qpos[model.joint(_drawer(*TARGET)).qposadr[0]])


def _in_dish(model, data) -> bool:
    p = data.xipos[model.body("coupling").id]           # (its middle: standing or lying in it)
    return np.hypot(p[0] - DISH_XY[0], p[1] - DISH_XY[1]) < DISH["od"] / 2 - 0.015 and p[2] < T + 0.035


def reset(model, data, rng) -> None:
    """The coupling a little off its spot in the drawer, turned any way."""
    a = model.joint("coupling").qposadr[0]
    th = rng.uniform(0, 2 * math.pi)
    data.qpos[a] += rng.uniform(-0.003, 0.003)
    data.qpos[a + 1] += rng.uniform(-0.003, 0.003)
    data.qpos[a + 3:a + 7] = [math.cos(th / 2), 0, 0, math.sin(th / 2)]


def step(model, data) -> None:
    k = int(data.userdata[_U_STEP])
    z = data.xpos[model.body("coupling").id][2]
    if k == 0 and _slide(model, data) > OPENED:
        data.userdata[_U_STEP] = 1
    elif k == 1 and z > T + CAB_H + 0.01:
        data.userdata[_U_STEP] = 2
    elif k == 2 and _in_dish(model, data):
        data.userdata[_U_STEP] = 3
    elif k == 3 and _slide(model, data) < 0.004:
        data.userdata[_U_STEP] = 4


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def _contents(spec, body, kind: str, x0: float, y0: float, z0: float, steel: str, brass: str, rng) -> None:
    """Visual hardware lying in a drawer (fixed: only the coupling taken is free)."""
    vis = dict(contype=0, conaffinity=0, group=1)
    bw = CAB["dw"] - 0.02
    if kind in ("nut", "washer"):
        n, size = (7, 0.025) if kind == "nut" else (8, 0.027)
        for _ in range(n):
            x = x0 + rng.uniform(0.02, CAB["bin_depth"] - 0.02)
            y = y0 + rng.uniform(-bw / 2 + size / 2, bw / 2 - size / 2)
            a = rng.uniform(0, math.pi)
            body.add_geom(type=MESH, meshname=f"{kind}_m", pos=[x, y, z0], quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)],
                          material=steel, **vis)
    elif kind == "bolt":
        for i in range(4):
            a = rng.uniform(-0.25, 0.25)
            body.add_geom(type=MESH, meshname="bolt_m", pos=[x0 + 0.02 + 0.035 * i, y0 - 0.025 + rng.uniform(-0.01, 0.01), z0],
                          quat=[math.cos((a + math.pi / 2) / 2), 0, 0, math.sin((a + math.pi / 2) / 2)], material=steel, **vis)
    elif kind in ("coupling", "small_coupling"):
        for i, (dx, dy) in enumerate(((0.05, 0.024), (0.05, -0.024), (0.10, 0.024), (0.10, -0.024), (0.15, 0.0))):
            body.add_geom(type=MESH, meshname=kind + "_m", pos=[x0 + dx, y0 + dy, z0], material=brass, **vis)


@scene("parts_cabinet", robot_pos=(0.0, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer v2  |  parts cabinet: a drawer pulled open, a brass coupling taken out, the drawer shut",
       camera=((0.55, 0.0, 0.97), 1.8, 112, -22))
def build(spec: mujoco.MjSpec) -> None:
    add_workshop(spec)
    spec.nuserdata = 1
    world = spec.worldbody
    rng = np.random.default_rng(3)
    vis = dict(contype=0, conaffinity=0, group=1)
    col = dict(rgba=[0, 0, 0, 0], group=3)
    frame_mat = paint(spec, "cabinet_steel", [0.20, 0.30, 0.42, 1], specular=0.35, shininess=0.45)
    clear = paint(spec, "drawer_clear", [0.86, 0.90, 0.96, 0.32], specular=0.7, shininess=0.8)
    handle_mat = paint(spec, "handle_grey", [0.30, 0.32, 0.35, 1], specular=0.3, shininess=0.4)
    steel = paint(spec, "zinc_steel", [0.72, 0.73, 0.76, 1], specular=0.7, shininess=0.6)
    brass = paint(spec, "brass", [0.80, 0.62, 0.26, 1], specular=0.8, shininess=0.7)
    for name, file, sc in (("cabinet_frame", None, 1), ("cabinet_drawer", None, 1), ("cabinet_handle", None, 1),
                           ("coupling_m", "brass_coupling", 1), ("small_coupling_m", "brass_coupling", 0.8),
                           ("nut_m", "hex_nut", 1), ("washer_m", "washer", 1), ("bolt_m", "hex_bolt", 1),
                           ("parts_dish", None, 1)):
        mesh(spec, name, file, scale=sc)
    c = CAB
    dw, dh, gap, wall, depth = c["dw"], c["dh"], c["gap"], c["wall"], c["depth"]

    # the cabinet (fixed): the frame's mesh; boxes for its walls and the shelves and dividers between the cells
    cab = world.add_body(name="cabinet", pos=[CAB_X, CAB_Y, T])
    cab.add_geom(type=MESH, meshname="cabinet_frame", material=frame_mat, **vis)
    for g in ((depth / 2, 0, CAB_H - c["top"] / 2, depth / 2, CAB_W / 2, c["top"] / 2),
              (depth / 2, 0, c["plinth"] / 2, depth / 2, CAB_W / 2, c["plinth"] / 2),
              (depth / 2, CAB_W / 2 - wall / 2, CAB_H / 2, depth / 2, wall / 2, CAB_H / 2),
              (depth / 2, -CAB_W / 2 + wall / 2, CAB_H / 2, depth / 2, wall / 2, CAB_H / 2),
              (depth - 0.004, 0, CAB_H / 2, 0.004, CAB_W / 2, CAB_H / 2)):
        cab.add_geom(type=BOX, pos=list(g[:3]), size=list(g[3:]), **col)
    for j in range(1, c["rows"]):
        z = c["plinth"] + j * (dh + gap) - gap / 2
        cab.add_geom(type=BOX, pos=[depth / 2, 0, z], size=[depth / 2, CAB_W / 2, gap / 2], **col)
    for k in range(1, c["cols"]):
        y = CAB_W / 2 - wall - k * (dw + gap) + gap / 2
        cab.add_geom(type=BOX, pos=[depth / 2, y, CAB_H / 2], size=[depth / 2, gap / 2, CAB_H / 2], **col)
    kinds = {"NUT": "nut", "WASHER": "washer", "x": "bolt", '1"': "coupling", '3/4"': "small_coupling"}
    for (k, j), text in LABELS.items():
        y, z = _mm.cabinet_cell(k, j)
        lab = texture(spec, f"label_{k}{j}", mat=f"label_{k}{j}_m", specular=0.1)
        if (k, j) == TARGET:
            continue
        # the other drawers: shut, fixed in the frame (their fronts, pulls and labels collide/show; their contents)
        cab.add_geom(type=MESH, meshname="cabinet_drawer", pos=[0, y, z], material=clear, **vis)
        cab.add_geom(type=MESH, meshname="cabinet_handle", pos=[0, y, z], material=handle_mat, **vis)
        cab.add_geom(type=BOX, size=[0.0006, 0.026, 0.0072], pos=[-0.0012, y, z + dh - 0.0105], material=lab, **vis)
        cab.add_geom(type=BOX, size=[c["front_t"] / 2, dw / 2 - 0.002, dh / 2 - 0.002],
                     pos=[c["front_t"] / 2, y, z + dh / 2], **col)
        cab.add_geom(type=CAPSULE, size=[c["handle_r"], c["handle_w"] / 2 - c["handle_r"]],
                     pos=[-c["handle_out"], y, z + c["handle_z"]], quat=[0.7071, 0.7071, 0, 0], **col)
        kind = next((v for key, v in kinds.items() if key in text), None)
        if kind:
            _contents(spec, cab, kind, c["front_t"], y, z + 0.0035, steel, brass, rng)

    # the 1" coupling drawer, on its slide (drawn out toward the robot, -x)
    k, j = TARGET
    y, z = _mm.cabinet_cell(k, j)
    dr = world.add_body(name=_drawer(k, j), pos=[CAB_X, CAB_Y + y, T + z])
    dr.add_joint(name=_drawer(k, j), type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0], range=[0, 0.21],
                 limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=8.0, frictionloss=5.0, armature=0.02)
    dr.add_geom(type=MESH, meshname="cabinet_drawer", material=clear, mass=0.15, **vis)
    dr.add_geom(type=MESH, meshname="cabinet_handle", material=handle_mat, **vis)
    dr.add_geom(type=BOX, size=[0.0006, 0.026, 0.0072], pos=[-0.0012, 0, dh - 0.0105], material=f"label_{k}{j}_m", **vis)
    hw, ho, hr, hz = c["handle_w"], c["handle_out"], c["handle_r"], c["handle_z"]
    dr.add_geom(name="pull_c", type=CAPSULE, size=[hr, hw / 2 - hr], pos=[-ho, 0, hz], quat=[0.7071, 0.7071, 0, 0],
                friction=[2.0, 0.03, 0.003], condim=4, mass=0.01, **col)
    for s_ in (-1, 1):
        dr.add_geom(type=BOX, size=[ho / 2, 0.004, 0.0055], pos=[-ho / 2, s_ * (hw / 2 - 0.004), hz], **col)
    bw, bd, bh, ft = dw - 0.006, c["bin_depth"], c["bin_h"], c["front_t"]
    for g in ((ft / 2, 0, dh / 2, ft / 2, (dw - 0.003) / 2, (dh - 0.003) / 2),
              (ft + (bd - ft) / 2, 0, 0.0024, (bd - ft) / 2, bw / 2, 0.0009),
              (ft + (bd - ft) / 2, bw / 2 - 0.0009, 0.0015 + bh / 2, (bd - ft) / 2, 0.0009, bh / 2),
              (ft + (bd - ft) / 2, -bw / 2 + 0.0009, 0.0015 + bh / 2, (bd - ft) / 2, 0.0009, bh / 2),
              (bd - 0.0009, 0, 0.0015 + bh / 2, 0.0009, bw / 2, bh / 2)):
        dr.add_geom(type=BOX, pos=list(g[:3]), size=list(g[3:]), friction=[0.8, 0.02, 0.002], mass=0.02, **col)
    for dx, dy in ((0.19, 0.024), (0.19, -0.024)):                       # the couplings that stay (at the back)
        dr.add_geom(type=MESH, meshname="coupling_m", pos=[ft + dx, dy, 0.0033], material=brass, **vis)
        dr.add_geom(type=CYL, size=[COUPLING["od"] / 2, COUPLING["length"] / 2],
                    pos=[ft + dx, dy, 0.0033 + COUPLING["length"] / 2], mass=0.01, **col)
    dr.add_site(name="pull", pos=[-ho, 0, hz + hr])
    spec.add_exclude(bodyname1="cabinet", bodyname2=_drawer(k, j))
    # the coupling taken (free), standing at the front of the drawer
    cp = world.add_body(name="coupling", pos=[CAB_X + ft + TAKE_AT[0], CAB_Y + y + TAKE_AT[1], T + z + 0.0034])
    cp.add_freejoint(name="coupling")
    cp.add_geom(type=MESH, meshname="coupling_m", material=brass, **vis)
    cp.add_geom(type=CYL, size=[COUPLING["od"] / 2, COUPLING["length"] / 2], pos=[0, 0, COUPLING["length"] / 2],
                mass=0.14, friction=[1.2, 0.02, 0.002], condim=4, **col)
    cp.add_site(name="coupling_top", pos=[0, 0, COUPLING["length"]])

    # the parts dish: a stainless pan on the bench
    dish = world.add_body(name="dish", pos=[DISH_XY[0], DISH_XY[1], T])
    dish.add_geom(type=MESH, meshname="parts_dish", material=paint(spec, "stainless", [0.80, 0.81, 0.83, 1],
                                                                    specular=0.9, shininess=0.8), **vis)
    r, h = DISH["od"] / 2, DISH["h"]
    dish.add_geom(type=CYL, size=[r - 0.004, 0.0008], pos=[0, 0, 0.0008], friction=[0.8, 0.02, 0.002], **col)
    for i in range(12):
        a = 2 * math.pi * i / 12
        dish.add_geom(type=BOX, size=[0.0015, r * math.tan(math.pi / 12), h / 2],
                      pos=[r * math.cos(a), r * math.sin(a), h / 2], quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)],
                      **col)


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.controller import READY

    m, d = wb.m, wb.d
    wb.start_ready()
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.look([CAB_X - 0.10, -0.08, T + 0.06], dur=1.0)     # the cabinet and the dish: all in view from here
    wb.wait(0.8)
    # 1. the left hand hooks the drawer's pull, draws it out and holds out of the way
    _take_pull(wb)
    _move(wb, -OPEN)
    _let_go_pull(wb, park=PARK)
    notes["opened"] = round(_slide(m, d), 3)
    if _slide(m, d) < OPENED:
        raise S.TaskFailed(f"the drawer's open {_slide(m, d):.3f} m")
    # 2. the right hand takes a coupling out by its rim; 3. sets it in the dish
    _pick(wb)
    _place(wb)
    notes["in_dish"] = bool(_in_dish(m, d))
    # 4. the left hand hooks the pull again and pushes the drawer shut
    _take_pull(wb)
    _move(wb, _slide(m, d) + 0.02)                       # (a little past shut: it ends against the frame)
    _let_go_pull(wb)
    wb.wait(1.0)
    notes["shut"] = round(_slide(m, d), 4)
    if not _in_dish(m, d):
        raise S.TaskFailed("the coupling isn't in the dish")
    if _slide(m, d) > 0.004:
        raise S.TaskFailed(f"the drawer's still {_slide(m, d):.3f} m open")


def _take_pull(wb) -> None:
    """The left hand takes the pull from above, palm down, the fingers forward and curled ("knob_pre"), lowered so
    the tips go down between the bar and the drawer's face, then closed onto the bar's far side: a hook that pulls,
    a thumb that pushes."""
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.controller import palm_down
    r = palm_down(PULL_YAW)
    axis = S.site(wb, "pull") + [0, PULL_AT, -CAB["handle_r"]]
    p = axis + [0, 0, PULL_ABOVE] - r[:, 0] * PULL_BACK
    wb.fingers("l", "knob_pre", dur=0.5)
    for target, dur in ((p + [-0.08, 0, 0.035], 1.2), (p + [0, 0, 0.035], 0.6), (p, 0.5)):   # (in under the cabinet's top
                                                                                        # edge, the tips over the bar)
        wb.reach("l", target, r, dur)
        wb.settle()
    wb.fingers("l", "knob", amount=S.TWIST_SQUEEZE, dur=0.5)
    wb.settle()
    wb.wait(0.15)


def _move(wb, dx: float, step: float = 0.01, dt: float = 0.12) -> None:
    """The left hand straight along x by ``dx`` in 1 cm steps (the drawer's slide): the arm alone."""
    p, r = wb.hand_goal["l"]
    n = max(1, math.ceil(abs(dx) / step))
    for k in range(1, n + 1):
        wb.reach("l", p + [dx * k / n, 0, 0], r, dt)
        wb.wait(dt)
    wb.settle()


def _let_go_pull(wb, park=None) -> None:
    """The fingers eased off while the hand backs off as far as their tips open out (so they don't push the drawer's
    face), straight up (the curled tips come up from behind the bar); then held at ``park`` (out to the side, clear
    of the open drawer), or back to READY."""
    from pioneer_whole_body.controller import READY
    wb.wait(0.2)
    q, r = wb.hand_goal["l"]
    wb.fingers("l", "knob_pre", dur=0.4)
    wb.reach("l", q - r[:, 0] * RELEASE_BACK, r, 0.4)
    wb.settle()
    q, r = wb.hand_goal["l"]
    wb.reach("l", q + [0, 0, 0.06], r, 0.7)
    wb.settle()
    if park is None:
        wb.fingers("l", "relaxed", dur=0.4)
        wb.arm_joints_to("l", READY, 1.0)
        wb.settle()
        return
    wb.fingers("l", "fist", amount=0.8, dur=0.5)
    q, r = wb.hand_goal["l"]
    wb.reach("l", [q[0], park[1], park[2]], r, 0.8)
    wb.settle()
    wb.reach("l", park, r, 0.6)
    wb.settle()


def _pick(wb) -> None:
    """The right hand takes the coupling out of the open drawer: up and out to the side, in over the drawer with the
    fingers curled and the thumb flat, pitched down over the coupling (the forearm with it: the arm has no wrist),
    down onto it with the fingers open, closed round its rim ("lid"); lifted out."""
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.controller import palm_down
    r = palm_down(GRIP_YAW, GRIP_TILT)
    p = S.site(wb, "coupling_top") - r @ GRIP_AT
    pre = p - r[:, 2] * 0.04                             # (back off along the palm's normal)
    hi = p[2] + 0.07
    r0 = palm_down(GRIP_YAW)
    wb.fingers("r", TUCKED, dur=0.5)
    wb.arm_joints_to("r", RAISED, 0.9)                   # (up out of READY first: there the shoulder's at its limit)
    wb.settle()
    for target, rot, dur in (([pre[0], pre[1], hi], r0, 1.2), ([pre[0], pre[1], hi], r, 0.6), (pre, r, 0.6)):
        wb.reach("r", target, rot, dur)
        wb.settle()
    aim = pre
    for _ in range(2):                                   # (where the palm is, not where it was sent: the arm sags
        err = wb.palm_pose("r", False)[0] - pre          # this far out, so the aim's nudged by the miss)
        aim = aim - err
        wb.reach("r", aim, r, 0.4)
        wb.settle()
    wb.fingers("r", "lid_pre", dur=0.5)
    wb.settle()
    wb.reach("r", p + (aim - pre), r, 0.6)
    wb.settle()
    wb.fingers("r", "lid", amount=S.SQUEEZE, dur=0.5)
    wb.settle()
    wb.wait(0.2)
    z0 = S.body_pos(wb, "coupling")[2]
    for target, dur in ((p + [0, 0, 0.04], 0.5), (p + [-0.04, 0, 0.11], 0.7)):
        wb.reach("r", target, r, dur)
        wb.settle()
    if S.body_pos(wb, "coupling")[2] - z0 < 0.08:
        raise S.TaskFailed("the coupling didn't come up")


def _place(wb) -> None:
    """The right hand carries the coupling over the dish, sets it down in it, lets go, goes back to READY."""
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.controller import READY
    held = S.body_pos(wb, "coupling") - wb.palm_pose("r", False)[0]       # (felt: where it hangs in the hand)
    goal = np.array([DISH_XY[0], DISH_XY[1], T + 0.0025]) - held
    q, rr = wb.hand_goal["r"]
    for target, dur in (([q[0], goal[1], q[2]], 1.2), ([goal[0], goal[1], q[2]], 0.8), (goal + [0, 0, 0.03], 0.8),
                        (goal, 0.6)):
        wb.reach("r", target, rr, dur)
        wb.settle()
    wb.fingers("r", "lid_pre", dur=0.6)
    wb.settle()
    q, rr = wb.hand_goal["r"]
    for target, dur in ((q + [0, 0, 0.10], 0.8), (q + [-0.10, 0, 0.10], 0.7)):
        wb.reach("r", target, rr, dur)
        wb.settle()
    wb.fingers("r", "relaxed", dur=0.4)
    wb.arm_joints_to("r", READY, 1.0)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["parts_cabinet"].plan = plan
