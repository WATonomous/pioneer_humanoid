"""Drill battery (a workshop scene modelled on the real thing): a cordless drill lies on its side on the bench, its
big slide-on battery pack flat (its fuel gauge one red LED); to its right an empty charging dock, to its left a second
dock with a charged pack in it (its status LED green). The right hand takes the drill's pack by its carry handle,
slides it back off the drill's rails, lifts it across and sets it down in line with the empty dock's rails, pushes it
on till it latches (the dock's LED turns red: charging); the left hand does the same with the charged pack, from its
dock onto the drill.

The rails are modelled as they work: a pack on them moves only along them (held to the rail's line each step), latches
at the end, and comes free when slid back past the rail's start (the hand pressing the release as it pulls).
The props are CAD models (``assets/workshop/make_meshes.py``, build123d) in the workshop room (``workshop.py``).
The robot starts at the bench, its legs and base still: no walking.

Steps (``progress``): 1. the flat pack off the drill, 2. on the charger, 3. the charged pack on the drill.
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

_spec = importlib.util.spec_from_file_location("make_meshes", ASSETS / "make_meshes.py")
_mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mm)
BAT, DRILL, DOCK = _mm.BATTERY, _mm.DRILL, _mm.DOCK

ZC = T + BAT["h"] / 2                  # a pack's middle, lying on the bench
MOUNTS = {"drill": (0.60, 0.03), "dock_a": (0.60, -0.20), "dock_b": (0.60, 0.31)}   # their rail faces (x, y)
PACKS = ("pack_old", "pack_new")
RAIL = BAT["rail"]                     # how far a pack slides on its rails
SEAT = np.array([0.0, -BAT["w"] / 2, 0.0])   # a seated pack's middle, in its mount's frame
PAIRS = [(m, p) for m in MOUNTS for p in PACKS]
OFF, ON, LATCHED = 0, 1, 2
_U_STEP, _U_PAIR = 0, 1                # userdata: the step; then each pair's state (OFF / ON / LATCHED)
_U_RELEASE = _U_PAIR + len(PAIRS)      # ... then each mount's release (held by the hand pulling the pack)
GREEN, RED, DARK = [0.1, 0.9, 0.2, 1], [0.95, 0.12, 0.08, 1], [0.05, 0.05, 0.05, 1]
STEPS = ("the flat pack off the drill", "on the charger", "the charged pack on the drill")
PULL_YAW = {"l": 0.0, "r": 0.0}        # the fingers forward, square across the handle's bar
HOOK_BACK, HOOK_ABOVE = 0.05, 0.02     # the palm from the bar's axis: back along the fingers, up (as on the vise's knob)
LIFT = 0.03                            # carried this far up off the bench
BACK = 0.05                            # set down this far back off the next rails' seat (on them), then pushed on
RAISED = {**READY, "shoulder_pitch": -0.45, "shoulder_roll": 0.8}
_FINGERS = ("i_", "", "r_")
HOOK_PRE = {**SHAPES["knob_pre"], **{f"{f}mcp": 0.2 for f in _FINGERS}, **{f"{f}pip": 0.8 for f in _FINGERS}}
                                       # an open hook (the tips out past the bar's far side), the thumb ready


def _pair(mount: str, pack: str) -> int:
    return PAIRS.index((mount, pack))


def _in_mount(model, data, mount: str, pack: str) -> np.ndarray:
    """The pack's middle in its mount's frame (from its joint: up to date between steps, where xpos isn't)."""
    mb = model.body(mount).id
    a = model.joint(pack).qposadr[0]
    R = data.xmat[mb].reshape(3, 3)
    return R.T @ (data.qpos[a:a + 3] - data.xpos[mb])


def _rails(model, data) -> None:
    """The rails: a pack near a rail's start, lined up, goes on; on them it moves only along them (held to the rail's
    line, square to it: its other motion taken out each step); at the end it latches (held there, till the mount's
    release is pressed); slid back past the rail's start, it's off."""
    for k, (mount, pack) in enumerate(PAIRS):
        state = int(data.userdata[_U_PAIR + k])
        p = _in_mount(model, data, mount, pack) - SEAT
        s = -p[0]                                       # (how far back off the seat, along the rail)
        release = data.userdata[_U_RELEASE + list(MOUNTS).index(mount)]
        if state == OFF:
            busy = any(int(data.userdata[_U_PAIR + _pair(mount, q)]) for q in PACKS if q != pack) or \
                any(int(data.userdata[_U_PAIR + _pair(n, pack)]) for n in MOUNTS if n != mount)
            if not busy and not release and abs(p[1]) < 0.012 and abs(p[2]) < 0.012 and 0 <= s < RAIL - 0.01:
                data.userdata[_U_PAIR + k] = state = ON
        if state == ON and s <= 0.001 and not release:
            data.userdata[_U_PAIR + k] = state = LATCHED
        if state == LATCHED and release:
            data.userdata[_U_PAIR + k] = state = ON
        if state == ON and s > RAIL:
            data.userdata[_U_PAIR + k] = OFF
            continue
        if state:
            mb = model.body(mount).id
            R = data.xmat[mb].reshape(3, 3)
            ds = 0.0 if state == LATCHED else min(max(s, 0.0), RAIL)
            a = model.joint(pack).qposadr[0]
            v = model.joint(pack).dofadr[0]
            data.qpos[a:a + 3] = data.xpos[mb] + R @ (SEAT + [-ds, 0, 0])
            data.qpos[a + 3:a + 7] = data.xquat[mb]
            along = R[:, 0]
            data.qvel[v:v + 3] = 0.0 if state == LATCHED else along * (data.qvel[v:v + 3] @ along)
            data.qvel[v + 3:v + 6] = 0.0


def _leds(model, data) -> None:
    """Each dock's status LED: green over a charged pack, red over a charging one, dark with none (the lit cap of the
    colour raised in its window, the others sunk: so the video, which replays the joints, shows it too)."""
    for mount in ("dock_a", "dock_b"):
        held = [p for p in PACKS if int(data.userdata[_U_PAIR + _pair(mount, p)]) == LATCHED]
        lit = None if not held else ("red" if held[0] == "pack_old" else "green")
        for colour in ("red", "green"):
            j = model.joint(f"{mount}_led_{colour}")
            data.qpos[j.qposadr[0]] = 0.0 if colour == lit else -0.004
            data.qvel[j.dofadr[0]] = 0.0


def _state(model, data, mount: str, pack: str) -> int:
    return int(data.userdata[_U_PAIR + _pair(mount, pack)])


def step(model, data) -> None:
    _rails(model, data)
    _leds(model, data)
    k = int(data.userdata[_U_STEP])
    if k == 0 and _state(model, data, "drill", "pack_old") == OFF:
        data.userdata[_U_STEP] = 1
    elif k == 1 and _state(model, data, "dock_a", "pack_old") == LATCHED:
        data.userdata[_U_STEP] = 2
    elif k == 2 and _state(model, data, "drill", "pack_new") == LATCHED:
        data.userdata[_U_STEP] = 3


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def reset(model, data, rng) -> None:
    """The packs latched where they start (the drill's flat one, the dock's charged one); the drill and the docks
    each a few mm off their places (the packs with them), different with each seed."""
    for mount, pack in (("drill", "pack_old"), ("dock_b", "pack_new")):
        data.userdata[_U_PAIR + _pair(mount, pack)] = LATCHED
    for mount in MOUNTS:
        b = model.body(mount).id
        model.body_pos[b][:2] += rng.uniform(-0.005, 0.005, 2)
    mujoco.mj_kinematics(model, data)
    for mount, pack in (("drill", "pack_old"), ("dock_b", "pack_new")):
        b = model.body(mount).id
        a = model.joint(pack).qposadr[0]
        data.qpos[a:a + 3] = model.body_pos[b] + SEAT


@scene("drill_battery", robot_pos=(0.0, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer v2  |  drill battery: the flat pack onto the charger, the charged one onto the drill",
       camera=((0.50, 0.03, 1.08), 1.4, 130, -28))
def build(spec: mujoco.MjSpec) -> None:
    add_workshop(spec)
    spec.nuserdata = 1 + len(PAIRS) + len(MOUNTS)
    world = spec.worldbody
    vis = dict(contype=0, conaffinity=0, group=1, density=0)
    col = dict(rgba=[0, 0, 0, 0], group=3)
    for m_ in ("battery_pack", "battery_gauge", "drill_body", "drill_trim", "drill_steel", "dock_body"):
        mesh(spec, m_)
    yellow = paint(spec, "drill_yellow", [0.93, 0.70, 0.06, 1], specular=0.45, shininess=0.5)
    black = paint(spec, "tool_black", [0.08, 0.08, 0.09, 1], specular=0.3, shininess=0.4)
    pack_mat = paint(spec, "pack_grey", [0.20, 0.21, 0.23, 1], specular=0.35, shininess=0.4)
    dock_mat = paint(spec, "dock_grey", [0.30, 0.31, 0.33, 1], specular=0.3, shininess=0.4)
    L, W, H = BAT["l"], BAT["w"], BAT["h"]
    # the drill (lying on its side, too heavy to shift as a pack slides on: fixed): its foot's rail face at its frame
    x, y = MOUNTS["drill"]
    dr = world.add_body(name="drill", pos=[x, y, ZC])
    dr.add_geom(type=MESH, meshname="drill_body", material=yellow, **vis)
    dr.add_geom(type=MESH, meshname="drill_trim", material=black, **vis)
    dr.add_geom(type=MESH, meshname="drill_steel", material="chrome", **vis)
    c = DRILL
    yb = c["foot"] + c["grip_l"] + 0.026                # the motor's axis (the drill lies on its side: up is +y)
    for g in ((0.008, c["foot"] / 2, 0, 0.0525, c["foot"] / 2, 0.031),                       # the foot
              (0.0, c["foot"] + c["grip_l"] / 2, 0, 0.02, c["grip_l"] / 2, 0.023),          # the grip
              (0.03, yb, 0, 0.105, 0.031, 0.031)):                                          # the housing to the chuck
        dr.add_geom(type=BOX, pos=list(g[:3]), size=list(g[3:]), friction=[1.0, 0.02, 0.002], **col)
    # the docks (fixed): a block each, the status LED on top
    for mount in ("dock_a", "dock_b"):
        x, y = MOUNTS[mount]
        dk = world.add_body(name=mount, pos=[x, y, ZC])
        dk.add_geom(type=MESH, meshname="dock_body", material=dock_mat, **vis)
        dk.add_geom(type=BOX, pos=[0.01, DOCK["d"] / 2, -H / 2 + DOCK["h"] / 2],
                    size=[DOCK["l"] / 2, DOCK["d"] / 2, DOCK["h"] / 2], **col)
        led = [-0.04, DOCK["d"] / 2, -H / 2 + DOCK["h"]]
        dk.add_geom(type=BOX, pos=[led[0], led[1], led[2] + 0.0003], size=[0.0065, 0.0065, 0.0003], rgba=DARK, **vis)
        for colour, rgba in (("red", RED), ("green", GREEN)):           # (a lit cap each, on a slide: see _leds)
            lb = dk.add_body(name=f"{mount}_led_{colour}", pos=led)
            lb.add_joint(name=f"{mount}_led_{colour}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                         range=[-0.005, 0.001], damping=1.0)
            lb.add_geom(type=BOX, size=[0.006, 0.006, 0.001], pos=[0, 0, 0.0001], rgba=rgba, contype=0, conaffinity=0,
                        group=1, mass=0.001)
    # the packs (free): the flat one on the drill, the charged one in dock b
    for pack, mount in (("pack_old", "drill"), ("pack_new", "dock_b")):
        x, y = MOUNTS[mount]
        pk = world.add_body(name=pack, pos=[x + SEAT[0], y + SEAT[1], ZC])
        pk.add_freejoint(name=pack)
        pk.add_geom(type=MESH, meshname="battery_pack", material=pack_mat, **vis)
        pk.add_geom(type=MESH, meshname="battery_gauge", rgba=RED if pack == "pack_old" else GREEN, **vis)
        pk.add_geom(type=BOX, size=[L / 2, W / 2, H / 2], mass=0.3, friction=[0.3, 0.02, 0.002], condim=4, priority=1,
                    **col)                       # (its plastic case on the bench: it slides; its friction, not the bench's)
        hl, hh, hr = BAT["handle_l"], BAT["handle_h"], BAT["handle_r"]
        pk.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[hr, hl / 2 - hr], pos=[0, 0, H / 2 + hh],
                    quat=[0.7071, 0.7071, 0, 0], friction=[1.2, 0.02, 0.002], condim=4, mass=0.01, **col)
        for s_ in (-1, 1):
            pk.add_geom(type=BOX, size=[0.009, 0.004, (hh + hr) / 2], pos=[0, s_ * (hl / 2 - 0.004), H / 2 + (hh + hr) / 2],
                        mass=0.005, **col)
        pk.add_site(name=f"{pack}_handle", pos=[0, 0, H / 2 + hh])      # its carry handle's bar
    for mount in MOUNTS:
        for pack in PACKS:
            spec.add_exclude(bodyname1=mount, bodyname2=pack)


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    m, d = wb.m, wb.d
    wb.start_ready()
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.look([0.50, 0.04, T + 0.04], dur=1.0)
    wb.wait(0.6)
    # 1-2. the right hand: the flat pack off the drill, across, onto dock a
    _move_pack(wb, "r", "pack_old", "drill", "dock_a")
    if _state(m, d, "dock_a", "pack_old") != LATCHED:
        raise S.TaskFailed("the flat pack isn't on the charger")
    # 3. the left hand: the charged pack off dock b, across, onto the drill
    _move_pack(wb, "l", "pack_new", "dock_b", "drill")
    wb.wait(1.0)
    notes["states"] = {f"{m_}/{p}": _state(m, d, m_, p) for m_, p in PAIRS if _state(m, d, m_, p)}
    if _state(m, d, "drill", "pack_new") != LATCHED:
        raise S.TaskFailed("the charged pack isn't on the drill")


def _move_pack(wb, side: str, pack: str, src: str, dst: str) -> None:
    """Take the pack by its carry handle (the hook over the bar, the thumb behind it); slide it back off ``src``'s
    rails (the release pressed), lift it, carry it across and set it down in line with ``dst``'s rails, on their
    start; push it on till it latches; let go, back to READY."""
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.controller import palm_down
    m, d = wb.m, wb.d
    r = palm_down(PULL_YAW[side])
    off = np.array([0, 0, HOOK_ABOVE]) - r[:, 0] * HOOK_BACK
    p = S.site(wb, f"{pack}_handle") + off
    wb.fingers(side, HOOK_PRE, dur=0.5)
    if side == "r":
        wb.arm_joints_to("r", RAISED, 0.9)
        wb.settle()
    for target, dur in ((p + [0, 0, 0.10], 1.1), (p + [0, 0, 0.03], 0.6), (p, 0.5)):
        wb.reach(side, target, r, dur)
        wb.settle()
    wb.fingers(side, "knob", amount=S.TWIST_SQUEEZE, dur=0.5)
    wb.settle()
    wb.wait(0.15)
    d.userdata[_U_RELEASE + list(MOUNTS).index(src)] = 1          # (the release, pressed as it pulls)
    _slide(wb, side, [-(RAIL + 0.035), 0, 0])             # (well past the rails' end: off them, the hand lagging)
    d.userdata[_U_RELEASE + list(MOUNTS).index(src)] = 0
    if _state(m, d, src, pack) != OFF:
        raise S.TaskFailed(f"{pack} didn't come off the {src}")
    _slide(wb, side, [0, 0, LIFT], step=0.01)
    # across, over the start of dst's rails (where the pack hangs in the hand: felt)
    seat = np.array([*MOUNTS[dst], ZC]) + SEAT
    spot = seat + [-BACK, 0, 0]
    held = S.body_pos(wb, pack) - wb.palm_pose(side, False)[0]
    goal = spot - held + [0, 0, LIFT]
    q, _ = wb.hand_goal[side]
    for target, dur in (([q[0], goal[1], goal[2]], 1.4), (goal, 0.8)):
        wb.reach(side, target, r, dur)
        wb.settle()
    for _ in range(3):                                   # hovering: where the pack is (felt), nudged on
        wb.wait(0.2)
        err = spot[:2] - S.body_pos(wb, pack)[:2]
        if np.linalg.norm(err) < 0.002:
            break
        err *= min(1.0, 0.02 / np.linalg.norm(err))
        goal = goal + [err[0], err[1], 0]
        wb.reach(side, goal, r, 0.4)
        wb.settle()
    _slide(wb, side, [0, 0, -LIFT - 0.004], step=0.008)
    _slide(wb, side, [BACK + 0.012, 0, 0])               # (on along the rails, a little past the seat: it latches)
    wb.wait(0.2)
    wb.fingers(side, HOOK_PRE, dur=0.4)
    wb.settle()
    q, rr = wb.hand_goal[side]
    wb.reach(side, q + [0, 0, 0.08], rr, 0.7)
    wb.settle()
    wb.fingers(side, "relaxed", dur=0.4)
    if side == "r":
        wb.arm_joints_to("r", RAISED, 0.8)
        wb.settle()
    wb.arm_joints_to(side, READY, 1.0)
    wb.settle()


def _slide(wb, side: str, delta, step: float = 0.01, dt: float = 0.1) -> None:
    """The hand straight by ``delta`` in small steps (the pack slides along with it, on the bench or on rails)."""
    p, r = wb.hand_goal[side]
    delta = np.asarray(delta, float)
    n = max(1, math.ceil(np.linalg.norm(delta) / step))
    for k in range(1, n + 1):
        wb.reach(side, p + delta * k / n, r, dt)
        wb.wait(dt)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["drill_battery"].plan = plan
