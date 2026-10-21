"""A hinged lid (an articulated object turned about a hinge, the hand turning with it): the robot reads which button
to press off a lit tag on a box, lifts the box's lid open, presses that button inside, and lowers the lid shut.

The robot walks up to the bench. On it:

  box      in the middle: its lid hinged along the back edge (a damped hinge, a stay at 76 degrees: let go, the lid
           settles shut, so it has to be held open), a mushroom knob on its front edge
  buttons  inside the box, red, green and blue; under the lid, out of sight till it's open
  tag      a lit tag on the box's front, red, green or blue: which button to press (it changes with every seed,
           ``reset``)
  lamp     on the right: comes up when the right button was pressed (a wrong one first: it never does)

The plan: look at the box's front, read the tag's colour in the head camera's image (``vision.Eye``). Take the lid's
knob with the hand away from that button's side and lift the lid open on an arc about its hinge, the hand turning
with the lid (the grip fixed on it), up to its stay, and hold it there; the other hand reaches in under it and
presses the button with its thumb. Then the holding hand lets go: the lid settles shut.

Steps (``progress``): 1. read the tag, 2. open the lid, 3. press the button, 4. shut the lid.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH, READY
from pioneer_whole_body.parts import KNOB, KNOB_R, STEEL, quat_z
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

COLOURS = ("red", "green", "blue")
RGBA = {"red": [0.86, 0.13, 0.11, 1], "green": [0.13, 0.68, 0.25, 1], "blue": [0.13, 0.33, 0.86, 1]}
FRONT, BACK = 0.46, 0.63          # the box (x)
HALF_W = 0.15
H = 0.075                         # its walls' height (the hinge's)
LID_T = 0.008
REACH = 0.155                     # the knob from the hinge (along the lid)
BOSS = 0.045                      # the knob stands this high off the lid on a thin post: a mushroom, a short cap
CAP_H = 0.022                     # the wrapped fingers curl in under (a grip that can pull up on it)
OPEN = math.radians(80)           # (aimed for: the stay stops it at 76)
OPENED = math.radians(48)         # open enough
BUTTON_X = 0.515
BUTTON_Y = {"red": 0.075, "green": 0.0, "blue": -0.075}
TRAVEL = 0.006
LAMP = (0.60, -0.28)
STEPS = ("read the tag", "open the lid", "press the button", "shut the lid")
_U_STEP, _U_ANSWER, _U_PRESSED, _U_OK = 0, 1, 2, 3


def _lid(model, data) -> float:
    return float(data.qpos[model.joint("lid").qposadr[0]])


def _pressed(model, data, c) -> bool:
    return data.qpos[model.joint(f"button_{c}").qposadr[0]] < -0.6 * TRAVEL


def step(model, data) -> None:
    u = data.userdata
    for i, c in enumerate(COLOURS):
        if _pressed(model, data, c) and not u[_U_PRESSED]:
            u[_U_PRESSED] = i + 1                              # the first button pressed
            u[_U_OK] = i == int(u[_U_ANSWER])
    a = model.joint("lamp").qposadr[0]
    target = 0.0 if u[_U_OK] else -math.pi / 2
    data.qpos[a] += float(np.clip(target - data.qpos[a], -0.004, 0.004))
    data.qvel[model.joint("lamp").dofadr[0]] = 0.0
    k = int(u[_U_STEP])
    if k == 1 and _lid(model, data) > OPENED:
        u[_U_STEP] = 2
    elif k == 2 and u[_U_OK]:
        u[_U_STEP] = 3
    elif k == 3 and _lid(model, data) < 0.02:
        u[_U_STEP] = 4


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def reset(model, data, rng) -> None:
    """The tag's colour."""
    i = int(rng.integers(3))
    data.userdata[_U_ANSWER] = i
    model.mat_rgba[model.material("tag_lit").id] = RGBA[COLOURS[i]]
    data.qpos[model.joint("lamp").qposadr[0]] = -math.pi / 2


@scene("lid_box", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer v2  |  a hinged lid: the tag read by sight, the lid lifted open, its button pressed")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = 4
    world = spec.worldbody
    shell, dark, grey = [0.50, 0.52, 0.55, 1], [0.12, 0.13, 0.15, 1], [0.32, 0.34, 0.37, 1]
    cx = (FRONT + BACK) / 2
    hx = (BACK - FRONT) / 2
    box = world.add_body(name="box", pos=[cx, 0, T])
    box.add_geom(type=BOX, size=[hx, HALF_W, 0.002], pos=[0, 0, 0.002], rgba=shell)
    for s in (1, -1):
        box.add_geom(type=BOX, size=[hx, 0.005, H / 2], pos=[0, s * (HALF_W - 0.005), H / 2], rgba=shell)
        box.add_geom(type=BOX, size=[0.005, HALF_W, H / 2], pos=[s * (hx - 0.005), 0, H / 2], rgba=shell)
    # the tag: lit, on the front
    spec.add_material(name="tag_lit", rgba=RGBA["red"], emission=0.6)
    box.add_geom(type=BOX, size=[0.001, 0.035, 0.018], pos=[-hx - 0.001, 0, H / 2], material="tag_lit",
                 contype=0, conaffinity=0)
    # the buttons inside
    for c in COLOURS:
        h = box.add_body(name=f"button_{c}_housing", pos=[BUTTON_X - cx, BUTTON_Y[c], 0.004])
        h.add_geom(type=BOX, size=[0.025, 0.025, 0.027], pos=[0, 0, 0.027], rgba=dark)    # (tall: the caps
        cap = h.add_body(name=f"button_{c}", pos=[0, 0, 0.054])                         # just under the lid)
        cap.add_joint(name=f"button_{c}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-TRAVEL, 0],
                      limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=150.0, springref=0.0, damping=3.0)
        cap.add_geom(type=CYL, size=[0.018, 0.006, 0], pos=[0, 0, 0.006], rgba=RGBA[c], mass=0.02)
        cap.add_site(name=f"button_{c}_top", pos=[0, 0, 0.012])
        spec.add_exclude(bodyname1=f"button_{c}_housing", bodyname2=f"button_{c}")
    # the lid: hinged along the back edge, a mushroom knob on its front edge
    lid = box.add_body(name="lid", pos=[hx, 0, H])
    lid.add_joint(name="lid", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[0, 76],   # (degrees: MjSpec's default; a stay
                  # stops it at 67)
                  limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.08, frictionloss=0.004, armature=0.002)
    L = 2 * hx + 0.01
    lid.add_geom(type=BOX, size=[L / 2, HALF_W + 0.004, LID_T / 2], pos=[-L / 2, 0, LID_T / 2], rgba=shell,
                 mass=0.05)
    z = LID_T
    lid.add_geom(type=CYL, size=[0.009, BOSS / 2, 0], pos=[-REACH, 0, z + BOSS / 2], rgba=STEEL, mass=0.004)
    z += BOSS
    lid.add_geom(type=CYL, size=[KNOB_R, CAP_H / 2, 0], pos=[-REACH, 0, z + CAP_H / 2], rgba=KNOB, mass=0.015,
                 friction=[2.0, 0.03, 0.003], condim=4)
    for i in range(10):
        a = 2 * math.pi * i / 10
        lid.add_geom(type=BOX, size=[0.0025, 0.003, CAP_H / 2 - 0.001],
                     pos=[-REACH + KNOB_R * math.cos(a), KNOB_R * math.sin(a), z + CAP_H / 2], quat=quat_z(a),
                     rgba=KNOB, contype=0, conaffinity=0, mass=0.0001)
    lid.add_site(name="lid_knob_top", pos=[-REACH, 0, z + CAP_H + 0.0012])
    lid.add_site(name="hinge", pos=[0, 0, 0])
    spec.add_exclude(bodyname1="box", bodyname2="lid")
    # the lamp
    post = world.add_body(name="lamp_post", pos=[LAMP[0], LAMP[1], T])
    post.add_geom(type=CYL, size=[0.006, 0.05, 0], pos=[0, 0, 0.05], rgba=grey, contype=0, conaffinity=0)
    flag = post.add_body(name="lamp", pos=[0, 0, 0.1])
    flag.add_joint(name="lamp", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], ref=0.0)
    flag.add_geom(type=BOX, size=[0.003, 0.02, 0.03], pos=[0, 0, 0.03], rgba=[0.15, 0.85, 0.3, 1], contype=0,
                  conaffinity=0, mass=0.01)
    flag.gravcomp = 1.0


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.vision import Eye

    m, d = wb.m, wb.d
    eye = Eye(m)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    wb.walk(0.30)
    # 1. read the tag
    tag = np.array([FRONT, 0.0, T + H / 2])
    wb.look(tag, dur=0.7)
    wb.wait(0.35)
    seen = [b for b in eye.find(d, tag[2]) if b.colour in COLOURS and abs(b.world[1]) < 0.06]
    if not seen:
        raise S.TaskFailed("no tag seen")
    colour = max(seen, key=lambda b: b.area).colour
    notes["tag"] = colour
    d.userdata[_U_STEP] = 1
    # 2. open the lid (the hand away from the button's side), and hold it open
    side = "l" if BUTTON_Y[colour] > 0 else "r"           # the hand that presses
    holder = "r" if side == "l" else "l"
    pose = _take_lid(wb, holder)
    _turn(wb, holder, pose, OPEN)
    notes["opened"] = round(math.degrees(_lid(m, d)), 1)
    if _lid(m, d) < OPENED:
        raise S.TaskFailed(f"the lid's open {math.degrees(_lid(m, d)):.0f} deg")
    # 3. the other hand reaches in under it and presses the button
    top = S.site(wb, f"button_{colour}_top")
    wb.look(top, dur=0.5)
    wb.fingers(side, "fist", dur=0.3)
    S.press(wb, side, f"{side}_thumb_tip", top, depth=0.01, approach=0.1, hold=0.3)   # (in from high
                                                                                         # over the rim)
    wb.fingers(side, "relaxed", dur=0.3)
    wb.arm_joints_to(side, READY, 0.8)
    wb.settle()
    notes["pressed"] = COLOURS[int(d.userdata[_U_PRESSED]) - 1] if d.userdata[_U_PRESSED] else None
    if not d.userdata[_U_OK]:
        raise S.TaskFailed(f"pressed {notes['pressed']}, the tag is {colour}")
    # 4. let go: it settles shut on its damped hinge
    _let_go(wb, holder)
    wb.look([LAMP[0], LAMP[1], T + 0.12], dur=0.6)
    wb.wait(4.0)
    wb.arm_joints_to(holder, READY, 0.9)
    wb.settle()
    notes["shut"] = round(math.degrees(_lid(m, d)), 1)
    if _lid(m, d) > 0.02:
        raise S.TaskFailed(f"the lid's still {math.degrees(_lid(m, d)):.0f} deg open")


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _take_lid(wb, side: str):
    """Take the lid's knob (wrapped over its top, under the mushroom's rim); returns the hand's pose for any angle of
    the lid: the grip fixed on the lid, so the hand turns with it (the lid's turn about the hinge applied to the
    palm's pose with the lid shut)."""
    from pioneer_whole_body import skills as S
    m, d = wb.m, wb.d
    a0 = _lid(m, d)
    hinge = S.site(wb, "hinge")
    top = S.site(wb, "lid_knob_top")
    top0 = hinge + _ry(-a0) @ (top - hinge)               # the knob's top with the lid shut, the grasp there
    p0, r0 = S.knob_pose(top0, 0.5 if side == "r" else -0.5)
    pose = lambda a: (hinge + _ry(a) @ (p0 - hinge), _ry(a) @ r0)      # noqa: E731
    pa, ra = pose(a0)
    wb.look(top, dur=0.5)
    wb.fingers(side, "lid_pre", dur=0.5)
    wb.reach(side, pa + _ry(a0)[:, 2] * S.CLEAR, ra, 1.0)
    wb.settle()
    wb.reach(side, pa, ra, 0.6)
    wb.settle()
    wb.fingers(side, "lid", amount=S.SQUEEZE, dur=0.45)
    wb.settle()
    wb.wait(0.15)
    return pose


def _turn(wb, side: str, pose, goal: float, step: float = math.radians(3), dt: float = 0.15) -> None:
    """Carry the held lid to ``goal`` (rad) on its arc about the hinge, in small moves."""
    a0 = _lid(wb.m, wb.d)
    n = max(1, math.ceil(abs(goal - a0) / step))
    for k in range(1, n + 1):
        wb.reach(side, *pose(a0 + (goal - a0) * k / n), dt)
        wb.settle()
    wb.wait(0.3)


def _let_go(wb, side: str) -> None:
    """The fingers open right out (curled, they'd hook under the mushroom's rim), the hand straight back off the
    knob along its axis, up out of the lid's way."""
    axis = _ry(_lid(wb.m, wb.d))[:, 2]
    q, r = wb.hand_goal[side]
    wb.reach(side, q - axis * 0.01, r, 0.3)            # (in a little first: the fingertips out from under the rim)
    wb.settle()
    wb.fingers(side, "open", dur=0.7)
    wb.settle()
    wb.reach(side, q + axis * 0.08, r, 0.7)
    wb.settle()
    wb.fingers(side, "relaxed", dur=0.4)
    wb.arm_joints_to(side, HIGH, 0.9)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["lid_box"].plan = plan
