"""Watch and repeat (perception and sequencing, both hands): a panel shows a sequence of colours, one flag at a time;
the robot watches it with its head camera and presses the coloured buttons in the same order. Two rounds, the second
longer. What the sequence is comes only from the camera (it changes with every seed: ``reset``).

The robot walks up to the bench. On it:

  buttons   a row of four, red, green, blue, yellow (the left hand presses the two on the left, the right hand the
            two on the right), and a white start button in front of them
  display   behind them a flag of each colour, lying face down; when the panel plays its sequence, each flag of the
            sequence stands up for a moment, in turn
  lamp      on the right: comes up when both rounds were entered right

The panel (``step``): pressing start plays the round's sequence; then each button press is checked against it, in
order; a wrong press shows every flag (the round must be started again and played again); the whole sequence
entered: the round is done. The robot: presses start, watches the display (a camera image every 0.1 s: which flag is
up), and when the display has been still for a while, presses the buttons it saw, in order, then checks the
panel took them (round done) and goes on to the next round.

Steps (``progress``): round 1: watch, round 1: repeat, round 2: watch, round 2: repeat.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH, READY
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

COLOURS = ("red", "green", "blue", "yellow")
RGBA = {"red": [0.86, 0.13, 0.11, 1], "green": [0.13, 0.68, 0.25, 1], "blue": [0.13, 0.33, 0.86, 1],
        "yellow": [0.93, 0.8, 0.12, 1]}
BUTTON_X, BUTTON_Y = 0.53, (0.135, 0.045, -0.045, -0.135)      # red, green, blue, yellow (left to right)
START = (0.45, 0.0)
FLAG_X, FLAG_Z = 0.68, 0.06                                    # the display's flags (their hinge, height on a stand)
FLAG = (0.0015, 0.03, 0.036)                                   # half sizes of a flag
UP = -0.45                                                     # a standing flag leans back toward the robot a little
LAMP = (0.62, -0.27)
TRAVEL = 0.008
ROUNDS = (4, 6)                                                # the sequence's length in each round
ON, OFF = 0.9, 0.35                                            # a flag stands this long, then this long down
DELAY = 3.0                                                    # the sequence starts this long after start
DOWN = -math.pi / 2
STEPS = ("round 1: watch", "round 1: repeat", "round 2: watch", "round 2: repeat")
# userdata
_U_STEP, _U_PHASE, _U_ROUND, _U_T0, _U_IN, _U_OK = 0, 1, 2, 3, 4, 5
_U_LAST = 6                                    # 6..10: each button (4 colours, start) pressed last step
_U_SEQ = 12                                    # 12..: the sequences, round after round
WAIT, PLAY, INPUT, DONE, ERROR = 0, 1, 2, 3, 4


def _seq(data, rnd: int) -> list[int]:
    a = _U_SEQ + sum(ROUNDS[:rnd])
    return [int(v) for v in data.userdata[a:a + ROUNDS[rnd]]]


def _pressed(model, data, name) -> bool:
    return data.qpos[model.joint(name).qposadr[0]] < -0.6 * TRAVEL


def step(model, data) -> None:
    u = data.userdata
    names = [f"button_{c}" for c in COLOURS] + ["button_start"]
    edges = []
    for i, n in enumerate(names):
        now = _pressed(model, data, n)
        if now and not u[_U_LAST + i]:
            edges.append(i)
        u[_U_LAST + i] = now
    phase, rnd = int(u[_U_PHASE]), int(u[_U_ROUND])
    if 4 in edges and phase in (WAIT, ERROR) and rnd < len(ROUNDS):
        u[_U_PHASE], u[_U_T0], u[_U_IN] = PLAY, data.time, 0
    elif phase == PLAY and data.time - u[_U_T0] - DELAY > ROUNDS[rnd] * (ON + OFF):
        u[_U_PHASE] = INPUT
    elif phase == INPUT:
        for i in edges:
            if i == 4:
                continue
            k = int(u[_U_IN])
            if i == _seq(data, rnd)[k]:
                u[_U_IN] = k + 1
                if k + 1 == ROUNDS[rnd]:
                    u[_U_ROUND] = rnd + 1
                    u[_U_PHASE] = DONE if rnd + 1 == len(ROUNDS) else WAIT
                    u[_U_OK] = rnd + 1 == len(ROUNDS)
            else:
                u[_U_PHASE] = ERROR
            break
    # the flags: up while their colour is the one playing (or all, after a wrong press)
    phase, rnd = int(u[_U_PHASE]), int(u[_U_ROUND])
    show = set()
    if phase == PLAY:
        t = data.time - u[_U_T0] - DELAY
        k = int(t // (ON + OFF))
        if 0 <= k < ROUNDS[rnd] and t - k * (ON + OFF) < ON:
            show = {_seq(data, rnd)[k]}
    elif phase == ERROR:
        show = set(range(4))
    for i, c in enumerate(COLOURS):
        _drive(model, data, f"flag_{c}", UP if i in show else DOWN)
    _drive(model, data, "lamp", UP if u[_U_OK] else DOWN, rate=0.004)
    # progress: watch / repeat for each round, latched in order
    k = int(u[_U_STEP])
    target = 2 * int(u[_U_ROUND]) + (1 if phase == INPUT else 0)
    if target > k:
        u[_U_STEP] = target


def _drive(model, data, joint, target, rate=0.03):
    a = model.joint(joint).qposadr[0]
    data.qpos[a] += float(np.clip(target - data.qpos[a], -rate, rate))
    data.qvel[model.joint(joint).dofadr[0]] = 0.0


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def reset(model, data, rng) -> None:
    seq = []
    for n in ROUNDS:
        s = [int(rng.integers(4))]
        while len(s) < n:                        # no colour twice in a row (a flag that stays up reads as one)
            c = int(rng.integers(4))
            if c != s[-1]:
                s.append(c)
        seq += s
    data.userdata[_U_SEQ:_U_SEQ + len(seq)] = seq
    for c in COLOURS:
        data.qpos[model.joint(f"flag_{c}").qposadr[0]] = DOWN
    data.qpos[model.joint("lamp").qposadr[0]] = DOWN


def _button(world, spec, name, pos, rgba, r=0.02):
    x, y = pos
    h = world.add_body(name=f"{name}_housing", pos=[x, y, T])
    h.add_geom(type=BOX, size=[0.03, 0.03, 0.02], pos=[0, 0, 0.02], rgba=[0.3, 0.32, 0.35, 1])
    cap = h.add_body(name=name, pos=[0, 0, 0.04])
    cap.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-TRAVEL, 0],
                  limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=150.0, springref=0.0, damping=3.0)
    cap.add_geom(type=CYL, size=[r, 0.006, 0], pos=[0, 0, 0.006], rgba=rgba, mass=0.02)
    cap.add_site(name=f"{name}_top", pos=[0, 0, 0.012])
    spec.add_exclude(bodyname1=f"{name}_housing", bodyname2=name)


def _flag(spec, parent, name, pos, rgba):
    """A flag on a hinge: its face lit (a sign lamp), its back dark. Up: the face toward the robot; down: face down."""
    spec.add_material(name=f"{name}_lit", rgba=rgba, emission=0.6)
    f = parent.add_body(name=name, pos=list(pos))
    f.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], ref=0.0)
    hx, hy, hz = FLAG
    f.add_geom(type=BOX, size=[hx, hy, hz], pos=[-hx, 0, hz + 0.002], material=f"{name}_lit", contype=0,
               conaffinity=0, mass=0.01)                                         # the face (toward the robot)
    f.add_geom(type=BOX, size=[hx, hy, hz], pos=[hx, 0, hz + 0.002], rgba=[0.22, 0.23, 0.26, 1], contype=0,
               conaffinity=0, mass=0.01)                                         # the back
    f.gravcomp = 1.0


@scene("simon_panel", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress, reset=reset,
       title="Pioneer  |  watch and repeat: a colour sequence seen with its head camera, pressed back")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec)
    spec.nuserdata = _U_SEQ + sum(ROUNDS)
    world = spec.worldbody
    for c, y in zip(COLOURS, BUTTON_Y):
        _button(world, spec, f"button_{c}", (BUTTON_X, y), RGBA[c])
    _button(world, spec, "button_start", START, [0.92, 0.92, 0.9, 1], r=0.016)
    # the display: a dark stand, a flag of each colour on it
    stand = world.add_body(name="display", pos=[FLAG_X, 0, T])
    stand.add_geom(type=BOX, size=[0.04, 0.2, FLAG_Z / 2], pos=[0.015, 0, FLAG_Z / 2], rgba=[0.12, 0.13, 0.15, 1])
    for c, y in zip(COLOURS, BUTTON_Y):
        _flag(spec, stand, f"flag_{c}", [-0.025, y, FLAG_Z], RGBA[c])
    post = world.add_body(name="lamp_post", pos=[LAMP[0], LAMP[1], T])
    post.add_geom(type=CYL, size=[0.006, 0.05, 0], pos=[0, 0, 0.05], rgba=[0.32, 0.34, 0.37, 1], contype=0,
                  conaffinity=0)
    _flag(spec, post, "lamp", [0, 0, 0.1], [0.15, 0.85, 0.3, 1])          # (face down until both rounds are in)


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S
    from pioneer_whole_body.vision import Eye

    m, d = wb.m, wb.d
    eye = Eye(m, 320, 240)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    wb.walk(0.30)
    notes["rounds"] = []
    for rnd in range(len(ROUNDS)):
        for attempt in range(3):
            _press(wb, "r", "button_start")
            seen = _watch(wb, eye)
            notes["rounds"].append([COLOURS[i] for i in seen])
            for i in seen:
                _press(wb, "l" if BUTTON_Y[i] > 0 else "r", f"button_{COLOURS[i]}")
            wb.settle()
            wb.wait(0.3)
            if int(d.userdata[_U_ROUND]) > rnd:
                break
        else:
            raise S.TaskFailed(f"round {rnd + 1} not taken")
    for side in ("l", "r"):
        wb.arm_joints_to(side, READY, 1.0)
    wb.look([LAMP[0], LAMP[1], T + 0.12], dur=0.6)
    wb.wait(2.0)
    if not d.userdata[_U_OK]:
        raise S.TaskFailed("the lamp is not up")


def _press(wb, side: str, button: str) -> None:
    from pioneer_whole_body import skills as S
    top = S.site(wb, f"{button}_top")
    wb.look(top, dur=0.4)
    wb.fingers(side, "fist", dur=0.3)
    S.press(wb, side, f"{side}_thumb_tip", top, approach=0.05, hold=0.15)
    wb.arm_joints_to(side, READY, 0.7)


def _watch(wb, eye, quiet: float = 1.6, longest: float = 12.0) -> list[int]:
    """Look at the display and read the sequence: an image every 0.1 s, the colour of the flag that is up (a blob of
    a button colour above the buttons); a new entry each time a flag comes up after none. Done when nothing has
    stood for ``quiet`` s after something has."""
    plane = T + FLAG_Z + FLAG[2]                     # the middle of a standing flag's face
    wb.look([FLAG_X, 0.0, T + FLAG_Z], dur=0.5)
    seq, up, since, t0 = [], None, 0.0, wb.t
    while wb.t - t0 < longest:
        wb.wait(0.1)
        now = None
        for b in eye.find(wb.d, plane, min_area=25):
            if b.colour in COLOURS and b.world[0] > FLAG_X - 0.08 and abs(b.world[1]) < 0.2:
                now = COLOURS.index(b.colour)
                break
        if now is not None and now != up:
            seq.append(now)
        if now is None:
            since += 0.1
            if seq and since > quiet:
                break
        else:
            since = 0.0
        up = now
    return seq


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["simon_panel"].plan = plan
