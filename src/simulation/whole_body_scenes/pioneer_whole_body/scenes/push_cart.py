"""Pushing a cart (whole body: walking while holding): the robot takes a trolley by the two knobs on its handle and
walks it forward across the room into a marked bay, a tote of parts riding on it, then lets go and steps back. The
cart rolls on four ball casters; it moves only because the hands push it (the grip holds by friction) as the legs
walk the body after it, and it coasts a little when they stop: the robot looks where it ended up and pushes the rest.
Come to rest in the bay, the bay's latch holds it (scripted, like the fixtures in the other scenes).

Steps (``progress``): 1. take the handle, 2. push the cart into the bay (n cm to go), 3. let go and step back.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import READY
from pioneer_whole_body.parts import KNOB, KNOB_H, KNOB_R, STEEL, add_weld, quat_z
from pioneer_whole_body.studio import add_studio

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE

CART0 = (0.66, 0.0)                      # the cart's middle at the start
DECK = (0.24, 0.22, 0.012)               # half sizes of its deck
DECK_Z = 0.28
WHEEL_R = 0.03
HANDLE_X = -0.26                         # the handle posts behind the deck's middle
HANDLE_Y = 0.13
KNOB_Z = 0.86                            # the handle knobs' bottoms
BAY = (1.40, 0.0)                        # where the cart's middle goes
BAY_TOL = 0.06
STEPS = ("take the handle", "push the cart into the bay", "let go and step back")
BAR_Z = 0.86                             # the handle knobs' bottoms
_U_STEP, _U_HELD, _U_FREE = 0, 1, 2


def _cart(model, data):
    b = model.body("cart").id
    r = data.xmat[b].reshape(3, 3)
    return data.xpos[b][:2].copy(), math.atan2(r[1, 0], r[0, 0])


def _in_bay(model, data) -> bool:
    p, yaw = _cart(model, data)
    return np.linalg.norm(p - BAY) < BAY_TOL and abs(yaw) < math.radians(10)


def step(model, data) -> None:
    latch = model.equality("bay_latch").id
    if not data.eq_active[latch] and _in_bay(model, data):
        b = model.body("cart").id
        v = data.qvel[model.joint("cart").dofadr[0]:][:3]
        if np.linalg.norm(v) < 0.02:              # come to rest in the bay: the bay's latch holds it
            model.eq_data[latch, 0:3] = 0
            model.eq_data[latch, 3:6] = data.xpos[b]
            model.eq_data[latch, 6:10] = data.xquat[b]
            model.eq_data[latch, 10] = 1.0
            data.eq_active[latch] = 1
    k = int(data.userdata[_U_STEP])
    if k == 0 and data.userdata[_U_HELD]:
        data.userdata[_U_STEP] = 1
    elif k == 1 and _in_bay(model, data) and data.userdata[_U_FREE]:
        data.userdata[_U_STEP] = 2
    elif k == 2 and _in_bay(model, data) and _clear(model, data):
        data.userdata[_U_STEP] = 3


def _clear(model, data) -> bool:
    """Both hands well away from the handle."""
    return all(np.linalg.norm(data.site_xpos[model.site(f"{s}_palm").id] - data.site_xpos[model.site(f"handle_{s}_top").id])
               > 0.15 for s in ("l", "r"))


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    if k == 1:
        p, _ = _cart(model, data)
        return k, len(STEPS), f"push the cart into the bay ({100 * max(0.0, BAY[0] - p[0]):.0f} cm to go)"
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def _knob(body, pos, name):
    x, y, z = pos
    body.add_geom(type=CYL, size=[KNOB_R, 0.006, 0], pos=[x, y, z - 0.006], rgba=STEEL, mass=0.005)   # (a boss
    # as wide as the knob under it: no undercut for a thumb to hook under when letting go)
    body.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[x, y, z + KNOB_H / 2], rgba=KNOB, mass=0.02,
                  friction=[2.0, 0.03, 0.003], condim=4)
    for i in range(10):
        a = 2 * math.pi * i / 10
        body.add_geom(type=BOX, size=[0.0025, 0.003, KNOB_H / 2 - 0.001],
                      pos=[x + KNOB_R * math.cos(a), y + KNOB_R * math.sin(a), z + KNOB_H / 2], quat=quat_z(a),
                      rgba=KNOB, contype=0, conaffinity=0, mass=0.0001)
    body.add_site(name=f"{name}_top", pos=[x, y, z + KNOB_H + 0.0012])


@scene("push_cart", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress,
       title="Pioneer  |  pushing a cart into its bay: walking while holding")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    spec.nuserdata = 3
    world = spec.worldbody
    blue, dark, grey = [0.16, 0.3, 0.55, 1], [0.12, 0.13, 0.15, 1], [0.6, 0.62, 0.66, 1]
    # the bay: lines on the floor round where the cart goes
    bx, by = BAY
    hx, hy = DECK[0] + 0.05, DECK[1] + 0.05
    for (px, py, sx, sy) in ((bx - hx, by, 0.01, hy), (bx + hx, by, 0.01, hy), (bx, by - hy, hx, 0.01),
                             (bx, by + hy, hx, 0.01)):
        world.add_geom(type=BOX, size=[sx, sy, 0.0008], pos=[px, py, 0.0008], rgba=[0.95, 0.75, 0.1, 1],
                       contype=0, conaffinity=0)
    cx, cy = CART0
    cart = world.add_body(name="cart", pos=[cx, cy, 0.0])
    cart.add_freejoint(name="cart")
    dx, dy, dz = DECK
    cart.add_geom(type=BOX, size=[dx, dy, dz], pos=[0, 0, DECK_Z], rgba=blue, mass=2.0)
    cart.add_geom(type=BOX, size=[dx - 0.04, dy - 0.04, 0.02], pos=[0, 0, 0.09], rgba=dark, mass=2.0)  # ballast: it
    # can't tip over when the handle is pushed
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * (dx - 0.03), sy * (dy - 0.03)
            cart.add_geom(type=CYL, size=[0.012, (DECK_Z - 2 * WHEEL_R) / 2, 0], pos=[x, y, WHEEL_R + DECK_Z / 2 - WHEEL_R],
                          rgba=grey, mass=0.03, contype=0, conaffinity=0)
            cart.add_geom(type=SPHERE, size=[WHEEL_R, 0, 0], pos=[x, y, WHEEL_R], rgba=dark, mass=0.03,
                          friction=[0.06, 0.001, 0.0001], condim=3,
                          priority=1)       # (its own friction, not the floor's: a contact takes the higher one)
    # the handle: two posts from the deck's back edge, a bar between them, a knob on top of each post
    for s_, side in ((1, "l"), (-1, "r")):
        y = s_ * HANDLE_Y
        cart.add_geom(type=CYL, size=[0.011, (BAR_Z - DECK_Z) / 2, 0], pos=[HANDLE_X, y, (BAR_Z + DECK_Z) / 2],
                      rgba=grey, mass=0.05)
        _knob(cart, [HANDLE_X, y, BAR_Z], f"handle_{side}")
    cart.add_geom(type=CYL, size=[0.01, HANDLE_Y, 0], pos=[HANDLE_X, 0, BAR_Z - 0.06], quat=[0.7071, 0.7071, 0, 0],
                  rgba=grey, mass=0.05)
    cart.add_site(name="bar", pos=[HANDLE_X, 0, BAR_Z])
    cart.add_geom(type=BOX, size=[0.03, dy, 0.012], pos=[HANDLE_X + 0.012, 0, DECK_Z], rgba=blue, mass=0.05)
    for k in (-1, 1):                                 # a rim round the deck (the tote can't slide off)
        cart.add_geom(type=BOX, size=[0.006, dy, 0.02], pos=[k * (dx - 0.006), 0, DECK_Z + dz + 0.02], rgba=blue,
                      mass=0.05)
        cart.add_geom(type=BOX, size=[dx, 0.006, 0.02], pos=[0, k * (dy - 0.006), DECK_Z + dz + 0.02], rgba=blue,
                      mass=0.05)
    add_weld(spec, "bay_latch", "world", "cart", active=False)
    # the tote and its parts, riding on the deck (free: they stay on because the cart moves gently)
    tote = world.add_body(name="tote", pos=[cx, cy, DECK_Z + dz])
    tote.add_freejoint(name="tote")
    tw, td, th = 0.12, 0.16, 0.05
    tote.add_geom(type=BOX, size=[tw, td, 0.003], pos=[0, 0, 0.003], rgba=[0.85, 0.45, 0.12, 1], mass=0.08)
    for k in (-1, 1):
        tote.add_geom(type=BOX, size=[0.003, td, th / 2], pos=[k * tw, 0, th / 2], rgba=[0.85, 0.45, 0.12, 1], mass=0.02)
        tote.add_geom(type=BOX, size=[tw, 0.003, th / 2], pos=[0, k * td, th / 2], rgba=[0.85, 0.45, 0.12, 1], mass=0.02)
    for i, (px, py) in enumerate(((-0.05, -0.08), (0.05, -0.08), (-0.05, 0.0), (0.05, 0.0), (0.0, 0.08))):
        part = world.add_body(name=f"part{i}", pos=[cx + px, cy + py, DECK_Z + dz + 0.03])
        part.add_freejoint(name=f"part{i}")
        part.add_geom(type=CYL, size=[0.025, 0.022, 0], rgba=STEEL, mass=0.04)


# ----------------------------------------------------------------------------- the demo
def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S

    m, d = wb.m, wb.d
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    wb.walk(0.22)                                  # (the handle ~0.48 m in front of the pelvis)
    # take both knobs (the left, then the right)
    for side, yaw in (("l", -0.3), ("r", 0.3)):
        top = S.site(wb, f"handle_{side}_top")
        p, r = S.knob_pose(top, yaw)
        wb.look(top, dur=0.4)
        wb.fingers(side, "lid_pre", dur=0.5)
        wb.reach(side, p + [0, 0, S.CLEAR], r, 1.2)
        wb.settle()
        wb.reach(side, p, r, 0.7)
        wb.settle()
        wb.fingers(side, "lid", amount=S.SQUEEZE, dur=0.45)
        wb.settle()
    d.userdata[_U_HELD] = 1
    # push: walk forward carrying the hands along; then look where the cart is and push the rest
    wb.look([BAY[0], BAY[1], 0.4], dur=0.8)
    for _ in range(4):
        p, _ = _cart(m, d)
        togo = BAY[0] - p[0]
        if togo < 0.02:
            break
        wb.walk(togo, carry=True)
        wb.settle()
        wb.wait(0.8)                               # (it coasts)
    notes["cart"] = [round(float(v), 3) for v in _cart(m, d)[0]] + [round(math.degrees(_cart(m, d)[1]), 1)]
    # let go: the hands right open, up off the knobs, back toward the body, then down to the sides (the cart rolls
    # at a touch: nothing may brush it)
    for side in ("l", "r"):
        wb.fingers(side, "open", dur=0.6)
    wb.settle()
    wb.wait(0.3)
    for dp, dur in (([0, 0, 0.08], 0.8), ([-0.18, 0, 0.0], 0.8)):
        for side in ("l", "r"):
            p, r = wb.hand_goal[side]
            wb.reach(side, p + dp, r, dur)
        wb.settle()
    for side in ("l", "r"):
        wb.fingers(side, "relaxed", dur=0.5)
        wb.arm_joints_to(side, READY, 1.0)
    wb.settle()
    d.userdata[_U_FREE] = 1
    wb.walk(-0.3)
    wb.look([BAY[0], BAY[1], 0.3], dur=0.6)
    wb.wait(1.5)
    if not _in_bay(m, d):
        raise S.TaskFailed(f"the cart is at {notes['cart']}")


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["push_cart"].plan = plan
