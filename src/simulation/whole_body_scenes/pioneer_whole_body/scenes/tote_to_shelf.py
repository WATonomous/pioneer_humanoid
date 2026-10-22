"""Kit and stow (long horizon, whole body): load a tote with three cells, carry it to the low shelf, crouch and
stow it.

The robot walks up to the bench. On it:

  tote          in front, in a nest (corner posts): three cell slots, a handle knob on each end
  cells         in a rack on the right
  low shelf     beside the bench to the left, 12 cm lower (0.73 m): a fixture for the tote, latching

Steps, each checked from the simulation and latched in order (``progress``):
  1. load the tote               all three cells in its slots
  2. lift the tote               both hands, by its handles, clear of the bench
  3. stow it on the low shelf    side steps over, a crouch and a lean (the hands go 12 cm lower than the bench), set
                                 in the shelf's fixture (its latch snaps), the cells still in it
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_whole_body import scene
from pioneer_whole_body.controller import HIGH
from pioneer_whole_body.parts import KNOB, KNOB_H, KNOB_R, ORANGE, STEEL, add_weld, quat_z
from pioneer_whole_body.studio import BENCH_TOP_Z, add_bench, add_studio

T = BENCH_TOP_Z
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

TOTE_POS = (0.52, 0.0)
TOTE = (0.05, 0.13)                       # inner half sizes (x, y)
TOTE_WALL = 0.006
TOTE_H = 0.035
SLOTS_Y = (-0.10, 0.0, 0.10)
SLOT_R = 0.0225
CELL_R, CELL_H = 0.019, 0.062
HANDLE_Y = 0.20                           # the knobs stand on brackets out from the ends (a hand loading the
                                          # end slot clears them)
STEP = 0.15
RACK = [(0.48, -0.30), (0.48, -0.40), (0.48, -0.50)]
SHELF_SIDE = 6 * STEP                      # the shelf is this far to the left (clear of the bench)
SHELF_Z = 0.73
SHELF_X = (0.30, 0.70)
SHELF_HALF_Y = 0.28
FIX_PLAY = 0.03                            # the fixture's play round the tote (the latch takes it where it lies)
CROUCH = 0.09                              # pelvis down (m)
LEAN = 0.22                                # pelvis pitch (rad, forward)

STEPS = ("load the tote", "lift the tote", "stow it on the low shelf")
_U_STEP = 0


def cells_in(model, data) -> int:
    tote = model.body("tote").id
    tp, tr = data.xpos[tote], data.xmat[tote].reshape(3, 3)
    n = 0
    for k in range(3):
        rel = tr.T @ (data.xpos[model.body(f"cell{k}").id] - tp)
        up = data.xmat[model.body(f"cell{k}").id].reshape(3, 3)[:, 2] @ tr[:, 2]
        if any(math.hypot(rel[0], rel[1] - y) < SLOT_R for y in SLOTS_Y) and rel[2] < 0.016 and \
                up > math.cos(math.radians(25)):
            n += 1
    return n


def _in_fixture(model, data) -> bool:
    p = data.xpos[model.body("tote").id]
    flat = data.xmat[model.body("tote").id][8] > math.cos(math.radians(4))
    sx = (SHELF_X[0] + SHELF_X[1]) / 2
    return abs(p[0] - sx) < FIX_PLAY and abs(p[1] - SHELF_SIDE) < FIX_PLAY and p[2] < SHELF_Z + 0.004 and flat


def step(model, data) -> None:
    latch = model.equality("shelf_latch").id
    if not data.eq_active[latch] and _in_fixture(model, data):
        po = data.xpos[model.body("shelf").id]
        ro = data.xmat[model.body("shelf").id].reshape(3, 3)
        q, qn = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(qn, data.xquat[model.body("shelf").id])
        mujoco.mju_mulQuat(q, qn, data.xquat[model.body("tote").id])
        model.eq_data[latch, 0:3] = 0
        model.eq_data[latch, 3:6] = ro.T @ (data.xpos[model.body("tote").id] - po)
        model.eq_data[latch, 6:10] = q
        model.eq_data[latch, 10] = 1.0
        data.eq_active[latch] = 1
    k = int(data.userdata[_U_STEP])
    if k < len(STEPS) and _step_done(model, data, k):
        data.userdata[_U_STEP] = k + 1


def _step_done(model, data, k) -> bool:
    if k == 0:
        return cells_in(model, data) == 3
    if k == 1:
        return data.xpos[model.body("tote").id][2] > T + 0.05
    return bool(data.eq_active[model.equality("shelf_latch").id]) and cells_in(model, data) == 3


def progress(model, data):
    k = int(data.userdata[_U_STEP])
    return k, len(STEPS), STEPS[min(k, len(STEPS) - 1)]


def _ring(body, r, h, wall, n, rgba, z0=0.0):
    half_t = (r + wall) * math.tan(math.pi / n) + 0.0005
    for i in range(n):
        a = 2 * math.pi * i / n
        body.add_geom(type=BOX, size=[wall / 2, half_t, h / 2],
                      pos=[(r + wall / 2) * math.cos(a), (r + wall / 2) * math.sin(a), z0 + h / 2],
                      quat=quat_z(a), rgba=rgba, mass=0.002)


@scene("tote_to_shelf", robot_pos=(-0.3, 0.0, 0.0), step=step, progress=progress,
       title="Pioneer  |  kit and stow: load the tote, carry it with both hands, crouch, stow on the low shelf")
def build(spec: mujoco.MjSpec) -> None:
    add_studio(spec)
    add_bench(spec, half_y=0.6)
    spec.nuserdata = 1
    world = spec.worldbody
    grey, blue = [0.32, 0.34, 0.37, 1], [0.15, 0.35, 0.7, 1]
    steel_dark = [0.16, 0.17, 0.19, 1]

    # the low shelf, to the left of the bench (a frame and a top; the fixture on it)
    sx0, sx1 = SHELF_X
    sx, sy = (sx0 + sx1) / 2, SHELF_SIDE
    shelf = world.add_body(name="shelf", pos=[sx, sy, SHELF_Z])
    shelf.add_geom(type=BOX, size=[(sx1 - sx0) / 2, SHELF_HALF_Y, 0.012], pos=[0, 0, -0.012],
                   rgba=[0.74, 0.62, 0.47, 1])
    for ex in (-1, 1):
        for ey in (-1, 1):
            shelf.add_geom(type=BOX, size=[0.015, 0.015, SHELF_Z / 2 - 0.012],
                           pos=[ex * ((sx1 - sx0) / 2 - 0.02), ey * (SHELF_HALF_Y - 0.02), -SHELF_Z / 2],
                           rgba=steel_dark, contype=0, conaffinity=0)
    # the fixture: a low frame round the tote's footprint, 2 cm of play
    fx, fy = TOTE[0] + TOTE_WALL + FIX_PLAY, TOTE[1] + TOTE_WALL + FIX_PLAY
    for s_ in (-1, 1):
        shelf.add_geom(type=BOX, size=[0.006, fy + 0.012, 0.006], pos=[s_ * (fx + 0.006), 0, 0.006], rgba=ORANGE)
        shelf.add_geom(type=BOX, size=[fx, 0.006, 0.006], pos=[0, s_ * (fy + 0.006), 0.006], rgba=ORANGE)

    # the tote's nest on the bench: corner posts round its footprint (3 mm of play), so it neither slides nor
    # turns while it's loaded; it lifts straight out
    tx, ty = TOTE_POS
    nest = world.add_body(name="nest", pos=[tx, ty, T])
    nx_, ny_ = TOTE[0] + TOTE_WALL + 0.003, TOTE[1] + TOTE_WALL + 0.003
    for ex in (-1, 1):
        for ey in (-1, 1):
            nest.add_geom(type=BOX, size=[0.008, 0.008, 0.012], pos=[ex * (nx_ + 0.008), ey * (ny_ + 0.008) * 0.6, 0.012],
                          rgba=steel_dark)

    # the tote: floor, walls, three slots, end brackets with the handle knobs
    tote = world.add_body(name="tote", pos=[tx, ty, T])
    tote.add_freejoint(name="tote")
    hx, hy = TOTE[0] + TOTE_WALL, TOTE[1] + TOTE_WALL
    tote.add_geom(type=BOX, size=[hx, hy, 0.003], pos=[0, 0, 0.003], rgba=blue, mass=0.3, friction=[1.2, 0.01, 0.001])
    for s_ in (-1, 1):
        tote.add_geom(type=BOX, size=[TOTE_WALL / 2, hy, TOTE_H / 2], pos=[s_ * (hx - TOTE_WALL / 2), 0, TOTE_H / 2],
                      rgba=blue, mass=0.02)
        tote.add_geom(type=BOX, size=[hx, TOTE_WALL / 2, TOTE_H / 2], pos=[0, s_ * (hy - TOTE_WALL / 2), TOTE_H / 2],
                      rgba=blue, mass=0.02)
        # bracket out from the end wall, the knob on it
        arm = (HANDLE_Y + KNOB_R - hy) / 2
        tote.add_geom(type=BOX, size=[0.022, arm, 0.004], pos=[0, s_ * (hy + arm), TOTE_H - 0.004],
                      rgba=steel_dark, mass=0.01)
        tote.add_geom(type=CYL, size=[KNOB_R, KNOB_H / 2, 0], pos=[0, s_ * HANDLE_Y, TOTE_H + KNOB_H / 2],
                      rgba=KNOB, mass=0.015, friction=[2.0, 0.03, 0.003], condim=4)
        tote.add_site(name=f"tote_handle_{'l' if s_ > 0 else 'r'}", pos=[0, s_ * HANDLE_Y, TOTE_H + KNOB_H])
    for y in SLOTS_Y:
        r = tote.add_body(pos=[0, y, 0.006])
        _ring(r, SLOT_R, 0.028, 0.003, 12, [0.1, 0.22, 0.5, 1])
    tote.add_site(name="tote_base", pos=[0, 0, 0])
    tote.add_site(name="tote_slot0", pos=[0, SLOTS_Y[0], 0.006])

    # cells in their rack
    rack = world.add_body(name="cell_rack", pos=[0.48, -0.40, T])
    rack.add_geom(type=BOX, size=[0.05, 0.15, 0.002], pos=[0, 0, -0.0015], rgba=grey, contype=0, conaffinity=0)
    for k, (x, y) in enumerate(RACK):
        rr = rack.add_body(pos=[x - 0.48, y + 0.40, 0])
        _ring(rr, SLOT_R - 0.002, 0.012, 0.004, 12, grey)
        c = world.add_body(name=f"cell{k}", pos=[x, y, T + 0.0005])
        c.add_freejoint(name=f"cell{k}")
        c.add_geom(type=CYL, size=[CELL_R, CELL_H / 2 - 0.004, 0], pos=[0, 0, CELL_H / 2], rgba=[0.14, 0.4, 0.75, 1],
                   mass=0.045, friction=[1.6, 0.03, 0.003], condim=4)
        c.add_geom(type=CYL, size=[CELL_R * 0.98, 0.002, 0], pos=[0, 0, CELL_H - 0.002], rgba=ORANGE, mass=0.002,
                   friction=[1.6, 0.03, 0.003], condim=4)
        c.add_geom(type=CYL, size=[CELL_R * 0.98, 0.002, 0], pos=[0, 0, 0.002], rgba=STEEL, mass=0.002)
        c.add_site(name=f"cell{k}_top", pos=[0, 0, CELL_H])
        c.add_site(name=f"cell{k}_base", pos=[0, 0, 0])
    add_weld(spec, "shelf_latch", "shelf", "tote", active=False)


def plan(wb, notes: dict) -> None:
    from pioneer_whole_body import skills as S

    m, d = wb.m, wb.d
    wb.look([TOTE_POS[0], 0.0, T], dur=0.8)
    wb.fingers("l", "relaxed", dur=0.8)
    wb.fingers("r", "relaxed", dur=0.8)
    wb.ready(dur=2.4)
    _hands(wb, HIGH, 0.8)
    wb.walk(0.30)

    # 1. cells into the tote, right hand
    for k, y, yaw in zip(range(3), SLOTS_Y, (0.5, 0.5, 0.5)):
        S.pick_knob(wb, "r", f"cell{k}_top", f"cell{k}", yaw=yaw, lift=0.10)
        over = y - SLOTS_Y[0]                                    # the right hand reaches the right-hand slot only:
        _shuffle(wb, over, carry=True)                            # side-step along to the others
        tp = d.xpos[m.body("tote").id].copy()                    # where the tote is now
        slot = tp + d.xmat[m.body("tote").id].reshape(3, 3) @ np.array([0, y, 0.006])
        S.carry(wb, "r", f"cell{k}_base", slot, clear=0.03, dur=1.4, over=0.08)   # well over the tote's handle
        down = lambda c=k: d.xpos[m.body(f"cell{c}").id][2] < tp[2] + 0.008     # noqa: E731
        for _ in range(3):
            S.align_held(wb, "r", f"cell{k}", f"cell{k}_base", slot, yaw=None, rounds=2)
            S.lower_until(wb, "r", 0.045, down, dur=1.0)
            if down():
                break
            p, r = wb.hand_goal["r"]                 # on the slot's rim: up a little, line up again
            wb.reach("r", p + [0, 0, 0.02], r, 0.5)
            wb.settle()
        S.release(wb, "r", up=0.08)
        wb.arm_joints_to("r", HIGH, 0.8)
        wb.settle()
        _shuffle(wb, -over)
    if cells_in(m, d) < 3:
        raise S.TaskFailed(f"only {cells_in(m, d)} cells in the tote")

    # 2. lift it by its two handles
    wb.look(S.site(wb, "tote_base") + [0, 0, 0.05], dur=0.5)
    S.pick_two(wb, {"l": ("tote_handle_l", -0.5), "r": ("tote_handle_r", 0.5)}, "tote", lift=0.08, shape="lid",
               above=S.GRASP_ABOVE, squeeze=1.25)

    # 3. over to the shelf, crouch and lean, into the fixture
    wb.look([0.5, SHELF_SIDE, SHELF_Z], dur=0.8)
    for _ in range(round(SHELF_SIDE / STEP)):
        wb.step("l", 0.0, STEP, carry=True)
        wb.step("r", 0.0, STEP, carry=True)
    dest = np.array([(SHELF_X[0] + SHELF_X[1]) / 2, SHELF_SIDE, SHELF_Z])
    base = S.site(wb, "tote_base")
    # crouch and lean while the hands carry the tote down over the fixture
    wb.pelvis(z=wb.cmd["base_z"] - CROUCH, pitch=LEAN, dur=1.6)
    S.move_two(wb, [dest[0] - base[0], dest[1] - base[1], 0.03 + dest[2] - base[2]], dur=1.6)
    for dur in (0.5, 0.4):
        base = S.site(wb, "tote_base")
        S.move_two(wb, [dest[0] - base[0], dest[1] - base[1], 0.03 + dest[2] - base[2]], dur=dur)
        wb.wait(0.15)
    latched = lambda: bool(d.eq_active[m.equality("shelf_latch").id])      # noqa: E731
    for attempt in range(3):
        S.align_two(wb, "tote", "tote_base", dest, yaw=0.0, rounds=2)
        for side in ("l", "r"):
            p, r = wb.hand_goal[side]
            wb.reach(side, p - [0, 0, 0.05], r, 1.0)
        wb.until(latched, 1.4)
        wb.wait(0.2)
        if latched():
            break
        S.move_two(wb, [0, 0, 0.03], dur=0.6)       # on the fixture's rim: up a little, line up again
    notes["shelf_tries"] = attempt + 1
    S.let_go(wb, "tote", up=0.1, back=0.0)
    wb.pelvis(z=wb.cmd["base_z"] + CROUCH, pitch=0.0, dur=1.4)
    for side in ("l", "r"):
        p, r = wb.hand_goal[side]
        wb.reach(side, p + [-0.1, 0, 0.1], r, 1.4)
    wb.settle()
    wb.look(dest + [0, 0, 0.05], dur=0.5)
    wb.wait(1.5)
    notes["final"] = dict(cells=cells_in(m, d), latched=bool(d.eq_active[m.equality("shelf_latch").id]),
                          tote=np.round(d.xpos[m.body("tote").id], 3).tolist())


def _shuffle(wb, dy: float, carry: bool = False) -> None:
    """Side-step dy in two half steps (none for 0)."""
    if abs(dy) < 1e-6:
        return
    lead, trail = ("l", "r") if dy > 0 else ("r", "l")
    wb.step(lead, 0.0, dy, carry=carry)
    wb.step(trail, 0.0, dy, carry=carry)


def _hands(wb, pose, dur: float = 1.0) -> None:
    for side in ("l", "r"):
        wb.arm_joints_to(side, pose, dur)
    wb.settle()


from pioneer_whole_body._register import _REGISTRY  # noqa: E402

_REGISTRY["tote_to_shelf"].plan = plan
