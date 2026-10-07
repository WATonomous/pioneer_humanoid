"""What the synthetic operator (human_operator.py) does in each scene, for synthetic_teleop.py.

A plan is written like instructions to a person at the leader arm: look where things are, reach,
grasp, check it worked (and try again if not), carry, let go. Positions come from ``op.see`` (judged a
few mm off), never from exact state, and every move goes through the operator's human motion model.
``success`` decides whether the take is kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class TaskFailed(RuntimeError):
    """A plan gave up (e.g. too many failed grasps): the take is discarded."""


@dataclass
class Task:
    plan: Callable          # plan(op, model, data, notes): notes collects what happened (retries, ...)
    success: Callable       # success(model, data) -> bool
    instruction: str        # dataset task text (multi-step scenes label each frame with its step instead)


# Gripper pointing down: the tool point (finger-box centre) is this far above the jaws' lower ends.
JAW_HALF = 0.044


def _all_steps_done(scene: str) -> Callable:
    from humanoid_mujoco_scenes import scene_progress
    progress = scene_progress(scene)

    def success(model, data) -> bool:
        k, n, _ = progress(model, data)
        return k >= n
    return success


def _finish(op, rest=None) -> None:
    """Back off and settle, like an operator before pressing N."""
    if rest is not None:
        op.move(rest, grip=op.rng.uniform(0.0, 0.3), tol=0.03, corrections=0)
    op.pause(0.5, 1.2)


# ----------------------------------------------------------------------------- drawer_stow
def drawer_plan(op, model, data, notes) -> None:
    import humanoid_mujoco_scenes.drawer_stow.scene as S

    rng, T = op.rng, S.T
    high = T + 0.17 + rng.uniform(-0.01, 0.015)        # carrying height: a held block clears the drawer front
    tab_y = sum(S.CAB_Y) / 2
    tab_top = T + S.CAB_HEIGHT - 0.006
    tab_z = tab_top - S.TAB[2] / 2 + 0.005
    jx = 0.0305 + 0.010                    # jaw half-length (along X, gripper down) + margin off the front

    def front():                           # drawer frame origin = its front face
        return op.see("drawer")[0]

    def grab_tab():
        x = front() - S.FRONT_T - jx
        op.move((x, tab_y, high), R=op.down(), grip=rng.uniform(0.3, 0.4), tol=0.02, via=True)
        op.move((x, tab_y, tab_top + JAW_HALF + 0.015), tol=0.005, axes=(1, 1, 0))   # line up above the tab
        op.move((x, tab_y, tab_z), tol=0.004)
        op.set_grip(1.0)
        op.pause(0.1, 0.3)

    def slide_drawer(travel: float, done) -> None:
        """Pull/push the gripped tab until the drawer is at ``travel`` (watching the drawer, not the hand)."""
        for _ in range(4):
            before = S.drawer_travel(model, data)
            x = op.follower_tcp()[0] - (travel - before)
            op.move((x, tab_y + rng.normal(0, 0.002), tab_z), tol=0.01, corrections=0, speed=0.8)
            op.settle()
            if done():
                return
            notes.append(f"drawer at {S.drawer_travel(model, data) * 1000:.0f} mm: again")
            if abs(S.drawer_travel(model, data) - before) < 0.01:   # hand slid off the tab: let go, grab it again
                op.set_grip(0.3)
                op.move(op.follower_tcp() + (0, 0, 0.05), tol=0.02, via=True)
                grab_tab()
        raise TaskFailed("drawer stuck")

    op.pause(0.4, 1.0)                     # takes a moment to start after pressing S
    start = op.follower_tcp()

    # 1. open
    grab_tab()
    slide_drawer(S.DRAWER_TRAVEL + 0.005, lambda: S.drawer_travel(model, data) >= S.DRAWER_TRAVEL - 0.012)   # all the way out
    op.set_grip(rng.uniform(0.25, 0.4))
    op.move(op.follower_tcp() + (0, 0, 0.07), tol=0.02, via=True)

    # 2-4. blocks, in the announced order
    spots = [(0.045, -0.027), (0.045, 0.027), (0.093, 0.0)]
    for k, colour in enumerate(S._order(data)):
        body = f"block_{colour}"
        for attempt in range(3):
            b = op.see(body)
            yaw = op.see_yaw(body) * rng.uniform(0.4, 1.0)    # roughly lines the jaws up with the block
            op.move((b[0], b[1], high), R=op.down(yaw), grip=rng.uniform(0.35, 0.45), tol=0.02, via=True)
            op.move((b[0], b[1], T + S.BLOCK + JAW_HALF + 0.015), tol=0.005, axes=(1, 1, 0))   # hover, line up
            op.move((b[0], b[1], T + JAW_HALF + 0.004), tol=0.004)
            op.set_grip(1.0)
            op.move((b[0], b[1], high), tol=0.02, corrections=0, axes=(0, 0, 1))      # straight up first
            if data.xpos[model.body(body).id][2] > T + 0.06:    # it came up with the gripper
                break
            notes.append(f"regrasp {colour}")
            op.set_grip(0.4)
        else:
            raise TaskFailed(f"grasp {colour}")
        sx = min(front() + spots[k][0], S.CAB_X - S.BLOCK / 2 - 0.012) + rng.normal(0, 0.004)   # over the tray's open part
        sy = tab_y + spots[k][1] + rng.normal(0, 0.003)
        op.move((sx, sy, high), R=op.down(rng.normal(0, 0.05)), tol=0.01, axes=(1, 1, 0))
        op.move((sx, sy, T + 0.127), tol=0.006)
        op.set_grip(rng.uniform(0.35, 0.45))
        op.move((sx, sy, high), tol=0.03, via=True)

    # 5. close
    grab_tab()
    slide_drawer(-0.01, lambda: S.drawer_travel(model, data) <= S.CLOSED_TRAVEL - 0.002)
    op.set_grip(rng.uniform(0.25, 0.4))
    op.move(op.follower_tcp() + (0, 0, 0.08), tol=0.02, via=True)
    _finish(op, rest=start + rng.normal(0, 0.02, 3))


# ----------------------------------------------------------------------------- peg_insert
def peg_plan(op, model, data, notes) -> None:
    import humanoid_mujoco_scenes.peg_insert.scene as S

    rng, T = op.rng, S.TABLE_TOP_Z
    peg = model.body("peg").id
    grip_z = 0.085 + rng.uniform(-0.004, 0.004)   # tool point above the peg's bottom when held
    block_top = T + S.BLOCK_HEIGHT
    high = block_top + grip_z + 0.05

    def peg_bottom():
        return data.xpos[peg][2] - S.PEG_HEIGHT / 2

    def tilt_deg():
        return float(np.degrees(np.arccos(np.clip(data.xmat[peg][8], -1, 1))))

    def in_hole():
        hole = data.xpos[model.body("hole_block").id]
        return (np.all(np.abs(data.xpos[peg][:2] - hole[:2]) < (S.PEG_SIZE + S.CLEARANCE) / 2) and tilt_deg() < 10
                and peg_bottom() < block_top - 0.01)

    def grasp():
        for attempt in range(3):
            p = op.see("peg")
            op.move((p[0], p[1], high), R=op.down(), grip=rng.uniform(0.3, 0.45), tol=0.02, via=True)
            op.move((p[0], p[1], T + S.PEG_HEIGHT + JAW_HALF + 0.015), tol=0.004, axes=(1, 1, 0))
            op.move((p[0], p[1], T + grip_z), tol=0.003)
            op.set_grip(1.0)
            op.move((p[0], p[1], high), tol=0.02, corrections=0, axes=(0, 0, 1))
            if peg_bottom() > T + 0.03:
                return
            notes.append("regrasp peg")
            op.set_grip(0.4)
        raise TaskFailed("grasp peg")

    def set_down():
        """The peg sits crooked in the jaws: stand it back on the table and take it again."""
        p = op.see("peg")
        x, y = S.PEG_POS[0] + rng.uniform(-0.01, 0.01), S.PEG_POS[1] + rng.uniform(-0.01, 0.01)
        off = op.follower_tcp() - p
        op.move((x + off[0], y + off[1], high), tol=0.01, axes=(1, 1, 0))
        op.move((x + off[0], y + off[1], T + S.PEG_HEIGHT / 2 + off[2] + 0.003), tol=0.004, corrections=0, speed=0.6)
        op.set_grip(0.35)
        tcp = op.follower_tcp()
        op.move((tcp[0], tcp[1], high), tol=0.02, via=True)
        grasp()

    def align(tol=0.001, tries=6):
        """Line the peg up over the hole by eye: peg against hole, side by side, not the hand."""
        for _ in range(tries):
            err = op.see_offset("peg", "hole_block")[:2]
            if np.linalg.norm(err) < tol:
                return
            op.nudge((err[0], err[1], 0.0))

    def lower_peg_to(z, speed):
        tcp = op.follower_tcp()
        op.move((tcp[0], tcp[1], z + (tcp[2] - peg_bottom())), tol=0.003, corrections=0, speed=speed)
        op.settle()

    op.pause(0.4, 1.0)
    start = op.follower_tcp()
    grasp()
    for attempt in range(5):
        if tilt_deg() > 6:
            notes.append("peg crooked: set it down, take it again")
            set_down()
        h = op.see("hole_block")
        off = op.follower_tcp() - op.see("peg")
        op.move((h[0] + off[0], h[1] + off[1], block_top + 0.012 + S.PEG_HEIGHT / 2 + off[2]), tol=0.003, axes=(1, 1, 0), speed=0.8)
        align()
        lower_peg_to(block_top + 0.004, 0.6)     # just above the hole: look again
        align(tries=4)
        lower_peg_to(block_top - 0.004, 0.4)     # into the mouth, look again
        align(tries=2)
        lower_peg_to(block_top - 0.035, 0.5)
        # Caught on the rim: feel for the hole with small sideways nudges.
        for _ in range(6):
            if in_hole():
                break
            err = op.see_offset("peg", "hole_block")[:2]
            d = err / (np.linalg.norm(err) + 1e-9) * min(np.linalg.norm(err), 0.0015) + rng.normal(0, 0.0005, 2)
            op.nudge((d[0], d[1], 0.0))
        if in_hole():
            break
        notes.append("lift and retry peg")
        lower_peg_to(block_top + 0.012, 0.8)
    else:
        raise TaskFailed("insert")
    lower_peg_to(T + 0.003, 0.7)   # push it home, let go, back off
    op.set_grip(rng.uniform(0.25, 0.4))
    tcp = op.follower_tcp()
    op.move((tcp[0], tcp[1], high), tol=0.02, via=True)
    _finish(op, rest=start + rng.normal(0, 0.02, 3))


def _peg_success(model, data) -> bool:
    import humanoid_mujoco_scenes.peg_insert.scene as S
    return S.is_inserted(model, data)


# ----------------------------------------------------------------------------- zip_tie
def zip_plan(op, model, data, notes) -> None:
    import humanoid_mujoco_scenes.zip_tie.scene as S

    rng = op.rng
    op.pause(0.6, 1.2)
    start = op.follower_tcp()
    head_west = S.SLOT_CENTRE[0] - S.HEAD_DEPTH / 2
    R = op.R_home @ np.eye(3)   # fingers forward as at home, jaws closing sideways across the flat tail

    def tail_at(x):
        """Where the tail is at this x (judged by eye along the strap's last segments)."""
        segs = np.array([op.see(f"strap{i}") for i in range(S.N_SEGMENTS - 8, S.N_SEGMENTS)])
        segs = segs[np.argsort(segs[:, 0])]
        return np.array([x, np.interp(x, segs[:, 0], segs[:, 1]), np.interp(x, segs[:, 0], segs[:, 2])])

    for attempt in range(3):
        g = tail_at(head_west - rng.uniform(0.055, 0.065))   # jaws (89 mm long) clear of the head
        op.move(g + (0, 0, 0.08), R=R, grip=rng.uniform(0.0, 0.15), tol=0.02, via=True)
        op.move(g + (0, 0, 0.02), tol=0.004, axes=(1, 1, 0))
        op.move(g, tol=0.003)
        op.set_grip(1.0)
        pulled = 0.0
        while pulled < 0.075 and not S.is_tight(model, data):
            stroke = rng.uniform(0.02, 0.04)
            before = S.loop_length(model, data)
            tcp = op.follower_tcp()
            op.move((tcp[0] - stroke, tcp[1] + rng.normal(0, 0.002), tcp[2]), tol=0.006, corrections=0, speed=0.7)
            op.pause(0.1, 0.4)
            pulled += stroke
            if before - S.loop_length(model, data) < 0.003:
                break                                  # tail slipped out of the jaws
        if S.is_tight(model, data):
            break
        notes.append("regrab tail")
        op.set_grip(0.1)
        tcp = op.follower_tcp()
        op.move(tcp + (0, 0, 0.04), tol=0.02, via=True)
    else:
        raise TaskFailed("tighten")
    op.pause(0.2, 0.5)
    op.set_grip(rng.uniform(0.0, 0.2))
    tcp = op.follower_tcp()
    op.move(tcp + (rng.uniform(-0.03, 0.0), 0, 0.07), tol=0.02, via=True)
    _finish(op, rest=start + rng.normal(0, 0.02, 3))


def _zip_success(model, data) -> bool:
    import humanoid_mujoco_scenes.zip_tie.scene as S
    return S.is_tight(model, data)


# ----------------------------------------------------------------------------- duplo
def duplo_plan(op, model, data, notes) -> None:
    import math

    import humanoid_mujoco_scenes.duplo.scene as S

    rng, T = op.rng, S.T
    bz = S.BRICK[2]
    grip_z = JAW_HALF + 0.004 + rng.uniform(-0.002, 0.002)   # tool point above a brick's bottom when held
    high = T + 0.14 + rng.uniform(-0.01, 0.01)
    plate_top = T + S.PLATE_T

    def yaw_of(body):
        R = data.xmat[model.body(body).id].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    op.pause(0.4, 1.0)
    start = op.follower_tcp()
    hand_yaw = 0.0
    for k, colour in enumerate(S._order(data)):
        brick = f"brick_{colour}"
        support = "baseplate" if k == 0 else f"brick_{S._order(data)[k - 1]}"
        for attempt in range(3):
            b = op.see(brick)
            hand_yaw = op.see_yaw(brick, symmetry=math.pi) * rng.uniform(0.5, 1.0)
            op.move((b[0], b[1], high), R=op.down(hand_yaw), grip=rng.uniform(0.35, 0.45), tol=0.02, via=True)
            op.move((b[0], b[1], b[2] + bz + S.STUD_H + JAW_HALF + 0.012), tol=0.004, axes=(1, 1, 0))
            op.move((b[0], b[1], b[2] + grip_z), tol=0.003)
            op.set_grip(1.0)
            op.move((b[0], b[1], high), tol=0.02, corrections=0, axes=(0, 0, 1))
            if data.xpos[model.body(brick).id][2] > T + 0.03:
                break
            notes.append(f"regrasp {colour}")
            op.set_grip(0.4)
        else:
            raise TaskFailed(f"grasp {colour}")
        # Turn the wrist so the held brick comes square to what it goes on, then carry it over.
        turn = (yaw_of(support) - yaw_of(brick) + math.pi / 2) % math.pi - math.pi / 2 + math.radians(rng.normal(0, 1.5))
        hand_yaw += turn
        top = plate_top if k == 0 else data.xpos[model.body(support).id][2] + bz
        sp = op.see(support)
        off = op.follower_tcp() - op.see(brick)
        lift = top + S.STUD_H + 0.012 - (data.xpos[model.body(brick).id][2])   # brick bottom just over the studs
        op.move((sp[0] + off[0], sp[1] + off[1], op.follower_tcp()[2]), R=op.down(hand_yaw), tol=0.01, axes=(1, 1, 0))
        op.move(op.follower_tcp() + (0, 0, lift), tol=0.004, axes=(0, 0, 1), corrections=1, speed=0.8)
        for _ in range(5):   # line it up by eye, brick against the one below
            err = op.see_offset(brick, support)[:2]
            if np.linalg.norm(err) < 0.0015:
                break
            op.nudge((err[0], err[1], 0.0))
        tcp = op.follower_tcp()
        op.move((tcp[0], tcp[1], tcp[2] - (data.xpos[model.body(brick).id][2] - top) - 0.002), tol=0.003, corrections=0, speed=0.5)
        op.settle()
        op.set_grip(rng.uniform(0.35, 0.45))
        tcp = op.follower_tcp()
        op.move((tcp[0], tcp[1], tcp[2] + 0.03), tol=0.01, via=True)
        if S.snapped_to(model, data, colour) != support:
            # Didn't click: press it down with the closed jaws on its studs.
            notes.append(f"press {colour}")
            b = op.see(brick)
            op.move((b[0], b[1], b[2] + bz + S.STUD_H + JAW_HALF + 0.01), grip=1.0, tol=0.004, axes=(1, 1, 0))
            op.move((b[0], b[1], b[2] + bz + JAW_HALF - 0.004), tol=0.003, corrections=0, speed=0.5)
            op.settle()
            tcp = op.follower_tcp()
            op.move((tcp[0], tcp[1], tcp[2] + 0.04), grip=0.4, tol=0.01, via=True)
        op.move(op.follower_tcp() + (0, 0, high - op.follower_tcp()[2]), tol=0.02, via=True)
    _finish(op, rest=start + rng.normal(0, 0.02, 3))


TASKS: dict[str, Task] = {
    "drawer_stow": Task(drawer_plan, _all_steps_done("drawer_stow"), "put the blocks in the drawer"),
    "peg_insert": Task(peg_plan, _peg_success, "insert the peg into the hole"),
    "zip_tie": Task(zip_plan, _zip_success, "pull the zip tie tight"),
    "duplo": Task(duplo_plan, _all_steps_done("duplo"), "stack the duplo bricks"),
}
