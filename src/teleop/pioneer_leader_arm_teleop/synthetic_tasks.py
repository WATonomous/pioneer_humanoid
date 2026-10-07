"""What the synthetic operator (human_operator.py) does in each scene, for synthetic_teleop.py.

A plan is written like instructions to a person at the leader arm: look where things are, reach,
grasp, check it worked (and try again if not), carry, let go. Where to aim comes from ``op.see`` /
``op.see_offset`` (judged a few mm off); yes/no checks a person reads at a glance (did it come up, did it
click, is the drawer open, how deep is the peg) look at the scene directly. Every move goes through the
operator's human motion model. ``success`` decides whether the take is kept.
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
    op.pause(0.3, 0.8)


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
        op.arc((x, tab_y, tab_top + JAW_HALF + 0.015), high, R=op.down(), grip=rng.uniform(0.3, 0.4), tol=0.005,
               axes=(1, 1, 0))                                                     # over the tab, lined up
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

    op.pause(0.3, 0.8)                     # takes a moment to start after pressing S
    start = op.follower_tcp()

    # 1. open
    grab_tab()
    slide_drawer(S.DRAWER_TRAVEL + 0.005, lambda: S.drawer_travel(model, data) >= S.DRAWER_TRAVEL - 0.012)   # all the way out
    op.set_grip(rng.uniform(0.25, 0.4))
    op.move(op.follower_tcp() + (0, 0, 0.03), tol=0.02, via=True)   # off the tab before moving away

    # 2-4. blocks, in the announced order
    spots = [(0.045, -0.027), (0.045, 0.027), (0.093, 0.0)]
    for k, colour in enumerate(S._order(data)):
        body = f"block_{colour}"
        op.glance()
        for attempt in range(3):
            b = op.see(body)
            yaw = op.see_yaw(body) * rng.uniform(0.4, 1.0)    # roughly lines the jaws up with the block
            op.arc((b[0], b[1], T + S.BLOCK + JAW_HALF + 0.015), high, R=op.down(yaw), grip=rng.uniform(0.35, 0.45),
                   tol=0.005, axes=(1, 1, 0))                                     # over the block, lined up
            op.move((b[0], b[1], T + JAW_HALF + 0.004), tol=0.004)
            op.set_grip(1.0)
            op.move((b[0], b[1], T + 0.10), tol=0.02, corrections=0, axes=(0, 0, 1), via=True)   # up off the table
            if data.xpos[model.body(body).id][2] > T + 0.03:    # it came up with the gripper
                break
            notes.append(f"regrasp {colour}")
            op.set_grip(0.4)
        else:
            raise TaskFailed(f"grasp {colour}")
        sx = min(front() + spots[k][0], S.CAB_X - S.BLOCK / 2 - 0.012) + rng.normal(0, 0.004)   # over the tray's open part
        sy = tab_y + spots[k][1] + rng.normal(0, 0.003)
        op.arc((sx, sy, T + 0.127), high, R=op.down(rng.normal(0, 0.05)), tol=0.006, land=0.75)   # over the front, in
        op.set_grip(rng.uniform(0.35, 0.45))
        op.move((sx, sy, T + 0.16), tol=0.03, via=True)

    # 5. close
    op.glance()
    grab_tab()
    slide_drawer(-0.01, lambda: S.drawer_travel(model, data) <= S.CLOSED_TRAVEL - 0.002)
    op.set_grip(rng.uniform(0.25, 0.4))
    op.move(op.follower_tcp() + (0, 0, 0.03), tol=0.02, via=True)
    _finish(op, rest=start + rng.normal(0, 0.02, 3))


# ----------------------------------------------------------------------------- peg_insert
def peg_plan(op, model, data, notes) -> None:
    import humanoid_mujoco_scenes.peg_insert.scene as S

    rng, T = op.rng, S.TABLE_TOP_Z
    peg = model.body("peg").id
    # Tool point above the peg's bottom when held: the jaws (88 mm long) reach down to ~2 cm above its bottom, so most of
    # the grip is below its middle and pushing on it doesn't twist it in the jaws.
    grip_z = 0.064 + rng.uniform(-0.003, 0.003)
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
            op.arc((p[0], p[1], T + S.PEG_HEIGHT + JAW_HALF + 0.015), high, R=op.down(), grip=rng.uniform(0.3, 0.45),
                   tol=0.004, axes=(1, 1, 0))
            op.move((p[0], p[1], T + grip_z), tol=0.003)
            op.set_grip(1.0)
            op.move((p[0], p[1], T + grip_z + 0.03), tol=0.01, corrections=0, axes=(0, 0, 1))   # up off the table
            if peg_bottom() > T + 0.015:
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

    op.pause(0.3, 0.8)
    start = op.follower_tcp()
    grasp()
    for attempt in range(5):
        if tilt_deg() > 6:
            notes.append("peg crooked: set it down, take it again")
            set_down()
        h = op.see("hole_block")
        off = op.follower_tcp() - op.see("peg")
        op.arc((h[0] + off[0], h[1] + off[1], block_top + 0.012 + S.PEG_HEIGHT / 2 + off[2]), high, tol=0.003,
               speed=0.8, land=0.7)                                             # carried over the block's edge
        align()
        lower_peg_to(block_top + 0.004, 0.6)     # just above the hole: look again
        align(tries=4)
        lower_peg_to(block_top - 0.004, 0.4)     # into the mouth, look again
        align(tries=2)
        # Push it in a bit at a time, feeling for it: if it stops going down it's caught on the rim, so ease off
        # (don't lean on it, it twists in the jaws) and nudge it towards the hole.
        for _ in range(10):
            if in_hole():
                break
            before = peg_bottom()
            x, y = op.pos.goal()[:2]
            op.move((x, y, op.follower_tcp()[2] - 0.01), tol=0.003, corrections=0, speed=0.8)
            op.settle(limit=0.3)
            if peg_bottom() > before - 0.003:
                tcp = op.follower_tcp()
                op.move((x, y, tcp[2] + 0.002), tol=0.003, corrections=0)
                err = op.see_offset("peg", "hole_block")[:2]
                d = err / (np.linalg.norm(err) + 1e-9) * min(np.linalg.norm(err), 0.0015) + rng.normal(0, 0.0005, 2)
                op.nudge((d[0], d[1], 0.0))
        if in_hole():
            break
        notes.append("lift and retry peg")
        lower_peg_to(block_top + 0.012, 0.8)
    else:
        raise TaskFailed("insert")
    # It's in: let go and it slides down the hole; push it home with the closed jaws if it sticks.
    op.set_grip(rng.uniform(0.3, 0.45))
    tcp = op.follower_tcp()
    op.move((tcp[0], tcp[1], tcp[2] + 0.06), tol=0.02, via=True)
    op.settle(limit=0.3)
    if not S.is_inserted(model, data):
        notes.append("push peg home")
        p = op.see("peg")
        op.move((p[0], p[1], p[2] + S.PEG_HEIGHT / 2 + JAW_HALF + 0.01), grip=1.0, tol=0.004, axes=(1, 1, 0))
        op.move((p[0], p[1], block_top + JAW_HALF - 0.005), tol=0.004, corrections=0, speed=0.7)
        op.settle(limit=0.3)
        tcp = op.follower_tcp()
        op.move((tcp[0], tcp[1], high), grip=0.4, tol=0.02, via=True)
    else:
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
    op.pause(0.4, 0.9)
    start = op.follower_tcp()
    head_west = S.SLOT_CENTRE[0] - S.HEAD_DEPTH / 2
    # Fingers forward as at home, jaws closing sideways across the flat tail, tipped down so the palm stays above the
    # tail behind the grasp (level, the wrist housing sits on the strap and the table).
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("y", rng.uniform(20, 30), degrees=True).as_matrix() @ op.R_home

    def tail_at(x):
        """Where the tail is at this x (judged by eye along the strap's last segments)."""
        segs = np.array([op.see(f"strap{i}") for i in range(S.N_SEGMENTS - 8, S.N_SEGMENTS)])
        segs = segs[np.argsort(segs[:, 0])]
        return np.array([x, np.interp(x, segs[:, 0], segs[:, 1]), np.interp(x, segs[:, 0], segs[:, 2])])

    for attempt in range(3):
        g = tail_at(head_west - rng.uniform(0.055, 0.065))   # jaws (89 mm long) clear of the head
        op.arc(g + (0, 0, 0.02), g[2] + 0.07, R=R, grip=rng.uniform(0.0, 0.15), tol=0.004, axes=(1, 1, 0))
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
    grip_z = JAW_HALF + 0.008 + rng.uniform(-0.002, 0.002)   # tool point above a brick's bottom when held (jaw tips clear of the plate)
    high = T + 0.14 + rng.uniform(-0.01, 0.01)
    plate_top = T + S.PLATE_T

    def yaw_of(body):
        R = data.xmat[model.body(body).id].reshape(3, 3)
        return math.atan2(R[1, 0], R[0, 0])

    hand_yaw = [0.0]

    def grasp(colour: str) -> None:
        brick = f"brick_{colour}"
        for attempt in range(3):
            b = op.see(brick)
            hand_yaw[0] = op.see_yaw(brick, symmetry=math.pi) * rng.uniform(0.5, 1.0)
            op.arc((b[0], b[1], b[2] + bz + S.STUD_H + JAW_HALF + 0.012), high, R=op.down(hand_yaw[0]),
                   grip=rng.uniform(0.35, 0.45), tol=0.004, axes=(1, 1, 0))
            op.move((b[0], b[1], b[2] + grip_z), tol=0.003)
            op.set_grip(1.0)
            op.move((b[0], b[1], b[2] + grip_z + 0.035), tol=0.01, corrections=0, axes=(0, 0, 1), via=True)   # up, clear
            if data.xpos[model.body(brick).id][2] > b[2] + 0.015:   # it came up with the gripper
                return
            notes.append(f"regrasp {colour}")
            op.set_grip(0.4)
        raise TaskFailed(f"grasp {colour}")

    def place(colour: str, support: str) -> None:
        brick = f"brick_{colour}"
        # Turn the wrist so the held brick comes square to what it goes on, then carry it over.
        turn = (yaw_of(support) - yaw_of(brick) + math.pi / 2) % math.pi - math.pi / 2 + math.radians(rng.normal(0, 1.5))
        hand_yaw[0] += turn
        top = plate_top if support == "baseplate" else data.xpos[model.body(support).id][2] + bz
        sp = op.see(support)
        off = op.follower_tcp() - op.see(brick)
        lift = top + S.STUD_H + 0.012 - (data.xpos[model.body(brick).id][2])   # brick bottom just over the studs
        op.arc((sp[0] + off[0], sp[1] + off[1], op.follower_tcp()[2] + lift), high, R=op.down(hand_yaw[0]), tol=0.004,
               land=0.6)
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

    def press(colour: str) -> None:
        """Didn't click but nearly there: press it down with the closed jaws on its studs."""
        b = op.see(f"brick_{colour}")
        op.move((b[0], b[1], b[2] + bz + S.STUD_H + JAW_HALF + 0.01), grip=1.0, tol=0.004, axes=(1, 1, 0))
        op.move((b[0], b[1], b[2] + bz + JAW_HALF - 0.004), tol=0.003, corrections=0, speed=0.5)
        op.settle()
        tcp = op.follower_tcp()
        op.move((tcp[0], tcp[1], tcp[2] + 0.04), grip=0.4, tol=0.01, via=True)

    op.pause(0.3, 0.8)
    start = op.follower_tcp()
    order = S._order(data)
    for k, colour in enumerate(order):
        support = "baseplate" if k == 0 else f"brick_{order[k - 1]}"
        op.glance()
        grasp(colour)
        for attempt in range(3):
            place(colour, support)
            if S.snapped_to(model, data, colour) == support:
                break
            if np.linalg.norm(op.see_offset(f"brick_{colour}", support)[:2]) < 0.005:
                notes.append(f"press {colour}")
                press(colour)
                if S.snapped_to(model, data, colour) == support:
                    break
            notes.append(f"re-place {colour}")      # too far off to push in: pick it up again
            grasp(colour)
        else:
            raise TaskFailed(f"place {colour}")
        op.move(op.follower_tcp() + (0, 0, high - op.follower_tcp()[2]), tol=0.02, via=True)
    _finish(op, rest=start + rng.normal(0, 0.02, 3))


TASKS: dict[str, Task] = {
    "drawer_stow": Task(drawer_plan, _all_steps_done("drawer_stow"), "put the blocks in the drawer"),
    "peg_insert": Task(peg_plan, _peg_success, "insert the peg into the hole"),
    "zip_tie": Task(zip_plan, _zip_success, "pull the zip tie tight"),
    "duplo": Task(duplo_plan, _all_steps_done("duplo"), "stack the duplo bricks"),
}
