"""Manipulation skills built on the whole-body controller: what a demo plan is written in.

Everything works from what the robot can see (object poses from the simulation, as a perception stack would give
them) and acts only through the hands: grasps hold by finger friction, a bolt turns because the fingers twist it.
A skill that fails raises ``TaskFailed`` (the demo is then discarded, like a failed synthetic take).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .controller import WholeBody, palm_down, rot_z
from .parts import PITCH, Thread

# the knob / boss grasp (controller.SHAPES["lid"]): palm site 30 mm behind the knob axis along the fingers,
# palm 13 mm above the knob's top, fingertips squeezed 10 % past the rim
GRASP_BACK = 0.03
GRASP_ABOVE = 0.013
SQUEEZE = 1.1
# turning a knob: the palm lower on it and a firmer squeeze (the fingertips must not slip on the rim)
TWIST_ABOVE = 0.005
TWIST_SQUEEZE = 1.2
CLEAR = 0.07          # approach / retreat height above a grasp
DEBUG = False


class TaskFailed(RuntimeError):
    pass


def site(wb: WholeBody, name: str) -> np.ndarray:
    return wb.d.site_xpos[wb.m.site(name).id].copy()


def body_pos(wb: WholeBody, name: str) -> np.ndarray:
    return wb.d.xpos[wb.m.body(name).id].copy()


def knob_pose(top: np.ndarray, yaw: float, above: float | None = None):
    """Palm pose that holds a knob / boss whose top centre is ``top``, fingers pointing along ``yaw``."""
    r = palm_down(yaw)
    return top + np.array([0, 0, GRASP_ABOVE if above is None else above]) - r[:, 0] * GRASP_BACK, r


def pick_knob(wb: WholeBody, side: str, top_site: str, body: str, yaw: float, lift: float = 0.08,
              speed: float = 1.0, shape: str = "lid") -> None:
    """Grasp a knob-topped part from above and lift it ``lift`` m. ``shape``: "lid" (wrapped over the top, the
    firmest) or "knob" (fingertips round the rim, lets go straight outward: for parts set among others)."""
    top = site(wb, top_site)
    p, r = knob_pose(top, yaw, None if shape == "lid" else TWIST_ABOVE)
    wb.look(top, dur=0.4)
    wb.fingers(side, f"{shape}_pre", dur=0.5)
    wb.reach(side, p + [0, 0, CLEAR], r, 1.2 / speed)
    wb.settle()
    _dbg(wb, side, "pick above")
    wb.reach(side, p, r, 0.7 / speed)
    wb.settle()
    _dbg(wb, side, "pick at")
    wb.fingers(side, shape, amount=SQUEEZE if shape == "lid" else TWIST_SQUEEZE, dur=0.45)
    wb.settle()
    wb.wait(0.15)
    z0 = body_pos(wb, body)[2]
    wb.reach(side, p + [0, 0, lift], r, 0.8 / speed)
    wb.settle()
    if body_pos(wb, body)[2] - z0 < 0.6 * lift:
        raise TaskFailed(f"{side} hand lost {body} while lifting")


def _dbg(wb, side, tag):
    if DEBUG:
        g = wb.hand_goal[side][0]
        c, _ = wb.palm_pose(side, True)
        sm, _ = wb.palm_pose(side, False)
        print(f"{tag}: goal {np.round(g, 3)} ik {np.round(c - g, 3)} sim {np.round(sm - g, 3)}")


def pick_two(wb: WholeBody, grips: dict, body: str, lift: float = 0.06, shape: str = "knob",
             above: float = TWIST_ABOVE, squeeze: float = TWIST_SQUEEZE) -> None:
    """Both hands take a knob each (``grips``: {side: (top_site, yaw)}) and lift ``body`` together."""
    poses = {}
    for side, (top_site, yaw) in grips.items():
        poses[side] = knob_pose(site(wb, top_site), yaw, above)
        wb.fingers(side, f"{shape}_pre", dur=0.5)
        wb.reach(side, poses[side][0] + [0, 0, CLEAR], poses[side][1], 1.3)
    wb.settle()
    for side, (p, r) in poses.items():
        wb.reach(side, p, r, 0.7)
    wb.settle()
    for side in poses:
        wb.fingers(side, shape, amount=squeeze, dur=0.45)
    wb.settle()
    wb.wait(0.2)
    z0 = body_pos(wb, body)[2]
    for side, (p, r) in poses.items():
        wb.reach(side, p + [0, 0, lift], r, 1.0)
    wb.settle()
    if body_pos(wb, body)[2] - z0 < 0.6 * lift:
        raise TaskFailed(f"lost {body} while lifting it with both hands")


def align_two(wb: WholeBody, body: str, point_site: str, target, yaw: float, rounds: int = 3,
              dur: float = 0.6, sym: float = math.pi) -> None:
    """``align_held`` for a part held in both hands: both hands turn about the part's point and shift together."""
    for _ in range(rounds):
        wb.wait(0.15)
        dyaw = _wrap(yaw - body_yaw(wb, body))
        dyaw = (dyaw + sym / 2) % sym - sym / 2
        rz = rot_z(dyaw)
        pt = site(wb, point_site)
        shift = np.asarray(target, float)[:2] - pt[:2]
        for side in ("l", "r"):
            p, r = wb.hand_goal[side]
            p_new = pt + rz @ (p - pt)
            p_new[:2] += shift
            wb.reach(side, p_new, rz @ r, dur)
        wb.settle()
        if DEBUG:
            print(f"align_two: dyaw {math.degrees(dyaw):.1f} err {np.round(shift, 4)}")


def ease_off(wb: WholeBody, sides=("l", "r"), lift: float = 0.003) -> None:
    """Stop pushing: a hand that was lowered onto something until it stopped is still driven toward a target
    below it, pressing the part into its palm. Re-aim each hand where it really is (a hair up)."""
    for s_ in sides:
        p, r = wb.palm_pose(s_, commanded=False)
        _, r_goal = wb.hand_goal[s_]
        wb.reach(s_, p + [0, 0, lift], r_goal, 0.3)
    wb.settle()


def let_go(wb: WholeBody, body: str, sides=("l", "r"), up: float = 0.08, back: float = 0.05,
           tilt: float = 0.0) -> bool:
    """Let go of a part held by its knobs and set down somewhere that keeps it (a wheel on its axle, a plate in a
    tray): open the hands right up where they are, give the fingers time to come off, then lift.
    Returns whether the part stayed down."""
    z0 = body_pos(wb, body)[2]
    ease_off(wb, sides)
    for s_ in sides:            # open with the fingers spread: straight fingers otherwise straddle the knob
        wb.fingers(s_, {"i_abd": -0.26, "r_abd": 0.26, "l_abd": 0.26}, dur=0.6)
    wb.settle()
    wb.wait(0.3)
    if DEBUG:
        hands = {wb.m.body(b).id for b in range(wb.m.nbody) if "_hand_" in wb.m.body(b).name}
        print("let_go contacts:", sorted({(wb.m.body(wb.m.geom_bodyid[c.geom1]).name, wb.m.body(wb.m.geom_bodyid[c.geom2]).name,
                                           wb.m.geom(c.geom1).name or wb.m.geom(c.geom2).name)
                                          for c in wb.d.contact[:wb.d.ncon]
                                          if (wb.m.geom_bodyid[c.geom1] in hands) != (wb.m.geom_bodyid[c.geom2] in hands)}))
    for s_ in sides:            # fingers tipped up off the knob (and back off along them), then up
        p, r = wb.hand_goal[s_]
        away = -r[:, 0].copy()
        away[2] = 0.0
        c, s2 = math.cos(tilt), math.sin(tilt)
        r_up = r @ np.array([[c, 0, s2], [0, 1, 0], [-s2, 0, c]])
        wb.reach(s_, p + back * away / np.linalg.norm(away) + [0, 0, 0.012], r_up, 0.6)
    wb.settle()
    for s_ in sides:
        p, r = wb.hand_goal[s_]
        wb.reach(s_, p + [0, 0, up], r, 0.9)
    wb.settle()
    for s_ in sides:
        wb.fingers(s_, "relaxed", dur=0.4)
    return body_pos(wb, body)[2] - z0 < 0.004


def move_two(wb: WholeBody, shift, dur: float = 1.2) -> None:
    """Both hands (holding one thing) move together by ``shift``."""
    for side in ("l", "r"):
        p, r = wb.hand_goal[side]
        wb.reach(side, p + np.asarray(shift, float), r, dur)
    wb.settle()


def carry(wb: WholeBody, side: str, point_site: str, target: np.ndarray, yaw: float | None = None,
          clear: float = 0.04, dur: float = 1.4, look=True) -> None:
    """Move the held part so that its ``point_site`` comes to rest ``clear`` above ``target`` (an arc over)."""
    p_cmd, r_cmd = wb.hand_goal[side]
    offset = site(wb, point_site) - p_cmd                # where the part sits relative to the commanded palm
    r1 = r_cmd if yaw is None else palm_down(yaw)
    if yaw is not None:                                  # the part turns with the hand
        offset = rot_z(_yaw(r1) - _yaw(r_cmd)) @ offset
    goal = np.asarray(target, float) - offset + [0, 0, clear]
    if look:
        wb.look(target, dur=0.6)
    top_z = max(p_cmd[2], goal[2]) + 0.03
    wb.reach(side, [*(0.5 * (p_cmd[:2] + goal[:2])), top_z], _mid(r_cmd, r1), dur * 0.5)
    wb.settle()
    wb.reach(side, goal, r1, dur * 0.5)
    wb.settle()


def body_yaw(wb: WholeBody, body: str) -> float:
    r = wb.d.xmat[wb.m.body(body).id].reshape(3, 3)
    return math.atan2(r[1, 0], r[0, 0])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def align_held(wb: WholeBody, side: str, body: str, point_site: str, target, yaw: float | None, sym: float = math.pi,
               rounds: int = 3, dur: float = 0.5, level: bool = True) -> None:
    """Hover the held part's ``point_site`` over ``target``, level it and turn it to ``yaw`` (mod ``sym``): look,
    correct, look again, the way a person lines a part up over its seat before putting it down."""
    for _ in range(rounds):
        wb.wait(0.15)
        p, r = wb.hand_goal[side]
        rb = wb.d.xmat[wb.m.body(body).id].reshape(3, 3)
        corr = _rot_between(rb[:, 2], np.array([0.0, 0.0, 1.0])) if level else np.eye(3)
        dyaw = 0.0 if yaw is None else _wrap(yaw - body_yaw(wb, body))
        if sym < 2 * math.pi and yaw is not None:
            dyaw = (dyaw + sym / 2) % sym - sym / 2
        corr = rot_z(dyaw) @ corr
        pt = site(wb, point_site)
        # turn the hand about the part's point, then move the point over the target
        p_new = pt + corr @ (p - pt)
        p_new[:2] += np.asarray(target)[:2] - pt[:2]
        wb.reach(side, p_new, corr @ r, dur)
        wb.settle()
        if DEBUG:
            print(f"align: dyaw {math.degrees(dyaw):.1f} tilt {math.degrees(math.acos(min(1, rb[2, 2]))):.1f} "
                  f"part yaw {math.degrees(body_yaw(wb, body)):.1f} err {np.round(np.asarray(target)[:2] - pt[:2], 4)}")


def _yaw(r: np.ndarray) -> float:
    return math.atan2(r[1, 0], r[0, 0])


def _mid(r0, r1):
    from .controller import _slerp
    return _slerp(r0, r1, 0.5)


def lower_until(wb: WholeBody, side: str, depth: float, cond, dur: float = 1.2) -> bool:
    """Lower a hand by up to ``depth`` m, stopping as soon as ``cond()`` holds."""
    p, r = wb.hand_goal[side]
    wb.reach(side, p - [0, 0, depth], r, dur)
    if DEBUG:
        for _ in range(6):
            wb.wait(dur / 6)
            pc, rc = wb.palm_pose(side, True)
            ps, rs = wb.palm_pose(side, False)
            print(f"lower: goal yaw {math.degrees(_yaw(r)):.1f} cmd {math.degrees(_yaw(rc)):.1f} sim {math.degrees(_yaw(rs)):.1f}"
                  f" z cmd {pc[2]:.3f} sim {ps[2]:.3f} cond {cond()}")
    return wb.until(cond, dur + 0.3)


def press(wb: WholeBody, side: str, tip_site: str, target, depth: float = 0.012, hold: float = 0.6,
          approach: float = 0.06) -> None:
    """Press ``target`` with a fingertip (``tip_site``): hover over it, push ``depth`` in, hold, back off."""
    target = np.asarray(target, float)
    wb.reach_tip(side, tip_site, target + [0, 0, approach], 1.1)
    wb.settle()
    wb.reach_tip(side, tip_site, target - [0, 0, depth], 0.5)
    wb.wait(0.5 + hold)
    wb.reach_tip(side, tip_site, target + [0, 0, approach], 0.5)
    wb.settle()


def release(wb: WholeBody, side: str | tuple = "l", up: float = 0.06, dur: float = 0.5, held: bool = False,
            swing: bool = False, ease: bool = True) -> None:
    """Let go gently: the fingers uncurl, the thumb stays put (moving it would shove the part), then the thumb
    backs off, lift clear, relax. ``side`` may be both hands, ("l", "r"). ``held``: the part sits in something that
    holds it (a socket): open the whole hand first, then lift (a half-open hand would drag it out). ``swing``: then
    swing the thumb aside too. ``ease``: first stop pushing down (see ``ease_off``)."""
    from .robot import hand_joints
    sides = (side,) if isinstance(side, str) else tuple(side)
    if ease:
        ease_off(wb, sides)
    if held:
        for s_ in sides:
            wb.fingers(s_, "knob_pre", dur=0.4)
        wb.settle()
        for s_ in sides:
            p, r = wb.hand_goal[s_]
            wb.reach(s_, p + [0, 0, up], r, dur)
        wb.settle()
        for s_ in sides:
            wb.fingers(s_, "relaxed", dur=0.4)
        return
    uncurl = {f"{f}{j}": 0.15 for f in ("", "i_", "r_", "l_") for j in ("mcp", "pip")}
    keep = {s_: {j[2:]: wb.cmd[j] for j in hand_joints(s_) if j[2:].startswith("th_")} for s_ in sides}
    for s_ in sides:
        wb.fingers(s_, {**uncurl, **keep[s_]}, dur=0.4)
    wb.settle()
    for s_ in sides:
        wb.fingers(s_, {**uncurl, **keep[s_], "th_mcp": 0.0, "th_ip": 0.0}, dur=0.3)
    wb.settle()
    if swing:                                   # and swings out of the way
        for s_ in sides:
            wb.fingers(s_, {**uncurl, "th_c": 0.0, "th_f": 0.0, "th_mcp": 0.0, "th_ip": 0.0}, dur=0.3)
        wb.settle()
    if DEBUG:
        hands = {wb.m.body(b).id for b in range(wb.m.nbody) if "_hand_" in wb.m.body(b).name}
        print("release contacts:", sorted({(wb.m.body(wb.m.geom_bodyid[c.geom1]).name, wb.m.body(wb.m.geom_bodyid[c.geom2]).name)
                                          for c in wb.d.contact[:wb.d.ncon]
                                          if wb.m.geom_bodyid[c.geom1] in hands or wb.m.geom_bodyid[c.geom2] in hands}))
    for s_ in sides:
        p, r = wb.hand_goal[s_]
        wb.reach(s_, p + [0, 0, up], r, dur)
    wb.settle()
    for s_ in sides:
        wb.fingers(s_, "relaxed", dur=0.4)


def insert_bolt(wb: WholeBody, side: str, thread: Thread, bolt: str) -> None:
    """Hold the bolt over its hole, then lower it until the thread catches."""
    mouth = wb.d.xpos[thread.hole_id].copy()
    carry(wb, side, f"{bolt}_tip", mouth, clear=0.03)
    wb.look(mouth, dur=0.3)
    # stand the bolt upright over the hole (it hangs a little crooked in the fingers), watching the tip
    align_held(wb, side, bolt, f"{bolt}_tip", mouth, yaw=None, rounds=3, dur=0.4)
    if not lower_until(wb, side, 0.045, lambda: thread.engaged(wb.d) == bolt, dur=1.6):
        raise TaskFailed(f"{bolt} did not catch the thread of {thread.hole}")
    wb.wait(0.15)


def tighten(wb: WholeBody, side: str, thread: Thread, bolt: str, yaw0: float, yaw1: float,
            max_strokes: int = 12, speed: float = 1.0) -> int:
    """Screw the engaged bolt home: grip the knob at ``yaw0``, turn the hand to ``yaw1`` (clockwise), let go,
    turn back, regrasp; until the bolt is seated. Returns the number of strokes."""
    axis = wb.d.xpos[thread.hole_id].copy()
    top_site = f"{bolt}_top"
    first = True
    # let go of the carrying grip and take the knob the way it's turned
    wb.fingers(side, "knob_pre", dur=0.3)
    wb.settle()
    p0, r0 = wb.hand_goal[side]
    wb.reach(side, p0 + [0, 0, 0.05], r0, 0.4 / speed)
    wb.settle()
    first = False
    for k in range(max_strokes):
        if thread.seated(wb.d):
            return k
        top = site(wb, top_site)
        top[:2] = axis[:2]
        p, r = knob_pose(top, yaw0, TWIST_ABOVE)
        if not first:
            wb.fingers(side, "knob_pre", dur=0.3)
            wb.reach(side, p + [0, 0, 0.05], r, 0.55 / speed)
            wb.settle()
            wb.reach(side, p, r, 0.35 / speed)
            wb.settle()
            wb.fingers(side, "knob", amount=TWIST_SQUEEZE, dur=0.3)
            wb.settle()
        first = False
        t0 = wb.d.qpos[thread.turn]
        a = yaw1 - yaw0
        p2 = top + rot_z(a) @ (p - top)
        p2[2] -= PITCH * abs(a) / (2 * math.pi)            # follow the bolt down
        wb.reach(side, p2, rot_z(a) @ r, 0.85 / speed)
        wb.settle()
        t1 = wb.d.qpos[thread.turn]
        wb.fingers(side, "knob_pre", dur=0.25)            # straight back off the rim (opening fully would drag it)
        wb.settle()
        p3, r3 = wb.hand_goal[side]
        wb.reach(side, p3 + [0, 0, 0.05], r3, 0.3 / speed)       # clear of the knob before turning back
        wb.settle()
        if DEBUG:
            ps, rs = wb.palm_pose(side, False)
            gap = wb.d.site_xpos[thread.tips[thread.engaged(wb.d)]] - wb.d.xpos[thread.hole_id]
            print(f"   weld gap {np.round(gap * 1000, 1)} mm, feed {wb.d.qpos[thread.feed] * 1000:.1f}")
            print(f"stroke {k}: depth {thread.depth(wb.d) * 1000:.1f} mm, turn {t0:.2f} -> twist {t1:.2f} -> "
                  f"back {wb.d.qpos[thread.turn]:.2f} (hand turned {a:.2f})")
    if not thread.seated(wb.d):
        raise TaskFailed(f"{bolt} not seated after {max_strokes} strokes")
    return max_strokes


def tighten_two(wb: WholeBody, jobs: list, max_strokes: int = 12, speed: float = 1.0) -> list[int]:
    """Both hands screw their bolts home at the same time: ``jobs`` = [(side, thread, bolt, yaw0, yaw1), ...].
    Each stroke as in ``tighten``, the hands moving together; a hand whose bolt is home rests."""
    strokes = [0] * len(jobs)
    phase = lambda: wb.settle()                                         # noqa: E731
    for side, *_ in jobs:                                               # off the carrying grip first
        wb.fingers(side, "knob_pre", dur=0.3)
    phase()
    for side, *_ in jobs:
        p0, r0 = wb.hand_goal[side]
        wb.reach(side, p0 + [0, 0, 0.05], r0, 0.4 / speed)
    phase()
    for _ in range(max_strokes):
        active = [j for j in jobs if not j[1].seated(wb.d)]
        if not active:
            break
        plan = {}
        for side, thread, bolt, yaw0, yaw1 in active:
            axis = wb.d.xpos[thread.hole_id]
            top = site(wb, f"{bolt}_top")
            top[:2] = axis[:2]
            plan[side] = (top, *knob_pose(top, yaw0, TWIST_ABOVE), yaw1 - yaw0)
            wb.fingers(side, "knob_pre", dur=0.3)
            wb.reach(side, plan[side][1] + [0, 0, 0.05], plan[side][2], 0.55 / speed)
        phase()
        for side, (top, p, r, a) in plan.items():
            wb.reach(side, p, r, 0.35 / speed)
        phase()
        for side in plan:
            wb.fingers(side, "knob", amount=TWIST_SQUEEZE, dur=0.3)
        phase()
        for side, (top, p, r, a) in plan.items():
            p2 = top + rot_z(a) @ (p - top)
            p2[2] -= PITCH * abs(a) / (2 * math.pi)
            wb.reach(side, p2, rot_z(a) @ r, 0.85 / speed)
        phase()
        for side in plan:
            wb.fingers(side, "knob_pre", dur=0.25)
        phase()
        for side in plan:
            p3, r3 = wb.hand_goal[side]
            wb.reach(side, p3 + [0, 0, 0.05], r3, 0.3 / speed)
        phase()
        for k, j in enumerate(jobs):
            if j in active:
                strokes[k] += 1
        if DEBUG:
            print("strokes", strokes, [round(j[1].depth(wb.d) * 1000, 1) for j in jobs])
    if not all(j[1].seated(wb.d) for j in jobs):
        raise TaskFailed(f"not every bolt seated after {max_strokes} strokes")
    return strokes
