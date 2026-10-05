"""Type any text on the keyboard scene with both arms. Scripted (IK + smooth joint moves), no learning.

    MUJOCO_GL=egl python -m humanoid_mujoco_scenes.keyboard.typer --text "hello world" --video hello.mp4
    python -m humanoid_mujoco_scenes.keyboard.typer --check-reach     # every key reachable?

Each key goes to the arm on its side (left arm y >= 0); if that arm can't reach it, the other arm.
An arm picks up its stylus the first time it is needed and keeps it, parking above its holder while
the other arm types; it types with the stylus leaning 20 deg toward the middle (see LEAN).

Every move is a straight line for the stylus tip, solved by IK every centimetre so the arm can't
swing through the keyboard, and played as one smooth minimum-jerk motion. Before each press the tip
is measured and the command corrected for the arm's sag; after it, the typer checks which key
registered and retries a miss 1 mm deeper. A stylus that has slid or tipped in the hand is put back
in its holder and picked up again.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from typing import NamedTuple

import mujoco
import numpy as np

from humanoid_mujoco_scenes import make_model, scene_step
from pioneer_humanoid.arm_params import (
    DEFAULT_JOINT_POS, LEFT_ARM_JOINTS, LEFT_GRIPPER_CLOSED, LEFT_GRIPPER_OPEN, RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_CLOSED, RIGHT_GRIPPER_OPEN,
)
from pioneer_humanoid.mujoco_bimanual_arm import set_home

from . import scene as kb

ARMS = {
    "left": dict(joints=LEFT_ARM_JOINTS, wrist="link6l", fingers=("link7l", "link8l"),
                 open=LEFT_GRIPPER_OPEN, closed=LEFT_GRIPPER_CLOSED),
    "right": dict(joints=RIGHT_ARM_JOINTS, wrist="link6", fingers=("link7", "link8"),
                  open=RIGHT_GRIPPER_OPEN, closed=RIGHT_GRIPPER_CLOSED),
}
DOWN = np.array([0.0, 0.0, -1.0])
HOVER = 0.015          # stylus tip above the keycap between presses
PRESS_DEPTH = 0.004    # commanded below the keycap top (actuation at 2 mm; arm sag under the key spring eats ~1.5 mm); +1 mm per retry
GRASP_DEPTH = 0.06     # fingertips this far down the 100 mm handle
AIM_TOL = 0.0007       # stylus tip within this of the hover point (m) before pressing
AIM_ITERS = 4
NULL_GAIN = 0.1        # pull of the spare DOF toward home, per IK iteration
MAX_JUMP = 0.2        # rad, largest joint change accepted between two waypoints (STEP apart) of a move
STEP = 0.01           # IK waypoint spacing along a move (m)
CLEAR = 0.04          # stylus lifted this far after grasping; parked with its tip this far above the holder
# stylus-tip height for moves between keys > 3 cm apart and to/from the holders; reach shrinks higher up
TRAVEL_Z = kb.TABLE_TOP_Z + kb.KB_SIZE[2] + kb.STEM + 0.055
NOMINAL_TIP = 0.072    # stylus tip beyond the fingertips once held, before one is (for reach checks)
RETRIES = 2
# Move durations x PACE at --speed 1. The wrist motor (0.73 N m peak against a 341 N m/rad PD) saturates
# on 2 mrad of lag; at PACE 1.5 it never does (at 1.0 it did for ~0.03% of the time, briefly jolting the arm).
PACE = 1.5
# Typing direction of each stylus: leaning 20 deg toward the keyboard's centre line, like a slanted
# pen. Straight down, the shoulder's inward limit (joint 2) leaves the centre keys at the edge of
# reach; the lean adds 5-10 cm across the middle. Straight down only at the holders.
_LEAN = np.radians(20)
LEAN = {"left": np.array([0.0, -np.sin(_LEAN), -np.cos(_LEAN)]),
        "right": np.array([0.0, np.sin(_LEAN), -np.cos(_LEAN)])}
REGRIP_SLIP = 0.015    # m the stylus tip may slide in the hand before re-gripping (aim corrects the rest)
REGRIP_TILT = 5.0      # deg it may tip
# Hand yaw for grasping: jaws square to the stylus handle's faces (the handles stand at kb.STYLUS_YAW);
# 45 and 135 deg are each inside that arm's reachable range at the holder.
GRASP_YAW = {"left": np.radians(kb.STYLUS_YAW), "right": np.radians(kb.STYLUS_YAW + 90)}


class Tool(NamedTuple):
    """What the IK places, fixed in the wrist link's frame: a point, the direction it points (down,
    when placed), and the jaw direction (the fingers' travel axis), whose heading is the hand's yaw."""
    point: np.ndarray
    axis: np.ndarray
    jaw: np.ndarray


def _wrist(m, d, arm):
    b = m.body(ARMS[arm]["wrist"]).id
    return d.xpos[b], d.xmat[b].reshape(3, 3)


def _jaw(m, d, arm):
    """The fingers' travel axis (their link Y), world frame."""
    return d.xmat[m.body(ARMS[arm]["fingers"][0]).id].reshape(3, 3)[:, 1]


def finger_tool(m: mujoco.MjModel, d: mujoco.MjData, arm: str) -> Tool:
    """Centre between the fingertips, pointing along the fingers. Fixed in the wrist frame: the
    fingers close symmetrically along their own Y, so neither changes with the grip."""
    tips = []
    for body in ARMS[arm]["fingers"]:
        g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(body).id
                 and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_contype[i])
        axis = -d.geom_xmat[g].reshape(3, 3)[:, 2]  # finger boxes run along their link's z, tip at -z
        tips.append(d.geom_xpos[g] + axis * m.geom_size[g][2])
    pos, R = _wrist(m, d, arm)
    return Tool(R.T @ (np.mean(tips, axis=0) - pos), R.T @ axis, R.T @ _jaw(m, d, arm))


def stylus_tip(m: mujoco.MjModel, d: mujoco.MjData, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """World (tip point, direction from handle to tip) of an arm's stylus."""
    b = m.body(f"stylus_{arm}").id
    R = d.xmat[b].reshape(3, 3)
    return d.xpos[b] - R[:, 2] * kb.STYLUS_TIP[1], -R[:, 2]


def stylus_tool(m: mujoco.MjModel, d: mujoco.MjData, arm: str) -> Tool:
    """The held stylus's tip and axis in the wrist frame, as actually grasped."""
    tip, axis = stylus_tip(m, d, arm)
    pos, R = _wrist(m, d, arm)
    return Tool(R.T @ (tip - pos), R.T @ axis, R.T @ _jaw(m, d, arm))


def yaw_of(R: np.ndarray, tool: Tool) -> float:
    """The hand's yaw, as _jaw_goal defines it: heading of the jaw slid along the tool axis until level
    (so _jaw_goal(yaw_of(R, tool), R @ tool.axis) is the jaw itself, leaning stylus or not)."""
    jaw, axis = R @ tool.jaw, R @ tool.axis
    level = jaw - jaw[2] / axis[2] * axis
    return float(np.arctan2(level[1], level[0]))


def _jaw_goal(yaw, down):
    """Jaw direction with heading `yaw`, square to the pointing direction `down`."""
    h = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    h -= (h @ down) * down
    return h / np.linalg.norm(h)


def solve_ik(m, d, arm, target, tool: Tool, yaw=None, down=DOWN, iters=400):
    """Damped least squares on d.qpos (kinematics only): `tool` to `target`, pointing along `down`.

    yaw=None: the twist about the vertical is left free (5 of the 6 DOF fixed) and pulled toward the
    home posture -- finds a pose wherever one exists; used for a move's end point. yaw given: the
    pose is fully fixed (6 DOF), so from a nearby seed the answer is the nearby solution -- used for
    the waypoints of a move, so they form one smooth path instead of hopping between the postures a
    free twist allows (that hopping whipped the arm and shook the stylus loose).
    Returns (position error, orientation error, q)."""
    joints = ARMS[arm]["joints"]
    dofs = [m.joint(j).dofadr[0] for j in joints]
    qadr = [m.joint(j).qposadr[0] for j in joints]
    lim = np.array([m.joint(j).range for j in joints])
    rest = np.array([DEFAULT_JOINT_POS[j] for j in joints])
    wrist = m.body(ARMS[arm]["wrist"]).id
    jaw_goal = None if yaw is None else _jaw_goal(yaw, down)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def error():
        pos, R = _wrist(m, d, arm)
        p, a = pos + R @ tool.point, R @ tool.axis
        rot = np.cross(a, down) if jaw_goal is None else 0.5 * (np.cross(a, down) + np.cross(R @ tool.jaw, jaw_goal))
        return p, np.concatenate([target - p, rot])

    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)  # mj_jac needs both
        p, err = error()
        if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
            break
        mujoco.mj_jac(m, d, jp, jr, p, wrist)
        J = np.vstack([jp[:, dofs], jr[:, dofs]])
        J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
        dq = J_pinv @ err
        if jaw_goal is None:
            dq += (np.eye(len(dofs)) - J_pinv @ J) @ (NULL_GAIN * (rest - d.qpos[qadr]))
        d.qpos[qadr] = np.clip(d.qpos[qadr] + np.clip(dq, -0.2, 0.2), lim[:, 0], lim[:, 1])
    mujoco.mj_kinematics(m, d)
    _, err = error()
    return np.linalg.norm(err[:3]), np.linalg.norm(err[3:]), d.qpos[qadr].copy()


def _spline(s: np.ndarray, qs: np.ndarray):
    """Cubic spline through (s_i, q_i), C2, with zero slope at both ends. Returns q(u), u in [0, 1]."""
    n = len(s) - 1
    if n < 2:
        return lambda u: qs[0] + (qs[-1] - qs[0]) * u
    h = np.diff(s)
    slope = np.diff(qs, axis=0) / h[:, None]
    A, rhs = np.zeros((n + 1, n + 1)), np.zeros_like(qs)
    A[0, :2] = 2 * h[0], h[0]
    rhs[0] = 6 * slope[0]
    A[n, n - 1:] = h[-1], 2 * h[-1]
    rhs[n] = -6 * slope[-1]
    for i in range(1, n):
        A[i, i - 1:i + 2] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 6 * (slope[i] - slope[i - 1])
    M = np.linalg.solve(A, rhs)  # second derivatives at the knots

    def q(u):
        i = min(max(int(np.searchsorted(s, u) - 1), 0), n - 1)
        a, b, hi = s[i + 1] - u, u - s[i], h[i]
        return (M[i] * a**3 + M[i + 1] * b**3) / (6 * hi) + (qs[i] / hi - M[i] * hi / 6) * a \
            + (qs[i + 1] / hi - M[i + 1] * hi / 6) * b
    return q


class Typer:
    def __init__(self, video: str | None = None, size=(480, 640), fps=30, speed=1.0):
        self.m = make_model("keyboard")
        self.d = mujoco.MjData(self.m)
        set_home(self.m, self.d)
        self.scratch = mujoco.MjData(self.m)
        self.scene_step = scene_step("keyboard")
        self.held: dict[str, Tool] = {}      # arm -> its stylus, as currently held
        self.grasped: dict[str, Tool] = {}   # arm -> its stylus, as first grasped (to see it slip)
        self.regrips = 0
        self.slowdown = PACE / speed
        self.active: str | None = None       # arm currently over the keyboard
        self.typed: list[str] = []           # every key that registered, in order
        self._down = set()
        self._ffmpeg = self._renderer = None
        if video:
            self._open_video(video, size, fps)
        mujoco.mj_forward(self.m, self.d)
        self.fingers = {arm: finger_tool(self.m, self.d, arm) for arm in ARMS}

    # ------------------------------------------------------------------ low level
    def _open_video(self, path, size, fps):
        if not shutil.which("ffmpeg"):
            raise SystemExit("--video needs ffmpeg on PATH")
        h, w = size
        self._renderer = mujoco.Renderer(self.m, h, w)
        self._cam = mujoco.MjvCamera()
        self._cam.lookat[:] = [kb.KB_POS[0], kb.KB_POS[1], kb.TABLE_TOP_Z + 0.05]
        self._cam.distance, self._cam.azimuth, self._cam.elevation = 0.8, 160, -35
        self._every = max(1, round(1 / (fps * self.m.opt.timestep)))
        self._ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
            stdin=subprocess.PIPE)

    def _frame(self):
        from PIL import Image, ImageDraw, ImageFont
        self._renderer.update_scene(self.d, self._cam)
        im = Image.fromarray(self._renderer.render())
        draw, font = ImageDraw.Draw(im), ImageFont.load_default(size=22)
        draw.text((12, 10), f"text:  {kb.get_text(self.m, self.d)}", fill=(255, 255, 255), font=font)
        shown = "".join(" " if k == "SPACE" else k for k in self.typed)
        draw.text((12, 38), f"typed: {shown}", fill=(120, 230, 120), font=font)
        self._ffmpeg.stdin.write(np.asarray(im).tobytes())

    def close(self):
        if self._ffmpeg:
            self._ffmpeg.stdin.close()
            self._ffmpeg.wait()
            self._renderer.close()

    def tick(self):
        self.scene_step(self.m, self.d)
        mujoco.mj_step(self.m, self.d)
        down = set(kb.pressed_keys(self.m, self.d))
        self.typed += [k for k in kb.KEYS if k in down - self._down]
        self._down = down
        if self._ffmpeg and round(self.d.time / self.m.opt.timestep) % self._every == 0:
            self._frame()

    def wait(self, seconds):
        for _ in range(int(seconds / self.m.opt.timestep)):
            self.tick()

    def _act(self, arm):
        return [self.m.actuator(j).id for j in ARMS[arm]["joints"]]

    def grip(self, arm, closed):
        for joint, value in ARMS[arm]["closed" if closed else "open"].items():
            self.d.ctrl[self.m.actuator(joint).id] = value
        self.wait(0.6 if closed else 0.2)

    def _home(self, arm):
        return np.array([DEFAULT_JOINT_POS[j] for j in ARMS[arm]["joints"]])

    def ik(self, arm, target, tool: Tool, yaw=None, seed=None, down=DOWN):
        """Joint targets putting `tool` at `target` (with hand yaw `yaw`, or free), or None.
        Seeded from home unless `seed` is given, so an end point's posture depends only on the target."""
        joints = ARMS[arm]["joints"]
        self.scratch.qpos[:] = self.d.qpos
        self.scratch.qpos[[self.m.joint(j).qposadr[0] for j in joints]] = self._home(arm) if seed is None else seed
        ep, eo, q = solve_ik(self.m, self.scratch, arm, np.asarray(target, float), tool, yaw, down)
        return q if ep < 2e-3 and eo < 2e-2 else None

    def _fk(self, arm, q):
        self.scratch.qpos[:] = self.d.qpos
        self.scratch.qpos[[self.m.joint(j).qposadr[0] for j in ARMS[arm]["joints"]]] = q
        mujoco.mj_kinematics(self.m, self.scratch)
        return _wrist(self.m, self.scratch, arm)

    def tool_pose(self, arm, tool: Tool) -> tuple[np.ndarray, np.ndarray]:
        """(point, axis) of `tool` with the arm at its current joint targets (not its sagging actual pose)."""
        pos, R = self._fk(arm, self.d.ctrl[self._act(arm)])
        return pos + R @ tool.point, R @ tool.axis

    def go(self, arm, target, tool: Tool, seconds=None, lift=None, yaw=None, down=DOWN):
        """Move `tool` to `target` along a straight line (via height `lift` first, if given: up, across,
        down), ending with the tool pointing along `down` (blending to it from where it points now).
        IK every STEP along the way, with the hand's yaw fixed at each
        waypoint and turning from the start yaw to the end posture's, each seeded from the last, and
        any joint jump refused -- so the arm follows one smooth path and can't swing through the
        keyboard. End posture: `yaw` if given; else keep the current yaw if that path works, else the
        free-yaw IK pose. One minimum-jerk profile over the whole path."""
        target = np.asarray(target, float)
        q_now = self.d.ctrl[self._act(arm)].copy()
        start, axis = self.tool_pose(arm, tool)
        if axis @ DOWN < np.cos(np.radians(40)):
            # Not pointing down yet (from home: fingers forward), so no straight line from here; go
            # to the end posture in joint space -- the arm starts well above the table.
            q_end = self.ik(arm, target, tool, yaw, down=down)
            if q_end is None:
                raise RuntimeError(f"{arm} arm can't reach {np.round(target, 3)}")
            return self._follow(arm, [q_now, q_end], [0.0, 1.0], seconds or 1.5)

        corners = [start, target]
        if lift is not None:
            corners[1:1] = [np.r_[start[:2], lift], np.r_[target[:2], lift]]
        points, dist = [start], [0.0]
        for a, b in zip(corners, corners[1:]):
            if np.linalg.norm(b - a) < 1e-6:
                continue
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / STEP)))
            for i in range(1, n + 1):
                points.append(a + (b - a) * i / n)
                dist.append(dist[-1] + np.linalg.norm(b - a) / n)
        if len(points) == 1:
            return
        frac = np.array(dist[1:]) / max(dist[-1], 1e-9)
        downs = [(1 - f) * axis + f * down for f in frac]
        downs = [v / np.linalg.norm(v) for v in downs]

        yaw0 = yaw_of(self._fk(arm, q_now)[1], tool)
        ends = [yaw] if yaw is not None else [yaw0, None]   # None: whatever the free-yaw IK picks
        for end_yaw in ends:
            q_end = self.ik(arm, target, tool, end_yaw, seed=q_now if end_yaw == yaw0 else None, down=down)
            if q_end is None:
                continue
            turn = (yaw_of(self._fk(arm, q_end)[1], tool) - yaw0 + np.pi) % (2 * np.pi) - np.pi
            qs = self._path(arm, tool, q_now, points[1:], downs, yaw0, turn, frac)
            if qs is not None:
                break
        else:
            raise RuntimeError(f"{arm} arm can't move smoothly to {np.round(target, 3)}")
        if seconds is None:
            seconds = float(np.clip(0.3 + 4.0 * dist[-1] + 0.3 * abs(turn), 0.35, 2.5))
        self._follow(arm, qs, dist, seconds)

    def _path(self, arm, tool, q_now, points, downs, yaw0, turn, frac):
        """Joint waypoints through `points`, or None. Yaw: turn evenly the short way; failing that the
        long way; failing that hold the start yaw and turn over the last stretch."""
        long = turn - np.sign(turn) * 2 * np.pi if abs(turn) > 1e-6 else 2 * np.pi
        for profile in (turn * frac, long * frac, turn * np.clip((frac - 0.7) / 0.3, 0, 1)):
            qs = [q_now]
            for p, dn, dyaw in zip(points, downs, profile):
                q = self.ik(arm, p, tool, yaw0 + dyaw, seed=qs[-1], down=dn)
                if q is None or np.abs(q - qs[-1]).max() > MAX_JUMP:
                    break
                qs.append(q)
            else:
                return qs
        return None

    def _follow(self, arm, qs, dist, seconds):
        """Track joint waypoints `qs` (at path lengths `dist`) with one minimum-jerk profile."""
        seconds *= self.slowdown
        qs, s = np.array(qs), np.array(dist) / max(dist[-1], 1e-9)
        # A spline through the waypoints rather than straight segments, whose corners are steps in
        # joint velocity.
        path = _spline(s, qs)
        act, n = self._act(arm), max(1, int(seconds / self.m.opt.timestep))
        for i in range(1, n + 1):
            t = i / n
            self.d.ctrl[act] = path(10 * t**3 - 15 * t**4 + 6 * t**5)
            self.tick()

    # ------------------------------------------------------------------ skills
    def _stylus_above_holder(self, arm):
        hx, hy = kb.STYLUS_POS[arm]
        top = kb.TABLE_TOP_Z + kb.STYLUS_TIP[1] + 0.002 + kb.STYLUS_HANDLE[1]
        return np.array([hx, hy, top])

    def pick(self, arm):
        top, fingers, yaw = self._stylus_above_holder(arm), self.fingers[arm], GRASP_YAW[arm]
        self.grip(arm, False)
        self.go(arm, top + [0, 0, 0.06], fingers, lift=top[2] + 0.06, yaw=yaw)
        self.go(arm, top - [0, 0, GRASP_DEPTH], fingers, seconds=0.8, yaw=yaw)
        self.grip(arm, True)
        self.go(arm, top - [0, 0, GRASP_DEPTH - CLEAR], fingers, seconds=0.6, yaw=yaw)
        self.held[arm] = self.grasped[arm] = stylus_tool(self.m, self.d, arm)

    def regrip_if_slipped(self, arm) -> bool:
        """Re-measure the stylus in the hand (the IK aims with that); if it has slid or tipped too far,
        stand it back in its holder, let go and pick it up again. A pinched stylus creeps a little each
        move (here mostly MuJoCo's soft contacts; a real gripper loosens too). On hardware the slip
        would come from the wrist camera; here it is read from the sim."""
        self.held[arm] = now = stylus_tool(self.m, self.d, arm)
        first = self.grasped[arm]
        tilt = np.degrees(np.arccos(np.clip(now.axis @ first.axis, -1, 1)))
        if np.linalg.norm(now.point - first.point) < REGRIP_SLIP and tilt < REGRIP_TILT:
            return False
        hx, hy = kb.STYLUS_POS[arm]
        # Square handle into a square hole: turn the hand so the handle's faces line up with the holder
        # (allowing for any twist the stylus has picked up in the hand), then centre the tip over it.
        R = self.d.xmat[self.m.body(f"stylus_{arm}").id].reshape(3, 3)
        hand = yaw_of(_wrist(self.m, self.d, arm)[1], now)
        twist = (np.arctan2(R[1, 0], R[0, 0]) - hand) % (np.pi / 2)    # handle faces vs hand, mod 90 deg
        yaw = np.radians(kb.STYLUS_YAW) - twist
        yaw += np.round((GRASP_YAW[arm] - yaw) / (np.pi / 2)) * (np.pi / 2)  # the equivalent nearest the grasp yaw
        entry = self.aim(arm, [hx, hy, kb.TABLE_TOP_Z + kb.HOLDER_H + 0.01], yaw=yaw)
        self.go(arm, entry - [0, 0, kb.HOLDER_H + 0.005], now, seconds=0.8, yaw=yaw)  # tip 5 mm off the table
        self.grip(arm, False)
        self.wait(0.3)
        self.go(arm, self._stylus_above_holder(arm) + [0, 0, 0.06], self.fingers[arm], yaw=GRASP_YAW[arm])
        self.pick(arm)
        self.regrips += 1
        return True

    def park(self, arm):
        """Stylus still held, lifted clear above its holder, out of the other arm's way."""
        hx, hy = kb.STYLUS_POS[arm]
        self.go(arm, [hx, hy, kb.TABLE_TOP_Z + kb.HOLDER_H + CLEAR], self.held[arm], lift=TRAVEL_Z)

    def aim(self, arm, point, yaw=None, down=DOWN) -> np.ndarray:
        """Put the stylus tip at `point`, correcting for the arm's sag: move, measure where the tip
        actually is, shift the command by the error, repeat. Returns the corrected command point.
        (Here the tip is read from the sim; on hardware: joint encoders + the grasp measured at pick-up.)"""
        tool, cmd = self.held[arm], np.array(point, float)
        far = np.linalg.norm(self.tool_pose(arm, tool)[0][:2] - cmd[:2]) > 0.03
        for i in range(AIM_ITERS):
            self.go(arm, cmd, tool, seconds=None if i == 0 else 0.2, lift=TRAVEL_Z if far and i == 0 else None,
                    yaw=yaw, down=down)
            self.wait(0.12)
            err = point - stylus_tip(self.m, self.d, arm)[0]
            if np.linalg.norm(err) < AIM_TOL:
                break
            cmd = cmd + err
        return cmd

    def press(self, arm, key, depth=PRESS_DEPTH) -> list[str]:
        """One press; returns the keys that registered during it."""
        before = len(self.typed)
        down = LEAN[arm]
        hover = self.aim(arm, kb.key_top(self.m, self.d, key) + [0, 0, HOVER], down=down)
        self.go(arm, hover - [0, 0, HOVER + depth], self.held[arm], seconds=0.3, down=down)
        self.wait(0.08)
        self.go(arm, hover, self.held[arm], seconds=0.25, down=down)
        self.wait(0.03)
        return self.typed[before:]

    def arm_for(self, key) -> str:
        """The arm on the key's side, or the other one if that can't reach it."""
        top = kb.key_top(self.m, self.d, key)
        first = "left" if top[1] >= 0 else "right"
        for arm in (first, "right" if first == "left" else "left"):
            f = self.fingers[arm]
            tool = self.held.get(arm) or Tool(f.point + f.axis * NOMINAL_TIP, f.axis, f.jaw)
            if all(self.ik(arm, top + [0, 0, dz], tool, down=LEAN[arm]) is not None
                   for dz in (HOVER, -PRESS_DEPTH - 0.002)):
                return arm
        raise RuntimeError(f"neither arm can reach key {key}")

    def type_text(self, text: str, log=print) -> bool:
        kb.set_text(self.m, self.d, text)
        for char in text:
            key = kb.key_for(char)
            arm = self.arm_for(key)
            if self.active and self.active != arm:
                self.park(self.active)
            self.active = arm
            if arm not in self.held:
                self.pick(arm)
            elif self.regrip_if_slipped(arm):
                log(f"  ({arm} arm re-gripped its stylus)")
            for attempt in range(1 + RETRIES):
                got = self.press(arm, key, PRESS_DEPTH + 0.001 * attempt)
                if key in got:
                    break
                log(f"  {key}: miss (registered {got or 'nothing'}), retrying")
            extra = [k for k in got if k != key]
            log(f"  {key!s:5} {arm:5} arm  {'ok' if key in got else 'FAILED'}"
                + (f"  (also pressed {extra})" if extra else ""))
        if self.active:
            self.park(self.active)
        self.wait(0.3)
        done, total, _ = kb.progress(self.m, self.d)
        return done == total


def check_reach() -> bool:
    t = Typer()
    ok = True
    for key in kb.KEYS:
        try:
            print(f"  {key:5} {t.arm_for(key)}")
        except RuntimeError as e:
            print(f"  {key:5} UNREACHABLE ({e})")
            ok = False
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--text", default="hello", help="A-Z, 0-9 and spaces")
    p.add_argument("--video", help="write an mp4 of the run here (needs ffmpeg and a GL backend)")
    p.add_argument("--size", default="480x640", help="video height x width")
    p.add_argument("--speed", type=float, default=1.0,
                   help="move speed; 1 keeps every motor inside its torque limit, higher is faster but jerkier")
    p.add_argument("--check-reach", action="store_true", help="only check every key is reachable")
    args = p.parse_args()
    if args.check_reach:
        sys.exit(0 if check_reach() else 1)
    for c in args.text:
        try:
            kb.key_for(c)
        except ValueError as e:
            p.error(str(e))
    h, w = (int(v) for v in args.size.split("x"))
    typer = Typer(args.video, size=(h, w), speed=args.speed)
    try:
        print(f'typing "{args.text}"')
        ok = typer.type_text(args.text)
    finally:
        typer.close()
    shown = "".join(" " if k == "SPACE" else k for k in typer.typed)
    print(f'typed "{shown}" in {typer.d.time:.1f} s of sim time, {typer.regrips} re-grip(s) -- '
          f'{"done" if ok else "INCOMPLETE"}')
    if args.video:
        print(f"wrote {args.video}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
