"""Whole-body motion for the scripted demos: hands to poses, gaze, waist, pelvis and planted feet, finger synergies.

Every motion is a timed blend of *targets*; each control tick (``CONTROL_HZ``) the targets are turned into joint
position commands by inverse kinematics on a scratch ``MjData``, and the physics runs in between. Nothing is
teleported: objects move only because fingers and palms push them.

- hands: the palm site (``{side}_palm``: x along the fingers, z out of the palm) to a position and orientation,
  moved along a straight line with a minimum-jerk profile (orientation slerped); 7-joint arm IK (damped least
  squares, a pull toward a comfortable posture in the null space).
- gaze: the head's neck yaw / pitch aim the eyes at a point (or follow a hand / body), a little ahead of the hands.
- pelvis: x, y, z, yaw (a crouch, a lean, a turn, a step); the legs are solved each tick to keep both feet where
  they were planted. ``step_to`` moves the feet one at a time (lift, swing, plant) and shifts the pelvis over.
- fingers: named hand shapes (``SHAPES``) blended in joint space; grip force comes from the position error left
  when the fingers stop on an object (finger torque limits in robot.GAINS).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import mujoco
import numpy as np

from .robot import BASE, SIDES, arm_joints, hand_joints, leg_joints

CONTROL_HZ = 100
# arm IK stays inside these (radians; the URDF's ranges are wider than a comfortable working posture)
IK_LIMITS = {"shoulder_pitch": (-2.4, 0.5), "shoulder_roll": (-0.3, 1.2), "elbow": (-1.57, 0.0)}
# arms up in front, palms down, ready to work over the bench (joint space; the right arm mirrors by name)
# (the elbow bends 90 degrees at most, so the hands work with it near its limit and the elbows out)
WORK = {"shoulder_pitch": -0.29, "shoulder_roll": 0.61, "upper_arm_yaw": 0.76, "elbow": -1.57, "forearm_roll": 0.88,
        "wrist_dev": -0.2, "wrist_flex": -0.17}
ARM_REST = WORK
# idle: the hand up and out to the side of the work area (palm ~0.37, 0.41 out, 0.95 high), clear of the bench
READY = {"shoulder_pitch": -0.05, "shoulder_roll": 1.07, "upper_arm_yaw": 0.54, "elbow": -1.57, "forearm_roll": 0.69,
         "wrist_dev": 0.21, "wrist_flex": -0.29}
# hands up (forearms raised in front of the chest, ~1.3 m): an idle pose that clears everything on the bench, for
# side steps along it; reached from READY by the shoulder alone
HIGH = {**READY, "shoulder_pitch": -0.9, "shoulder_roll": 0.4}
# on the way from hanging to READY: forearm up first (it would swing into the bench edge otherwise)
LIFT = {"shoulder_pitch": 0.7, "shoulder_roll": 0.25, "upper_arm_yaw": 0.3, "elbow": -1.57, "forearm_roll": 0.5,
        "wrist_dev": 0.0, "wrist_flex": 0.0}


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def frame(x, z) -> np.ndarray:
    """Rotation with x and z columns along the given directions (z kept, x orthogonalised)."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x, float) - np.dot(x, z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def palm_down(yaw: float = 0.0, tilt: float = 0.0) -> np.ndarray:
    """Palm facing down, fingers pointing along ``yaw`` (0: +X, away from the robot), fingers pitched ``tilt`` down."""
    f = np.array([math.cos(yaw) * math.cos(tilt), math.sin(yaw) * math.cos(tilt), -math.sin(tilt)])
    n = np.array([math.cos(yaw) * math.sin(tilt), math.sin(yaw) * math.sin(tilt), -math.cos(tilt)])
    return frame(f, n)


def _slerp(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, r0.flatten())
    mujoco.mju_mat2Quat(q1, r1.flatten())
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = min(1.0, float(np.dot(q0, q1)))
    th = math.acos(dot)
    if th < 1e-6:
        q = q1
    else:
        q = (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)
    q /= np.linalg.norm(q)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def _rot_err(r_cur: np.ndarray, r_des: np.ndarray) -> np.ndarray:
    """Angular velocity (world) that turns r_cur into r_des in unit time."""
    q_cur, q_des, q_err = np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q_cur, r_cur.flatten())
    mujoco.mju_mat2Quat(q_des, r_des.flatten())
    mujoco.mju_negQuat(q_cur, q_cur)
    mujoco.mju_mulQuat(q_err, q_des, q_cur)
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q_err, 1.0)
    return v


# Finger synergies: joint -> angle for the LEFT hand's names without the side prefix (the right hand mirrors by
# name: its joint axes are mirrored in the URDF, so the same angles give the mirror shape).
_CURL = lambda m, p, d, fingers=("", "i_", "r_", "l_"): {f"{f}{j}": a for f in fingers for j, a in (("mcp", m), ("pip", p), ("dip", d))}
SHAPES: dict[str, dict[str, float]] = {
    "open": {},
    "relaxed": {**_CURL(0.25, 0.35, 0.2), "th_c": 0.3, "th_f": 0.3},
    "spread": {"i_abd": -0.15, "r_abd": 0.1, "l_abd": 0.2, "th_c": 0.0},
    # around a ~30 mm bar lying across the fingers, thumb wrapped over from the other side (squeezed past contact)
    "power": {**_CURL(1.25, 1.45, 1.1), "th_c": 0.87, "th_f": 1.05, "th_mcp": 0.9, "th_ip": 1.0},
    "power_pre": {**_CURL(0.55, 0.55, 0.35), "th_c": 0.87, "th_f": 0.6, "th_mcp": 0.2, "th_ip": 0.2},
    # tips of the thumb, index and middle together (a tripod for small parts)
    "pinch": {**_CURL(0.75, 0.9, 0.6, ("", "i_")), **_CURL(1.3, 1.5, 1.2, ("r_", "l_")),
              "th_c": 0.87, "th_f": 1.0, "th_mcp": 0.55, "th_ip": 0.45},
    "pinch_pre": {**_CURL(0.35, 0.45, 0.3, ("", "i_")), **_CURL(1.1, 1.3, 1.0, ("r_", "l_")),
                  "th_c": 0.87, "th_f": 0.55, "th_mcp": 0.1, "th_ip": 0.1},
    # a knob (~42 mm) from above, like a jar lid: index / middle / ring tips on its far rim, thumb on the near rim.
    # Solved for the knob centre 20 mm out along the fingers from the palm site, its mid-height 25 mm below the palm
    # (fingertip targets 4 mm inside the rim: the squeeze). Little finger tucked.
    "lid": {"i_mcp": 1.0, "i_pip": 1.09, "mcp": 1.25, "pip": 0.99, "r_mcp": 0.97, "r_pip": 1.1, "l_mcp": 1.3,
            "l_pip": 1.3, "th_c": 0.12, "th_f": 0.1, "th_mcp": 1.29, "th_ip": 1.16},
    "lid_pre": {"i_mcp": 0.3, "i_pip": 0.35, "mcp": 0.4, "pip": 0.3, "r_mcp": 0.3, "r_pip": 0.35, "l_mcp": 1.0,
                "l_pip": 1.0, "th_c": 0.12, "th_f": 0.1, "th_mcp": 0.0, "th_ip": 0.0},
    # turning a knob: solved for its axis 30 mm out along the fingers and the contacts 24 mm below the palm.
    # "knob_pre" puts every tip 12 mm outside the rim, so closing moves them straight in: a sideways sweep of the
    # thumb (as from "lid_pre") would turn the knob back before the hand turns it.
    "knob": {"i_mcp": 0.8, "i_pip": 1.145, "mcp": 1.036, "pip": 1.087, "r_mcp": 0.778, "r_pip": 1.148, "l_mcp": 1.3,
             "l_pip": 1.3, "th_c": 0.36, "th_f": 0.0, "th_mcp": 1.028, "th_ip": 1.396},
    "knob_pre": {"i_mcp": 0.536, "i_pip": 1.137, "mcp": 0.736, "pip": 1.145, "r_mcp": 0.522, "r_pip": 1.134,
                 "l_mcp": 1.3, "l_pip": 1.3, "th_c": 0.332, "th_f": 0.261, "th_mcp": 1.329, "th_ip": 1.233},
    "fist": {**_CURL(1.5, 1.55, 1.3), "th_c": 0.5, "th_f": 0.9, "th_mcp": 1.0, "th_ip": 1.0},
    "point": {**_CURL(1.3, 1.5, 1.2, ("", "r_", "l_")), "th_c": 0.5, "th_f": 0.9, "th_mcp": 0.8, "th_ip": 0.8},
}


def shape_targets(side: str, shape: str | dict, amount: float = 1.0) -> dict[str, float]:
    s = SHAPES[shape] if isinstance(shape, str) else shape
    return {j: amount * s.get(j[2:], 0.0) for j in hand_joints(side)}


@dataclass
class _Track:
    """One blended target: start value, goal value, start time, duration."""
    v0: object
    v1: object
    t0: float
    dur: float

    def s(self, t: float) -> float:
        return 1.0 if self.dur <= 0 else min_jerk((t - self.t0) / self.dur)

    def done(self, t: float) -> bool:
        return t >= self.t0 + self.dur


@dataclass
class Hold:
    """Something the gaze can follow: a body, site or a fixed point (evaluated every tick)."""
    fn: Callable[[], np.ndarray]


class WholeBody:
    """Drives the humanoid in ``data``; ``on_tick(t)`` runs after each control tick (scene step hooks, video)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, scene_step=None, on_frame=None, fps: float = 30.0):
        self.m, self.d = model, data
        self.k = mujoco.MjData(model)                     # kinematic scratch for IK
        self.scene_step, self.on_frame, self.fps = scene_step, on_frame, fps
        self.t = 0.0
        self._next_frame = 0.0
        self.substeps = max(1, round(1.0 / CONTROL_HZ / model.opt.timestep))
        self.jq = {model.joint(i).name: model.jnt_qposadr[i] for i in range(model.njnt)
                   if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE}
        self.jv = {model.joint(i).name: model.jnt_dofadr[i] for i in range(model.njnt)
                   if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE}
        self.act = {model.actuator(i).name: i for i in range(model.nu)}
        self.cmd = {name: float(data.qpos[self.jq[name]]) for name in self.act}   # joint position commands
        self.arm_dofs = {s: np.array([self.jv[j] for j in arm_joints(s)]) for s in SIDES}
        self.leg_dofs = {s: np.array([self.jv[j] for j in leg_joints(s)]) for s in SIDES}
        self.palm = {s: model.site(f"{s}_palm").id for s in SIDES}
        self.hand_site = dict(self.palm)            # the site an arm's Cartesian target is for
        self.foot_id = {s: model.body(f"{s}_foot").id for s in SIDES}
        self.hand_track: dict[str, Optional[_Track]] = {s: None for s in SIDES}
        self.hand_goal: dict[str, Optional[tuple]] = {s: None for s in SIDES}    # (pos, rot) once a hand is commanded
        self.finger_track: dict[str, Optional[_Track]] = {s: None for s in SIDES}
        self.arm_track: dict[str, Optional[_Track]] = {s: None for s in SIDES}
        self.base_track: Optional[_Track] = None
        self.waist_track: Optional[_Track] = None
        self.gaze_target: Optional[Callable[[], np.ndarray]] = None
        self.gaze_track: Optional[_Track] = None
        self.foot_goal: dict[str, tuple] = {}
        self.foot_track: dict[str, Optional[tuple]] = {s: None for s in SIDES}
        # comfortable arm posture (null-space pull): upper arm a little forward, elbow bent, palm turned down
        self.arm_rest = {s: np.array([ARM_REST[j] for j in ("shoulder_pitch", "shoulder_roll", "upper_arm_yaw",
                                                             "elbow", "forearm_roll", "wrist_dev", "wrist_flex")])
                         for s in SIDES}
        self.base0 = data.xpos[model.body("pelvis").id].copy() - np.array(
            [self.cmd["base_x"], self.cmd["base_y"], self.cmd["base_z"]])
        self._sync_kin()
        for s in SIDES:
            self.foot_goal[s] = (self.k.xpos[self.foot_id[s]].copy(), self.k.xmat[self.foot_id[s]].reshape(3, 3).copy())
        for name, i in self.act.items():
            data.ctrl[i] = self.cmd[name]

    # ------------------------------------------------------------------ kinematics
    def _sync_kin(self):
        self.k.qpos[:] = self.d.qpos
        for name, q in self.cmd.items():
            self.k.qpos[self.jq[name]] = q
        mujoco.mj_kinematics(self.m, self.k)
        mujoco.mj_comPos(self.m, self.k)

    def palm_pose(self, side: str, commanded: bool = True):
        """The palm site's pose: of the commanded joints (default) or the simulated ones."""
        src = self.k if commanded else self.d
        if commanded:
            self._sync_kin()
        return src.site_xpos[self.palm[side]].copy(), src.site_xmat[self.palm[side]].reshape(3, 3).copy()

    def _ik_site(self, site: int, dofs: np.ndarray, names: list[str], pos, rot, rest, iters=12, w_rot=0.35):
        """Damped least squares on the scratch data: put ``site`` at pos / rot using ``dofs``."""
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        lo = np.array([max(self.m.jnt_range[self.m.joint(n).id][0], IK_LIMITS.get(n[2:], (-9, 9))[0]) for n in names])
        hi = np.array([min(self.m.jnt_range[self.m.joint(n).id][1], IK_LIMITS.get(n[2:], (-9, 9))[1]) for n in names])
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, self.k)
            mujoco.mj_comPos(self.m, self.k)
            ep = pos - self.k.site_xpos[site]
            er = _rot_err(self.k.site_xmat[site].reshape(3, 3), rot) if rot is not None else np.zeros(3)
            mujoco.mj_jacSite(self.m, self.k, jp, jr, site)
            if rot is not None:
                J = np.vstack([jp[:, dofs], w_rot * jr[:, dofs]])
                e = np.concatenate([ep, w_rot * er])
            else:
                J, e = jp[:, dofs], ep
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                break
            lam = 1e-4
            JJt = J @ J.T + lam * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            q = np.array([self.k.qpos[self.jq[n]] for n in names])
            if rest is not None:   # null space: drift toward the rest posture
                N = np.eye(len(names)) - np.linalg.pinv(J) @ J
                dq += N @ (0.05 * (rest - q))
            dq = np.clip(dq, -0.25, 0.25)
            q = np.clip(q + dq, lo, hi)
            for n, v in zip(names, q):
                self.k.qpos[self.jq[n]] = v
        return {n: float(self.k.qpos[self.jq[n]]) for n in names}

    # ------------------------------------------------------------------ commands (non-blocking)
    def reach(self, side: str, pos, rot=None, dur: float = 1.0):
        """Move a palm to ``pos`` (and ``rot``) in a straight line over ``dur`` s."""
        if self.arm_track[side] is not None:          # finish a joint-space move first
            self.wait(self.arm_track[side].t0 + self.arm_track[side].dur - self.t)
        if self.hand_site[side] != self.palm[side]:   # back from a fingertip move: the palm takes over again
            self.hand_site[side] = self.palm[side]
            self.hand_goal[side] = None
            self.hand_track[side] = None
        p0, r0 = self.hand_goal[side] if self.hand_goal[side] is not None else self.palm_pose(side)
        if self.hand_track[side] is not None and not self.hand_track[side].done(self.t):
            p0, r0 = self._hand_now(side)
        rot = r0 if rot is None else np.asarray(rot, float)
        self.hand_track[side] = _Track((p0, r0), (np.asarray(pos, float), rot), self.t, dur)
        self.hand_goal[side] = (np.asarray(pos, float), rot)

    def reach_tip(self, side: str, tip_site: str, pos, dur: float = 1.0):
        """Move a fingertip site (position only; the arm's posture fills in the rest) in a straight line."""
        if self.arm_track[side] is not None:
            self.wait(self.arm_track[side].t0 + self.arm_track[side].dur - self.t)
        sid = self.m.site(tip_site).id
        if self.hand_site[side] != sid or self.hand_goal[side] is None:
            self._sync_kin()
            p0 = self.k.site_xpos[sid].copy()
        else:
            p0 = self._hand_now(side)[0] if not self.hand_track[side].done(self.t) else self.hand_goal[side][0]
        self.hand_site[side] = sid
        self.hand_track[side] = _Track((p0, None), (np.asarray(pos, float), None), self.t, dur)
        self.hand_goal[side] = (np.asarray(pos, float), None)

    def _hand_now(self, side):
        tr = self.hand_track[side]
        s = tr.s(self.t)
        (p0, r0), (p1, r1) = tr.v0, tr.v1
        return p0 + s * (p1 - p0), (None if r1 is None else _slerp(r0, r1, s))

    def arm_joints_to(self, side: str, q: dict, dur: float = 1.0):
        """Joint-space move of one arm (e.g. from hanging to the ready pose, where straight lines would hit things).
        Afterwards the palm's Cartesian target is where the joints put it."""
        names = arm_joints(side)
        v0 = np.array([self.cmd[n] for n in names])
        v1 = np.array([q.get(n[2:], self.cmd[n]) for n in names])
        self.arm_track[side] = _Track(v0, v1, self.t, dur)
        self.hand_track[side] = None
        self.hand_goal[side] = None
        self.hand_site[side] = self.palm[side]

    def ready(self, sides=SIDES, dur: float = 2.0):
        """Both arms from hanging up into the ready pose, forearms raised first."""
        for s in sides:
            self.arm_joints_to(s, LIFT, dur * 0.5)
        self.settle()
        for s in sides:
            self.arm_joints_to(s, READY, dur * 0.5)
        self.settle()

    def fingers(self, side: str, shape, amount: float = 1.0, dur: float = 0.6):
        names = hand_joints(side)
        v0 = np.array([self.cmd[n] for n in names])
        goal = shape_targets(side, shape, amount)
        self.finger_track[side] = _Track(v0, np.array([goal[n] for n in names]), self.t, dur)

    def look(self, target, dur: float = 0.5):
        """Aim the eyes at ``target``: a point, or a callable returning one (followed every tick)."""
        fn = target if callable(target) else (lambda p=np.asarray(target, float): p)
        y0, p0 = self.cmd["neck_yaw"], self.cmd["neck_pitch"]
        self.gaze_target = fn
        self.gaze_track = _Track((y0, p0), None, self.t, dur)

    def waist(self, yaw: float, dur: float = 1.0):
        self.waist_track = _Track(self.cmd["waist_yaw"], yaw, self.t, dur)

    def pelvis(self, x=None, y=None, z=None, yaw=None, pitch=None, dur: float = 1.0):
        """Move the pelvis (z: crouch, pitch: lean forward at the hips); the feet stay planted."""
        cur = np.array([self.cmd[b] for b in BASE])
        goal = cur.copy()
        for i, v in enumerate((x, y, z, yaw, pitch)):
            if v is not None:
                goal[i] = v
        self.base_track = _Track(cur, goal, self.t, dur)

    def foot(self, side: str, pos, yaw: float, dur: float = 0.6, lift: float = 0.05):
        p0, r0 = self.foot_goal[side]
        r1 = rot_z(yaw) @ self._foot_rest_rot
        self.foot_track[side] = (_Track((p0.copy(), r0.copy()), (np.asarray(pos, float), r1), self.t, dur), lift)
        self.foot_goal[side] = (np.asarray(pos, float), r1)

    def pelvis_world(self) -> np.ndarray:
        """The pelvis's commanded world x, y (its joints are offsets from where the scene put it)."""
        return self.base0[:2] + np.array([self.cmd["base_x"], self.cmd["base_y"]])

    def step(self, side: str, dx: float, dy: float = 0.0, dur: float = 0.9, lift: float = 0.045,
             carry: bool = False):
        """One step of ``side``'s foot by (dx, dy) in the world: weight onto the other foot, swing, plant, re-centre.
        ``carry``: the hands keep their place relative to the pelvis (they hold something), else they stay put."""
        other = "r" if side == "l" else "l"
        sp = self.foot_goal[other][0]
        cur = self.pelvis_world()

        def move_pelvis(x=None, y=None, z=None, d=0.3):
            if carry:
                shift = np.array([0.0 if x is None else x + self.base0[0] - self.pelvis_world()[0],
                                  0.0 if y is None else y + self.base0[1] - self.pelvis_world()[1],
                                  0.0 if z is None else z - self.cmd["base_z"]])
                for s_ in SIDES:
                    if self.hand_goal[s_] is not None and self.hand_goal[s_][1] is not None:
                        p_, r_ = self.hand_goal[s_]
                        self.reach(s_, p_ + shift, r_, d)
            self.pelvis(x=x, y=y, z=z, dur=d)

        # weight over the stance foot (a quasi-static gait: the balance itself isn't simulated)
        move_pelvis(y=cur[1] + 0.55 * (sp[1] - cur[1]) - self.base0[1], z=self.cmd["base_z"] - 0.01, d=0.3 * dur)
        self.settle()
        p0, _ = self.foot_goal[side]
        self.foot(side, p0 + np.array([dx, dy, 0.0]), self.cmd["base_yaw"], dur=0.45 * dur, lift=lift)
        mid = (self.foot_goal[side][0][:2] + sp[:2]) / 2
        move_pelvis(x=cur[0] + dx / 2 - self.base0[0], d=0.45 * dur)
        self.settle()
        move_pelvis(y=mid[1] - self.base0[1], z=self.cmd["base_z"] + 0.01, d=0.25 * dur)
        self.settle()

    def turn(self, angle: float, step: float = math.radians(30), dur: float = 0.9, lift: float = 0.04,
             carry: bool = False):
        """Turn on the spot by ``angle`` (rad, + is left) in steps of at most ``step``: each step the leading foot
        swings to its place turned about the pelvis, then the other; the pelvis turns with them. ``carry``: the
        hands turn with the pelvis (they hold something); else they should be in a joint-space pose (they follow
        the body)."""
        n = max(1, math.ceil(abs(angle) / step - 1e-9))
        da = angle / n
        lead, trail = ("l", "r") if angle > 0 else ("r", "l")
        for _ in range(n):
            c = np.array([*self.pelvis_world(), 0.0])
            yaw0 = self.cmd["base_yaw"]
            for k, side in enumerate((lead, trail)):
                other = "r" if side == "l" else "l"
                sp = self.foot_goal[other][0]
                cur = self.pelvis_world()
                # weight over the stance foot
                shift = 0.5 * (sp[:2] - cur)
                self.pelvis(x=cur[0] + shift[0] - self.base0[0], y=cur[1] + shift[1] - self.base0[1],
                            z=self.cmd["base_z"] - 0.01, dur=0.3 * dur)
                self.settle()
                p0, _ = self.foot_goal[side]
                rel = p0 - c
                p1 = c + rot_z(da) @ rel
                p1[2] = p0[2]
                self.foot(side, p1, yaw0 + da, dur=0.45 * dur, lift=lift)
                # the pelvis turns half way with each foot
                y1 = yaw0 + da * (k + 1) / 2
                if carry:
                    self._turn_hands(c, y1 - self.cmd["base_yaw"], 0.45 * dur)
                self.pelvis(yaw=y1, dur=0.45 * dur)
                self.settle()
                self.pelvis(x=c[0] - self.base0[0], y=c[1] - self.base0[1], z=self.cmd["base_z"] + 0.01,
                            dur=0.25 * dur)
                self.settle()

    def _turn_hands(self, c, da: float, dur: float):
        """Turn both hands' targets by ``da`` about the vertical through ``c`` (they hold something)."""
        R = rot_z(da)
        for s_ in SIDES:
            if self.hand_goal[s_] is not None and self.hand_goal[s_][1] is not None:
                p_, r_ = self.hand_goal[s_]
                p1 = c + R @ (p_ - c)
                p1[2] = p_[2]
                self.reach(s_, p1, R @ r_, dur)

    def walk(self, dx: float, step: float = 0.14, dur: float = 0.9, carry: bool = False):
        """Walk ``dx`` m along +X: the trailing foot steps at most ``step`` past the other, feet together at the end.
        ``carry``: as for ``step``."""
        goal = {f: self.foot_goal[f][0][0] + dx for f in SIDES}
        while True:
            x = {f: self.foot_goal[f][0][0] for f in SIDES}
            todo = [f for f in SIDES if abs(goal[f] - x[f]) > 1e-4]
            if not todo:
                break
            side = min(todo, key=lambda f: (x[f] - goal[f]) * math.copysign(1, dx))
            other = "r" if side == "l" else "l"
            lead = x[other] - x[side] + (goal[side] - goal[other])
            nxt = goal[side] - (goal[other] - x[other]) + math.copysign(step, dx)
            nxt = min(nxt, goal[side]) if dx > 0 else max(nxt, goal[side])
            self.step(side, nxt - x[side], dur=dur, carry=carry)

    @property
    def _foot_rest_rot(self):
        if not hasattr(self, "_frr"):
            self._frr = rot_z(-self.cmd["base_yaw"]) @ self.foot_goal["l"][1]
        return self._frr

    # ------------------------------------------------------------------ the loop
    def _update_commands(self):
        t = self.t
        if self.base_track is not None:
            tr = self.base_track
            v = tr.v0 + tr.s(t) * (tr.v1 - tr.v0)
            for b, q in zip(BASE, v):
                self.cmd[b] = float(q)
        if self.waist_track is not None:
            tr = self.waist_track
            self.cmd["waist_yaw"] = tr.v0 + tr.s(t) * (tr.v1 - tr.v0)
        for s in SIDES:
            tr = self.arm_track[s]
            if tr is not None:
                v = tr.v0 + tr.s(t) * (tr.v1 - tr.v0)
                for n, q in zip(arm_joints(s), v):
                    self.cmd[n] = float(q)
                if tr.done(t):
                    self.arm_track[s] = None
            tr = self.finger_track[s]
            if tr is not None:
                v = tr.v0 + tr.s(t) * (tr.v1 - tr.v0)
                for n, q in zip(hand_joints(s), v):
                    self.cmd[n] = float(q)
        self._sync_kin()
        # legs: feet stay at their goals (or travel along their swing)
        for s in SIDES:
            ft = self.foot_track[s]
            if ft is not None:
                tr, lift = ft
                sv = tr.s(t)
                (p0, r0), (p1, r1) = tr.v0, tr.v1
                pos = p0 + sv * (p1 - p0)
                pos = pos + np.array([0, 0, lift * math.sin(math.pi * min(1.0, max(0.0, (t - tr.t0) / tr.dur)))])
                rot = _slerp(r0, r1, sv)
            else:
                pos, rot = self.foot_goal[s]
            res = self._ik_site_body(self.foot_id[s], self.leg_dofs[s], leg_joints(s), pos, rot)
            self.cmd.update(res)
        # arms
        for s in SIDES:
            if self.hand_track[s] is None:
                continue
            pos, rot = self._hand_now(s)
            res = self._ik_site(self.hand_site[s], self.arm_dofs[s], arm_joints(s), pos, rot, self.arm_rest[s])
            self.cmd.update(res)
        # gaze
        if self.gaze_target is not None:
            yaw, pitch = self._gaze_angles(self.gaze_target())
            tr = self.gaze_track
            sv = tr.s(t)
            self.cmd["neck_yaw"] = tr.v0[0] + sv * (yaw - tr.v0[0])
            self.cmd["neck_pitch"] = tr.v0[1] + sv * (pitch - tr.v0[1])

    def _ik_site_body(self, body, dofs, names, pos, rot, iters=10):
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        lo = np.array([self.m.jnt_range[self.m.joint(n).id][0] for n in names])
        hi = np.array([self.m.jnt_range[self.m.joint(n).id][1] for n in names])
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, self.k)
            mujoco.mj_comPos(self.m, self.k)
            ep = pos - self.k.xpos[body]
            er = _rot_err(self.k.xmat[body].reshape(3, 3), rot)
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jacBody(self.m, self.k, jp, jr, body)
            J = np.vstack([jp[:, dofs], jr[:, dofs]])
            e = np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(6), e)
            q = np.clip(np.array([self.k.qpos[self.jq[n]] for n in names]) + np.clip(dq, -0.3, 0.3), lo, hi)
            for n, v in zip(names, q):
                self.k.qpos[self.jq[n]] = v
        return {n: float(self.k.qpos[self.jq[n]]) for n in names}

    def _gaze_angles(self, target):
        """Neck yaw / pitch that point the eyes at ``target`` (URDF frame: the face looks along -Y, pitch + is down)."""
        torso = self.m.body("torso").id
        neck = self.k.xpos[self.m.body("neck_link").id] + self.k.xmat[torso].reshape(3, 3) @ np.array([0, -0.03, 0.25])
        v = self.k.xmat[torso].reshape(3, 3).T @ (np.asarray(target) - neck)
        yaw = math.atan2(v[0], -v[1])
        pitch = math.atan2(-v[2], math.hypot(v[0], v[1]))
        lo_y, hi_y = self.m.jnt_range[self.m.joint("neck_yaw").id]
        lo_p, hi_p = self.m.jnt_range[self.m.joint("neck_pitch").id]
        return float(np.clip(yaw, lo_y, hi_y)), float(np.clip(pitch, lo_p, hi_p))

    def tick(self):
        self._update_commands()
        for name, i in self.act.items():
            self.d.ctrl[i] = self.cmd[name]
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, self.d)
            if self.scene_step is not None:
                self.scene_step(self.m, self.d)
        self.t += self.substeps * self.m.opt.timestep
        if self.on_frame is not None and self.t >= self._next_frame:
            self.on_frame(self)
            self._next_frame += 1.0 / self.fps

    def until(self, cond: Callable[[], bool], timeout: float) -> bool:
        """Run until ``cond()`` or ``timeout`` s; on ``cond``, every arm stops where it is commanded now."""
        end = self.t + timeout
        while self.t < end:
            self.tick()
            if cond():
                self.hold()
                return True
        return False

    def watch(self, cond: Callable[[], bool], dur: float) -> bool:
        """``wait(dur)``, but stop every arm where it is commanded now as soon as ``cond()`` (returns whether)."""
        end = self.t + dur
        while self.t < end - 1e-9:
            self.tick()
            if cond():
                self.hold()
                return True
        return False

    def hold(self):
        """Every arm moving in a straight line stops where it is commanded now."""
        for s in SIDES:
            if self.hand_track[s] is not None and not self.hand_track[s].done(self.t):
                p, r = self._hand_now(s)
                self.hand_track[s] = _Track((p, r), (p, r), self.t, 0.0)
                self.hand_goal[s] = (p, r)

    def wait(self, dur: float):
        end = self.t + dur
        while self.t < end - 1e-9:
            self.tick()

    def settle(self):
        """Run until every hand / finger / base / waist / foot motion has finished."""
        ends = [tr.t0 + tr.dur for tr in (*self.hand_track.values(), *self.finger_track.values(),
                                         *self.arm_track.values(), self.base_track,
                                         self.waist_track, self.gaze_track) if tr is not None]
        ends += [ft[0].t0 + ft[0].dur for ft in self.foot_track.values() if ft is not None]
        self.wait(max([0.0, *[e - self.t for e in ends]]))
