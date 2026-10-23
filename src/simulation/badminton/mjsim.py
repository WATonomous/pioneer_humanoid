"""Shared MuJoCo sim helpers: model loading, drag hook, episode stepping.

The drag hook applies only the aerodynamic force (mujoco supplies gravity):
    F_drag = -m * k * |v| * v
which equals m * (drag_accel(v) - g_vec) with aero.drag_accel's convention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

import aero

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(ROOT, "scene", "badminton.xml")

ARM_PREFIX = "arm_"


@dataclass
class Sim:
    model: mujoco.MjModel
    data: mujoco.MjData
    params: dict
    k: float = field(init=False)
    mass: float = field(init=False)

    def __post_init__(self):
        p = self.params
        self.k = p["shuttle"]["k"]
        self.mass = p["shuttle"]["mass"]
        self.orient_tau = p["shuttle"]["orient_tau"]
        m = self.model
        self.shuttle_bid = m.body("shuttle").id
        self.shuttle_jid = m.joint("shuttle_free").id
        self.shuttle_qadr = m.jnt_qposadr[self.shuttle_jid]
        self.shuttle_vadr = m.jnt_dofadr[self.shuttle_jid]
        self.face_sid = m.site("face_center").id
        self.face_gid = m.geom("racket_face").id
        self.shuttle_gid = m.geom("shuttle_col").id
        self.cork_gid = m.geom("shuttle_cork").id
        self.floor_gid = m.geom("floor").id
        self.net_gid = m.geom("net").id
        arm_joints = p["arm"]["joints"]
        self.arm_jids = [m.joint(ARM_PREFIX + n).id for n in arm_joints]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in self.arm_jids])
        self.arm_vadr = np.array([m.jnt_dofadr[j] for j in self.arm_jids])
        self.arm_act_ids = np.array([m.actuator(n).id for n in arm_joints])
        # left arm: the scripted baseline drives only the right arm; the
        # workspace builder samples both
        left_joints = p["arm"]["joints_left"]
        self.face_l_sid = m.site("face_center_l").id
        self.arm_l_jids = [m.joint(ARM_PREFIX + n).id for n in left_joints]
        self.arm_l_qadr = np.array([m.jnt_qposadr[j] for j in self.arm_l_jids])

    # -- state access ------------------------------------------------------
    @property
    def shuttle_pos(self) -> np.ndarray:
        return self.data.qpos[self.shuttle_qadr:self.shuttle_qadr + 3].copy()

    @property
    def shuttle_vel(self) -> np.ndarray:
        return self.data.qvel[self.shuttle_vadr:self.shuttle_vadr + 3].copy()

    @property
    def face_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.face_sid].copy()

    @property
    def face_normal(self) -> np.ndarray:
        # site z axis is the face normal
        return self.data.site_xmat[self.face_sid].reshape(3, 3)[:, 2].copy()

    @property
    def face_vel(self) -> np.ndarray:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                 self.face_sid, vel, 0)
        return vel[3:6].copy()  # linear part, world frame

    @property
    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    # -- control -----------------------------------------------------------
    def set_arm_targets(self, q_targets: np.ndarray) -> None:
        self.data.ctrl[self.arm_act_ids] = q_targets

    def hold_arm(self, q: np.ndarray) -> None:
        self.data.qpos[self.arm_qadr] = q
        self.data.qvel[self.arm_vadr] = 0.0
        self.set_arm_targets(q)

    # -- episode -----------------------------------------------------------
    def reset(self, p0=None, v0=None, arm_q=None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if arm_q is not None:
            self.hold_arm(np.asarray(arm_q, dtype=float))
        if p0 is not None:
            self.set_shuttle(p0, v0 if v0 is not None else np.zeros(3))
        mujoco.mj_forward(self.model, self.data)

    def set_shuttle(self, p, v) -> None:
        qa, va = self.shuttle_qadr, self.shuttle_vadr
        self.data.qpos[qa:qa + 3] = p
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        v = np.asarray(v, dtype=float)
        speed = np.linalg.norm(v)
        if speed > 1e-9:
            mujoco.mju_quatZ2Vec(quat, v / speed)
        self.data.qpos[qa + 3:qa + 7] = quat
        self.data.qvel[va:va + 6] = 0.0
        self.data.qvel[va:va + 3] = v

    def apply_drag(self) -> None:
        # Second-order hook: evaluate drag at the half-step velocity estimate
        # (v advanced half a step under gravity + current drag). Under
        # semi-implicit Euler this cancels the leading O(dt) error and keeps
        # flight parity with the RK4 reference below 1 cm at 1.5 s.
        dt = self.model.opt.timestep
        v = self.data.qvel[self.shuttle_vadr:self.shuttle_vadr + 3]
        a = aero.drag_accel(v, self.k)          # includes gravity
        v_est = v + 0.5 * dt * a
        speed = float(np.linalg.norm(v_est))
        self.data.xfrc_applied[self.shuttle_bid, :3] = \
            -self.mass * self.k * speed * v_est

    def orient_shuttle(self) -> None:
        # Kinematic cork-first alignment. The collision geom is a sphere, so
        # orientation is visual only: rotate the body +z (cork) axis toward
        # the velocity direction with a first-order lag (orient_tau), which
        # makes the skirt trail the cork and the post-impact flip take
        # ~3*orient_tau. Angular velocity is zeroed so the orientation is
        # fully kinematic (contact friction cannot spin the shuttle up).
        qa, va = self.shuttle_qadr, self.shuttle_vadr
        self.data.qvel[va + 3:va + 6] = 0.0
        v = self.data.qvel[va:va + 3]
        speed = float(np.linalg.norm(v))
        if speed < 0.5:
            return
        quat = self.data.qpos[qa + 3:qa + 7]
        xmat = np.zeros(9)
        mujoco.mju_quat2Mat(xmat, quat)
        z_world = xmat.reshape(3, 3)[:, 2]
        v_hat = v / speed
        cross = np.cross(z_world, v_hat)
        s = float(np.linalg.norm(cross))
        c = float(np.clip(z_world @ v_hat, -1.0, 1.0))
        angle = np.arctan2(s, c)
        if angle < 1e-6:
            return
        if s < 1e-8:
            # antiparallel: any axis perpendicular to z_world works
            axis = np.cross(z_world, [1.0, 0.0, 0.0])
            if np.linalg.norm(axis) < 1e-8:
                axis = np.cross(z_world, [0.0, 1.0, 0.0])
            axis /= np.linalg.norm(axis)
        else:
            axis = cross / s
        frac = 1.0 - np.exp(-self.model.opt.timestep / self.orient_tau)
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, axis, frac * angle)
        new_q = np.zeros(4)
        mujoco.mju_mulQuat(new_q, dq, quat)   # dq is a world-frame rotation
        mujoco.mju_normalize4(new_q)
        self.data.qpos[qa + 3:qa + 7] = new_q

    def step(self) -> None:
        self.apply_drag()
        mujoco.mj_step(self.model, self.data)
        self.orient_shuttle()

    # -- contacts ----------------------------------------------------------
    def contacts(self) -> set:
        """Set of geom-id pairs currently in contact with the shuttle."""
        shuttle_gids = (self.shuttle_gid, self.cork_gid)
        out = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if pair[0] in shuttle_gids:
                out.add(pair[1])
            elif pair[1] in shuttle_gids:
                out.add(pair[0])
        return out

    def face_contact(self) -> bool:
        return self.face_gid in self.contacts()


def load(scene_path: str = SCENE_PATH, params_path: str | None = None) -> Sim:
    """Load the scene on plain CPU MuJoCo (the local mode; GPU training runs
    through the mjlab task in mjlab_task/ instead)."""
    params = aero.load_params(params_path) if params_path else aero.load_params()
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)   # open at the ready pose
    mujoco.mj_forward(model, data)
    return Sim(model=model, data=data, params=params)
