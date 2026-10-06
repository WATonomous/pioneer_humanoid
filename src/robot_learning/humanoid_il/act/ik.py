"""Gripper-pointing-down inverse kinematics for the Pioneer LEFT arm in MuJoCo (damped least squares). Sim only.

The grasp point is midway between the finger pads, GRASP_POINT in link6l's frame; "pointing down" is link6l's
frame equal to a turn of ``yaw`` about world Z (its -Z, the approach axis, then points straight down and the
fingers close along the turned world Y). The arm has no wrist roll, so that yaw is limited: about -45..+30 deg.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS

GRASP_POINT = np.array([0.0, 0.053, -0.1138])   # link6l frame, measured between the open finger pads
FINGERS_BELOW_GRASP = 0.087                       # finger tips sit this far below the grasp point
YAW_RANGE = (math.radians(-45), math.radians(30))  # pointing down, the yaw the arm reaches over most of the table


class LeftArmIK:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        jid = [model.joint(j).id for j in LEFT_ARM_JOINTS]
        self.qadr = [model.joint(j).qposadr[0] for j in LEFT_ARM_JOINTS]
        self.dadr = [model.joint(j).dofadr[0] for j in LEFT_ARM_JOINTS]
        self.lo, self.hi = model.jnt_range[jid].T
        self.body = model.body("link6l").id
        self.scratch = mujoco.MjData(model)

    def fk(self, qpos_full: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.scratch
        s.qpos[:] = qpos_full
        s.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, s)
        mujoco.mj_comPos(self.m, s)
        R = s.xmat[self.body].reshape(3, 3)
        return s.xpos[self.body] + R @ GRASP_POINT, R.copy()

    def solve(self, qpos_full, q0, pos, yaw, iters: int = 30, tol: float = 5e-4):
        """Joint angles putting the grasp point at ``pos`` pointing down turned ``yaw``; (q, position error)."""
        Rt = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        q = np.array(q0, dtype=float)
        for _ in range(iters):
            p, R = self.fk(qpos_full, q)
            ep = np.asarray(pos) - p
            Re = Rt @ R.T
            ang = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            if np.linalg.norm(ep) < tol and np.linalg.norm(ang) < 5e-3:
                break
            jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.scratch, jp, jr, p, self.body)
            J = np.vstack([jp[:, self.dadr], 0.6 * jr[:, self.dadr]])
            err = np.concatenate([ep, 0.6 * ang])
            q = np.clip(q + 0.7 * J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err), self.lo, self.hi)
        p, _ = self.fk(qpos_full, q)
        return q, float(np.linalg.norm(np.asarray(pos) - p))

    @staticmethod
    def yaw_of(R: np.ndarray) -> float:
        return math.atan2(R[1, 0], R[0, 0])
