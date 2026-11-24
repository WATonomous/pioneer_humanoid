"""The left arm in MuJoCo with the agreed sim settings, shared by the baseline, A3 tuning and the RL environment."""
import mujoco
import numpy as np

from arm_model import LeftArm, set_home
from metrics import CONTROL_DT

PHYSICS_DT = 1 / 600                          # s per physics step
SUBSTEPS = round(CONTROL_DT / PHYSICS_DT)     # physics steps per control step: 20
EPISODE_STEPS = 160                           # control steps per episode (5.3 s)


class Sim:
    """The left arm in MuJoCo, stepped at the control rate."""

    def __init__(self, gravity_comp=False):
        self.arm = LeftArm()
        self.model, self.data = self.arm.model, self.arm.data
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST   # until arm_model.py sets it
        self.model.opt.timestep = PHYSICS_DT
        self.gravity_comp = gravity_comp
        # where each left-arm motor sits in data.ctrl, and each joint in data.qfrc_* (forces)
        self.act_ids = np.array([self.model.actuator(n).id for n in self.arm.joint_names])
        self.dof_ids = np.array([self.model.joint(n).dofadr[0] for n in self.arm.joint_names])

    def reset(self, q):
        """Start a new episode: left arm still at angles q, motors holding q, everything else at home."""
        mujoco.mj_resetData(self.model, self.data)
        set_home(self.model, self.data)
        self.data.qpos[self.arm.qadr] = q
        self.data.ctrl[self.act_ids] = q
        mujoco.mj_forward(self.model, self.data)

    def step(self, q_cmd):
        """One control step (1/30 s): command the left arm to angles q_cmd, then run SUBSTEPS physics steps."""
        self.data.ctrl[self.act_ids] = q_cmd
        for _ in range(SUBSTEPS):
            if self.gravity_comp:
                self.data.qfrc_applied[self.dof_ids] = self.data.qfrc_bias[self.dof_ids]
            mujoco.mj_step(self.model, self.data)

    def tcp_pose(self):
        """Current tool point position (3,) and orientation (3, 3), read from the simulation."""
        p = self.data.site_xpos[self.arm.tcp_id].copy()
        R = self.data.site_xmat[self.arm.tcp_id].reshape(3, 3).copy()
        return p, R

