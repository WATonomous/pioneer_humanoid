"""The 8-DoF arm (pioneer_bimanual_arm_8dof, left arm) in plain MuJoCo, with a tool point (TCP) on the gripper.

Built from the repo's arm_spec() (URDF + joint limits + PD position actuators), plus one added site, "tcp".
Needs only `mujoco` and `numpy`. Poses are in the rig's base_link frame, which is the MuJoCo world frame.

    arm = LeftArm()
    position, rotation = arm.tcp_pose(q)       # q: 7 joint angles in rad, shoulder pitch .. wrist yaw
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

# <repo>/src/simulation/inverse_kinematics/arm_model.py -> <repo>/src
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC / "pioneer_humanoid"))

from pioneer_humanoid.arm_8dof_params import DEFAULT_JOINT_POS, LEFT_ARM_JOINTS  # noqa: E402
from pioneer_humanoid.mujoco_bimanual_arm_8dof import arm_spec, set_home  # noqa: E402

# The URDF has no tool frame: the fingers hang off left_link7 at its origin. With the gripper open the finger
# meshes run along -z of left_link7 to about -0.247 m, with a gap of about 6.4 cm between them (along y).
# This point is on the line between the fingers, a few cm back from the tips. A first guess: confirm with the team.
TCP_BODY = "left_link7"
TCP_OFFSET = (0.0, 0.0, -0.20)   # m, in the left_link7 frame


class LeftArm:
    def __init__(self):
        spec = arm_spec()
        spec.body(TCP_BODY).add_site(name="tcp", pos=list(TCP_OFFSET))
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        set_home(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.joint_names = list(LEFT_ARM_JOINTS)                                   # 7 revolute joints, in chain order
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in self.joint_names])
        self.limits = np.array([self.model.joint(n).range for n in self.joint_names])    # (7, 2) rad
        self.rest = np.array([DEFAULT_JOINT_POS[n] for n in self.joint_names])    # home pose
        self.tcp_id = self.model.site("tcp").id
        self.link7_id = self.model.body(TCP_BODY).id

    def forward(self, q):
        """Put the left arm at joint angles q (7,) and run MuJoCo's forward kinematics."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)

    def tcp_pose(self, q):
        """Tool point position (3,) in m and orientation (3, 3) in the base frame, at joint angles q."""
        self.forward(q)
        return self.data.site_xpos[self.tcp_id].copy(), self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()

    def wrist_pivot(self, q):
        """Wrist pivot (the wrist-pitch joint's origin) in m, at joint angles q. What the repo's mink IK tracks."""
        self.forward(q)
        return self.data.xanchor[self.model.joint("left_wrist_pitch").id].copy()

    def link7_rotation(self, q):
        """Orientation (3, 3) of the last link (left_link7) in the base frame, at joint angles q."""
        self.forward(q)
        return self.data.xmat[self.link7_id].reshape(3, 3).copy()

    def random_q(self, rng, margin=0.0):
        """Joint angles drawn uniformly inside the limits (shrunk by `margin` rad at each end)."""
        return rng.uniform(self.limits[:, 0] + margin, self.limits[:, 1] - margin)
