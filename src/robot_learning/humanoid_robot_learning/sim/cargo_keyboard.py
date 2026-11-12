"""Keyboard expert returning commands; the shared environment owns actuator writes."""
import numpy as np
from scipy.spatial.transform import Rotation
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg


class KeyboardExpert:
    def __init__(self, kinematics):
        self.kin = kinematics
        self.keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=.003, rot_sensitivity=.025, gripper_term=False))
        self.keyboard.add_callback("K", self.toggle_gripper)
        self.grip = 0.
        self.fine = False

    def toggle_gripper(self):
        self.grip = 0. if self.grip > .5 else 1.

    def seed(self, env):
        self.keyboard.reset()
        self.grip = float(env.last[6])
        self.q = env.robot.data.joint_pos[0, env.arm_ids].cpu().numpy().copy()
        self.pos, self.rot = self.kin.fk(self.q)

    def action(self, env):
        cmd = self.keyboard.advance()[:6].cpu().numpy() * (.2 if self.fine else 1.)
        self.pos += cmd[:3]
        self.rot = self.rot * Rotation.from_rotvec(cmd[3:])
        measured_q = env.robot.data.joint_pos[0, env.arm_ids].cpu().numpy()
        measured_pos, _ = self.kin.fk(measured_q)
        self.pos = measured_pos + np.clip(self.pos - measured_pos, -.04, .04)
        q, error, angle = self.kin.ik(self.pos, self.rot, self.q)
        if error < .03 and angle < .25:
            self.q = q
        return np.r_[self.q, self.grip].astype(np.float32)
