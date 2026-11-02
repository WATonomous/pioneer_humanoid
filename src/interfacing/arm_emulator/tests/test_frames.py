"""Frames and signs against the real calibration: the mismatch that has broken this arm before."""
from __future__ import annotations

import math
import unittest

from arm_emulator.config import REPO_CALIBRATION, load_motors
from arm_emulator.drive import MIT_CONTROL, MIT_ENTER
from arm_emulator.physics import EmulatedArm
from joint_config import cmd_to_motor_deg, load_joint_map, motor_to_cmd_deg


class TestFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.arm = EmulatedArm(seed=0)
        cls.joint_map = load_joint_map(REPO_CALIBRATION, "left")

    def test_slot_order_is_arm_pose_order(self) -> None:
        self.assertEqual([m.name for m in self.arm.specs],
                         ["shoulder.pitch", "shoulder.yaw", "shoulder.roll", "elbow.pitch", "elbow.roll", "wrist.pitch",
                          "gripper.open_close"])

    def test_motor_frame_matches_joint_config_both_ways(self) -> None:
        arm = self.arm
        for spec in arm.specs[:-1]:
            info = self.joint_map[spec.motor_id]
            for cmd_deg in (-20.0, 0.0, 35.0):
                arm.data.qpos[arm._qadr[spec.motor_id]] = math.radians(cmd_deg)
                motor_rad, _ = arm._motor_state(spec)
                self.assertAlmostEqual(math.degrees(motor_rad), cmd_to_motor_deg(info, cmd_deg), places=6)
                self.assertAlmostEqual(motor_to_cmd_deg(info, math.degrees(motor_rad)), cmd_deg, places=6)
            arm.data.qpos[arm._qadr[spec.motor_id]] = 0.0

    def test_gripper_closure_matches_gripper_position(self) -> None:
        arm = self.arm
        spec = arm.specs[-1]
        arm.data.qpos[arm._finger_qadr] = 0.5 * arm._finger_closed
        motor_rad, _ = arm._motor_state(spec)
        info = self.joint_map[spec.motor_id]
        self.assertAlmostEqual(motor_to_cmd_deg(info, math.degrees(motor_rad)) / info["upper"], 0.5, places=6)
        arm.data.qpos[arm._finger_qadr] = 0.0

    def test_positive_motor_torque_moves_each_joint_its_own_way(self) -> None:
        """A joint with direction -1 must move negative in the command frame for a positive motor torque."""
        for spec in load_motors()[0][:-1]:
            with self.subTest(spec.name):
                arm = EmulatedArm(seed=0, start_pose_deg=[0, 0, 0, -90, 0, 0])  # clear of the table
                for other in arm.specs:
                    arm.command(other.motor_id, MIT_ENTER, 0, 0, 0, 0, 0)
                q0, _ = arm._motor_state(spec)
                # Far above gravity and friction here: only a few ticks, so the sign is all that shows.
                peak = 0.5 * spec.model["peak_torque"]
                for _ in range(20):
                    arm.command(spec.motor_id, MIT_CONTROL, q0, 0.0, 0.0, 0.0, min(peak, spec.profile["t_max"]))
                    arm.step()
                q1, _ = arm._motor_state(spec)
                self.assertGreater(q1, q0)


if __name__ == "__main__":
    unittest.main()
