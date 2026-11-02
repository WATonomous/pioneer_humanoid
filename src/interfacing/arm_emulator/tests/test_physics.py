"""The arm under its real gains, with the offline stand-in for joint_command. No ROS."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

from arm_emulator.drive import MIT_CONTROL
from arm_emulator.faults import parse
from arm_emulator.physics import EmulatedArm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenarios"))
from offline_command import OfflineCommand  # noqa: E402

HOME = [0.0, 0.0, 0.0, -90.0, 0.0, 0.0]


class TestHolding(unittest.TestCase):
    def test_hangs_where_it_started(self) -> None:
        arm = EmulatedArm(seed=0)
        command = OfflineCommand(arm)
        command.run(1.0, [0.0] * 6)
        np.testing.assert_allclose(arm.joints_deg(), [0.0] * 6, atol=1.0)

    def test_limp_arm_falls(self) -> None:
        arm = EmulatedArm(seed=0, start_pose_deg=HOME)
        for _ in range(1000):
            arm.step()
        self.assertGreater(arm.joints_deg()[3], -80.0, "the elbow must drop with no motor torque")

    def test_elbow_reaches_home_with_sag(self) -> None:
        arm = EmulatedArm(seed=0, start_pose_deg=[0, 0, 0, -75, 0, 0])
        command = OfflineCommand(arm)
        command.ramp(4.0, [0, 0, 0, -75, 0, 0], HOME)
        command.run(2.0, HOME)
        elbow = arm.joints_deg()[3]
        self.assertLess(abs(elbow + 90.0), 5.0)
        self.assertNotAlmostEqual(elbow, -90.0, delta=0.2, msg="real gains are soft: some sag must show")


class TestWorkcell(unittest.TestCase):
    def test_raising_the_elbow_from_hanging_goes_through_the_table(self) -> None:
        """Why the emulator starts at home: the path teleop's elbow ramp takes, in this workcell."""
        arm = EmulatedArm(seed=0)
        touching = []
        for elbow in range(0, -91, -5):
            arm.data.qpos[arm._qadr[11]] = np.radians(elbow)
            mujoco.mj_forward(arm.model, arm.data)
            if any(arm.model.geom_bodyid[arm.data.contact[i].geom1] in arm._arm_bodies
                   or arm.model.geom_bodyid[arm.data.contact[i].geom2] in arm._arm_bodies for i in range(arm.data.ncon)):
                touching.append(elbow)
        self.assertTrue(touching, "if the table moved, the start pose can go back to hanging")
        self.assertNotIn(-90, touching)

    def test_held_by_hand_until_the_first_stiff_command(self) -> None:
        arm = EmulatedArm(seed=0, start_pose_deg=HOME, supported=True)
        for _ in range(300):
            arm.step()
        np.testing.assert_allclose(arm.joints_deg(), HOME, atol=1e-6)
        OfflineCommand(arm).run(2.0, HOME)
        self.assertFalse(arm.supported)


class TestLoadMonitor(unittest.TestCase):
    def test_driving_the_arm_into_the_table_is_seen(self) -> None:
        arm = EmulatedArm(seed=0, start_pose_deg=HOME)
        spec = arm.specs[3]  # elbow.pitch: lowers the forearm onto the table
        for _ in range(1500):
            q, _ = arm._motor_state(spec)
            arm.command(spec.motor_id, MIT_CONTROL, q, 0.0, 0.0, 0.0, spec.direction * 18.0)
            arm.step()
        self.assertGreater(arm.peak_contact, 1.0, "the forearm must have reached the table")
        self.assertGreater(arm.peak_torque[spec.motor_id], 10.0)

    def test_starts_clear_of_the_table(self) -> None:
        for pose in ([0.0] * 6, HOME):
            arm = EmulatedArm(seed=0, start_pose_deg=pose)
            self.assertEqual(arm.data.ncon, 0, f"in contact at {pose}")


class TestFaults(unittest.TestCase):
    def test_dead_output_lets_the_joint_fall(self) -> None:
        arm = EmulatedArm(seed=0, start_pose_deg=HOME)
        command = OfflineCommand(arm)
        command.run(1.0, HOME)
        arm.inject(parse("dead:11"))
        command.run(1.0, HOME)
        self.assertGreater(arm.joints_deg()[3], -80.0)

    def test_stall_freezes_the_joint(self) -> None:
        arm = EmulatedArm(seed=0)
        command = OfflineCommand(arm)
        arm.inject(parse("stall:11"))
        command.ramp(2.0, [0.0] * 6, [0, 0, 0, -30, 0, 0])
        self.assertAlmostEqual(arm.joints_deg()[3], 0.0, places=3)

    def test_zero_shift_reads_in_the_feedback(self) -> None:
        arm = EmulatedArm(seed=0)
        before = arm.command(11, MIT_CONTROL, 0, 0, 0, 0, 0).position
        arm.inject(parse("zero_shift:11:30"))
        after = arm.command(11, MIT_CONTROL, 0, 0, 0, 0, 0).position
        self.assertAlmostEqual(after - before, 30.0, delta=0.11)


class TestObjects(unittest.TestCase):
    def test_reset_moves_the_cube_not_the_arm(self) -> None:
        arm = EmulatedArm(seed=0, start_pose_deg=[0, 0, 0, -30, 0, 0])
        joints = arm.joints_deg().copy()
        cube = arm.model.joint("rubiks_cube").qposadr[0]
        positions = set()
        for seed in range(5):
            arm.reset_objects(seed)
            positions.add(round(float(arm.data.qpos[cube]), 5))
        self.assertGreater(len(positions), 1)
        np.testing.assert_allclose(arm.joints_deg(), joints)

    def test_not_successful_at_rest(self) -> None:
        self.assertFalse(EmulatedArm(seed=0).success())


if __name__ == "__main__":
    unittest.main()
