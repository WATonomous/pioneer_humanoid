"""The real joint_command against the emulator: normal motion, and each fault its watchdog must catch.

    python3 -m unittest scenarios/test_joint_command.py        # in the emulator container
"""
from __future__ import annotations

import time
import unittest

import numpy as np

from arm_emulator import config  # noqa: F401 -- puts the arm's script folder on sys.path
from harness import HOME_DEG, Client, Stack

ELBOW_PITCH, SHOULDER_PITCH, WRIST_PITCH = 11, 14, 22
FAULT_TEXT = "MIT FAULT"


class JointCommandTestCase(unittest.TestCase):
    faults: list[str] = []

    def setUp(self) -> None:
        self.stack = Stack(self.faults)
        self.client = Client(self.stack)
        self.addCleanup(self.stack.stop)
        self.addCleanup(self.client.close)

    def wait_for_fault(self, text: str, q_deg, seconds: float = 6.0) -> None:
        """Keep commanding, as teleop would, until joint_command reports `text`."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.client.send(q_deg)
            time.sleep(0.02)
            if self.stack.joint_command_says(FAULT_TEXT) and self.stack.joint_command_says(text):
                return
        self.fail(f"joint_command did not report {text!r}.\n{self.stack.logs()[-3000:]}")


class TestNormalMotion(JointCommandTestCase):
    def test_seeds_moves_home_and_tracks(self) -> None:
        start = self.client.measured()
        np.testing.assert_allclose(start, HOME_DEG, atol=2.0, err_msg="the arm starts at home, held by hand")
        faulted = lambda: self.stack.joint_command_says(FAULT_TEXT)  # noqa: E731
        reach = [10.0, 5.0, -5.0, -75.0, 10.0, 10.0]
        self.client.ramp(start, reach, 6.0, stop=faulted)
        self.client.hold(1.0, reach, stop=faulted)
        np.testing.assert_allclose(self.client.measured(), reach, atol=8.0)
        self.client.ramp(reach, HOME_DEG, 6.0, stop=faulted)
        self.client.hold(2.0, HOME_DEG, stop=faulted)
        self.assertFalse(faulted(), self.stack.logs()[-3000:])
        np.testing.assert_allclose(self.client.measured(), HOME_DEG, atol=8.0)
        self.assertTrue(self.stack.joint_command_says("Rate-limiter seeded from feedback"))
        self.assertEqual(self.client.status.get("breaches"), [], self.client.status)

    def test_limits_a_target_far_beyond_the_range(self) -> None:
        self.client.measured()
        seen = []
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            self.client.send([0.0, 0.0, 0.0, -500.0, 0.0, 0.0])
            time.sleep(0.02)
            if ELBOW_PITCH in self.client.commands:
                seen.append(self.client.commands[ELBOW_PITCH].position)
        # The commanded angle, rad in the motor frame, may move at most velocity_max: 60 deg/s here,
        # and delta_max 1.6 deg a tick would allow 80, so 60 bounds this second, whatever was asked.
        self.assertLess(abs(np.degrees(seen[-1] - seen[0])), 70.0)
        self.assertGreater(abs(np.degrees(seen[-1] - seen[0])), 5.0, "it should still have moved")

    def test_non_finite_pose_is_ignored(self) -> None:
        start = self.client.measured()
        self.client.hold(0.5, start)
        self.client.hold(0.5, [float("nan")] * 6)
        self.assertTrue(self.stack.joint_command_says("non-finite"))
        np.testing.assert_allclose(self.client.measured(), start, atol=5.0)


class TestFeedbackTimeout(JointCommandTestCase):
    faults = [f"mute:{ELBOW_PITCH}"]

    def test_excluded_from_seed(self) -> None:
        # Silent from the start: seeded without it and held at kp = 0, not faulted, as joint_command documents.
        time.sleep(1.0)
        self.client.hold(3.0, HOME_DEG)
        self.assertTrue(self.stack.joint_command_says("EXCLUDING joint elbow.pitch"), self.stack.logs()[-3000:])


class TestDriveFault(JointCommandTestCase):
    faults = [f"code:{SHOULDER_PITCH}:2"]

    def test_reported_status_halts(self) -> None:
        self.wait_for_fault("drive reports fault status 2", self.client.measured())


class TestMotorGoesDead(JointCommandTestCase):
    def test_tracking_error_halts(self) -> None:
        start = self.client.measured()
        # Holding home, then the motor dies: it falls and the tracking check must catch it.
        self.client.hold(1.0, start)
        self.client.control(f"fault dead:{ELBOW_PITCH}")
        self.wait_for_fault("tracking error", start)


class TestStall(JointCommandTestCase):
    def test_tracking_error_halts(self) -> None:
        start = self.client.measured()
        self.client.hold(0.5, start)
        self.client.control(f"fault stall:{ELBOW_PITCH}")
        reach = [0, 0, 0, -60, 0, 0]
        self.client.ramp(start, reach, 4.0, stop=lambda: self.stack.joint_command_says(FAULT_TEXT))
        self.wait_for_fault("elbow.pitch", reach)


class TestStaleCalibration(JointCommandTestCase):
    # Far enough that shoulder.yaw (direction -1) reads -40 deg, outside its limits [-9.9, 175.2].
    faults = ["zero_shift:12:40"]

    def test_out_of_range_joint_is_excluded(self) -> None:
        self.client.measured()
        self.client.hold(2.0, HOME_DEG)
        self.assertTrue(self.stack.joint_command_says("EXCLUDING joint shoulder.yaw"), self.stack.logs()[-3000:])


class TestOverheat(JointCommandTestCase):
    faults = [f"hot:{SHOULDER_PITCH}:60"]

    def test_temperature_halts(self) -> None:
        self.wait_for_fault("reached mit_max_temperature", self.client.measured())


class TestMidRunFeedbackLoss(JointCommandTestCase):
    def test_timeout_halts(self) -> None:
        start = self.client.measured()
        self.client.hold(1.0, start)
        self.client.control(f"fault mute:{ELBOW_PITCH}")
        self.wait_for_fault("MIT feedback is", start)


class TestCommandsStop(JointCommandTestCase):
    def test_arm_goes_soft_after_command_timeout(self) -> None:
        start = self.client.measured()
        self.client.hold(1.0, start)
        time.sleep(11.0)  # command_timeout_sec 10
        self.assertTrue(self.stack.joint_command_says("ArmPose stream stale"), self.stack.logs()[-3000:])
        self.assertEqual(self.client.commands[ELBOW_PITCH].kp, 0.0)


if __name__ == "__main__":
    unittest.main()
