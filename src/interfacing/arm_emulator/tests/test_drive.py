"""One drive on its own: the law, the rounding, the family differences. No ROS, no MuJoCo.

    python -m unittest discover -s tests -t .       # from src/interfacing/arm_emulator
"""
from __future__ import annotations

import math
import unittest

from arm_emulator import mit_native
from arm_emulator.config import load_motors
from arm_emulator.drive import MIT_CONTROL, MIT_ENTER, MIT_EXIT, Drive

MOTORS, CATALOGUE = load_motors()
BY_ID = {m.motor_id: m for m in MOTORS}
ELBOW_AK80, SHOULDER_AK10, WRIST_GL40 = 11, 14, 22


def drive(motor_id: int) -> Drive:
    return Drive(BY_ID[motor_id], CATALOGUE["ambient_temperature"], CATALOGUE["ak_can_timeout_s"])


class TestLaw(unittest.TestCase):
    def test_matches_hand_calculation(self) -> None:
        d = drive(ELBOW_AK80)
        d.command(MIT_CONTROL, 0.5, 0.0, 60.0, 0.4, 0.0, now=0.0)
        tau = d.step(q=0.45, qd=0.1, now=0.001, dt=0.001)
        # 60 * 0.05 - 0.4 * 0.1 = 2.96; rounding of the 12-bit gains and 16-bit position moves it by under 1%.
        self.assertAlmostEqual(tau, 2.96, delta=0.03)

    def test_gains_use_the_real_quantisation(self) -> None:
        d = drive(WRIST_GL40)
        d.command(MIT_ENTER, 0, 0, 0, 0, 0, now=0.0)
        d.command(MIT_CONTROL, 0.0, 0.0, 0.22, 0.0098, 0.0, now=0.0)
        # packMitGain: 0.22 * 4096 / 500 rounds to raw 2, so about 0.244 is what the drive applies.
        self.assertAlmostEqual(d.kp, 2 * 500.0 / 4095.0, places=6)

    def test_zero_feed_forward_lands_near_zero(self) -> None:
        d = drive(ELBOW_AK80)
        d.command(MIT_CONTROL, 0.0, 0.0, 0.0, 0.0, 0.0, now=0.0)
        self.assertLess(abs(d.t_ff), 0.01)

    def test_peak_torque_limits_the_output(self) -> None:
        d = drive(ELBOW_AK80)
        d.command(MIT_CONTROL, 12.0, 0.0, 500.0, 0.0, 0.0, now=0.0)
        self.assertAlmostEqual(d.step(0.0, 0.0, 0.001, 0.001), BY_ID[ELBOW_AK80].model["peak_torque"])

    def test_no_torque_left_at_no_load_speed(self) -> None:
        d = drive(ELBOW_AK80)
        d.command(MIT_CONTROL, 12.0, 0.0, 500.0, 0.0, 0.0, now=0.0)
        self.assertAlmostEqual(d.step(0.0, BY_ID[ELBOW_AK80].model["no_load_speed"], 0.001, 0.001), 0.0)


class TestFamilies(unittest.TestCase):
    def test_gl2_ignores_mit_until_enter(self) -> None:
        d = drive(WRIST_GL40)
        self.assertFalse(d.command(MIT_CONTROL, 1.0, 0.0, 1.0, 0.1, 0.0, now=0.0), "no answer before ENTER")
        self.assertEqual(d.step(0.0, 0.0, 0.001, 0.001), 0.0)
        d.command(MIT_ENTER, 0, 0, 0, 0, 0, now=0.001)
        d.command(MIT_CONTROL, 1.0, 0.0, 1.0, 0.1, 0.0, now=0.001)
        self.assertGreater(d.step(0.0, 0.0, 0.002, 0.001), 0.0)

    def test_gl2_goes_limp_on_exit(self) -> None:
        d = drive(WRIST_GL40)
        d.command(MIT_ENTER, 0, 0, 0, 0, 0, now=0.0)
        d.command(MIT_CONTROL, 1.0, 0.0, 1.0, 0.1, 0.0, now=0.0)
        d.command(MIT_EXIT, 0, 0, 0, 0, 0, now=0.001)
        self.assertEqual(d.step(0.0, 0.0, 0.002, 0.001), 0.0)

    def test_ak_needs_no_enter_and_stops_on_can_timeout(self) -> None:
        d = drive(ELBOW_AK80)
        d.command(MIT_CONTROL, 1.0, 0.0, 10.0, 0.1, 0.0, now=0.0)
        self.assertGreater(d.step(0.0, 0.0, 0.1, 0.001), 0.0)
        self.assertEqual(d.step(0.0, 0.0, CATALOGUE["ak_can_timeout_s"] + 0.1, 0.001), 0.0)

    def test_non_finite_command_never_reaches_the_drive(self) -> None:
        d = drive(ELBOW_AK80)
        self.assertFalse(d.command(MIT_CONTROL, math.nan, 0.0, 10.0, 0.1, 0.0, now=0.0))
        self.assertIsNone(d.last_command_t)


class TestFeedback(unittest.TestCase):
    def test_ak_servo_frame_rounding_and_torque(self) -> None:
        d = drive(ELBOW_AK80)
        d.command(MIT_CONTROL, 1.0, 0.0, 10.0, 0.0, 0.0, now=0.0)
        d.step(0.0, 0.0, 0.001, 0.001)
        fb = d.feedback(math.radians(12.34), 0.0)
        self.assertAlmostEqual(fb.position, 12.3, places=6)
        self.assertAlmostEqual(fb.torque, round(d.tau / 0.5701 / 0.01) * 0.01 * 0.5701, places=6)
        self.assertEqual(fb.error_code, 0)

    def test_ak80_torque_reading_saturates_below_its_peak(self) -> None:
        d = drive(ELBOW_AK80)
        d.tau = 21.0
        self.assertAlmostEqual(d.feedback(0.0, 0.0).torque, 18.0)

    def test_gl2_status_nibble(self) -> None:
        d = drive(WRIST_GL40)
        self.assertEqual(d.feedback(0.0, 0.0).error_code, 0)
        d.command(MIT_ENTER, 0, 0, 0, 0, 0, now=0.0)
        self.assertEqual(d.feedback(0.0, 0.0).error_code, 1)

    def test_mute_drive_answers_nothing(self) -> None:
        d = drive(ELBOW_AK80)
        d.mute = True
        self.assertIsNone(d.feedback(0.0, 0.0))

    def test_faulted_drive_reports_and_produces_nothing(self) -> None:
        d = drive(SHOULDER_AK10)
        d.command(MIT_CONTROL, 1.0, 0.0, 10.0, 0.1, 0.0, now=0.0)
        d.fault_code = 2
        self.assertEqual(d.step(0.0, 0.0, 0.001, 0.001), 0.0)
        self.assertEqual(d.feedback(0.0, 0.0).error_code, 2)


class TestNativePacking(unittest.TestCase):
    """The binding reaches the real mit_protocol.cpp: values from can/test/test_mit_protocol.cpp's rules."""

    def test_full_scale_does_not_wrap(self) -> None:
        self.assertEqual(mit_native.pack_value(12.5, -12.5, 12.5, 16), 65535)

    def test_mid_scale(self) -> None:
        self.assertEqual(mit_native.pack_value(0.0, -12.5, 12.5, 16), 32768)

    def test_ak_payload_is_kp_first(self) -> None:
        profile = dict(p_max=12.56, v_max=65.0, t_max=18.0, kp_max=500.0, kd_max=5.0)
        ak = mit_native.pack_command(True, 0.0, 0.0, 500.0, 0.0, 0.0, profile)
        self.assertEqual(ak[0], 0xFF)
        gl2 = mit_native.pack_command(False, 0.0, 0.0, 500.0, 0.0, 0.0, profile)
        self.assertEqual(gl2[0], 0x80)


if __name__ == "__main__":
    unittest.main()
