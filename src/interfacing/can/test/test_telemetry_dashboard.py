"""Health rules of the telemetry dashboard's store (no ROS needed).

    python3 -m unittest src/interfacing/can/test/test_telemetry_dashboard.py
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import telemetry_dashboard as td  # noqa: E402

CALIBRATION = """
left:
  elbow:
    pitch: {can_id: 33, direction: -1, zero_offset: 10.0, lower_limit: -20.0, upper_limit: 120.0, limit_range: true}
  gripper:
    open_close: {can_id: 22, direction: 1, zero_offset: 0.0, lower_limit: 0.0, upper_limit: 70.0, limit_range: true}
"""
LEGS = """
left:
  hip:
    pitch: {can_id: 40, direction: null, zero_offset: null, lower_limit: -20.0, upper_limit: 90.0, limit_range: true}
"""
ACTUATORS = """
safety:
  units: {angle: deg}
  global: {mit_max_torque: 10, mit_max_track_err: 10.0, mit_feedback_timeout: 0.2,
           mit_max_temperature: 55, mit_family: gl2, control_type: -1}
  left:
    elbow:
      pitch: {control_type: 0, mit_family: ak, mit_kp: 60.0, mit_kd: 0.4, mit_max_torque: 9.0,
              mit_peak_torque: 18.0, active: true}
"""


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make_store():
    with tempfile.TemporaryDirectory() as tmp:
        arm, leg, act = (Path(tmp) / n for n in ("arm.yaml", "leg.yaml", "act.yaml"))
        arm.write_text(CALIBRATION)
        leg.write_text(LEGS)
        act.write_text(ACTUATORS)
        joints = td.build_joint_table([(arm, "arm"), (leg, "leg")], act, None)
    clock = Clock()
    return td.TelemetryStore(joints, clock=clock), clock


def joint(store, motor_id):
    return next(j for j in store.snapshot()["joints"] if j["id"] == motor_id)


class JointTable(unittest.TestCase):
    def test_limits_merge_over_global(self):
        store, _ = make_store()
        elbow = store.by_id[33]
        self.assertEqual((elbow["max_torque"], elbow["peak_torque"], elbow["max_temp"]),
                         (9.0, 18.0, 55.0))
        self.assertEqual(elbow["family"], "ak")
        self.assertEqual(elbow["slot"], 3)

    def test_uncalibrated_leg_is_listed_not_fatal(self):
        store, _ = make_store()
        hip = store.by_id[40]
        self.assertFalse(hip["calibrated"])
        self.assertEqual(hip["limb"], "left leg")
        self.assertIsNone(hip["max_torque"])  # arm_actuators.yaml does not cover legs


class Health(unittest.TestCase):
    def feed(self, store, clock, motor_deg=0.0, torque=0.0, temp=30, code=0, steps=3):
        for _ in range(steps):
            clock.now += 0.02
            store.on_feedback(33, motor_deg, 0.0, 0.0, torque, temp, code)
        store.tick()

    def test_frames(self):
        store, clock = make_store()
        # motor = direction * (cmd - zero): cmd 40 deg -> motor -30 deg
        store.on_cmd(33, td.MIT_CONTROL, math.radians(-30.0), 60.0, 0.4, 0.0)
        self.feed(store, clock, motor_deg=-28.0)
        j = joint(store, 33)
        self.assertAlmostEqual(j["sp"], 40.0, places=2)
        self.assertAlmostEqual(j["pos"], 38.0, places=2)
        self.assertAlmostEqual(j["err"], -2.0, places=2)
        self.assertEqual(j["st"], "ok")

    def test_offline_until_feedback_then_idle(self):
        store, clock = make_store()
        store.tick()
        self.assertEqual(joint(store, 33)["st"], "offline")
        self.assertEqual(store.overall()[0], "offline")
        self.feed(store, clock)
        self.assertEqual(joint(store, 33)["st"], "idle")
        self.assertEqual(store.overall()[0], "nominal")

    def test_stale_feedback(self):
        store, clock = make_store()
        self.feed(store, clock)
        clock.now += 0.5
        store.tick()
        self.assertEqual(joint(store, 33)["st"], "stale")
        self.assertEqual(store.overall()[0], "degraded")

    def test_temperature_warn_then_fault(self):
        store, clock = make_store()
        self.feed(store, clock, temp=48)
        self.assertEqual(joint(store, 33)["st"], "warn")
        self.feed(store, clock, temp=55)
        self.assertEqual(joint(store, 33)["st"], "fault")
        self.assertEqual(store.overall()[0], "fault")

    def test_torque_rated_is_a_warning_peak_is_a_fault(self):
        store, clock = make_store()
        self.feed(store, clock, torque=-12.0)   # above rated 9, below peak 18
        self.assertEqual(joint(store, 33)["st"], "warn")
        self.feed(store, clock, torque=19.0)
        self.assertEqual(joint(store, 33)["st"], "fault")

    def test_tracking_error_only_while_driven(self):
        store, clock = make_store()
        self.feed(store, clock, motor_deg=-60.0)            # far from any setpoint, not driven
        self.assertEqual(joint(store, 33)["st"], "idle")
        store.on_cmd(33, td.MIT_CONTROL, 0.0, 60.0, 0.4, 0.0)   # setpoint cmd 10, measured 70
        self.feed(store, clock, motor_deg=-60.0)
        self.assertEqual(joint(store, 33)["st"], "fault")
        store.on_cmd(33, td.MIT_CONTROL, 0.0, 0.0, 2.0, 0.0)    # damp frame: no setpoint
        self.feed(store, clock, motor_deg=-60.0)
        j = joint(store, 33)
        self.assertEqual((j["st"], j["mode"], j["sp"]), ("idle", "MIT damp", None))

    def test_status_word_differs_by_family(self):
        store, clock = make_store()
        self.feed(store, clock, code=1)                     # AK: 1 = motor over-temp
        self.assertEqual(joint(store, 33)["st"], "fault")
        clock.now += 0.02
        store.on_feedback(22, 5.0, 0.0, 0.0, 0.0, 30, 1)    # GL II: 1 = Enabled
        store.tick()
        self.assertEqual(joint(store, 22)["st"], "idle")
        store.on_feedback(22, 5.0, 0.0, 0.0, 0.0, 30, 9)    # under-voltage
        store.tick()
        self.assertEqual(joint(store, 22)["why"], "drive fault: Under-voltage")

    def test_events_on_transitions_and_unknown_ids(self):
        store, clock = make_store()
        self.feed(store, clock)
        self.feed(store, clock, temp=60)
        self.feed(store, clock, temp=30)
        messages = [e["msg"] for e in store.events_after(0)]
        self.assertEqual(len(messages), 3)
        self.assertIn("online", messages[0])
        self.assertIn("recovered", messages[2])
        store.on_feedback(99, 0.0, 0.0, 0.0, 0.0, 30, 0)
        self.assertEqual(store.snapshot()["unknownIds"], {99: 1})

    def test_requested_targets_and_gripper(self):
        store, clock = make_store()
        store.on_pose("left", [1, 2, 3, 44.0, 5, 6], True, 0.5)
        self.assertEqual(joint(store, 33)["req"], 44.0)
        self.assertEqual(joint(store, 22)["req"], 35.0)


if __name__ == "__main__":
    unittest.main()
