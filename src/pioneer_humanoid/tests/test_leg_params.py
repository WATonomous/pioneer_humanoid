"""Tests for pioneer_humanoid.leg_params. Run: pytest src/pioneer_humanoid/tests (env: watonomous_env)."""
import math
import os
import xml.etree.ElementTree as ET

import pytest

from pioneer_humanoid.leg_params import AK60_39, AK80_64, AKH70_48, JOINT_MOTOR, MotorSpec, check_gains

_URDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "assets", "whole_body_humanoid", "whole_body_humanoid.urdf"
)


# --- Armature: J_rotor * ratio^2, against the plan's fact sheet (rounded values, so 1%) -------
@pytest.mark.parametrize(
    "motor, expected",
    [(AKH70_48, 0.27), (AK80_64, 0.23), (AK60_39, 0.046)],
)
def test_armature_matches_fact_sheet(motor, expected):
    assert motor.armature == pytest.approx(expected, rel=0.01)


# --- Unit conversions, on a made-up motor with round numbers -----------------------------------
def _fake_motor(**overrides) -> MotorSpec:
    fields = dict(
        name="fake", gear_ratio=10, peak_torque=1.0, rated_torque=1.0, no_load_speed_rpm=60.0,
        rotor_inertia_gcm2=1e7, backdrive_torque=0.0, p_max=1.0, v_max=1.0, t_max=1.0,
    )
    fields.update(overrides)
    return MotorSpec(**fields)


def test_rpm_to_rad_s():
    assert _fake_motor(no_load_speed_rpm=60.0).no_load_speed == pytest.approx(2 * math.pi)


def test_gcm2_to_kgm2():
    assert _fake_motor(rotor_inertia_gcm2=1e7).rotor_inertia == pytest.approx(1.0)


def test_armature_scales_with_ratio_squared():
    assert _fake_motor(gear_ratio=10, rotor_inertia_gcm2=1e7).armature == pytest.approx(100.0)


def test_motor_spec_is_frozen():
    with pytest.raises(AttributeError):
        AKH70_48.peak_torque = 300.0


# --- Every URDF leg joint has exactly one motor ------------------------------------------------
def test_all_urdf_joints_mapped():
    root = ET.parse(_URDF_PATH).getroot()
    urdf_joints = {j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"}
    assert urdf_joints == set(JOINT_MOTOR)


# --- check_gains -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kp, kd, r",
    [
        (500.0, 5.0, 1.0),  # exactly at both limits
        (0.0, 0.0, 1.0),  # zero is allowed
        (2000.0, 20.0, 2.0),  # joint side; motor sees 500 and 5
    ],
)
def test_check_gains_accepts(kp, kd, r):
    check_gains("Ankle_P_L", kp, kd, r)  # passing = no exception


@pytest.mark.parametrize(
    "kp, kd, r",
    [
        (100.0, 5.01, 1.0),  # kd just over
        (501.0, 1.0, 1.0),  # kp just over
        (-1.0, 1.0, 1.0),  # negative kp
        (100.0, -0.1, 1.0),  # negative kd
        (100.0, 21.0, 2.0),  # motor sees 21 / 4 = 5.25
        (100.0, 1.0, 0.0),  # r must be > 0
    ],
)
def test_check_gains_rejects(kp, kd, r):
    with pytest.raises(ValueError):
        check_gains("Ankle_P_L", kp, kd, r)


def test_check_gains_unknown_joint():
    with pytest.raises(KeyError):
        check_gains("Elbow_L", 10.0, 1.0)
