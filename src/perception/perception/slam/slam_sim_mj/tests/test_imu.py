"""Sign-convention tests for the synthetic IMU.

Every one of these catches a mistake that is invisible in a headless run and shows
up hours later as a map that is tilted, mirrored or slowly falling through the floor.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imu import GRAVITY, imu_sample  # noqa: E402
from orbit_path import look_at_quat, orbit_pose  # noqa: E402

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_stationary_reads_gravity_up_not_zero():
    """The classic accelerometer trap: at rest it reads +g, never 0."""
    omega, accel = imu_sample(lambda t: (np.array([1.0, 2.0, 3.0]), IDENTITY), 0.0)
    assert np.allclose(omega, 0.0, atol=1e-6)
    assert np.allclose(accel, [0.0, 0.0, GRAVITY], atol=1e-4)


def test_constant_velocity_still_reads_only_gravity():
    """Moving is not accelerating. A drift here means the differencing is wrong."""
    _, accel = imu_sample(lambda t: (np.array([2.0 * t, 0.0, 0.0]), IDENTITY), 5.0)
    assert np.allclose(accel, [0.0, 0.0, GRAVITY], atol=1e-3)


def test_free_fall_reads_zero():
    """Falling under gravity is weightlessness -- the specific force cancels."""
    def pose(t):
        return np.array([0.0, 0.0, -0.5 * GRAVITY * t * t]), IDENTITY
    _, accel = imu_sample(pose, 1.0)
    assert np.allclose(accel, 0.0, atol=1e-2)


def test_gravity_follows_the_sensor_when_it_tilts():
    """Rotated sensor, same world gravity: the vector must move in the sensor frame,
    and its magnitude must not change."""
    # Optical frame looking horizontally: world -Z (down) lands on optical +Y (down).
    q = look_at_quat(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    _, accel = imu_sample(lambda t: (np.zeros(3), q), 0.0)
    assert np.isclose(np.linalg.norm(accel), GRAVITY, atol=1e-4)
    assert np.isclose(accel[1], -GRAVITY, atol=1e-4)  # "up" is optical -Y


def test_spin_gives_angular_velocity_about_the_right_axis():
    rate = 0.7

    def pose(t):
        a = rate * t
        # rotation about world Z, as (w, x, y, z)
        return np.zeros(3), np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])

    omega, _ = imu_sample(pose, 1.0)
    assert np.allclose(omega, [0.0, 0.0, rate], atol=1e-4)


def test_orbit_gravity_magnitude_is_constant_around_the_lap():
    """Integration check against the real trajectory: whatever the camera is doing,
    an accelerometer on it never sees gravity change size."""
    center = np.zeros(3)

    def pose(t):
        return orbit_pose(t, center, radius=2.0, height=1.5, period=60.0,
                          target=np.array([0.0, 0.0, 0.6]))

    for t in np.linspace(0.0, 60.0, 25):
        omega, accel = imu_sample(pose, float(t))
        # Orbit motion is slow, so specific force is gravity plus a small centripetal
        # term -- within a few percent of g, and never near zero or double.
        assert 9.0 < np.linalg.norm(accel) < 10.6
        assert np.all(np.isfinite(omega))
