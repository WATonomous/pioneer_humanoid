"""Synthetic IMU derived from the scripted camera trajectory.

This exists because of a measured failure, not for completeness. Running RGB-D
RTAB-Map with no IMU drifts steadily in z on a path that is flat by construction.
`Grid/MaxGroundHeight` is 0.15 m, so that drift swamps the ground/obstacle threshold
and the 2D occupancy map degenerates into a blob with no border. A real D455 has an
IMU; the sim needs one too, and RTAB-Map uses it as a gravity prior that pins roll
and pitch.

Sim-specific note: this is synthesised from the trajectory rather than read off a
simulated sensor. MuJoCo *has* accelerometer/gyro sensors, but they read from
physics state, and this camera is kinematically posed (a mocap body) rather than
dynamically simulated -- so those sensors would read zero. Differentiating the path
we already know is exact and does not require faking a physics body.

Pure numpy, no MuJoCo, no ROS -- so the sign conventions are testable in
milliseconds. Getting a sign wrong here is invisible at runtime and surfaces hours
later as a map tilted by a constant angle.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

GRAVITY = 9.80665

PoseFn = Callable[[float], "tuple[np.ndarray, np.ndarray]"]


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def imu_sample(pose_fn: PoseFn, t: float, dt: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Angular velocity and specific force at time `t`, both in the SENSOR frame.

    `pose_fn(t)` returns (position, quaternion_wxyz) for the sensor frame in world
    coordinates -- pass the same function that drives the camera, so the IMU cannot
    describe a different motion than the images do.

    Central differences rather than analytic derivatives: the trajectory functions
    are already closed-form but differentiating them by hand is four more chances to
    get a sign wrong, and at these step sizes the numerical error is far below what
    a real IMU's noise floor would be.

    Returns (omega, accel):
      omega  rad/s about the sensor's own axes
      accel  m/s^2 SPECIFIC FORCE -- what an accelerometer actually measures, i.e.
             acceleration minus gravity. A stationary sensor reads +9.8 along its
             own "up", NOT zero. This is the whole point: that vector is the gravity
             prior RTAB-Map levels the map with.
    """
    p_prev, q_prev = pose_fn(t - dt)
    p_curr, q_curr = pose_fn(t)
    p_next, q_next = pose_fn(t + dt)

    r_prev, r_curr, r_next = (_quat_to_matrix(q) for q in (q_prev, q_curr, q_next))

    # Angular velocity from the rotation matrix derivative: R^T * dR/dt is
    # skew-symmetric, and its off-diagonal entries are omega in the sensor frame.
    r_dot = (r_next - r_prev) / (2.0 * dt)
    skew = r_curr.T @ r_dot
    omega = np.array([skew[2, 1] - skew[1, 2],
                      skew[0, 2] - skew[2, 0],
                      skew[1, 0] - skew[0, 1]]) / 2.0

    accel_world = (p_next - 2.0 * p_curr + p_prev) / (dt * dt)
    gravity_world = np.array([0.0, 0.0, -GRAVITY])
    return omega, r_curr.T @ (accel_world - gravity_world)
