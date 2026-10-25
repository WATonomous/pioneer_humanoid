"""
IMU measurement (gyro + accel, timestamp).

Units are SI: gyro in rad/s, accel in m/s^2. The accelerometer measures specific force,
so a still IMU reads about 9.81 m/s^2 pointing *up* (away from the floor).

On the D455 the IMU axes are aligned with the left IR camera's optical frame (x right,
y down, z forward); only its position differs. See slam_core.io.rosbag_reader for the
camera-to-IMU transform.
"""
from dataclasses import dataclass

import numpy as np


# One IMU sample: gyro and accel at the same instant. The D455 measures them at
# different rates (gyro 200 Hz, accel 100 Hz), so loaders interpolate accel onto the
# gyro timestamps to give these.
@dataclass
class ImuSample:
    stamp: float        # seconds, same clock as Frame.stamp
    gyro: np.ndarray    # (3,) rad/s, angular velocity
    accel: np.ndarray   # (3,) m/s^2, specific force (includes gravity)

    def __post_init__(self):
        self.gyro = np.asarray(self.gyro, dtype=np.float64).ravel()
        self.accel = np.asarray(self.accel, dtype=np.float64).ravel()
        if not np.isfinite(self.stamp):
            raise ValueError("imu sample: stamp must be finite")
        for name in ("gyro", "accel"):
            v = getattr(self, name)
            if v.shape != (3,) or not np.all(np.isfinite(v)):
                raise ValueError(f"imu sample at {self.stamp}: {name} must be 3 finite values, got {v}")
