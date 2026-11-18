from .imu import read_imu, imu_between
import numpy as np
import gtsam
from slam_core.types.pose import quat_to_rot, rotation_angle
from .imu import read_extrinsics

# IMU noise, from our Kalibr run's imu yaml (d455_imu_calibration/kalibr_output/imu-datad455_imu_kalibr_mono.yaml,
# update_rate 200 Hz). Used as given, with no inflation factor.
# Previously: OpenVINS rs_d455 values (gyro 2.05e-4, accel 2.08e-3, gyro RW 1.11e-5, accel RW 4.13e-4).

GYRO_NOISE_DENSITY = 4.16e-4   # rad/s/sqrt(Hz)
ACCEL_NOISE_DENSITY = 1.60e-3   # m/s^2/sqrt(Hz)
GYRO_RANDOM_WALK = 2.57e-6   # rad/s^2/sqrt(Hz)
ACCEL_RANDOM_WALK = 1.41e-4   # m/s^3/sqrt(Hz)
INTEGRATION_SIGMA_SQ = 1e-8  # covariance added for integration error (a fudge term)
GRAVITY = 9.81               # m/s^2


def preintegrate(gyro_seg, accel_seg, dt, params, bias):
    pim = gtsam.PreintegratedImuMeasurements(params, bias)

    for n in range(len(dt)):
        pim.integrateMeasurement(accel_seg[n], gyro_seg[n], dt[n])

    return pim


def make_params(T_cam_imu):
    params = gtsam.PreintegrationParams(np.array([0, GRAVITY, 0]))

    params.setGyroscopeCovariance(GYRO_NOISE_DENSITY**2 * np.eye(3))
    params.setAccelerometerCovariance(ACCEL_NOISE_DENSITY**2 * np.eye(3))
    params.setIntegrationCovariance(INTEGRATION_SIGMA_SQ * np.eye(3))

    params.setBodyPSensor(gtsam.Pose3(T_cam_imu))

    return params


if __name__ == "__main__":

    stamp, gyro, accel = read_imu("data/run1/imu.txt")
    keyframes = np.loadtxt("data/run1/keyframes.txt")
    extrinsics = read_extrinsics("data/run1/imu_extrinsics.txt")
    params = make_params(extrinsics)
    bias = gtsam.imuBias.ConstantBias()
    preintegrate_error = []

    for k in range(len(keyframes)-1):
        gyro_seg, accel_seg, dt = imu_between(stamp, gyro, accel, keyframes[k, 2], keyframes[k+1, 2])
        pim = preintegrate(gyro_seg, accel_seg, dt, params, bias)

        R_imu = pim.deltaRij().matrix()
        R_vo = quat_to_rot(keyframes[k+1, 6:10])

        err = np.degrees(rotation_angle(R_vo.T @ R_imu))

        preintegrate_error.append(err)

    print("preintegration max error", max(preintegrate_error))
    print("preintegration median error", np.median(preintegrate_error))

    # Gravity-sign test: a still IMU for 1 s should predict no motion
    n = 200                                # number of samples in 1 second
    still_gyro = np.zeros((n, 3))
    still_accel = np.tile([0, -GRAVITY, 0], (n, 1))
    still_dt = np.full(n, 0.005)          # each sample's duration in seconds

    pim = preintegrate(still_gyro, still_accel, still_dt, params, bias)
    start = gtsam.NavState(gtsam.Pose3(), np.zeros(3))
    end = pim.predict(start, bias)

    print("still test position", end.position())
    print("still test velocity", end.velocity())
