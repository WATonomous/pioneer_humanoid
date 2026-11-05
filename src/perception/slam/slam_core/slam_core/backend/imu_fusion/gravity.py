# Gravity alignment without a still period: finds which way is down in the first keyframe's

import numpy as np
import gtsam
from slam_core.types.pose import from_tum, rotation_angle
from .imu import read_imu, imu_between, read_extrinsics
from .preintegration import preintegrate, make_params, GRAVITY


# Chains keyframes.txt rows (each T_prev_kf) into T_kf0_kfk for every keyframe, like pose_graph.py
def chain_vo(keyframes):
    poses = [np.eye(4)]
    for k in range(1, len(keyframes)):
        poses.append(poses[-1] @ from_tum(keyframes[k, 3:10]))
    return poses


def estimate_gravity(stamp, gyro, accel, keyframes, params):
    poses = chain_vo(keyframes)
    kf_stamps = keyframes[:, 2]
    bias = gtsam.imuBias.ConstantBias()   # gyro/accel bias assumed 0 here; the graph estimates it

    dv_sum = np.zeros(3)
    for k in range(len(keyframes) - 1):
        gyro_seg, accel_seg, dt = imu_between(stamp, gyro, accel, kf_stamps[k], kf_stamps[k + 1])
        pim = preintegrate(gyro_seg, accel_seg, dt, params, bias)
        dv_sum += poses[k][:3, :3] @ pim.deltaVij()

    # Remove the robot's own velocity change (end minus start), estimated from VO positions.
    v_start = (poses[1][:3, 3] - poses[0][:3, 3]) / (kf_stamps[1] - kf_stamps[0])
    v_end = (poses[-1][:3, 3] - poses[-2][:3, 3]) / (kf_stamps[-1] - kf_stamps[-2])
    total_time = kf_stamps[-1] - kf_stamps[0]

    return (v_end - v_start - dv_sum) / total_time


# The smallest rotation that turns the gravity direction into +y (straight down in odom).
# "Smallest" removes only the tilt (roll and pitch) and keeps the first camera's yaw.
def level_rotation(g_kf0):
    g_hat = g_kf0 / np.linalg.norm(g_kf0)
    down = np.array([0.0, 1.0, 0.0])

    axis = np.cross(g_hat, down)          # the line to turn around
    sin_angle = np.linalg.norm(axis)
    cos_angle = np.dot(g_hat, down)
    if sin_angle < 1e-9:                  # already pointing down, nothing to turn
        return np.eye(3)

    angle = np.arctan2(sin_angle, cos_angle)
    return gtsam.Rot3.AxisAngle(axis / sin_angle, angle).matrix()


if __name__ == "__main__":
    stamp, gyro, accel = read_imu("data/run1/imu.txt")
    keyframes = np.loadtxt("data/run1/keyframes.txt")
    params = make_params(read_extrinsics("data/run1/imu_extrinsics.txt"))

    # Check 1: chained VO rotation of KF1 should match the 8.25 deg from the preintegration check
    poses = chain_vo(keyframes)
    print("KF1 rotation from KF0 (deg, expect ~8.25)", np.degrees(rotation_angle(poses[1][:3, :3])))

    # Check 2: gravity's length should be ~9.81, and its tilt within the 9.5-18.6 deg quick estimates
    g_kf0 = estimate_gravity(stamp, gyro, accel, keyframes, params)
    tilt = np.degrees(np.arccos(g_kf0[1] / np.linalg.norm(g_kf0)))
    print("gravity in KF0", g_kf0)
    print("gravity length (expect ~9.81)", np.linalg.norm(g_kf0))
    print("KF0 tilt from level (deg)", tilt)

    # Check 3: leveling should turn gravity into [0, ~9.81, 0]
    R_odom_kf0 = level_rotation(g_kf0)
    print("leveled gravity (expect [0, ~9.81, 0])", R_odom_kf0 @ g_kf0)

    # Check 4: the smallest rotation only removes the tilt, so its angle should equal the tilt
    print("leveling rotation angle (deg, expect = tilt)", np.degrees(rotation_angle(R_odom_kf0)))
