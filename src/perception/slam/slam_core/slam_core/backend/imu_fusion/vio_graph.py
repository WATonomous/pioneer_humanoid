# Batch VIO graph: pose_graph.py's VO and loop factors, plus a velocity and an IMU bias per keyframe,
# an ImuFactor between consecutive keyframes and a bias random-walk factor.

import numpy as np
import gtsam
from gtsam.symbol_shorthand import B, V, X   # X(i) pose, V(i) velocity, B(i) bias of keyframe i

from .imu import imu_between
from .preintegration import preintegrate, ACCEL_RANDOM_WALK, GYRO_RANDOM_WALK

# Priors on KF0. Position and yaw are free choices (we define odom to start there), so they're pinned.
PRIOR_TILT_SIGMA = np.radians(3.0)   # rad, roll/pitch of KF0
PRIOR_PINNED_SIGMA = 1e-6            # rad and m, yaw and position of KF0
PRIOR_VEL_SIGMA = 0.5                # m/s, KF0 velocity (guess from VO)
PRIOR_ACCEL_BIAS_SIGMA = 0.2         # m/s^2, KF0 accel bias (gravity.py saw |g| 0.07 short)
PRIOR_GYRO_BIAS_SIGMA = 0.02         # rad/s, KF0 gyro bias


# KF0's pose prior noise: tight for turning about vertical (yaw) and for position, loose for tilting.
def kf0_prior_noise(R_odom_kf0):
    up = R_odom_kf0.T @ np.array([0.0, 1.0, 0.0])
    rot_cov = PRIOR_TILT_SIGMA**2 * np.eye(3) + (PRIOR_PINNED_SIGMA**2 - PRIOR_TILT_SIGMA**2) * np.outer(up, up)
    cov = np.zeros((6, 6))
    cov[:3, :3] = rot_cov
    cov[3:, 3:] = PRIOR_PINNED_SIGMA**2 * np.eye(3)
    return gtsam.noiseModel.Gaussian.Covariance(cov)


# Bias noise: GTSAM's ConstantBias vector is accel first, then gyro
def bias_sigmas(accel, gyro):
    return gtsam.noiseModel.Diagonal.Sigmas(np.array([accel] * 3 + [gyro] * 3))



# Returns the graph, the initial guess, and {id: T_odom_kf} before (leveled VO) and after solving.
def solve_vio(kfs, loops, imu, params, R_odom_kf0, loop_rot=0.02, loop_trans=0.03):
    stamp, gyro, accel = imu
    by_id = {kf.id: kf for kf in kfs}

    # Initial guess: VO poses turned by the leveling rotation (about KF0, which is at the origin)
    T_level = np.eye(4)
    T_level[:3, :3] = R_odom_kf0
    before = {kf.id: T_level @ kf.T_odom_kf for kf in kfs}

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    zero_bias = gtsam.imuBias.ConstantBias()

    for i, kf in enumerate(kfs):
        # Velocity guess in odom: VO distance to the next keyframe over the time (previous for the last)
        j0, j1 = (i, i + 1) if i + 1 < len(kfs) else (i - 1, i)
        a, b = kfs[j0], kfs[j1]
        v_guess = (before[b.id][:3, 3] - before[a.id][:3, 3]) / (b.stamp - a.stamp)

        initial.insert(X(kf.id), gtsam.Pose3(before[kf.id]))
        initial.insert(V(kf.id), v_guess)
        initial.insert(B(kf.id), zero_bias)

        if kf.is_first:
            graph.add(gtsam.PriorFactorPose3(X(kf.id), gtsam.Pose3(before[kf.id]), kf0_prior_noise(R_odom_kf0)))
            graph.add(gtsam.PriorFactorVector(V(kf.id), v_guess, gtsam.noiseModel.Isotropic.Sigma(3, PRIOR_VEL_SIGMA)))
            graph.add(gtsam.PriorFactorConstantBias(B(kf.id), zero_bias,
                                                    bias_sigmas(PRIOR_ACCEL_BIAS_SIGMA, PRIOR_GYRO_BIAS_SIGMA)))
            continue

        prev = by_id[kf.prev_id]

        # VO edge, exactly as in pose_graph.py. Leveling turns both poses the same way, so T_prev_kf is unchanged
        graph.add(gtsam.BetweenFactorPose3(X(prev.id), X(kf.id), gtsam.Pose3(kf.T_prev_kf),
                                           gtsam.noiseModel.Gaussian.Covariance(kf.covariance)))

        # IMU edge: everything the IMU measured between the two keyframes, as one factor.
        # Preintegrated with zero bias; GTSAM corrects it for the estimated B(prev) to first order.
        gyro_seg, accel_seg, dt = imu_between(stamp, gyro, accel, prev.stamp, kf.stamp)
        pim = preintegrate(gyro_seg, accel_seg, dt, params, zero_bias)
        graph.add(gtsam.ImuFactor(X(prev.id), V(prev.id), X(kf.id), V(kf.id), B(prev.id), pim))

        # Bias can drift between keyframes: random walk grows with sqrt(time)
        sq = np.sqrt(kf.stamp - prev.stamp)
        graph.add(gtsam.BetweenFactorConstantBias(B(prev.id), B(kf.id), zero_bias,
                                                  bias_sigmas(ACCEL_RANDOM_WALK * sq, GYRO_RANDOM_WALK * sq)))

    # Loop edges, exactly as in pose_graph.py (Huber so one wrong loop can't drag the map)
    huber = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
    loop_noise = gtsam.noiseModel.Robust.Create(
        huber, gtsam.noiseModel.Diagonal.Sigmas(np.array([loop_rot] * 3 + [loop_trans] * 3)))
    for old_id, new_id, T_old_new in loops:
        graph.add(gtsam.BetweenFactorPose3(X(old_id), X(new_id), gtsam.Pose3(T_old_new), loop_noise))

    lm = gtsam.LevenbergMarquardtParams()
    lm.setMaxIterations(100)
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial, lm).optimize()

    after = {kf.id: result.atPose3(X(kf.id)).matrix() for kf in kfs}
    return graph, initial, result, before, after
