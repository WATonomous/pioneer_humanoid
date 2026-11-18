"""Incremental GTSAM pose-graph backend.

This module accepts one Keyframe at a time and updates an iSAM2 factor graph.
The first keyframe receives a prior, and later keyframes receive VO
BetweenFactorPose3 constraints using their relative pose and covariance.
"""

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, V, X

from slam_core.backend.imu_fusion.imu import imu_between
from slam_core.backend.imu_fusion.preintegration import make_params, preintegrate
from slam_core.backend.imu_fusion.preintegration import (
    ACCEL_RANDOM_WALK, GYRO_RANDOM_WALK)

from slam_core.types import Keyframe, LoopClosure

# T_cam_imu: pose of the IMU in the left IR camera frame (p_ir1 = T_cam_imu @ p_imu), Kalibr convention.
# Derived, not measured by Kalibr: T_infra1_color (factory, from /tf_static) x T_color_imu (Kalibr result).
# Source: d455_imu_calibration/kalibr_output/T_infra1_imu.yaml
T_CAM_IMU = np.array([
    [0.9999892768399253, -0.004148029012896373, 0.0020591407152132164, 0.02816988878101279],
    [0.004143765359911823, 0.9999892709290251, 0.0020705640504841225, -0.008068887543765038],
    [-0.0020677073823052936, -0.0020620092515342657, 0.9999957363429309, -0.014059076567611293],
    [0.0, 0.0, 0.0, 1.0],
])


# starting priors on the first keyframe, copied from imu_fusion/vio_graph.py
PRIOR_VEL_SIGMA = 0.5          # m/s
PRIOR_ACCEL_BIAS_SIGMA = 0.2   # m/s^2
PRIOR_GYRO_BIAS_SIGMA = 0.02   # rad/s


class PoseGraphBackend:
    def __init__(
        self,
        prior_rot_sigma=1e-6,
        prior_trans_sigma=1e-6,
        loop_kernel="cauchy",
        loop_kernel_scale=1.345,
    ):
        # add custom params
        params = gtsam.ISAM2Params()
        params.relinearizeSkip = 1
        self._extra_updates = 20

        # imu preintergration
        self._params = make_params(T_CAM_IMU)
        self._bias = gtsam.imuBias.ConstantBias()

        self._isam = gtsam.ISAM2(params)
        self._seen_ids = set()
        self._stamps = {}
        self._loop_pairs = set()
        self._estimate = gtsam.Values()
        self._prior_sigmas = np.array(
            [prior_rot_sigma] * 3 + [prior_trans_sigma] * 3,
            dtype=float,
        )
        self._loop_kernel = loop_kernel.lower()
        self._loop_kernel_scale = float(loop_kernel_scale)
        if self._loop_kernel not in {"huber", "cauchy"}:
            raise ValueError("loop_kernel must be 'huber' or 'cauchy'")

    def bias_sigmas(self, accel, gyro):
        return gtsam.noiseModel.Diagonal.Sigmas(np.array([accel] * 3 + [gyro] * 3))

    def add_keyframe(self, keyframe: Keyframe, imu=None):
        if keyframe.id in self._seen_ids:
            raise ValueError(f"keyframe {keyframe.id} was already added")
        if not keyframe.is_first and keyframe.prev_id not in self._seen_ids:
            raise ValueError(
                f"keyframe {keyframe.id} refers to unseen previous keyframe "
                f"{keyframe.prev_id}"
            )

        factors = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        key = X(keyframe.id)
        pose_guess = gtsam.Pose3(keyframe.T_odom_kf)
        initial.insert(key, pose_guess)

        #imu: starting guesses for this keyframe's velocity and bias
        if imu is not None:
            if keyframe.is_first:
                v_guess = np.zeros(3)
            else:
                prev_pos = self._estimate.atPose3(X(keyframe.prev_id)).translation()
                dt_kf = keyframe.stamp - self._stamps[keyframe.prev_id]
                v_guess = (pose_guess.translation() - prev_pos) / dt_kf
            initial.insert(V(keyframe.id), v_guess)
            initial.insert(B(keyframe.id), self._bias)

        if keyframe.is_first:
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(self._prior_sigmas)
            factors.add(gtsam.PriorFactorPose3(key, pose_guess, prior_noise))

            if imu is not None:
                factors.add(gtsam.PriorFactorVector(V(keyframe.id), v_guess,
                            gtsam.noiseModel.Isotropic.Sigma(3, PRIOR_VEL_SIGMA)))
                factors.add(gtsam.PriorFactorConstantBias(B(keyframe.id), self._bias,
                            self.bias_sigmas(PRIOR_ACCEL_BIAS_SIGMA, PRIOR_GYRO_BIAS_SIGMA)))
        else:
            prev_key = X(keyframe.prev_id)
            relative_pose = gtsam.Pose3(keyframe.T_prev_kf)
            covariance = np.asarray(keyframe.covariance, dtype=float)
            vo_noise = gtsam.noiseModel.Gaussian.Covariance(covariance)
            factors.add(
                gtsam.BetweenFactorPose3(
                    prev_key,
                    key,
                    relative_pose,
                    vo_noise,
                )
            )

        # imu preintergration
        if not keyframe.is_first and imu is not None:
            stamp, gyro, accel = imu
            t0 = self._stamps[keyframe.prev_id]
            t1 = keyframe.stamp
            # imu stuff
            gyro_seg, accel_seg, dt = imu_between(stamp, gyro, accel, t0, t1)
            pim = preintegrate(gyro_seg, accel_seg, dt, self._params, self._bias)
            factors.add(gtsam.ImuFactor(X(keyframe.prev_id), V(keyframe.prev_id),
                            key, V(keyframe.id), B(keyframe.prev_id), pim))
            sq = np.sqrt(t1 - t0)
            factors.add(gtsam.BetweenFactorConstantBias(B(keyframe.prev_id), B(keyframe.id), self._bias,
                        self.bias_sigmas(ACCEL_RANDOM_WALK * sq, GYRO_RANDOM_WALK * sq)))

        self._isam.update(factors, initial)
        self._estimate = self._isam.calculateEstimate()

        # add seen ids and past timestamps
        self._seen_ids.add(keyframe.id)
        self._stamps[keyframe.id] = keyframe.stamp
        return self.optimized_pose(keyframe.id)

    def add_loop_constraint(self, loop: LoopClosure) -> dict[int, gtsam.Pose3]:
        pair = (loop.old_id, loop.new_id)
        if loop.old_id not in self._seen_ids or loop.new_id not in self._seen_ids:
            raise ValueError(
                f"loop X{loop.old_id} -> X{loop.new_id} refers to an unseen keyframe"
            )
        if pair in self._loop_pairs:
            raise ValueError(f"loop X{loop.old_id} -> X{loop.new_id} was already added")

        covariance = np.asarray(loop.covariance, dtype=float)
        base_noise = gtsam.noiseModel.Gaussian.Covariance(covariance)
        if self._loop_kernel == "huber":
            estimator = gtsam.noiseModel.mEstimator.Huber.Create(
                self._loop_kernel_scale
            )
        else:
            estimator = gtsam.noiseModel.mEstimator.Cauchy.Create(
                self._loop_kernel_scale
            )
        loop_noise = gtsam.noiseModel.Robust.Create(estimator, base_noise)

        factors = gtsam.NonlinearFactorGraph()
        factors.add(
            gtsam.BetweenFactorPose3(
                X(loop.old_id),
                X(loop.new_id),
                gtsam.Pose3(loop.T_old_new),
                loop_noise,
            )
        )
        self._isam.update(factors, gtsam.Values())
        for _ in range(self._extra_updates):
            self._isam.update(gtsam.NonlinearFactorGraph(), gtsam.Values())  # no new data,
        self._estimate = self._isam.calculateEstimate()
        self._loop_pairs.add(pair)
        return self.optimized_poses()

    def optimized_pose(self, keyframe_id: int) -> gtsam.Pose3:
        if keyframe_id not in self._seen_ids:
            raise KeyError(f"unknown keyframe {keyframe_id}")
        return self._estimate.atPose3(X(keyframe_id))

    def optimized_poses(self) -> dict[int, gtsam.Pose3]:
        return {
            keyframe_id: self.optimized_pose(keyframe_id)
            for keyframe_id in sorted(self._seen_ids)
        }
