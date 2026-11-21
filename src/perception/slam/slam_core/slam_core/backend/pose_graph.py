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
    ACCEL_RANDOM_WALK, GYRO_RANDOM_WALK, T_CAM_IMU)
from slam_core.backend.imu_fusion.gravity import gravity_in_frame, level_rotation

from slam_core.types import Keyframe, LoopClosure


# starting priors on the first keyframe, copied from imu_fusion/vio_graph.py
PRIOR_VEL_SIGMA = 0.5          # m/s
PRIOR_ACCEL_BIAS_SIGMA = 0.2   # m/s^2
PRIOR_GYRO_BIAS_SIGMA = 0.02   # rad/s
PRIOR_TILT_SIGMA = np.radians(3.0)   # rad, roll/pitch of keyframe 0 (the leveling can be a few degrees off)
PRIOR_PINNED_SIGMA = 1e-4            # rad and m, yaw and position of keyframe 0


# Keyframe 0 pose prior: tight for yaw and position, loose for tilt (as imu_fusion/vio_graph.py's kf0_prior_noise).
# Pinned at 1e-4, not 1e-6: with 1e-6 iSAM2 raised IndeterminateSystemException once the leveling
# axis was not aligned with a coordinate axis (tested on a synthetic 10 degree tilt).
def kf0_prior_noise(R_level):
    up = R_level.T @ np.array([0.0, 1.0, 0.0])
    rot_cov = PRIOR_TILT_SIGMA**2 * np.eye(3) + (PRIOR_PINNED_SIGMA**2 - PRIOR_TILT_SIGMA**2) * np.outer(up, up)
    cov = np.zeros((6, 6))
    cov[:3, :3] = rot_cov
    cov[3:, 3:] = PRIOR_PINNED_SIGMA**2 * np.eye(3)
    return gtsam.noiseModel.Gaussian.Covariance(cov)


class ImuNotReadyError(RuntimeError):
    """add_keyframe with add_imu: the IMU has not reached the keyframe's stamp yet. Keep the keyframe, retry later."""


class PoseGraphBackend:
    def __init__(
        self,
        prior_rot_sigma=1e-6,
        prior_trans_sigma=1e-6,
        loop_kernel="cauchy",
        loop_kernel_scale=1.345,
        R_odom_kf0=None,
        gravity_keyframes=4,
        T_cam_imu=T_CAM_IMU,   # 4x4, IMU pose in the left IR camera frame; default is the Kalibr calibration
        live_imu=False,        # True: IMU comes in through add_imu, and every keyframe waits for it (ImuNotReadyError)
    ):
        # add custom params
        params = gtsam.ISAM2Params()
        params.relinearizeSkip = 1
        # QR, not the default Cholesky: the bias random-walk sigmas (~1e-6 per keyframe) next to the
        # 0.02 bias prior made Cholesky raise IndeterminateSystemException (data/run1, at B(139)).
        params.setFactorization("QR")
        self._extra_updates = 20

        # imu preintergration
        self._params = make_params(np.asarray(T_cam_imu, dtype=float))
        self._bias = gtsam.imuBias.ConstantBias()

        # gravity leveling: rotation that makes +y point down in the graph's frame (None = VO frame as is).
        # Pass it in, or leave it None and give add_keyframe imu data: the first gravity_keyframes keyframes are then
        # held back, gravity is estimated from them (gravity.gravity_in_frame), and they are inserted leveled.
        self._R_level = None if R_odom_kf0 is None else np.asarray(R_odom_kf0, dtype=float)
        self._T_level = np.eye(4)
        if self._R_level is not None:
            self._T_level[:3, :3] = self._R_level
        if gravity_keyframes < 3:
            raise ValueError("gravity_keyframes must be at least 3")
        self._gravity_keyframes = gravity_keyframes
        self._pending = []   # (keyframe, imu) held back until gravity is known
        self._pending_loops = []   # loops touching a held-back keyframe, added once it is inserted

        # Live IMU buffer (add_imu): samples kept from the last inserted keyframe on, trimmed as keyframes go in
        self._imu_stamp = np.empty(0)
        self._imu_gyro = np.empty((0, 3))
        self._imu_accel = np.empty((0, 3))
        self._imu_streaming = bool(live_imu)   # every keyframe takes its IMU from the buffer
        self._last_id = None          # newest inserted keyframe: the IMU factor's start (not VO's prev_id)

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

    # Live IMU input: push samples as they arrive (one sample as scalars/3-vectors, or arrays of N). Stamps must keep
    # increasing, on the same clock as keyframe stamps. Needs live_imu=True.
    def add_imu(self, stamp, gyro, accel):
        if not self._imu_streaming:
            raise ValueError("add_imu needs PoseGraphBackend(live_imu=True)")
        stamp = np.atleast_1d(np.asarray(stamp, dtype=float))
        gyro = np.asarray(gyro, dtype=float).reshape(-1, 3)
        accel = np.asarray(accel, dtype=float).reshape(-1, 3)
        if not (len(stamp) == len(gyro) == len(accel)):
            raise ValueError("add_imu: stamp, gyro and accel need the same number of samples")
        if np.any(np.diff(stamp) <= 0) or (self._imu_stamp.size and stamp[0] <= self._imu_stamp[-1]):
            raise ValueError("add_imu: IMU stamps must be strictly increasing")
        self._imu_stamp = np.concatenate([self._imu_stamp, stamp])
        self._imu_gyro = np.vstack([self._imu_gyro, gyro])
        self._imu_accel = np.vstack([self._imu_accel, accel])

    # The buffered IMU as add_keyframe's (stamp, gyro, accel), after checking it covers this keyframe.
    def _buffered_imu(self, keyframe):
        if self._imu_stamp.size == 0 or self._imu_stamp[-1] < keyframe.stamp:
            raise ImuNotReadyError(
                f"keyframe {keyframe.id} at {keyframe.stamp:.6f}: IMU only up to "
                f"{self._imu_stamp[-1] if self._imu_stamp.size else 'nothing yet'}; add_imu more, then retry")
        if self._imu_stamp[0] > keyframe.stamp:
            raise ValueError(f"keyframe {keyframe.id}: IMU starts after it ({self._imu_stamp[0]:.6f}); "
                             "start the IMU stream before the first keyframe")
        return self._imu_stamp, self._imu_gyro, self._imu_accel

    # Drop samples nothing will need: keep the last one at or before the oldest keyframe an IMU factor still starts from.
    def _trim_imu(self):
        if not self._imu_streaming or self._pending or self._last_id is None:
            return
        keep_from = max(np.searchsorted(self._imu_stamp, self._stamps[self._last_id], side="right") - 1, 0)
        self._imu_stamp = self._imu_stamp[keep_from:]
        self._imu_gyro = self._imu_gyro[keep_from:]
        self._imu_accel = self._imu_accel[keep_from:]

    # imu: the whole recording as (stamp, gyro, accel) for replay, or None. With live_imu, leave it None: the buffer is
    # used, and ImuNotReadyError means the IMU hasn't reached this keyframe's stamp yet (keep the keyframe and retry).
    def add_keyframe(self, keyframe: Keyframe, imu=None):
        if self._imu_streaming:
            if imu is not None:
                raise ValueError("add_keyframe: give imu or use add_imu, not both")
            imu = self._buffered_imu(keyframe)
        if imu is not None and self._last_id is not None and keyframe.stamp <= self._stamps[self._last_id]:
            raise ValueError(f"keyframe {keyframe.id}: stamp must be after the previous keyframe's")
        if imu is not None and self._pending and keyframe.stamp <= self._pending[-1][0].stamp:
            raise ValueError(f"keyframe {keyframe.id}: stamp must be after the previous keyframe's")
        pose = self._add_keyframe(keyframe, imu)
        self._trim_imu()
        return pose

    def _add_keyframe(self, keyframe: Keyframe, imu):
        # No IMU data, or the leveling is already known: insert right away.
        if imu is None or self._R_level is not None:
            if imu is None and self._pending:
                raise ValueError("imu must be given for every keyframe once buffering has started")
            return self._insert_keyframe(keyframe, imu)
        return self._buffer_keyframe(keyframe, imu)

    # Hold the first keyframes back until gravity can be estimated, then insert them all leveled.
    def _buffer_keyframe(self, keyframe: Keyframe, imu):
        pending_ids = {kf.id for kf, _ in self._pending}
        if keyframe.id in pending_ids:
            raise ValueError(f"keyframe {keyframe.id} was already added")
        if not keyframe.is_first and keyframe.prev_id not in pending_ids:
            raise ValueError(
                f"keyframe {keyframe.id} refers to unseen previous keyframe "
                f"{keyframe.prev_id}"
            )
        self._pending.append((keyframe, imu))
        if len(self._pending) < self._gravity_keyframes:
            return gtsam.Pose3(keyframe.T_odom_kf)   # provisional: VO's own pose, not optimized yet

        # Enough keyframes: gravity in the first keyframe's frame, turned into the odom frame, then the leveling rotation.
        kfs = [kf for kf, _ in self._pending]
        imu = self._pending[-1][1]   # the newest IMU snapshot covers every held-back keyframe
        stamp, gyro, accel = imu
        T0_inv = np.linalg.inv(kfs[0].T_odom_kf)
        g_kf0 = gravity_in_frame(stamp, gyro, accel, np.array([kf.stamp for kf in kfs]),
                                 [T0_inv @ kf.T_odom_kf for kf in kfs], self._params)
        self._R_level = level_rotation(kfs[0].T_odom_kf[:3, :3] @ g_kf0)
        self._T_level[:3, :3] = self._R_level

        # Insert the held-back keyframes through the normal path, now leveled.
        for kf, _ in self._pending:
            self._insert_keyframe(kf, imu)
        self._pending = []

        # Loops that arrived while their keyframes were held back can go in now.
        loops, self._pending_loops = self._pending_loops, []
        for loop in loops:
            self.add_loop_constraint(loop)
        return self.optimized_pose(keyframe.id)

    def _insert_keyframe(self, keyframe: Keyframe, imu=None):
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
        pose_guess = gtsam.Pose3(self._T_level @ keyframe.T_odom_kf)
        initial.insert(key, pose_guess)

        #imu: starting guesses for this keyframe's velocity and bias
        if imu is not None:
            if keyframe.is_first:
                v_guess = np.zeros(3)
            else:
                last_pos = self._estimate.atPose3(X(self._last_id)).translation()
                dt_kf = keyframe.stamp - self._stamps[self._last_id]
                v_guess = (pose_guess.translation() - last_pos) / dt_kf
            initial.insert(V(keyframe.id), v_guess)
            initial.insert(B(keyframe.id), self._bias)

        if keyframe.is_first:
            if imu is not None and self._R_level is not None:
                prior_noise = kf0_prior_noise(self._R_level)   # tight yaw and position, loose tilt
            else:
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

        # imu preintergration, from the newest keyframe in time. Not VO's prev_id: after a tracking recovery VO can
        # measure from an older keyframe, but the IMU (and the velocity/bias chain) runs through time.
        if not keyframe.is_first and imu is not None:
            stamp, gyro, accel = imu
            last = self._last_id
            t0 = self._stamps[last]
            t1 = keyframe.stamp
            gyro_seg, accel_seg, dt = imu_between(stamp, gyro, accel, t0, t1)
            pim = preintegrate(gyro_seg, accel_seg, dt, self._params, self._bias)
            factors.add(gtsam.ImuFactor(X(last), V(last), key, V(keyframe.id), B(last), pim))
            sq = np.sqrt(t1 - t0)
            factors.add(gtsam.BetweenFactorConstantBias(
                B(last),
                B(keyframe.id),
                self._bias,
                self.bias_sigmas(ACCEL_RANDOM_WALK * sq, GYRO_RANDOM_WALK * sq),
            ))

        self._isam.update(factors, initial)
        self._estimate = self._isam.calculateEstimate()

        # add seen ids and past timestamps
        self._seen_ids.add(keyframe.id)
        self._stamps[keyframe.id] = keyframe.stamp
        self._last_id = keyframe.id
        return self.optimized_pose(keyframe.id)

    def add_loop_constraint(self, loop: LoopClosure) -> dict[int, gtsam.Pose3]:
        pair = (loop.old_id, loop.new_id)
        if pair in self._loop_pairs or any(
            (queued.old_id, queued.new_id) == pair for queued in self._pending_loops
        ):
            raise ValueError(f"loop X{loop.old_id} -> X{loop.new_id} was already added")

        # A keyframe held back for gravity leveling isn't in the graph yet: queue the loop until it is.
        known_ids = self._seen_ids | {kf.id for kf, _ in self._pending}
        if loop.old_id not in known_ids or loop.new_id not in known_ids:
            raise ValueError(
                f"loop X{loop.old_id} -> X{loop.new_id} refers to an unseen keyframe"
            )
        if loop.old_id not in self._seen_ids or loop.new_id not in self._seen_ids:
            self._pending_loops.append(loop)
            return self.optimized_poses()   # nothing optimized for the held-back keyframes yet

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
