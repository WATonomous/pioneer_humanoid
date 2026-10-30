"""Incremental GTSAM pose-graph backend.

This module accepts one Keyframe at a time and updates an iSAM2 factor graph.
The first keyframe receives a prior, and later keyframes receive VO
BetweenFactorPose3 constraints using their relative pose and covariance.
"""

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X

from slam_core.types import Keyframe, LoopClosure


class PoseGraphBackend:
    def __init__(
        self,
        prior_rot_sigma=1e-6,
        prior_trans_sigma=1e-6,
        loop_kernel="huber",
        loop_kernel_scale=1.345,
    ):
        self._isam = gtsam.ISAM2(gtsam.ISAM2Params())
        self._seen_ids = set()
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

    def add_keyframe(self, keyframe: Keyframe) -> gtsam.Pose3:
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

        if keyframe.is_first:
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(self._prior_sigmas)
            factors.add(gtsam.PriorFactorPose3(key, pose_guess, prior_noise))
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

        self._isam.update(factors, initial)
        self._estimate = self._isam.calculateEstimate()
        self._seen_ids.add(keyframe.id)
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
