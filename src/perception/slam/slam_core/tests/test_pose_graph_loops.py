"""BE-5 checks for loop correction and robust handling of false loop closures."""

import unittest

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X

from slam_core.backend import PoseGraphBackend
from slam_core.types import Keyframe, LoopClosure


def rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def make_sequence(step_count=36):
    truth = []
    for index in range(step_count + 1):
        angle = 2.0 * np.pi * index / step_count
        position = np.array([
            1.8 * (1.0 - np.cos(angle)),
            1.2 * np.sin(angle),
            0.65 * np.sin(2.0 * angle),
        ])
        rotation = rotation_matrix(
            0.08 * np.sin(angle),
            0.12 * np.sin(2.0 * angle),
            angle,
        )
        truth.append(transform(rotation, position))

    drift_per_step = transform(
        rotation_matrix(0.001, -0.0015, 0.004),
        np.array([0.012, -0.006, 0.008]),
    )
    vo_poses = [truth[0].copy()]
    relative_measurements = [np.eye(4)]
    for index in range(1, len(truth)):
        true_relative = np.linalg.inv(truth[index - 1]) @ truth[index]
        measured_relative = true_relative @ drift_per_step
        relative_measurements.append(measured_relative)
        vo_poses.append(vo_poses[-1] @ measured_relative)

    return truth, vo_poses, relative_measurements


def pose_covariance(rotation_sigma=0.025, translation_sigma=0.04):
    sigmas = np.array([rotation_sigma] * 3 + [translation_sigma] * 3)
    return np.diag(sigmas**2)


def score_trajectory(estimated_matrices, truth):
    position_errors = np.array([
        estimated[:3, 3] - actual[:3, 3]
        for estimated, actual in zip(estimated_matrices, truth)
    ])
    translation_ate = float(np.sqrt(np.mean(np.sum(position_errors**2, axis=1))))
    rotation_errors = []
    for estimated, actual in zip(estimated_matrices, truth):
        rotation_error = actual[:3, :3].T @ estimated[:3, :3]
        cosine = np.clip((np.trace(rotation_error) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(np.degrees(np.arccos(cosine)))
    rotation_ate = float(np.sqrt(np.mean(np.square(rotation_errors))))
    endpoint_translation = float(np.linalg.norm(
        estimated_matrices[-1][:3, 3] - estimated_matrices[0][:3, 3]
    ))
    relative_rotation = estimated_matrices[0][:3, :3].T @ estimated_matrices[-1][:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    endpoint_rotation = float(np.degrees(np.arccos(cosine)))
    return translation_ate, rotation_ate, endpoint_translation, endpoint_rotation


def make_loop_measurement(truth, false_loop=False, false_translation_m=None,
                          false_rotation=True, false_rotation_error_deg=None):
    measurement = np.linalg.inv(truth[0]) @ truth[-1]
    if false_loop:
        direction = np.array([8.0, -6.0, 9.0])
        magnitude = np.linalg.norm(direction) if false_translation_m is None else false_translation_m
        translation_error = direction / np.linalg.norm(direction) * magnitude
        if false_rotation_error_deg is not None:
            rotation_error = rotation_matrix(
                0.0, 0.0, np.radians(false_rotation_error_deg)
            )
        else:
            rotation_error = (
                rotation_matrix(0.7, -0.4, 0.9) if false_rotation else np.eye(3)
            )
        measurement = measurement @ transform(rotation_error, translation_error)
    return measurement


def run_backend(loop_kernel=None, false_loop=False, false_translation_m=None,
                false_rotation=True, false_rotation_error_deg=None):
    truth, vo_poses, relative_measurements = make_sequence()
    backend = PoseGraphBackend(loop_kernel=loop_kernel or "huber")
    for index, (vo_pose, relative_pose) in enumerate(zip(vo_poses, relative_measurements)):
        keyframe = Keyframe(
            id=index,
            prev_id=index - 1,
            stamp=index / 10.0,
            T_prev_kf=relative_pose,
            covariance=pose_covariance(),
            T_odom_kf=vo_pose,
        )
        backend.add_keyframe(keyframe)

    loop_measurement = make_loop_measurement(
        truth, false_loop, false_translation_m, false_rotation,
        false_rotation_error_deg,
    )

    loop = LoopClosure(
        old_id=0,
        new_id=len(truth) - 1,
        stamp=(len(truth) - 1) / 10.0,
        T_old_new=loop_measurement,
        covariance=pose_covariance(rotation_sigma=0.04, translation_sigma=0.06),
        n_inliers=100,
    )
    if not false_loop and loop_kernel is not None:
        backend.add_loop_constraint(loop)
    elif false_loop:
        backend.add_loop_constraint(loop)

    estimates = backend.optimized_poses()
    estimated_matrices = [estimates[index].matrix() for index in range(len(truth))]
    return score_trajectory(estimated_matrices, truth)


def run_gaussian_false_loop(false_translation_m=None, false_rotation=True,
                            false_rotation_error_deg=None):
    """Reference graph with identical VO inputs and an un-robust Gaussian loop."""
    truth, vo_poses, relative_measurements = make_sequence()
    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    vo_noise = gtsam.noiseModel.Gaussian.Covariance(pose_covariance())
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))

    for index, (vo_pose, relative_pose) in enumerate(zip(vo_poses, relative_measurements)):
        factors = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        key = X(index)
        pose_guess = gtsam.Pose3(vo_pose)
        initial.insert(key, pose_guess)
        if index == 0:
            factors.add(gtsam.PriorFactorPose3(key, pose_guess, prior_noise))
        else:
            factors.add(gtsam.BetweenFactorPose3(
                X(index - 1), key, gtsam.Pose3(relative_pose), vo_noise
            ))
        isam.update(factors, initial)

    false_loop = make_loop_measurement(
        truth, True, false_translation_m, false_rotation,
        false_rotation_error_deg,
    )
    factors = gtsam.NonlinearFactorGraph()
    factors.add(gtsam.BetweenFactorPose3(
        X(0), X(len(truth) - 1), gtsam.Pose3(false_loop),
        gtsam.noiseModel.Gaussian.Covariance(
            pose_covariance(rotation_sigma=0.04, translation_sigma=0.06)
        ),
    ))
    isam.update(factors, gtsam.Values())
    estimate = isam.calculateEstimate()
    matrices = [estimate.atPose3(X(index)).matrix() for index in range(len(truth))]
    return score_trajectory(matrices, truth)


class PoseGraphLoopTests(unittest.TestCase):
    def test_true_loop_reduces_error_for_huber_and_cauchy(self):
        baseline = run_backend()
        huber = run_backend(loop_kernel="huber", false_loop=False)
        cauchy = run_backend(loop_kernel="cauchy", false_loop=False)

        print(f"\nVO only: ATE={baseline[0]:.4f} m / {baseline[1]:.2f} deg, end={baseline[2]:.4f} m / {baseline[3]:.2f} deg")
        print(f"Correct loop, Huber: ATE={huber[0]:.4f} m / {huber[1]:.2f} deg, end={huber[2]:.4f} m / {huber[3]:.2f} deg")
        print(f"Correct loop, Cauchy: ATE={cauchy[0]:.4f} m / {cauchy[1]:.2f} deg, end={cauchy[2]:.4f} m / {cauchy[3]:.2f} deg")

        self.assertLess(huber[0], baseline[0] * 0.8)
        self.assertLess(cauchy[0], baseline[0] * 0.8)
        self.assertLess(huber[1], baseline[1] * 0.8)
        self.assertLess(cauchy[1], baseline[1] * 0.8)
        self.assertLess(huber[2], baseline[2])
        self.assertLess(cauchy[2], baseline[2])

    def test_huber_limits_false_loop(self):
        gaussian = run_gaussian_false_loop()
        false_loop = run_backend(loop_kernel="huber", false_loop=True)
        print(f"False loop, Gaussian: ATE={gaussian[0]:.4f} m / {gaussian[1]:.2f} deg, end={gaussian[2]:.4f} m / {gaussian[3]:.2f} deg")
        print(f"False loop, Huber: ATE={false_loop[0]:.4f} m / {false_loop[1]:.2f} deg, end={false_loop[2]:.4f} m / {false_loop[3]:.2f} deg")

        self.assertLess(false_loop[0], gaussian[0] * 0.5)
        self.assertLess(false_loop[1], gaussian[1] * 0.5)
        self.assertLess(false_loop[2], gaussian[2] * 0.5)

    def test_cauchy_limits_false_loop(self):
        gaussian = run_gaussian_false_loop()
        false_loop = run_backend(loop_kernel="cauchy", false_loop=True)
        print(f"False loop, Gaussian: ATE={gaussian[0]:.4f} m / {gaussian[1]:.2f} deg, end={gaussian[2]:.4f} m / {gaussian[3]:.2f} deg")
        print(f"False loop, Cauchy: ATE={false_loop[0]:.4f} m / {false_loop[1]:.2f} deg, end={false_loop[2]:.4f} m / {false_loop[3]:.2f} deg")

        self.assertLess(false_loop[0], gaussian[0] * 0.5)
        self.assertLess(false_loop[1], gaussian[1] * 0.5)
        self.assertLess(false_loop[2], gaussian[2] * 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
