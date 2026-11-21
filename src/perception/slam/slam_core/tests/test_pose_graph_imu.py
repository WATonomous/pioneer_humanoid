"""Checks for the IMU path of PoseGraphBackend (add_keyframe with imu), on synthetic data.

A smooth, known camera motion is made up. From it come the VO measurements (with a slow
pitch drift, the kind of error the IMU should remove) and the IMU readings an ideal D455 IMU
would give, mounted at the Kalibr T_CAM_IMU (so the lever arm is exercised too).
"""

import unittest

import gtsam
import numpy as np
from gtsam.symbol_shorthand import V, X

from slam_core.backend import PoseGraphBackend
from slam_core.backend.imu_fusion.imu import imu_between
from slam_core.backend.imu_fusion.preintegration import GRAVITY, T_CAM_IMU, make_params, preintegrate
from slam_core.types import Keyframe, LoopClosure
from slam_core.types.pose import covariance_from_sigmas, rotation_angle

KF_DT = 0.2            # s between keyframes (run1 is mostly 0.2 s)
IMU_RATE = 200.0       # Hz, as the D455
N_KEYFRAMES = 250      # longer than the 139 where Cholesky failed on run1
TILT0 = np.radians(15.0)        # camera pitched at keyframe 0, as run1 (~16 deg)
PITCH_DRIFT = np.radians(0.1)   # VO error per keyframe step; 25 deg after 250 steps
G_WORLD = np.array([0.0, GRAVITY, 0.0])   # world frame: y down, as the backend's leveled frame


def rot_x(a):
    return gtsam.Rot3.Rx(a).matrix()


def rot_y(a):
    return gtsam.Rot3.Ry(a).matrix()


def rot_z(a):
    return gtsam.Rot3.Rz(a).matrix()


# True camera pose in the world (y down) at time t. At t = 0 it is a pure tilt at the origin,
# so the backend's leveled frame (which keeps keyframe 0's yaw and position) is this world frame.
def T_world_cam(t):
    T = np.eye(4)
    yaw = 0.6 * np.sin(0.25 * t)                       # turning left and right about vertical
    pitch = TILT0 + 0.05 * np.sin(0.7 * t)             # nodding
    roll = 0.05 * np.sin(0.9 * t)
    T[:3, :3] = rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)
    T[:3, 3] = [0.4 * np.sin(0.3 * t),                 # side to side
                0.03 * np.sin(1.3 * t),                # bobbing
                0.25 * t + 0.1 * np.sin(0.5 * t)]      # walking forward, ~0.3 m/s
    return T


def T_world_imu(t):
    return T_world_cam(t) @ T_CAM_IMU


# What an ideal IMU reads at time t: angular velocity and specific force, both in the IMU frame.
def imu_reading(t, h=1e-3):
    R = T_world_imu(t)[:3, :3]
    R_prev, R_next = T_world_imu(t - h)[:3, :3], T_world_imu(t + h)[:3, :3]
    gyro = gtsam.Rot3.Logmap(gtsam.Rot3(R_prev.T @ R_next)) / (2 * h)
    p = lambda s: T_world_imu(s)[:3, 3]
    accel_world = (p(t + h) - 2 * p(t) + p(t - h)) / h**2
    accel = R.T @ (accel_world - G_WORLD)   # at rest this is [0, -9.81, 0], as in imu.txt
    return gyro, accel


def make_imu(t_end):
    stamp = np.arange(-0.05, t_end + 0.05, 1.0 / IMU_RATE)
    readings = [imu_reading(t) for t in stamp]
    return stamp, np.array([g for g, _ in readings]), np.array([a for _, a in readings])


# Keyframes as VO would send them: relative poses from the truth plus a pitch drift each step.
def make_keyframes(n=N_KEYFRAMES):
    drift = np.eye(4)
    drift[:3, :3] = rot_x(PITCH_DRIFT)
    truth = [T_world_cam(k * KF_DT) for k in range(n)]
    keyframes, T_odom = [], np.eye(4)
    for k in range(n):
        T_prev_kf = np.eye(4) if k == 0 else np.linalg.inv(truth[k - 1]) @ truth[k] @ drift
        T_odom = T_odom @ T_prev_kf
        keyframes.append(Keyframe(id=k, prev_id=k - 1, stamp=k * KF_DT, T_prev_kf=T_prev_kf,
                                  covariance=covariance_from_sigmas(0.01, 0.02), T_odom_kf=T_odom))
    return keyframes, truth


# Angle (deg) between where each pose thinks "down" is and where it really is: roll/pitch error only.
def tilt_error_deg(T_est, T_true):
    down_est = T_est[:3, :3].T @ np.array([0.0, 1.0, 0.0])
    down_true = T_true[:3, :3].T @ np.array([0.0, 1.0, 0.0])
    return np.degrees(np.arccos(np.clip(down_est @ down_true, -1.0, 1.0)))


def loop(old_id, new_id, truth):
    return LoopClosure(old_id=old_id, new_id=new_id, stamp=new_id * KF_DT,
                       T_old_new=np.linalg.inv(truth[old_id]) @ truth[new_id],
                       covariance=covariance_from_sigmas(0.02, 0.03), n_inliers=100)


# Wraps the backend's iSAM2 and remembers the starting values handed over with the last update.
class RecordingIsam:
    def __init__(self, isam):
        self._isam = isam
        self.last_initial = None

    def update(self, *args):
        if len(args) >= 2:
            self.last_initial = args[1]
        return self._isam.update(*args)

    def __getattr__(self, name):
        return getattr(self._isam, name)


class PoseGraphImuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.keyframes, cls.truth = make_keyframes()
        cls.imu = make_imu(cls.keyframes[-1].stamp)

        cls.vio = PoseGraphBackend()
        for kf in cls.keyframes:
            cls.vio.add_keyframe(kf, cls.imu)
        cls.vio_poses = [cls.vio.optimized_pose(kf.id).matrix() for kf in cls.keyframes]

        vo = PoseGraphBackend()
        for kf in cls.keyframes:
            vo.add_keyframe(kf)
        # VO alone has no idea which way is down: give it the true start so only drift is compared.
        cls.vo_poses = [cls.truth[0] @ vo.optimized_pose(kf.id).matrix() for kf in cls.keyframes]

    def test_synthetic_imu_matches_backend_conventions(self):
        # Check the test's own IMU model first (gravity sign, frames, lever arm): preintegrating it
        # from the true state at one keyframe must land on the true pose of the next.
        params = make_params(T_CAM_IMU)
        bias = gtsam.imuBias.ConstantBias()
        h = 1e-4
        for k in (5, 100, 200):
            t0, t1 = k * KF_DT, (k + 1) * KF_DT
            pim = preintegrate(*imu_between(*self.imu, t0, t1), params, bias)
            v0 = (T_world_cam(t0 + h)[:3, 3] - T_world_cam(t0 - h)[:3, 3]) / (2 * h)
            pred = pim.predict(gtsam.NavState(gtsam.Pose3(self.truth[k]), v0), bias).pose().matrix()
            self.assertLess(np.linalg.norm(pred[:3, 3] - self.truth[k + 1][:3, 3]), 1e-3)
            self.assertLess(np.degrees(rotation_angle(self.truth[k + 1][:3, :3].T @ pred[:3, :3])), 0.01)

    def test_long_run_does_not_raise(self):
        # Regression for the IndeterminateSystemException iSAM2 raised with Cholesky (run1, B(139)).
        self.assertEqual(len(self.vio_poses), N_KEYFRAMES)

    def test_gravity_leveling_finds_initial_tilt(self):
        err = tilt_error_deg(self.vio_poses[0], self.truth[0])
        print(f"\nkeyframe 0 tilt error after leveling: {err:.3f} deg (true tilt {np.degrees(TILT0):.1f} deg)")
        self.assertLess(err, 0.5)

    def test_imu_removes_vo_tilt_drift(self):
        vo_err = tilt_error_deg(self.vo_poses[-1], self.truth[-1])
        vio_err = tilt_error_deg(self.vio_poses[-1], self.truth[-1])
        print(f"\nlast keyframe tilt error: VO only {vo_err:.2f} deg, VO + IMU {vio_err:.2f} deg")
        self.assertGreater(vo_err, 10.0)   # the drift is really there
        self.assertLess(vio_err, 1.0)      # and the IMU takes it out

    def test_imu_keeps_trajectory_close_to_truth(self):
        pos = [np.linalg.norm(e[:3, 3] - t[:3, 3]) for e, t in zip(self.vio_poses, self.truth)]
        rot = [np.degrees(rotation_angle(t[:3, :3].T @ e[:3, :3])) for e, t in zip(self.vio_poses, self.truth)]
        vo_pos = [np.linalg.norm(e[:3, 3] - t[:3, 3]) for e, t in zip(self.vo_poses, self.truth)]
        print(f"\nposition error max: VO only {max(vo_pos):.3f} m, VO + IMU {max(pos):.3f} m;"
              f" VO + IMU rotation error max {max(rot):.2f} deg")
        self.assertLess(max(pos), 0.25 * max(vo_pos))
        self.assertLess(max(rot), 2.0)   # yaw is unobservable from gravity, so it may drift a little

    def test_loop_to_held_back_keyframe_is_queued_then_added(self):
        backend = PoseGraphBackend()   # gravity_keyframes=4: keyframes 0-3 are held back
        for kf in self.keyframes[:3]:
            backend.add_keyframe(kf, self.imu)
        early = loop(0, 2, self.truth)
        self.assertEqual(backend.add_loop_constraint(early), {})   # queued, nothing optimized yet
        with self.assertRaises(ValueError):
            backend.add_loop_constraint(loop(0, 2, self.truth))   # same loop twice
        with self.assertRaises(ValueError):
            backend.add_loop_constraint(loop(0, 7, self.truth))   # keyframe 7 not added at all

        backend.add_keyframe(self.keyframes[3], self.imu)   # 4th keyframe: gravity, insert, then the loop
        self.assertIn((0, 2), backend._loop_pairs)
        self.assertEqual(backend._pending_loops, [])
        self.assertEqual(len(backend.optimized_poses()), 4)

    def test_new_keyframe_starting_guess_is_in_the_optimized_frame(self):
        # Regression: a new keyframe's starting pose and velocity were built from VO's own T_odom_kf, which drifts away
        # from the optimized frame once the IMU (or a loop) corrects the trajectory. By keyframe 100 the velocity guess was
        # ~1.8 m/s for a camera moving at 0.24 m/s. It must come from the previous optimized pose plus VO's relative step.
        k = 100
        for with_loop in (False, True):
            with self.subTest(with_loop=with_loop):
                backend = PoseGraphBackend()
                for kf in self.keyframes[:k + 1]:
                    backend.add_keyframe(kf, self.imu)
                if with_loop:
                    backend.add_loop_constraint(loop(0, k, self.truth))   # moves the whole map
                recorder = RecordingIsam(backend._isam)
                backend._isam = recorder

                kf = self.keyframes[k + 1]
                prev_position = backend.optimized_pose(k).translation()   # before the update moves it slightly
                backend.add_keyframe(kf, self.imu)
                step = np.linalg.norm(kf.T_prev_kf[:3, 3])               # VO's measured motion to this keyframe
                guess = recorder.last_initial
                self.assertAlmostEqual(np.linalg.norm(guess.atPose3(X(kf.id)).translation() - prev_position), step, places=6)
                self.assertAlmostEqual(np.linalg.norm(guess.atVector(V(kf.id))), step / KF_DT, places=6)


if __name__ == "__main__":
    unittest.main()
