import math
from types import SimpleNamespace

import numpy as np
import pytest

from synthetic import synthetic_config, true_grip_pose
from umi_gripper import geometry
from umi_gripper.calibration.calibrate_camera import angular_speed, estimate_lag, solve_hand_eye
from umi_gripper.calibration.calibrate_tcp import fit_pivot
from umi_gripper.collect import Session
from umi_gripper.quest_tracker import ButtonEvent
from umi_gripper.take import PoseSample, load_take
from umi_gripper.width import TriggerWidth


class FakeTracker:
    def __init__(self):
        self.samples, self.events, self.latest, self.text = [], [], None, ""

    def push(self, t, tracked=True):
        s = PoseSample(host_s=t, client_ms=t * 1000, epoch=0, tracked=tracked, position=np.array([0, 1, 0.0]),
                       quat_xyzw=np.array([0, 0, 0, 1.0]), trigger=0.4)
        self.samples.append(s)
        self.latest = s

    def press(self, name, t):
        self.events.append(ButtonEvent(name, t))

    def drain(self):
        out, self.samples = self.samples, []
        return out

    def poll_events(self):
        out, self.events = self.events, []
        return out

    def set_status(self, text):
        self.text = text


class FakeCamera:
    def __init__(self):
        self.recording, self.frames, self.error = False, [], None

    def set_recording(self, on):
        self.recording, self.frames = on, []

    def push(self, t):
        if self.recording:
            self.frames.append((t, np.zeros((480, 640, 3), np.uint8)))

    def drain(self):
        out, self.frames = self.frames, []
        return out


def test_session_start_save_discard(tmp_path):
    tracker, camera = FakeTracker(), FakeCamera()
    session = Session(synthetic_config(), "pick", tmp_path, tracker, TriggerWidth(), camera, log=lambda *_: None)
    tracker.push(0.0)
    session.tick(0.0)
    assert session.take is None and "Ready" in tracker.text
    tracker.press("primary", 0.01)
    session.tick(0.01)
    assert session.take is not None and camera.recording
    for i in range(1, 30):
        tracker.push(i / 72)
        camera.push(i / 72)
        session.tick(i / 72)
    assert tracker.text.startswith("REC")
    tracker.press("primary", 0.5)
    session.tick(0.5)
    assert session.take is None and not camera.recording and len(session.saved) == 1
    take = load_take(session.saved[0])
    assert take.meta["task"] == "pick" and take.meta["width_source"] == "trigger"
    assert len(take.poses["host_s"]) == 29 and len(take.camera_s) == 29 and len(take.width["host_s"]) == 29
    tracker.press("primary", 0.6)
    session.tick(0.6)
    discarded = session.take.path
    tracker.press("secondary", 0.7)
    session.tick(0.7)
    assert session.take is None and not discarded.exists() and len(session.saved) == 1


def test_session_surfaces_stream_errors(tmp_path):
    camera = FakeCamera()
    session = Session(synthetic_config(), "x", tmp_path, FakeTracker(), TriggerWidth(), camera, log=lambda *_: None)
    camera.error = RuntimeError("unplugged")
    with pytest.raises(RuntimeError, match="unplugged"):
        session.tick(0.0)


def test_pivot_fit_recovers_offset():
    rng = np.random.default_rng(3)
    offset, point = np.array([0.02, -0.06, -0.19]), np.array([0.3, 0.9, -0.4])
    rotations = np.array([geometry.rotation_exp(rng.normal(scale=0.5, size=3)) for _ in range(300)])
    positions = point - rotations @ offset + rng.normal(scale=0.0005, size=(300, 3))
    got, got_point, residual = fit_pivot(positions, rotations)
    np.testing.assert_allclose(got, offset, atol=1e-3)
    np.testing.assert_allclose(got_point, point, atol=1e-3)
    assert residual < 0.002


def test_pivot_fit_rejects_twist_only_and_sliding():
    offset, point = np.array([0.0, 0.0, -0.2]), np.zeros(3)
    twist = np.array([geometry.rotation_exp([0, 0, a]) for a in np.linspace(0, 2, 100)])
    with pytest.raises(ValueError, match="rotation"):
        fit_pivot(point - twist @ offset, twist)
    rng = np.random.default_rng(4)
    rotations = np.array([geometry.rotation_exp(rng.normal(scale=0.5, size=3)) for _ in range(100)])
    with pytest.raises(ValueError, match="residual"):
        fit_pivot(point - rotations @ offset + rng.normal(scale=0.01, size=(100, 3)), rotations)


def test_lag_estimate():
    t_pose = np.arange(0, 20, 1 / 72)
    t_cam = np.arange(0, 20, 1 / 30)

    def signal(t):
        return np.abs(np.sin(1.7 * t)) ** 3 + 0.5 * np.abs(np.sin(4.3 * t + 1))

    lag, score = estimate_lag(t_pose, signal(t_pose), t_cam + 0.042, signal(t_cam))
    assert lag == pytest.approx(0.042, abs=0.004) and score > 0.95


def test_angular_speed_is_frame_independent():
    times = np.linspace(0, 2, 100)
    rotations = np.array([geometry.rotation_exp([0.5 * t, 0.2 * t * t, 0]) for t in times])
    mount = geometry.rotation_exp([0.3, -1.0, 2.0])
    _, speed_a = angular_speed(times, rotations)
    _, speed_b = angular_speed(times, np.array([r @ mount for r in rotations]))
    np.testing.assert_allclose(speed_a, speed_b, atol=1e-9)


def test_hand_eye_recovers_camera_mount():
    pytest.importorskip("cv2")
    grip_cam = geometry.make_pose(geometry.rpy_deg_to_matrix([175, 5, -88]), [0.02, -0.04, -0.1])
    world_marker = geometry.make_pose(geometry.rpy_deg_to_matrix([90, 0, 0]), [0.0, 0.8, -0.6])
    world_grip = np.array([true_grip_pose(t) for t in np.linspace(0, 8, 60)])
    cam_marker = np.array([geometry.invert(g @ grip_cam) @ world_marker for g in world_grip])
    got, spread = solve_hand_eye(world_grip, cam_marker)
    np.testing.assert_allclose(got, grip_cam, atol=1e-4)
    assert spread < 1e-4


def test_marker_detection_pose():
    cv2 = pytest.importorskip("cv2")
    from umi_gripper.calibration.calibrate_camera import detect_marker_poses
    size, distance, f = 0.1, 0.5, 723.0
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 200)
    side = int(round(f * size / distance))   # pixels the marker spans at that distance
    image = np.full((480, 640), 255, np.uint8)
    small = cv2.resize(marker, (side, side), interpolation=cv2.INTER_NEAREST)
    top, left = 240 - side // 2, 320 - side // 2
    image[top:top + side, left:left + side] = small
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    take = SimpleNamespace(camera_s=np.zeros(1), load_frame=lambda i: rgb)
    k = np.array([[f, 0, 319.5], [0, f, 239.5], [0, 0, 1]])
    views = detect_marker_poses(take, k, "DICT_4X4_50", 0, size)
    assert len(views) == 1
    pose = views[0][1]
    np.testing.assert_allclose(pose[:3, 3], [0, 0, distance], atol=0.01)
    # marker faces the camera: its +Z points back at the lens, its +Y up the image
    np.testing.assert_allclose(pose[:3, :3], np.diag([1, -1, -1]), atol=0.05)


def test_reachability_small_motion_reaches():
    pytest.importorskip("mink")
    pytest.importorskip("mujoco")
    from umi_gripper.postprocess.reachability import check_reach
    relative = np.array([geometry.make_pose(geometry.rotation_exp([0, 0, 0.3 * i / 50]), [0.0, 0.05 * i / 50, -0.05 * i / 50])
                         for i in range(51)])
    reach = check_reach(relative, 25.0, "left")
    assert reach.reachable.all(), reach.summary()
    far = np.array([geometry.make_pose(None, [0, 0, -1.5 * i / 50]) for i in range(51)])
    assert not check_reach(far, 25.0, "left").reachable.all()
    assert math.isfinite(reach.summary()["max_orientation_error_deg"])
