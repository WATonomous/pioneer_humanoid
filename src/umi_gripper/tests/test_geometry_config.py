import json

import numpy as np
import pytest

from umi_gripper import config as umi_config
from umi_gripper import geometry


def random_rotation(rng):
    return geometry.rotation_exp(rng.normal(size=3))


def test_quaternion_round_trip_and_sign():
    rng = np.random.default_rng(1)
    for _ in range(50):
        r = random_rotation(rng)
        q = geometry.matrix_to_quat(r)
        assert q[3] >= 0
        np.testing.assert_allclose(geometry.quat_to_matrix(q), r, atol=1e-9)
        np.testing.assert_allclose(geometry.quat_to_matrix(-q), r, atol=1e-9)


def test_rotation_vector_exp_inverse_including_pi():
    for v in ([0.3, -0.2, 0.1], [np.pi - 1e-9, 0, 0], [0, 0, 0]):
        np.testing.assert_allclose(geometry.rotation_exp(geometry.rotation_vector(geometry.rotation_exp(v))),
                                   geometry.rotation_exp(v), atol=1e-6)


def test_rot6d_round_trip_and_unnormalised_input():
    rng = np.random.default_rng(2)
    r = random_rotation(rng)
    np.testing.assert_allclose(geometry.rot6d_to_matrix(geometry.rot6d(r)), r, atol=1e-12)
    noisy = geometry.rot6d(r) * 3.0 + rng.normal(scale=1e-3, size=6)
    out = geometry.rot6d_to_matrix(noisy)
    np.testing.assert_allclose(out @ out.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(out) == pytest.approx(1.0)


def test_slerp_endpoints_and_shortest_path():
    q0 = geometry.matrix_to_quat(geometry.rotation_exp([0, 0, 0.2]))
    q1 = geometry.matrix_to_quat(geometry.rotation_exp([0, 0, 1.0]))
    np.testing.assert_allclose(geometry.slerp(q0, q1, 0), q0, atol=1e-12)
    np.testing.assert_allclose(abs(geometry.slerp(q0, -q1, 1) @ q1), 1.0, atol=1e-12)
    mid = geometry.quat_to_matrix(geometry.slerp(q0, -q1, 0.5))
    np.testing.assert_allclose(mid, geometry.rotation_exp([0, 0, 0.6]), atol=1e-9)


def test_invert_and_rpy():
    pose = geometry.make_pose(geometry.rpy_deg_to_matrix([10, 20, 30]), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pose @ geometry.invert(pose), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(geometry.rpy_deg_to_matrix([0, 0, 90]) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_default_config_loads_both_headsets():
    for headset in umi_config.HEADSETS:
        cfg = umi_config.load_config(calibration=None, overrides={"headset": headset})
        assert umi_config.grip_to_tcp(cfg).shape == (4, 4)


def test_default_camera_mount_is_the_robots_wrist_camera():
    """tcp_to_camera's default must match arm_8dof_params (camera along -Z of the tcp, the fingers' direction)."""
    cfg = umi_config.load_config(calibration=None)
    tcp_cam = umi_config.tcp_to_camera(cfg)
    np.testing.assert_allclose(tcp_cam[:3, :3] @ [0, 0, 1], [0, 0, -1], atol=1e-6)   # optical axis -> fingers
    from umi_gripper._paths import PIONEER_HUMANOID, add
    add(PIONEER_HUMANOID)
    params = pytest.importorskip("pioneer_humanoid.arm_8dof_params")
    expected = np.asarray(params._WRIST_POS) - np.asarray(params.TCP_LOCAL)
    np.testing.assert_allclose(tcp_cam[:3, 3], expected, atol=1e-4)
    w, x, y, z = params._WRIST_ROT
    mj_cam = geometry.quat_to_matrix([x, y, z, w])
    np.testing.assert_allclose(tcp_cam[:3, :3], mj_cam @ np.diag([1, -1, -1]), atol=1e-5)


def test_calibration_file_overrides_and_invalid_values(tmp_path):
    calibration = tmp_path / "cal.json"
    calibration.write_text(json.dumps({"mounts": {"quest2": {"grip_to_tcp": {"xyz": [0.01, 0.02, -0.2]}}},
                                       "width": {"open_count": 100}}))
    cfg = umi_config.load_config(calibration=calibration)
    np.testing.assert_allclose(umi_config.grip_to_tcp(cfg)[:3, 3], [0.01, 0.02, -0.2])
    assert cfg["width"]["open_count"] == 100 and cfg["width"]["baud"] == 1_000_000
    with pytest.raises(ValueError):
        umi_config.load_config(calibration=None, overrides={"headset": "quest1"})
    with pytest.raises(ValueError):
        umi_config.load_config(calibration=None, overrides={"tcp_to_camera": {"xyz": [2.0, 0, 0]}})


def test_update_calibration_merges(tmp_path):
    path = tmp_path / "cal.json"
    umi_config.update_calibration({"width": {"open_count": 1}}, path)
    umi_config.update_calibration({"width": {"closed_count": 2}, "latency": {"camera_s": 0.03}}, path)
    assert json.loads(path.read_text()) == {"width": {"open_count": 1, "closed_count": 2},
                                            "latency": {"camera_s": 0.03}}
