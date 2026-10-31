from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pipeline = pytest.importorskip("terrain_core.preprocess.pipeline")
crop = pytest.importorskip("terrain_core.preprocess.crop")
depth = pytest.importorskip("terrain_core.preprocess.depth")
downsample = pytest.importorskip("terrain_core.preprocess.downsample")
transform = pytest.importorskip("terrain_core.preprocess.transform")

TERRAIN_YAML = Path(__file__).resolve().parents[2] / "config" / "terrain.yaml"

K = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])


def camera_looking_down(height):
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    T[2, 3] = height
    return T


def voxel_ids(points, size):
    return np.floor(points / size).astype(int)


# crop -------------------------------------------------------------------------


def test_crop_keeps_only_points_in_depth_range():
    points = np.array([[0, 0, 0.1], [0, 0, 1.0], [0, 0, 5.0]])
    assert crop.in_depth_range(points).tolist() == [False, True, False]


def test_crop_limits_are_inclusive():
    points = np.array([[0, 0, 0.3], [0, 0, 3.0]])
    assert crop.in_depth_range(points).all()


def test_crop_uses_depth_not_sideways_distance():
    points = np.array([[2.0, -1.0, 1.0]])
    assert crop.in_depth_range(points).all()


def test_crop_custom_limits():
    points = np.array([[0, 0, 0.5], [0, 0, 1.5]])
    assert crop.in_depth_range(points, z_min=1.0, z_max=2.0).tolist() == [False, True]


def test_crop_returns_one_bool_per_point(flat_floor):
    mask = crop.in_depth_range(flat_floor)
    assert mask.shape == (len(flat_floor),)
    assert mask.dtype == bool


# depth image -> points ----------------------------------------------------------


def test_pixel_at_image_centre_is_straight_ahead():
    image = np.zeros((3, 5), dtype=np.float32)
    image[1, 2] = 2.0
    assert np.allclose(depth.depth_to_points(image, K), [[0.0, 0.0, 2.0]])


def test_pixel_right_of_centre_has_positive_x():
    image = np.zeros((3, 5), dtype=np.float32)
    image[1, 3] = 2.0
    assert np.allclose(depth.depth_to_points(image, K), [[0.02, 0.0, 2.0]])


def test_pixel_below_centre_has_positive_y():
    image = np.zeros((3, 5), dtype=np.float32)
    image[2, 2] = 2.0
    assert np.allclose(depth.depth_to_points(image, K), [[0.0, 0.02, 2.0]])


def test_pixels_without_depth_are_skipped():
    image = np.zeros((3, 5), dtype=np.float32)
    image[0, 0] = image[2, 4] = 1.0
    assert depth.depth_to_points(image, K).shape == (2, 3)


def test_empty_depth_image_gives_no_points():
    points = depth.depth_to_points(np.zeros((3, 5), dtype=np.float32), K)
    assert points.shape == (0, 3)


def test_points_are_float32():
    image = np.ones((3, 5), dtype=np.float32)
    assert depth.depth_to_points(image, K).dtype == np.float32


# voxel downsample -------------------------------------------------------------


def test_points_in_same_voxel_become_one():
    points = np.array([[0.001, 0.001, 0.001], [0.015, 0.015, 0.015]])
    assert len(downsample.voxel_downsample(points, 0.02)) == 1


def test_points_in_different_voxels_are_all_kept():
    points = np.array([[0.01, 0.0, 0.0], [0.03, 0.0, 0.0], [0.05, 0.0, 0.0]])
    assert len(downsample.voxel_downsample(points, 0.02)) == 3


def test_at_most_one_point_per_voxel(flat_floor):
    out = downsample.voxel_downsample(flat_floor, 0.05)
    assert len(np.unique(voxel_ids(out, 0.05), axis=0)) == len(out)
    assert len(np.unique(voxel_ids(flat_floor, 0.05), axis=0)) == len(out)


def test_downsample_keeps_real_points_not_averages(step):
    out = downsample.voxel_downsample(step, 0.05)
    as_set = {tuple(p) for p in step}
    assert all(tuple(p) in as_set for p in out)


def test_downsample_empty_cloud():
    assert downsample.voxel_downsample(np.zeros((0, 3)), 0.02).shape == (0, 3)


def test_downsample_rejects_points_too_far_away():
    with pytest.raises(ValueError):
        downsample.voxel_downsample(np.array([[1e5, 0.0, 0.0]]), 0.02)


# transform --------------------------------------------------------------------


def test_identity_transform_changes_nothing(flat_floor):
    assert np.allclose(transform.transform_points(np.eye(4), flat_floor), flat_floor)


def test_transform_rotates_and_moves(make_transform):
    T = make_transform(x=1.0, z=0.5, yaw=np.pi / 2)
    out = transform.transform_points(T, np.array([[1.0, 0.0, 0.0]]))
    assert np.allclose(out, [[1.0, 1.0, 0.5]])


def test_transform_then_inverse_returns_original(step, make_transform):
    T = make_transform(x=0.3, y=-2.0, z=1.1, roll=0.1, pitch=-0.2, yaw=2.0)
    back = transform.transform_points(np.linalg.inv(T), transform.transform_points(T, step))
    assert np.allclose(back, step)


def test_transform_matches_4x4_matrix_multiply(ramp, make_transform):
    T = make_transform(x=0.5, y=0.2, z=-0.1, roll=0.3, pitch=0.1, yaw=-1.0)
    homogeneous = np.column_stack([ramp, np.ones(len(ramp))])
    expected = (T @ homogeneous.T).T[:, :3]
    assert np.allclose(transform.transform_points(T, ramp), expected)


# whole pipeline ---------------------------------------------------------------


def make_frame(depth_value, stamp=12.5, shape=(48, 64)):
    image = np.full(shape, depth_value, dtype=np.float32)
    K_frame = np.array([[50.0, 0.0, 31.5], [0.0, 50.0, 23.5], [0.0, 0.0, 1.0]])
    return SimpleNamespace(depth=image, K=K_frame, stamp=stamp)


def test_camera_looking_down_sees_floor_at_zero():
    cloud = pipeline.preprocess(make_frame(1.0), camera_looking_down(height=1.0))
    assert len(cloud.points) > 0
    assert np.allclose(cloud.points[:, 2], 0.0, atol=1e-5)


def test_cloud_keeps_the_frame_timestamp():
    cloud = pipeline.preprocess(make_frame(1.0, stamp=42.0), camera_looking_down(1.0))
    assert cloud.stamp == 42.0


def test_cloud_points_are_float32_n_by_3():
    cloud = pipeline.preprocess(make_frame(1.0), camera_looking_down(1.0))
    assert cloud.points.ndim == 2 and cloud.points.shape[1] == 3
    assert cloud.points.dtype == np.float32


def test_pipeline_crops_far_points():
    cloud = pipeline.preprocess(make_frame(5.0), camera_looking_down(5.0))
    assert len(cloud.points) == 0


def test_pipeline_downsamples():
    frame = make_frame(1.0)
    fine = pipeline.preprocess(frame, np.eye(4), pipeline.PreprocessConfig(voxel_size=0.001))
    coarse = pipeline.preprocess(frame, np.eye(4), pipeline.PreprocessConfig(voxel_size=0.1))
    assert len(coarse.points) < len(fine.points)


# config -----------------------------------------------------------------------


def test_config_from_yaml_reads_values(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("preprocess:\n  z_max: 2.0\n")
    cfg = pipeline.PreprocessConfig.from_yaml(path)
    assert cfg.z_max == 2.0
    assert cfg.z_min == pipeline.PreprocessConfig().z_min


def test_team_terrain_yaml_is_sensible():
    if not TERRAIN_YAML.exists():
        pytest.skip("config/terrain.yaml not in this branch yet")
    cfg = pipeline.PreprocessConfig.from_yaml(TERRAIN_YAML)
    assert 0 < cfg.z_min < cfg.z_max
    assert cfg.voxel_size > 0