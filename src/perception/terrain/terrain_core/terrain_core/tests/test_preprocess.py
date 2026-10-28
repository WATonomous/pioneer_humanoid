"""Tests for terrain_core.preprocess, on tiny inputs whose answers can be worked out by hand.

Run from src/perception/terrain:  python -m pytest terrain_core
"""
import numpy as np
import pytest
from slam_core.types import Frame

from terrain_core.preprocess.crop import in_depth_range
from terrain_core.preprocess.depth import depth_to_points
from terrain_core.preprocess.downsample import voxel_downsample
from terrain_core.preprocess.pipeline import PreprocessConfig, preprocess
from terrain_core.preprocess.transform import transform_points


# ---------- depth_to_points ----------

def test_depth_to_points_by_hand():
    # fx = fy = 1, centre pixel at column 1, row 0: easy numbers to check by hand
    K = np.array([[1.0, 0, 1.0], [0, 1.0, 0.0], [0, 0, 1]])
    depth = np.array([[1.0, 1.0, 0.0],      # 0.0 = no reading, so no point
                      [2.0, 1.0, 1.0]], dtype=np.float32)
    points = depth_to_points(depth, K)
    assert points.shape == (5, 3)
    # row 1, column 0, depth 2: x = (0 - 1) * 2 / 1 = -2, y = (1 - 0) * 2 / 1 = 2
    assert [-2.0, 2.0, 2.0] in points.tolist()


def test_centre_pixel_is_straight_ahead():
    K = np.array([[500.0, 0, 2.0], [0, 500.0, 1.0], [0, 0, 1]])
    depth = np.zeros((3, 5), dtype=np.float32)
    depth[1, 2] = 1.5                        # only the centre pixel (row cy=1, column cx=2)
    np.testing.assert_allclose(depth_to_points(depth, K), [[0.0, 0.0, 1.5]])


# ---------- in_depth_range ----------

def test_in_depth_range_keeps_only_0_3_to_3_m():
    points = np.array([[0, 0, 0.1], [0, 0, 0.3], [0, 0, 1.0], [0, 0, 3.0], [0, 0, 4.0]])
    assert in_depth_range(points, 0.3, 3.0).tolist() == [False, True, True, True, False]


# ---------- voxel_downsample ----------

def test_voxel_downsample_keeps_one_point_per_cube():
    points = np.array([[0.010, 0.010, 1.000],
                       [0.050, 0.000, 1.000],
                       [0.015, 0.012, 1.005],    # same 2 cm cube as point 0
                       [0.055, 0.005, 1.010],    # same 2 cm cube as point 1
                       [0.300, 0.200, 1.500]])
    np.testing.assert_array_equal(voxel_downsample(points, 0.02), points[[0, 1, 4]])


def test_voxel_downsample_negative_side_is_a_different_cube():
    # -0.001 and +0.001 are 2 mm apart but on either side of 0, so in different cubes
    points = np.array([[-0.001, 0.0, 0.0], [0.001, 0.0, 0.0]])
    assert len(voxel_downsample(points, 0.02)) == 2


def test_voxel_downsample_refuses_points_too_far_away():
    with pytest.raises(ValueError):
        voxel_downsample(np.array([[1e5, 0.0, 0.0]]), 0.02)


# ---------- transform_points ----------

def test_transform_points_rotates_then_moves():
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]    # turn 90 degrees around z
    T[:3, 3] = [10.0, 0.0, 0.0]                      # then move 10 m along x
    # [1, 0, 0] turns to [0, 1, 0], then moves to [10, 1, 0]
    np.testing.assert_allclose(transform_points(T, np.array([[1.0, 0.0, 0.0]])), [[10.0, 1.0, 0.0]])


# ---------- preprocess (the whole chain) ----------

def test_preprocess_flat_floor_lands_at_height_zero():
    # A camera 1 m above a flat floor, looking straight down: every pixel reads 1 m.
    K = np.array([[100.0, 0, 20.0], [0, 100.0, 15.0], [0, 0, 1]])
    depth = np.full((30, 40), 1.0, dtype=np.float32)
    depth[0, 0] = 5.0                                # one pixel too far away: cropped
    frame = Frame(stamp=12.5, image=np.zeros((30, 40), np.uint8), depth=depth, K=K)
    # Camera axes in odom: x right = +x, y down = -y, z forward = -z (looking down), 1 m up.
    T_odom_cam = np.eye(4)
    T_odom_cam[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    T_odom_cam[:3, 3] = [0.0, 0.0, 1.0]

    cloud = preprocess(frame, T_odom_cam, PreprocessConfig(z_min=0.3, z_max=3.0, voxel_size=0.02))

    assert cloud.stamp == 12.5                        # the image's timestamp is kept
    np.testing.assert_allclose(cloud.points[:, 2], 0.0, atol=1e-6)   # the floor is at height 0
    assert len(cloud.points) < 30 * 40                # cropped and downsampled


def test_config_from_yaml(tmp_path):
    path = tmp_path / "terrain.yaml"
    path.write_text("preprocess:\n  z_max: 2.5\n")   # only z_max set; the rest keep defaults
    assert PreprocessConfig.from_yaml(path) == PreprocessConfig(z_min=0.3, z_max=2.5, voxel_size=0.02)
