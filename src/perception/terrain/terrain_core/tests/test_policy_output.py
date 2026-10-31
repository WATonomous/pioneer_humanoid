import numpy as np
import pytest

from terrain_core import config

policy_output = pytest.importorskip("terrain_core.policy_output")

BASE_HEIGHT = 0.9
TOL = 0.01


def scan(points_world, T_world_base):
    return np.asarray(policy_output.height_scan(points_world, T_world_base))


def rotate_scene(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


@pytest.fixture
def box_scene(make_floor, add_box):
    return add_box(make_floor(z=0.0), center=(0.5, 0.0), size=(0.2, 0.2), height=0.1)


def test_shape_matches_config(make_floor, make_transform):
    out = scan(make_floor(z=0.0), make_transform(z=BASE_HEIGHT))
    assert out.shape == config.grid_shape()


def test_heights_are_floating_point(make_floor, make_transform):
    out = scan(make_floor(z=0.0), make_transform(z=BASE_HEIGHT))
    assert np.issubdtype(out.dtype, np.floating)


def test_flat_floor_reads_minus_base_height(make_floor, make_transform):
    out = scan(make_floor(z=0.0), make_transform(z=BASE_HEIGHT))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, -BASE_HEIGHT, atol=TOL)


def test_heights_do_not_depend_on_world_height(make_floor, make_transform):
    out = scan(make_floor(z=0.6), make_transform(z=0.6 + BASE_HEIGHT))
    assert np.allclose(out, -BASE_HEIGHT, atol=TOL)


def test_heights_do_not_depend_on_where_the_robot_is(make_floor, make_transform):
    floor = make_floor(z=0.0, center=(5.0, -3.0))
    out = scan(floor, make_transform(x=5.0, y=-3.0, z=BASE_HEIGHT))
    assert np.allclose(out, -BASE_HEIGHT, atol=TOL)


def test_box_shows_up(box_scene, make_transform):
    out = scan(box_scene, make_transform(z=BASE_HEIGHT))
    assert np.nanmax(out) == pytest.approx(-BASE_HEIGHT + 0.1, abs=TOL)


def test_box_in_front_and_behind_give_different_maps(make_floor, add_box, make_transform):
    floor = make_floor(z=0.0)
    T = make_transform(z=BASE_HEIGHT)
    in_front = scan(add_box(floor, (0.5, 0.0), (0.2, 0.2), 0.1), T)
    behind = scan(add_box(floor, (-0.3, 0.0), (0.2, 0.2), 0.1), T)
    assert not np.allclose(in_front, behind, equal_nan=True)


@pytest.mark.parametrize("yaw_deg", [90, 45, -30, 180])
def test_map_turns_with_the_robot(box_scene, make_transform, yaw_deg):
    T_upright = make_transform(z=BASE_HEIGHT)
    turn = make_transform(yaw=np.radians(yaw_deg))
    before = scan(box_scene, T_upright)
    after = scan(rotate_scene(box_scene, turn), turn @ T_upright)
    assert np.allclose(before, after, atol=TOL, equal_nan=True)


def test_turning_only_the_robot_changes_the_map(box_scene, make_transform):
    before = scan(box_scene, make_transform(z=BASE_HEIGHT))
    after = scan(box_scene, make_transform(z=BASE_HEIGHT, yaw=np.pi / 2))
    assert not np.allclose(before, after, equal_nan=True)


@pytest.mark.parametrize("roll_deg, pitch_deg", [(0, 10), (0, -10), (8, 0), (5, 5)])
def test_leaning_does_not_change_the_map(box_scene, make_transform, roll_deg, pitch_deg):
    upright = scan(box_scene, make_transform(z=BASE_HEIGHT))
    lean = make_transform(z=BASE_HEIGHT, roll=np.radians(roll_deg), pitch=np.radians(pitch_deg))
    leaning = scan(box_scene, lean)
    assert np.allclose(upright, leaning, atol=TOL, equal_nan=True)


def test_same_input_gives_same_output(box_scene, make_transform):
    T = make_transform(z=BASE_HEIGHT)
    assert np.array_equal(scan(box_scene, T), scan(box_scene, T), equal_nan=True)