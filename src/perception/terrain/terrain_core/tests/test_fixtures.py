import numpy as np

from conftest import FLOOR_Z, RAMP_ANGLE_DEG, STEP_HEIGHT, STEP_START_X


def apply(T, point):
    return T[:3, :3] @ point + T[:3, 3]


# make_transform -------------------------------------------------------------


def test_zero_transform_is_identity(make_transform):
    assert np.allclose(make_transform(), np.eye(4))


def test_yaw_90_turns_forward_into_left(make_transform):
    T = make_transform(yaw=np.pi / 2)
    assert np.allclose(apply(T, np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0])


def test_translation_moves_point(make_transform):
    T = make_transform(x=1.0, y=2.0, z=3.0)
    assert np.allclose(apply(T, np.zeros(3)), [1.0, 2.0, 3.0])


def test_rotation_part_is_a_valid_rotation(make_transform):
    R = make_transform(yaw=0.7)[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3))  # no stretching
    assert np.isclose(np.linalg.det(R), 1.0)  # no mirroring


# scenes ----------------------------------------------------------------------


def test_flat_floor_is_flat(flat_floor):
    assert flat_floor.ndim == 2 and flat_floor.shape[1] == 3
    assert np.allclose(flat_floor[:, 2], FLOOR_Z)


def test_step_heights(step):
    before = step[step[:, 0] <= STEP_START_X]
    after = step[step[:, 0] > STEP_START_X]
    assert np.allclose(before[:, 2], FLOOR_Z)
    assert np.allclose(after[:, 2], FLOOR_Z + STEP_HEIGHT)


def test_ramp_slope(ramp):
    rising = ramp[ramp[:, 0] > 0]
    slope, _ = np.polyfit(rising[:, 0], rising[:, 2], 1)
    assert np.isclose(np.degrees(np.arctan(slope)), RAMP_ANGLE_DEG)


def test_fixtures_are_independent(flat_floor, step):
    flat_floor[:, 2] = 99.0  # mutate one fixture...
    assert not np.any(step[:, 2] == 99.0)  # ...the other is unaffected