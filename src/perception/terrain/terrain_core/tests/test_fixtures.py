import numpy as np

from conftest import FLOOR_Z, RAMP_ANGLE_DEG, STEP_HEIGHT, STEP_START_X

FORWARD = np.array([1.0, 0.0, 0.0])
LEFT = np.array([0.0, 1.0, 0.0])


def apply(T, point):
    return T[:3, :3] @ point + T[:3, 3]


# make_transform -------------------------------------------------------------


def test_zero_transform_is_identity(make_transform):
    assert np.allclose(make_transform(), np.eye(4))


def test_yaw_90_turns_forward_into_left(make_transform):
    T = make_transform(yaw=np.pi / 2)
    assert np.allclose(apply(T, np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0])

def test_pitch_90_tips_forward_into_down(make_transform):
    T = make_transform(pitch=np.pi / 2)
    assert np.allclose(apply(T, FORWARD), [0.0, 0.0, -1.0])


def test_roll_90_tips_left_into_up(make_transform):
    T = make_transform(roll=np.pi / 2)
    assert np.allclose(apply(T, LEFT), [0.0, 0.0, 1.0])


def test_translation_moves_point(make_transform):
    T = make_transform(x=1.0, y=2.0, z=3.0)
    assert np.allclose(apply(T, np.zeros(3)), [1.0, 2.0, 3.0])


def test_rotation_part_is_a_valid_rotation(make_transform):
    R = make_transform(roll=0.3, pitch=-0.2, yaw=0.7)[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


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
    flat_floor[:, 2] = 99.0 
    assert not np.any(step[:, 2] == 99.0) 

def test_make_floor_is_centered_where_asked(make_floor):
    floor = make_floor(z=0.6, center=(5.0, -3.0))
    assert np.allclose(floor[:, 2], 0.6)
    assert np.isclose(floor[:, 0].mean(), 5.0, atol=0.02)
    assert np.isclose(floor[:, 1].mean(), -3.0, atol=0.02)


def test_add_box_raises_only_its_footprint(make_floor, add_box):
    floor = make_floor()
    scene = add_box(floor, center=(0.5, 0.0), size=(0.2, 0.2), height=0.1)
    raised = scene[:, 2] > 0.05
    assert raised.any()
    assert np.all(np.abs(scene[raised, 0] - 0.5) < 0.1)
    assert np.all(np.abs(scene[raised, 1]) < 0.1)
    assert np.allclose(floor[:, 2], 0.0)