import numpy as np
import pytest

FLOOR_Z = -0.9 
SCENE_HALF_SIZE = 1.0
POINT_SPACING = 0.02
STEP_START_X = 0.5
STEP_HEIGHT = 0.10
RAMP_ANGLE_DEG = 10.0


def _floor_grid(z=FLOOR_Z):
    """(N, 3) points on a flat, evenly spaced grid at height z."""
    coords = np.arange(-SCENE_HALF_SIZE, SCENE_HALF_SIZE + 1e-9, POINT_SPACING)
    xs, ys = np.meshgrid(coords, coords)
    zs = np.full_like(xs, z)
    return np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])


@pytest.fixture
def make_transform():
    """Return a function that builds a 4x4 transform from (x, y, z, yaw).

    yaw is in radians, counter-clockwise when viewed from above (rotation
    about +z). Roll and pitch are zero.
    """

    def _make(x=0.0, y=0.0, z=0.0, yaw=0.0):
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4)
        T[0, 0], T[0, 1] = c, -s
        T[1, 0], T[1, 1] = s, c
        T[:3, 3] = [x, y, z]
        return T

    return _make


@pytest.fixture
def flat_floor():
    """Flat ground 0.9 m below the base, 2 m x 2 m, points every 2 cm."""
    return _floor_grid()


@pytest.fixture
def step():
    """Flat floor with everything beyond x = 0.5 m raised by 10 cm."""
    points = _floor_grid()
    points[points[:, 0] > STEP_START_X, 2] += STEP_HEIGHT
    return points


@pytest.fixture
def ramp():
    """Floor that is flat behind the robot and rises at 10 degrees for x > 0."""
    points = _floor_grid()
    x = points[:, 0]
    points[:, 2] += np.clip(x, 0.0, None) * np.tan(np.radians(RAMP_ANGLE_DEG))
    return points