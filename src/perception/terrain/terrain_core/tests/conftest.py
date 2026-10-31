import numpy as np
import pytest

FLOOR_Z = -0.9
SCENE_HALF_SIZE = 1.0
POINT_SPACING = 0.02
GRID_OFFSET = 0.005  # keeps points off 5 cm cell edges
STEP_START_X = 0.5
STEP_HEIGHT = 0.10
RAMP_ANGLE_DEG = 10.0


def _floor_grid(z=FLOOR_Z, half_size=SCENE_HALF_SIZE, center=(0.0, 0.0)):
    coords = np.arange(-half_size + GRID_OFFSET, half_size, POINT_SPACING)
    xs, ys = np.meshgrid(coords + center[0], coords + center[1])
    zs = np.full_like(xs, z)
    return np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])


@pytest.fixture
def make_transform():
    def _make(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = [x, y, z]
        return T

    return _make


@pytest.fixture
def flat_floor():
    return _floor_grid()


@pytest.fixture
def step():
    points = _floor_grid()
    points[points[:, 0] > STEP_START_X, 2] += STEP_HEIGHT
    return points


@pytest.fixture
def ramp():
    points = _floor_grid()
    x = points[:, 0]
    points[:, 2] += np.clip(x, 0.0, None) * np.tan(np.radians(RAMP_ANGLE_DEG))
    return points


@pytest.fixture
def make_floor():
    def _make(z=0.0, half_size=2.0, center=(0.0, 0.0)):
        return _floor_grid(z=z, half_size=half_size, center=center)

    return _make


@pytest.fixture
def add_box():
    def _add(points, center, size, height):
        out = points.copy()
        inside = (np.abs(out[:, 0] - center[0]) < size[0] / 2) & (
            np.abs(out[:, 1] - center[1]) < size[1] / 2
        )
        out[inside, 2] += height
        return out

    return _add