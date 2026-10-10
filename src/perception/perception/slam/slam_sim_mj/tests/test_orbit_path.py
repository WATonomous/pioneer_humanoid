"""Conventions tests for the SLAM sandbox camera path.

Every assertion here corresponds to a failure that is invisible in a headless run
and only shows up as a wrong map hours later: a boresight pointing the wrong way,
an upside-down image, a mirrored (left-handed) frame, or a drifting orbit.

Run from the repo root:  pytest src/perception/perception/slam/slam_sim_mj/tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbit_path import (  # noqa: E402
    corridor_pose,
    rect_perimeter,
    _rect_point,
    WORLD_UP,
    look_at_quat,
    orbit_pose,
    quat_to_matrix,
)

CENTER = np.array([1.0, 2.0, 0.5])
RADIUS = 2.5
HEIGHT = 1.2
PERIOD = 40.0

# One lap, deliberately not hitting only the axis-aligned angles — a sign error can
# cancel itself at theta = 0 and reappear everywhere else.
TIMES = np.linspace(0.0, PERIOD, 23)


def camera_axes(quat):
    """World-frame (right, down, forward) for an optical-convention quaternion."""
    m = quat_to_matrix(quat)
    return m[:, 0], m[:, 1], m[:, 2]


@pytest.mark.parametrize("t", TIMES)
def test_boresight_points_at_target(t):
    """+Z must aim from the camera at the thing we said to look at."""
    eye, quat = orbit_pose(t, CENTER, RADIUS, HEIGHT, PERIOD)
    _, _, forward = camera_axes(quat)

    expected = CENTER - eye
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(forward, expected, atol=1e-9)


@pytest.mark.parametrize("t", TIMES)
def test_image_is_upright(t):
    """+Y is image-down, so it must point downward in the world, never up.

    This is the assertion that catches a 180-degree roll flip — the failure that
    renders a perfectly valid-looking map upside down.
    """
    _, quat = orbit_pose(t, CENTER, RADIUS, HEIGHT, PERIOD)
    _, down, _ = camera_axes(quat)
    assert np.dot(down, WORLD_UP) < 0.0


@pytest.mark.parametrize("t", TIMES)
def test_frame_is_right_handed(t):
    """right x down == forward, and det == +1.

    A left-handed frame mirrors the image. Feature matching still succeeds on a
    mirrored image, so this fails as a subtly wrong reconstruction rather than an
    obvious crash.
    """
    _, quat = orbit_pose(t, CENTER, RADIUS, HEIGHT, PERIOD)
    right, down, forward = camera_axes(quat)

    np.testing.assert_allclose(np.cross(right, down), forward, atol=1e-9)
    assert np.linalg.det(quat_to_matrix(quat)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("t", TIMES)
def test_axes_are_orthonormal(t):
    _, quat = orbit_pose(t, CENTER, RADIUS, HEIGHT, PERIOD)
    m = quat_to_matrix(quat)
    np.testing.assert_allclose(m.T @ m, np.eye(3), atol=1e-9)
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("t", TIMES)
def test_orbit_geometry(t):
    """Constant radius about the vertical axis through center, constant height."""
    eye, _ = orbit_pose(t, CENTER, RADIUS, HEIGHT, PERIOD)
    assert np.linalg.norm(eye[:2] - CENTER[:2]) == pytest.approx(RADIUS, abs=1e-9)
    assert eye[2] == pytest.approx(CENTER[2] + HEIGHT, abs=1e-9)


def test_orbit_closes_after_one_period():
    """Loop closure depends on returning to the same viewpoint, exactly."""
    start_pos, start_quat = orbit_pose(0.0, CENTER, RADIUS, HEIGHT, PERIOD)
    end_pos, end_quat = orbit_pose(PERIOD, CENTER, RADIUS, HEIGHT, PERIOD)

    np.testing.assert_allclose(start_pos, end_pos, atol=1e-9)
    # q and -q are the same rotation; compare via the matrices to avoid a false fail.
    np.testing.assert_allclose(
        quat_to_matrix(start_quat), quat_to_matrix(end_quat), atol=1e-9
    )


def test_camera_above_target_looks_down():
    """A camera above its target must have a downward boresight.

    Directly pins the sign that a naive 'pitch' argument gets backwards.
    """
    eye = np.array([0.0, 0.0, 3.0])
    target = np.array([1.0, 0.0, 0.0])
    _, _, forward = camera_axes(look_at_quat(eye, target))
    assert forward[2] < 0.0


def test_looking_straight_down_is_finite():
    """Boresight parallel to world up leaves roll undefined; must not emit NaNs."""
    quat = look_at_quat(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0]))
    assert np.all(np.isfinite(quat))
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-12)

    _, _, forward = camera_axes(quat)
    np.testing.assert_allclose(forward, np.array([0.0, 0.0, -1.0]), atol=1e-9)


def test_degenerate_eye_equals_target_raises():
    with pytest.raises(ValueError):
        look_at_quat(np.zeros(3), np.zeros(3))


def test_phase_offsets_the_orbit():
    """Two cameras a half-period apart sit on opposite sides of the circle."""
    a, _ = orbit_pose(0.0, CENTER, RADIUS, HEIGHT, PERIOD)
    b, _ = orbit_pose(0.0, CENTER, RADIUS, HEIGHT, PERIOD, phase=np.pi)
    np.testing.assert_allclose(a[:2] - CENTER[:2], -(b[:2] - CENTER[:2]), atol=1e-9)


# ── Forward-facing mode (look_ahead > 0) ──────────────────────────────────────
# A person walking looks where they are going. These pin the conventions down:
# "forward" must mean the direction of travel, not its reverse, and the horizon
# must stay level and the image upright -- all invisible headless.

class TestForwardFacing:
    CENTER = np.zeros(3)

    def _pose(self, t, **kw):
        return orbit_pose(t, self.CENTER, radius=2.0, height=1.4,
                          period=60.0, look_ahead=0.6, **kw)

    def test_boresight_is_level(self):
        """Gaze is horizontal: the camera looks along the floor, not down at it."""
        for t in (0.0, 7.5, 15.0, 42.0):
            _, q = self._pose(t)
            forward = quat_to_matrix(q)[:, 2]
            assert abs(forward[2]) < 1e-9, f"t={t}: boresight has vertical component"

    def test_looks_along_direction_of_travel(self):
        """Forward, not backward -- the sign error that silently reverses the path."""
        for t in (0.0, 12.0, 30.0, 55.0):
            eye, q = self._pose(t)
            nxt, _ = self._pose(t + 0.5)
            velocity = nxt - eye
            forward = quat_to_matrix(q)[:, 2]
            assert np.dot(forward, velocity) > 0, f"t={t}: camera faces backwards"

    def test_boresight_is_tangential_not_radial(self):
        """Distinguishes forward mode from the inward default."""
        eye, q = self._pose(0.0)
        forward = quat_to_matrix(q)[:, 2]
        radial = (self.CENTER - eye)[:2]
        radial = radial / np.linalg.norm(radial)
        assert abs(np.dot(forward[:2] / np.linalg.norm(forward[:2]), radial)) < 0.6

    def test_image_stays_upright(self):
        for t in (0.0, 10.0, 25.0, 47.0):
            _, q = self._pose(t)
            down = quat_to_matrix(q)[:, 1]
            assert np.dot(down, WORLD_UP) < 0, f"t={t}: image is upside down"

    def test_rotation_is_right_handed(self):
        for t in (0.0, 20.0, 40.0):
            _, q = self._pose(t)
            assert np.isclose(np.linalg.det(quat_to_matrix(q)), 1.0)

    def test_default_is_still_inward(self):
        """look_ahead=0 must not change the existing tested behaviour."""
        eye_a, q_a = orbit_pose(9.0, self.CENTER, radius=2.0, height=1.4, period=60.0)
        eye_b, q_b = orbit_pose(9.0, self.CENTER, radius=2.0, height=1.4,
                                period=60.0, look_ahead=0.0)
        assert np.allclose(eye_a, eye_b) and np.allclose(q_a, q_b)

    def test_forward_differs_from_inward(self):
        _, q_in = orbit_pose(9.0, self.CENTER, radius=2.0, height=1.4, period=60.0)
        _, q_fw = self._pose(9.0)
        assert not np.allclose(q_in, q_fw)


# ── Corridor circuit ──────────────────────────────────────────────────────────
# The realistic trajectory: walk a building loop once, looking ahead and glancing
# around. Geometry is provable here; on the GPU it costs 90s a lap to find out.

class TestRectPath:
    def test_constant_speed(self):
        """Arc-length parameterised: a speed-up at corners would jerk odometry."""
        pts = np.array([_rect_point(s, 6, 4, 1.2) for s in np.linspace(0, 1, 600)])
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert steps.max() / steps.min() < 1.02

    def test_extents_and_closure(self):
        pts = np.array([_rect_point(s, 6, 4, 1.2) for s in np.linspace(0, 1, 600)])
        assert np.isclose(pts[:, 0].max(), 6.0, atol=1e-6)
        assert np.isclose(pts[:, 1].max(), 4.0, atol=1e-6)
        assert np.allclose(_rect_point(0.0, 6, 4, 1.2), _rect_point(1.0, 6, 4, 1.2))

    def test_perimeter_matches_samples(self):
        pts = np.array([_rect_point(s, 6, 4, 1.2) for s in np.linspace(0, 1, 4000)])
        sampled = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        assert np.isclose(sampled, rect_perimeter(6, 4, 1.2), rtol=1e-3)

    def test_corner_radius_clamped(self):
        """A corner larger than the box must not produce NaNs or fly off."""
        pt = _rect_point(0.3, 2.0, 2.0, 99.0)
        assert np.all(np.isfinite(pt)) and np.linalg.norm(pt) < 4.0


class TestCorridorPose:
    C = np.zeros(3)

    def test_walks_at_requested_height(self):
        for t in (0.0, 22.0, 61.0):
            eye, _ = corridor_pose(t, self.C, height=1.5, period=90.0)
            assert np.isclose(eye[2], 1.5)

    def test_faces_direction_of_travel(self):
        for t in (0.0, 15.0, 33.0, 70.0):
            eye, q = corridor_pose(t, self.C, period=90.0)
            nxt, _ = corridor_pose(t + 0.4, self.C, period=90.0)
            fwd = quat_to_matrix(q)[:, 2]
            assert np.dot(fwd, nxt - eye) > 0, f"t={t}: walking backwards"

    def test_image_upright_all_the_way_round(self):
        for t in np.linspace(0, 90, 60):
            _, q = corridor_pose(float(t), self.C, period=90.0)
            assert np.dot(quat_to_matrix(q)[:, 1], WORLD_UP) < 0

    def test_right_handed_all_the_way_round(self):
        for t in np.linspace(0, 90, 40):
            _, q = corridor_pose(float(t), self.C, period=90.0)
            assert np.isclose(np.linalg.det(quat_to_matrix(q)), 1.0)

    def test_scanning_moves_the_gaze_but_keeps_it_upright(self):
        """Head sweep must change the view without ever flipping the horizon."""
        moved = False
        for t in np.linspace(0, 90, 80):
            _, plain = corridor_pose(float(t), self.C, period=90.0)
            _, scan = corridor_pose(float(t), self.C, period=90.0,
                                    scan_yaw=0.45, scan_pitch=0.20)
            if not np.allclose(plain, scan, atol=1e-3):
                moved = True
            assert np.dot(quat_to_matrix(scan)[:, 1], WORLD_UP) < 0
            assert np.isclose(np.linalg.det(quat_to_matrix(scan)), 1.0)
        assert moved, "scan parameters had no effect"

    def test_scanning_stays_within_requested_amplitude(self):
        """A runaway sweep would spin the camera and destroy odometry."""
        for t in np.linspace(0, 90, 120):
            _, plain = corridor_pose(float(t), self.C, period=90.0)
            _, scan = corridor_pose(float(t), self.C, period=90.0,
                                    scan_yaw=0.45, scan_pitch=0.20)
            angle = np.arccos(np.clip(
                np.dot(quat_to_matrix(plain)[:, 2], quat_to_matrix(scan)[:, 2]), -1, 1))
            assert angle < 0.45 + 0.20 + 0.05

    def test_returns_to_start_after_one_lap(self):
        a, _ = corridor_pose(0.0, self.C, period=90.0)
        b, _ = corridor_pose(90.0, self.C, period=90.0)
        assert np.allclose(a, b, atol=1e-9)
