"""Camera frame-convention tests.

A mirrored or upside-down camera produces a map that looks plausible and is wrong.
These render a single landmark and assert it lands where the ROS optical convention
says it must (+Z boresight, +X right in the image, +Y down).

Needs a GL context: run with MUJOCO_GL=egl. Skipped if unavailable, because the
trajectory and IMU tests must stay runnable anywhere.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mujoco = pytest.importorskip("mujoco")

from mj_camera import MjCamera  # noqa: E402
from orbit_path import look_at_quat  # noqa: E402

# One small landmark in an otherwise empty world, plus a backdrop so there is
# something for depth to hit.
SCENE = """<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="backdrop" type="box" pos="8 0 0" size="0.05 12 12" rgba="0.5 0.5 0.5 1"/>
    <geom name="landmark" type="box" pos="3 0 0" size="0.25 0.25 0.25" rgba="1 0 0 1"/>
  </worldbody>
</mujoco>"""


@pytest.fixture(scope="module")
def cam(tmp_path_factory):
    scene = tmp_path_factory.mktemp("scene") / "s.xml"
    scene.write_text(SCENE)
    try:
        return MjCamera(scene, width=320, height=240)
    except Exception as exc:  # no GL context available
        pytest.skip(f"no GL context: {exc}")


def _landmark_centroid(rgb):
    """(u, v) of the red landmark in pixels, or None if it is not in frame."""
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    mask = (r > 110) & (r - g > 55) & (r - b > 55)
    if mask.sum() < 20:
        return None
    vs, us = np.nonzero(mask)
    return us.mean(), vs.mean()


def test_boresight_is_optical_plus_z(cam):
    """Looking straight at the landmark puts it in the middle of the image."""
    cam.set_pose(np.zeros(3), look_at_quat(np.zeros(3), np.array([3.0, 0.0, 0.0])))
    rgb, _ = cam.render()
    c = _landmark_centroid(rgb)
    assert c is not None, "landmark not visible along the boresight"
    assert abs(c[0] - cam.width / 2) < 12
    assert abs(c[1] - cam.height / 2) < 12


def test_landmark_to_the_cameras_left_appears_on_the_image_left(cam):
    """Optical +X is image right, so a landmark at world +Y (left of a camera facing
    world +X) must land at u < centre. Getting this backwards mirrors the whole map."""
    eye = np.array([0.0, -0.7, 0.0])
    cam.set_pose(eye, look_at_quat(eye, eye + np.array([3.0, 0.0, 0.0])))
    rgb, _ = cam.render()
    c = _landmark_centroid(rgb)
    assert c is not None
    assert c[0] < cam.width / 2 - 10, f"landmark at u={c[0]:.0f}, expected left of centre"


def test_landmark_above_the_camera_appears_in_the_image_top(cam):
    """Optical +Y is image DOWN, so something higher in the world is higher in the
    image, i.e. smaller v. An upside-down camera fails here."""
    eye = np.array([0.0, 0.0, -0.7])
    cam.set_pose(eye, look_at_quat(eye, eye + np.array([3.0, 0.0, 0.0])))
    rgb, _ = cam.render()
    c = _landmark_centroid(rgb)
    assert c is not None
    assert c[1] < cam.height / 2 - 10, f"landmark at v={c[1]:.0f}, expected above centre"


def test_depth_is_metric_distance_to_the_image_plane(cam):
    """A flat surface normal to the boresight must read one constant distance edge to
    edge. Radial range would bow it outward and quietly warp the map."""
    cam.set_pose(np.zeros(3), look_at_quat(np.zeros(3), np.array([1.0, 0.0, 0.0])))
    _, depth = cam.render()
    h, w = depth.shape
    centre = depth[h // 2, w // 2]
    assert np.isclose(centre, 2.75, atol=0.05), f"landmark face at {centre:.3f} m, expected 2.75"
    corner = depth[6, 6]  # backdrop, well off-axis
    assert np.isclose(corner, 7.95, atol=0.05), f"backdrop corner at {corner:.3f} m, expected 7.95"


def test_intrinsics_match_the_real_d455_recording(cam):
    k = cam.intrinsic_matrix
    # Focal length scales with resolution, so compare the ratio (equivalently, the
    # field of view) rather than the raw pixel value.
    assert np.isclose(k[0, 0] / cam.width, 383.682 / 640, rtol=1e-3)
    assert np.isclose(k[0, 2], cam.width / 2)
    assert np.isclose(k[1, 2], cam.height / 2)
