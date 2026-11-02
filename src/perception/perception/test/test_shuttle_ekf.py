"""Tests for the shuttle EKF and the depth reader, on simulated flights (no ROS, no GPU).

A simulated D455 (640x480, fx 376) watches a shuttle flying at it from about 9 m; the EKF gets the
shuttle's pixel (plus noise) as a TrackNet hotspot each frame.
"""
import numpy as np

from perception.depth_reader import depth_reader
from perception.shuttle_ekf import Camera, ShuttleEKF

CAM = Camera(376., 376., 320., 240., 640, 480)
FPS = 30.


def flight(frames=40):
    """3D states [X, Y, Z, VX, VY, VZ] of a shuttle hit towards the camera, using the EKF's own physics."""
    x = np.array([0.5, -1.5, 9., -0.3, -5., -18.])
    physics = ShuttleEKF(CAM)
    path = []
    for _ in range(frames):
        path.append(x)
        x, _ = physics._motion(x, 1 / FPS)
    return path


def run(path, extra_hotspot=None, depth=True, noise=2., seed=0):
    """Feed the EKF one frame at a time; return it and the frames where it used the shuttle's pixel."""
    rng = np.random.default_rng(seed)
    ekf = ShuttleEKF(CAM, fps=FPS)
    for k, x in enumerate(path):
        u, v = CAM.project(x) + rng.normal(0, noise, 2)
        hotspots = [(u, v, 0.8)] + ([extra_hotspot] if extra_hotspot else [])
        ekf.update(k / FPS, hotspots, (lambda a, b, z=x[2]: z) if depth else None)
    return ekf


def test_follows_flight_and_predicts_arrival():
    path = flight()
    ekf = run(path[:30])
    assert ekf.status() == 'tracking'
    assert np.linalg.norm(ekf.x[:3] - path[29][:3]) < 0.2              # where it is now
    future = ekf.x
    for _ in range(9):                                                   # where it will be 0.3 s later
        future, _ = ekf._motion(future, 1 / FPS)
    assert np.linalg.norm(future[:3] - path[38][:3]) < 0.3


def test_ignores_static_false_hotspot():
    path = flight()
    wall_mark = (100., 400., 0.9)                                        # stronger than the shuttle
    with_mark = run(path[:30], extra_hotspot=wall_mark)
    without = run(path[:30])
    assert np.allclose(with_mark.x, without.x)


def test_coasts_after_shuttle_leaves_view_then_stops():
    ekf = run(flight()[:30])
    for k in range(30, 30 + int(FPS * 1.5)):                             # no hotspots for 1.5 s
        ekf.update(k / FPS, [])
    assert ekf.status() == 'none'


def test_time_jump_drops_track():
    ekf = run(flight()[:30])
    ekf.update(30 / FPS + 5., [])                                        # camera stalled for 5 s
    assert ekf.status() == 'none'


def test_depth_reader_keeps_shuttle_and_rejects_wall():
    wall = np.full((480, 640), 12000, np.uint16)                         # wall at 12 m, in mm
    seen = wall.copy()
    seen[200:203, 300:303] = 5000                                        # shuttle at 5 m: 3x3 pixels
    earlier = [wall] * 10
    assert depth_reader(seen, earlier)(301.4, 202.6) == 5.               # off-centre pixel still finds it
    assert depth_reader(wall, earlier)(301., 201.) is None               # sensor missed it: only wall
