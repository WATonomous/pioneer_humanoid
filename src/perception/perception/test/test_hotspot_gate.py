"""Tests for the hotspot gate and the EKF under it, on simulated flights (no ROS, no GPU, no depth).

A simulated D455 (640x480, fx 376) watches a shuttle flying at it from about 9 m; the gate gets
the shuttle's pixel (plus noise) as a TrackNet hotspot each frame, sometimes with false ones.
"""
import numpy as np

from perception.hotspot_gate import HotspotGate
from perception.shuttle_ekf import Camera, ShuttleEKF

CAM = Camera(376., 376., 320., 240., 640, 480)
FPS = 30.
WALL_MARK = (100., 400., 0.9)        # stronger than the shuttle, never moves


def flight(frames=40):
    """3D states [X, Y, Z, VX, VY, VZ] of a shuttle hit towards the camera, using the EKF's own physics."""
    x = np.array([0.5, -1.5, 9., -0.3, -5., -18.])
    physics = ShuttleEKF(CAM)
    path = []
    for _ in range(frames):
        path.append(x)
        x, _ = physics._motion(x, 1 / FPS)
    return path


def run(path, extra=(), noise=2., seed=0):
    """Feed the gate one frame at a time. Returns the gate and, per frame, (output, true pixel)."""
    rng = np.random.default_rng(seed)
    gate = HotspotGate(CAM, fps=FPS)
    outputs = []
    for k, x in enumerate(path):
        truth = CAM.project(x)
        u, v = truth + rng.normal(0, noise, 2)
        hotspots = sorted([(u, v, 0.8), *extra], key=lambda h: -h[2])
        outputs.append((gate.update(k / FPS, hotspots), truth))
    return gate, outputs


def test_follows_flight_and_outputs_the_shuttle_pixel():
    _, outputs = run(flight()[:30])
    assert outputs[0][0] is None                              # a track needs a few frames to start
    tracked = [(out, truth) for out, truth in outputs[6:]]
    assert all(out is not None for out, _ in tracked)
    assert all(np.hypot(out[0] - truth[0], out[1] - truth[1]) < 10 for out, truth in tracked)


def test_ignores_stronger_static_false_hotspot():
    _, outputs = run(flight()[:30], extra=[WALL_MARK])
    for out, truth in outputs[6:]:
        assert out is not None
        assert np.hypot(out[0] - truth[0], out[1] - truth[1]) < 10     # never the wall mark


def test_rejects_one_frame_false_blip_far_from_the_track():
    path = flight()
    gate, outputs = run(path[:20])
    blip = gate.update(20 / FPS, [(50., 60., 0.95)])           # shuttle missing, a strong blip instead
    assert blip is None
    assert gate.status() == 'coasting'
    truth = CAM.project(path[21])
    after = gate.update(21 / FPS, [(truth[0], truth[1], 0.8)])
    assert after is not None and np.hypot(after[0] - truth[0], after[1] - truth[1]) < 10


def test_coasts_without_output_then_stops():
    gate, _ = run(flight()[:30])
    out = gate.update(30 / FPS, [])
    assert out is None and gate.status() == 'coasting'
    for k in range(31, 31 + int(FPS * 1.5)):                   # no hotspots for 1.5 s
        gate.update(k / FPS, [])
    assert gate.status() == 'none'


def test_time_jump_drops_track():
    gate, _ = run(flight()[:30])
    assert gate.update(30 / FPS + 5., [WALL_MARK]) is None     # camera stalled for 5 s
    assert gate.status() == 'none'


if __name__ == '__main__':             # run without pytest: python test_hotspot_gate.py
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
