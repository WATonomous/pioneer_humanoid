"""Hotspot gate: picks which of TrackNet's top hotspots is the shuttle. No ROS in here.

TrackNet reports several candidate pixels per frame and sometimes a stronger one is a mark on the
wall. The gate follows the shuttle with ShuttleEKF (a physics model of the flight projected into the
image) and only accepts a hotspot that lies where the flight says the shuttle should be.

It works on pixels alone, with no depth: the 3D position stays the localizer's job. The depth
inputs of ShuttleEKF are simply left unused.

    gate = HotspotGate(Camera(fx, fy, cx, cy, width, height), fps=30.)
    point = gate.update(time_s, [(u, v, score), ...])    # (u, v, score) of the shuttle, or None
"""
from perception.shuttle_ekf import ShuttleEKF


class HotspotGate:
    def __init__(self, camera, fps=30., **settings):
        self.ekf = ShuttleEKF(camera, fps=fps, **settings)

    def update(self, time, hotspots):
        """time: image time in seconds; hotspots: [(u, v, score), ...] strongest first.
        Returns (u, v, score) of the hotspot judged to be the shuttle in this frame, or None when
        there is no track yet, or the track found nothing matching this frame (coasting)."""
        self.ekf.update(time, hotspots)
        ekf = self.ekf
        if ekf.status() != 'tracking' or ekf.last_match is None or ekf.last_match[0] != ekf.frame:
            return None
        u, v = ekf.last_match[1]
        score = next((h[2] for h in hotspots if h[0] == u and h[1] == v), 0.)
        return u, v, score

    def status(self):
        """'none' (no track), 'tracking' or 'coasting'"""
        return self.ekf.status()
