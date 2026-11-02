"""Shuttle EKF: follows one shuttle in 3D from TrackNet hotspots (pixels). No ROS in here.

State x = [X, Y, Z, VX, VY, VZ]: position (m) and velocity (m/s) in the camera frame
(x right, y down, z forward). Physics: gravity plus quadratic air drag, dv/dt = g - |v| v / L.

Each frame, update(time, hotspots, depth_at):
  1. predict:  move the state forward to this frame's time;
  2. project:  turn the predicted 3D position into a pixel, with an uncertainty ellipse;
  3. gate:     ignore hotspots outside the ellipse (Mahalanobis distance > gate);
  4. choose:   keep the hotspot with the lowest  distance^2 - 2 * score_weight * ln(score);
  5. depth:    read the depth sensor at that hotspot (depth_at); keep it only if it is close to the
               predicted depth (the sensor often sees the wall behind a small, fast shuttle);
  6. correct:  pull the state towards the pixel (and depth). No hotspot fits -> coast on physics.
Track rules:
  start:   no track yet -> 3 hotspots in 4 frames moving steadily (not static, not jumping);
  restart: after a racket hit the shuttle leaves from the last matched pixel in a new direction ->
           a hotspot near that pixel followed by one moving away restarts the track;
  stop:    max_misses frames without a matching hotspot while in view, or max_coast seconds after
           it left the view (kept so the impact prediction survives the last moments).
A new track starts at a guessed depth (start_depth, very uncertain); depth readings that pass the
check in step 5, and the physics over the following frames, then pin it down.

Settings in pixels are for a 1920-px-wide image and scale with the camera's width.
"""
import math
from collections import deque

import numpy as np

SETTINGS = dict(
    L=3.4,                # aerodynamic length (m)
    substeps=4,           # physics steps per frame
    pixel_sigma=10.,      # px, TrackNet position noise
    accel_sigma=8.,       # m/s^2, motion the physics does not explain (spin, model error)
    gate=6.,              # Mahalanobis^2 cut-off for a hotspot to count as the shuttle
    score_weight=3.,      # how much TrackNet's confidence counts against distance
    min_score=0.2,        # weaker hotspots are ignored
    start_score=0.3,      # hotspots that start a track must reach this
    start_tol=15.,        # px a start triple may stray from steady motion (+ 30% of the step)
    min_step=3.,          # px/frame: slower hotspots count as static
    max_step=250.,        # px/frame: faster pairs are not one shuttle
    hit_radius=30.,       # px: a restart must leave from this close to the last matched pixel
    hit_window=3,         # frames after the last match in which a restart is tried
    static_radius=6.,     # px: a spot seen in static_frames of the last 30 frames is static
    static_frames=15,
    start_depth=6.,       # m, first depth guess
    start_depth_sigma=3.,
    depth_sigma=0.005,    # m, depth noise at 1 m; grows with distance^2 (RealSense D455: <2% at 4 m)
    depth_gate=9.,        # (difference / expected spread)^2 above which a depth reading is ignored
    max_misses=5,         # frames without a match (while in view) before the track is dropped
    max_coast=1.,         # s to keep predicting a shuttle that has left the camera's view
    max_gap=0.5,          # s between frames above which the track is dropped (camera stall or restart)
    show_misses=1,        # coasted frames still reported as visible
)
PIXEL_SETTINGS = ('pixel_sigma', 'start_tol', 'min_step', 'max_step', 'hit_radius', 'static_radius')


class Camera:
    """Pinhole camera. pitch_deg: how far it looks up from level; sets gravity in its frame."""

    def __init__(self, fx, fy, cx, cy, width, height, pitch_deg=0.):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.width, self.height = width, height
        p = math.radians(pitch_deg)
        self.gravity = 9.81 * np.array([0., math.cos(p), -math.sin(p)])

    def project(self, x):
        X, Y, Z = x[:3]
        return np.array([self.fx * X / Z + self.cx, self.fy * Y / Z + self.cy])

    def jacobian(self, x):
        """d(pixel) / d(state)"""
        X, Y, Z = x[:3]
        H = np.zeros((2, 6))
        H[0, 0], H[0, 2] = self.fx / Z, -self.fx * X / Z ** 2
        H[1, 1], H[1, 2] = self.fy / Z, -self.fy * Y / Z ** 2
        return H

    def inside(self, uv, margin=0.):
        return -margin <= uv[0] < self.width + margin and -margin <= uv[1] < self.height + margin


class ShuttleEKF:
    def __init__(self, camera, fps=30., **settings):
        self.cam, self.fps = camera, fps
        self.s = {**SETTINGS, **settings}
        for k in PIXEL_SETTINGS:
            self.s[k] *= camera.width / 1920
        self.R = np.eye(2) * self.s['pixel_sigma'] ** 2
        self.recent = deque(maxlen=4)       # (frame, start-worthy hotspots) of the last 4 frames
        self.spots = deque(maxlen=30)       # every hotspot pixel of the last 30 frames
        self.frame, self.time = -1, None
        self.last_match = None              # (frame, pixel) of the last hotspot used
        self.x = self.P = None
        self.misses = 0

    # ---------------- per frame ----------------
    def update(self, time, hotspots, depth_at=None):
        """time: seconds; hotspots: [(u, v, score), ...] strongest first;
        depth_at(u, v): depth sensor reading in m at that pixel, or None if invalid (optional).
        Returns 'none', 'tracking' or 'coasting'. The estimate is in self.x and self.P."""
        dt = 1. / self.fps if self.time is None else time - self.time
        if not 0 < dt <= self.s['max_gap']:
            dt, self.x = 1. / self.fps, None                      # time went back or jumped: start over
        self.frame, self.time = self.frame + 1, time
        hotspots = [h for h in hotspots if h[2] >= self.s['min_score']]
        self.spots.append([h[:2] for h in hotspots])
        movers = [h for h in hotspots if h[2] >= self.s['start_score'] and not self._is_static(h)]
        self.recent.append((self.frame, movers))

        used = None
        if self.x is not None:
            self._predict(dt)
            if self.x[2] <= 0.3:
                self.x = None                                     # passed the camera
        if self.x is not None:
            used = self._correct(hotspots, depth_at)
            self.misses = 0 if used is not None else self.misses + 1
            # Lost while it should be in view -> drop. Out of view (often just before it reaches the
            # robot) -> keep coasting on physics so the impact prediction stays alive, up to max_coast.
            in_view = self.cam.inside(self.cam.project(self.x))
            if self.misses > (self.s['max_misses'] if in_view else self.s['max_coast'] * self.fps):
                self.x = None

        start = self._find_restart() if used is None else None
        if start is None and (self.x is None or self.misses >= 1):
            start = self._find_start()
        if start is not None:
            self._start(*start, depth_at)
        return self.status()

    def status(self):
        if self.x is None:
            return 'none'
        return 'tracking' if self.misses == 0 else 'coasting'

    def visible(self):
        """Whether to report the shuttle: tracked now or coasting only briefly, and in view."""
        return self.x is not None and self.misses <= self.s['show_misses'] and \
            self.cam.inside(self.cam.project(self.x))

    # ---------------- EKF ----------------
    def _motion(self, x, dt):
        """Move state x forward by dt seconds; also return the Jacobian d(new x) / d(x)."""
        h, L, F = dt / self.s['substeps'], self.s['L'], np.eye(6)
        for _ in range(self.s['substeps']):
            v = x[3:]
            speed = np.linalg.norm(v) + 1e-9
            Fi = np.eye(6)
            Fi[:3, 3:] = np.eye(3) * h
            Fi[3:, 3:] = np.eye(3) - h / L * (speed * np.eye(3) + np.outer(v, v) / speed)
            x = np.concatenate([x[:3] + v * h, v + (self.cam.gravity - speed / L * v) * h])
            F = Fi @ F
        return x, F

    def _predict(self, dt):
        self.x, F = self._motion(self.x, dt)
        q = self.s['accel_sigma'] ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * q * dt ** 4 / 4
        Q[:3, 3:] = Q[3:, :3] = np.eye(3) * q * dt ** 3 / 2
        Q[3:, 3:] = np.eye(3) * q * dt ** 2
        self.P = F @ self.P @ F.T + Q

    def _correct(self, hotspots, depth_at=None):
        """Gate and choose a hotspot, add its depth reading if it is plausible, and correct the
        state with them. Returns the hotspot's index or None."""
        H = self.cam.jacobian(self.x)
        S_inv = np.linalg.inv(H @ self.P @ H.T + self.R)
        predicted = self.cam.project(self.x)
        best = None
        for i, (u, v, score) in enumerate(hotspots):
            y = np.array([u, v]) - predicted
            d2 = y @ S_inv @ y
            cost = d2 - 2 * self.s['score_weight'] * math.log(score)
            if d2 <= self.s['gate'] and (best is None or cost < best[0]):
                best = (cost, i, y)
        if best is None:
            return None
        _, i, y = best
        u, v = hotspots[i][:2]

        # Depth reading at the chosen pixel: keep it only if it is close to the predicted depth
        R = self.R
        depth = depth_at(u, v) if depth_at else None
        if depth is not None:
            r_depth = (self.s['depth_sigma'] * depth ** 2) ** 2           # noise grows with distance^2
            dz = depth - self.x[2]
            if dz ** 2 / (self.P[2, 2] + r_depth) <= self.s['depth_gate']:
                H = np.vstack([H, [0, 0, 1, 0, 0, 0]])                     # measure [u, v, Z]
                y = np.append(y, dz)
                R = np.diag([R[0, 0], R[1, 1], r_depth])

        S_inv = np.linalg.inv(H @ self.P @ H.T + R)
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.last_match = (self.frame, (u, v))
        return i

    # ---------------- track start ----------------
    def _is_static(self, h):
        r2 = self.s['static_radius'] ** 2
        seen = sum(any((h[0] - a) ** 2 + (h[1] - b) ** 2 <= r2 for a, b in frame) for frame in self.spots)
        return seen >= self.s['static_frames']

    def _find_start(self):
        """Best steady triple a -> b -> c in 3 of the last 4 frames, c in this frame.
        Returns (c, velocity in px/frame) or None."""
        frames = list(self.recent)
        if len(frames) < 3 or frames[-1][0] != self.frame:
            return None
        f3, cs = frames[-1]
        best = None
        for i in range(len(frames) - 2):
            for j in range(i + 1, len(frames) - 1):
                (f1, as_), (f2, bs) = frames[i], frames[j]
                for a in as_:
                    for b in bs:
                        vel = ((b[0] - a[0]) / (f2 - f1), (b[1] - a[1]) / (f2 - f1))
                        step = math.hypot(*vel)
                        if not self.s['min_step'] <= step <= self.s['max_step']:
                            continue
                        guess = (b[0] + vel[0] * (f3 - f2), b[1] + vel[1] * (f3 - f2))
                        for c in cs:
                            err = math.hypot(c[0] - guess[0], c[1] - guess[1])
                            if err > self.s['start_tol'] + 0.3 * step * (f3 - f2):
                                continue
                            cost = err - 10 * (a[2] + b[2] + c[2])
                            if best is None or cost < best[0]:
                                best = (cost, c, ((c[0] - b[0]) / (f3 - f2), (c[1] - b[1]) / (f3 - f2)))
        return best[1:] if best else None

    def _find_restart(self):
        """After a racket hit: hotspot b (1-2 frames ago) near the last matched pixel, then hotspot c now
        moving away from it. Returns (c, velocity in px/frame) or None."""
        if self.last_match is None or self.frame - self.last_match[0] > self.s['hit_window']:
            return None
        frames = list(self.recent)
        if len(frames) < 2 or frames[-1][0] != self.frame:
            return None
        _, cs = frames[-1]
        last = self.last_match[1]
        best = None
        for f2, bs in frames[-3:-1]:
            gap = self.frame - f2
            if gap > 2:
                continue
            for b in bs:
                if math.hypot(b[0] - last[0], b[1] - last[1]) > self.s['hit_radius']:
                    continue
                for c in cs:
                    step = math.hypot(c[0] - b[0], c[1] - b[1]) / gap
                    if self.s['min_step'] <= step <= self.s['max_step'] and (best is None or b[2] + c[2] > best[0]):
                        best = (b[2] + c[2], c, ((c[0] - b[0]) / gap, (c[1] - b[1]) / gap))
        return best[1:] if best else None

    def _start(self, c, vel_px, depth_at=None):
        """New track at pixel c moving vel_px (px/frame). Depth comes from the depth sensor if it has a
        reading there, else from the guess start_depth; depth speed starts unknown.
        The 3D state and its uncertainty come from q = [u, v, Z, du/dt, dv/dt, dZ/dt]."""
        cam, sp = self.cam, self.s['pixel_sigma']
        depth = depth_at(c[0], c[1]) if depth_at else None
        if depth is not None:
            Z, sZ = depth, self.s['depth_sigma'] * depth ** 2
        else:
            Z, sZ = self.s['start_depth'], self.s['start_depth_sigma']

        def to_state(q):
            u, v, Z, du, dv, dZ = q
            X, Y = (u - cam.cx) * Z / cam.fx, (v - cam.cy) * Z / cam.fy
            return np.array([X, Y, Z, du * Z / cam.fx + X / Z * dZ, dv * Z / cam.fy + Y / Z * dZ, dZ])

        q = np.array([c[0], c[1], Z, vel_px[0] * self.fps, vel_px[1] * self.fps, 0.])
        J = np.zeros((6, 6))                                   # d(state) / d(q), numerically
        for k in range(6):
            e = np.zeros(6)
            e[k] = 1e-4 * max(1., abs(q[k]))
            J[:, k] = (to_state(q + e) - to_state(q - e)) / (2 * e[k])
        Pq = np.diag([sp ** 2, sp ** 2, sZ ** 2, (2 * sp * self.fps) ** 2, (2 * sp * self.fps) ** 2, 10. ** 2])
        self.x, self.P, self.misses = to_state(q), J @ Pq @ J.T, 0
        self.last_match = (self.frame, c[:2])

    # ---------------- impact ----------------
    def predict_impact(self, radius=2., horizon=2., step=0.01):
        """Roll the physics forward until the shuttle comes within radius (m) of the camera.
        Returns (seconds until then, position, velocity) or None if not within horizon seconds."""
        if self.x is None:
            return None
        x = self.x.copy()
        for i in range(int(horizon / step) + 1):
            if np.linalg.norm(x[:3]) <= radius:
                return i * step, x[:3], x[3:]
            x, _ = self._motion(x, step)
        return None
