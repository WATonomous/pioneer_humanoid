"""Camera direction for the demo videos: a slow, smooth camera that follows what the robot is looking at (its gaze
target, recorded on the tape), pulls back when the robot walks and closes in while it works."""
from __future__ import annotations

import numpy as np

from .video import Shot, render_tape

_CACHE: dict = {}


def _smooth(x: np.ndarray, fps: float, tau: float) -> np.ndarray:
    """Forward-backward exponential smoothing (no lag)."""
    a = 1 - np.exp(-1 / (fps * tau))
    y = x.copy()
    for i in range(1, len(y)):
        y[i] = y[i - 1] + a * (y[i] - y[i - 1])
    for i in range(len(y) - 2, -1, -1):
        y[i] = y[i + 1] + a * (y[i] - y[i + 1])
    return y


def _plan(tape) -> dict:
    q = np.array(tape.qpos)
    fps = tape.fps
    focus = np.array(tape.focus, float)
    pelvis = q[:, :2] + np.array(tape.base0[:2]) if hasattr(tape, "base0") else q[:, :2]
    # gaps (no gaze yet): hold the first known focus
    good = ~np.isnan(focus[:, 0])
    first = focus[good][0] if good.any() else np.array([0.5, 0.0, 0.9])
    last = first
    for i in range(len(focus)):
        if good[i]:
            last = focus[i]
        else:
            focus[i] = last
    look = _smooth(focus, fps, 1.2)
    speed = np.r_[0, np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1)] * fps   # pelvis x/y speed (m/s)
    wide = _smooth((speed > 0.02).astype(float), fps, 1.5)
    wide = np.clip(_smooth(wide, fps, 1.0) * 1.6, 0, 1)
    t = np.array(tape.t)
    return dict(look=look, wide=wide, t=t, T=t[-1])


def follow(t: float, tape, i: int) -> Shot:
    key = id(tape)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = _plan(tape)
    p = _CACHE[key]
    look, w, T = p["look"][i], p["wide"][i], p["T"]
    u = t / max(T, 1e-6)
    az = 205 - 55 * u + 8 * np.sin(2 * np.pi * t / 37)          # a slow orbit from front-right to front-left
    el = -24 + 6 * np.sin(2 * np.pi * t / 53)
    dist = 1.05 + 1.5 * w
    target = (1 - 0.55 * w) * look + 0.55 * w * np.array([look[0] - 0.25, look[1], 1.0])
    return Shot(tuple(target), dist, az, el)


def render(scene: str, tape_path: str, out: str, size=(1280, 720), workers: int = 4, title: str | None = None,
           inset: str | None = "head_cam") -> str:
    from . import entry
    title = title or entry(scene).title or scene
    return render_tape(scene, tape_path, out, ("pioneer_whole_body.shots", "follow"), title, size=size,
                       workers=workers, inset=inset)
