"""Put the headset's sample times on the host clock.

Each pose packet carries the headset's own ``performance.now()`` (``time_ms``) and is stamped with the host's
``time.monotonic()`` on arrival. Arrival time = sample time + offset + transport delay, and the delay is never
negative, so the smallest ``host - headset`` difference seen over a window is the clock offset plus the *minimum*
delay. Mapping every sample through that keeps the headset's even frame spacing instead of the USB/WebSocket
jitter; the constant left over (minimum delay + WebXR's display-time prediction) is ``latency.pose_s``, measured
with calibrate_camera.py. A window rather than one global minimum follows the two clocks' slow drift.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def sliding_min(times: np.ndarray, values: np.ndarray, half_window: float) -> np.ndarray:
    """min(values[j]) over |times[j] - times[i]| <= half_window, for every i. ``times`` ascending. O(n)."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(times)
    out = np.empty(n)
    window: deque[int] = deque()   # indices, values increasing
    right = 0
    left = 0
    for i in range(n):
        while right < n and times[right] <= times[i] + half_window:
            while window and values[window[-1]] >= values[right]:
                window.pop()
            window.append(right)
            right += 1
        while times[left] < times[i] - half_window:
            left += 1
        while window[0] < left:
            window.popleft()
        out[i] = values[window[0]]
    return out


def headset_to_host(client_ms: np.ndarray, host_s: np.ndarray, window_s: float = 30.0) -> np.ndarray:
    """Host-clock time (s) of each headset sample, from its headset time and host arrival time."""
    client_s = np.asarray(client_ms, dtype=float) / 1000.0
    host_s = np.asarray(host_s, dtype=float)
    if client_s.shape != host_s.shape:
        raise ValueError("client and host times must have the same shape")
    if len(client_s) == 0:
        return client_s
    if np.any(np.diff(client_s) < 0):
        raise ValueError("headset times must be non-decreasing (one page session per take)")
    offset = sliding_min(client_s, host_s - client_s, window_s / 2)
    return client_s + offset
