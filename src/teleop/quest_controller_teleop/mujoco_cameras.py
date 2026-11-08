"""Headset camera frames rendered on their own thread, so the control loop never waits on them.

Rendering three cameras and encoding them as JPEG takes ~30 ms on a laptop, three control ticks.
Done inline, it stalled arm commands every frame (measured: p95 tick 29 ms against a 10 ms period).
The control loop now only copies the sim state (microseconds); this thread renders a private
MjData rebuilt from that copy. The model is shared read-only.
"""
from __future__ import annotations

import threading

import numpy as np


class CameraStream:
    def __init__(self, model, driver, names, size):
        self.model, self.driver, self.names, self.size = model, driver, tuple(names), size
        self._lock = threading.Lock()
        self._state = None  # (qpos, qvel, act, mocap_pos, mocap_quat), latest only
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="quest-cameras", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def push(self, data) -> None:
        """Hand the current sim state to the camera thread; an unrendered older state is dropped."""
        state = tuple(np.copy(x) for x in (data.qpos, data.qvel, data.act, data.mocap_pos, data.mocap_quat))
        with self._lock:
            self._state = state
        self._ready.set()

    def close(self) -> None:
        self._stop.set()
        self._ready.set()
        self._thread.join(timeout=2.)

    def _run(self) -> None:
        import mujoco

        # GL contexts belong to the thread that made them: create and use the renderers here only.
        data = mujoco.MjData(self.model)
        renderers = {name: mujoco.Renderer(self.model, *self.size) for name in self.names}
        try:
            while self._ready.wait() and not self._stop.is_set():
                self._ready.clear()
                with self._lock:
                    state, self._state = self._state, None
                if state is None:
                    continue
                data.qpos[:], data.qvel[:], data.act[:], data.mocap_pos[:], data.mocap_quat[:] = state
                mujoco.mj_forward(self.model, data)
                self.driver.serve_frames(**{name: self._render(r, data, name) for name, r in renderers.items()})
        finally:
            for renderer in renderers.values():
                renderer.close()

    @staticmethod
    def _render(renderer, data, name):
        import mujoco

        renderer.update_scene(data, name)
        # Shadows and floor reflection cost most of a frame and barely show at headset-panel size.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        return renderer.render()
