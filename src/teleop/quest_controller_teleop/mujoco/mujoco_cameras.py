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


class CameraWindow:
    """The operator's window: ego camera on top, both wrist cameras below (50/50), at ~30 fps.

    Replaces the MuJoCo viewer when its 3D UI is not needed. The control loop pushes state copies (microseconds); a
    thread renders the three cameras off-screen into one image; show() displays it and must run on the main thread
    (macOS: windows live there). Measured on an M2: ego ~14 ms + wrists ~13 ms per frame.
    """

    TITLE = "Pioneer teleop (6 resets)"
    EGO_SIZE = (540, 960)     # (height, width): the full width over both wrist panels
    WRIST_SIZE = (360, 480)
    HEAVY_GROUP = 5           # render-only group for the 70k-vertex arm meshes: hidden in the wrist views
    # Display upscale. OpenCV shows one image pixel per Retina pixel (half size); 1.9x fills a 14" MacBook's height
    # (956 pt). Upscaling costs ~6 ms on the display loop and no render time; rendering at 2x instead took ~40% more.
    DISPLAY_SCALE = 1.9

    def __init__(self, model):
        import copy
        import mujoco

        # A private model copy, so regrouping its meshes never touches the sim's (or the headset cameras') model.
        # The wrist cameras never see the upper arm, whose meshes cost half of each wrist render.
        self.model = copy.deepcopy(model)
        for i in range(self.model.ngeom):
            if (self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and self.model.geom_group[i] < 3
                    and self.model.mesh_vertnum[self.model.geom_dataid[i]] > 20000):
                self.model.geom_group[i] = self.HEAVY_GROUP
        self._lock, self._state, self._ready = threading.Lock(), None, threading.Event()
        self._frame, self._frame_id = None, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-window", daemon=True)
        self._thread.start()

    def push(self, data) -> None:
        state = tuple(np.copy(x) for x in (data.qpos, data.qvel, data.act, data.mocap_pos, data.mocap_quat))
        with self._lock:
            self._state = state
        self._ready.set()

    def show(self, stop, reset_key, running) -> None:
        """Display frames until the window is closed or `running()` is false; key 6 sets `reset_key`."""
        import cv2

        cv2.namedWindow(self.TITLE, cv2.WINDOW_NORMAL)
        shown = 0
        try:
            while not stop.is_set() and running():
                with self._lock:
                    frame, frame_id = self._frame, self._frame_id
                if frame is not None and frame_id != shown:
                    cv2.imshow(self.TITLE, cv2.resize(frame[:, :, ::-1], None, fx=self.DISPLAY_SCALE,
                                                      fy=self.DISPLAY_SCALE, interpolation=cv2.INTER_LINEAR))
                    shown = frame_id
                if cv2.waitKey(5) & 0xFF == ord("6"):
                    reset_key.set()
                if shown and cv2.getWindowProperty(self.TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break   # window closed
        finally:
            self._stop.set()
            self._ready.set()
            cv2.destroyAllWindows()

    def _run(self) -> None:
        import mujoco

        data = mujoco.MjData(self.model)
        ego, wrist = mujoco.Renderer(self.model, *self.EGO_SIZE), mujoco.Renderer(self.model, *self.WRIST_SIZE)
        full, light = mujoco.MjvOption(), mujoco.MjvOption()
        full.geomgroup[self.HEAVY_GROUP] = 1
        try:
            while self._ready.wait() and not self._stop.is_set():
                self._ready.clear()
                with self._lock:
                    state, self._state = self._state, None
                if state is None:
                    continue
                data.qpos[:], data.qvel[:], data.act[:], data.mocap_pos[:], data.mocap_quat[:] = state
                mujoco.mj_forward(self.model, data)
                top = self._render(ego, data, "ego", full)
                bottom = np.hstack([self._render(wrist, data, name, light) for name in ("wrist_left", "wrist_right")])
                with self._lock:
                    self._frame, self._frame_id = np.vstack([top, bottom]), self._frame_id + 1
        finally:
            ego.close()
            wrist.close()

    @staticmethod
    def _render(renderer, data, name, option):
        import mujoco

        renderer.update_scene(data, name, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        return renderer.render()
