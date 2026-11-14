"""The operator's camera window, rendered on its own thread so the control loop never waits on it."""
from __future__ import annotations

import threading

import numpy as np


class CameraWindow:
    """The operator's window: ego camera on top, both wrist cameras below; also the recorder's camera.

    One render thread serves both, so they never compete for the GPU. The control loop pushes state copies; while
    recording it also queues 25 Hz samples, rendered in order and handed to the recorder with their action and state.
    show() must run on the main thread (macOS).
    """

    TITLE = "Pioneer teleop"
    CAMERAS = ("ego", "wrist_left", "wrist_right")
    SIZE = (480, 640)         # (height, width) per camera: the dataset schema's
    HEAVY_GROUP = 5           # the 70k-vertex upper-arm meshes, hidden in the wrist views (out of their sight)
    DISPLAY_HEIGHT = 1710     # px; OpenCV draws 1 px per Retina pixel, so this fills a 14" MacBook's height

    def __init__(self, model, sizes=None, recorder=None, on_saved=None):
        """`sizes`: {camera: (height, width)}; `recorder`: a SimLeRobotRecorder fed by push_sample();
        `on_saved()`: called after the recorder saves a take."""
        import copy
        import queue
        import mujoco

        self.sizes = {name: tuple((sizes or {}).get(name, self.SIZE)) for name in self.CAMERAS}
        self.recorder, self.on_saved = recorder, on_saved
        self.status = ""          # overlay text, e.g. the recording state
        # A private model copy, so regrouping meshes never touches the sim's model.
        self.model = copy.deepcopy(model)
        for i in range(self.model.ngeom):
            if (self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and self.model.geom_group[i] < 3
                    and self.model.mesh_vertnum[self.model.geom_dataid[i]] > 20000):
                self.model.geom_group[i] = self.HEAVY_GROUP
        self._lock, self._state, self._ready = threading.Lock(), None, threading.Event()
        self._samples = queue.Queue()
        self._frame, self._frame_id = None, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-window", daemon=True)
        self._thread.start()

    @staticmethod
    def _copy(data):
        return tuple(np.copy(x) for x in (data.qpos, data.qvel, data.act, data.mocap_pos, data.mocap_quat))

    def push(self, data) -> None:
        """The latest state, for display."""
        with self._lock:
            self._state = self._copy(data)
        self._ready.set()

    def push_sample(self, data, action, state, task=None, extras=None) -> None:
        """A recorded frame: rendered from this exact sim state, then given to the recorder.

        `task`: this frame's instruction (a multi-step scene's current step; None: the session's task);
        `extras`: the recorder's extra features. Both are read in the control loop: the render thread's state copy
        has no userdata, where scenes keep their step counter."""
        self._samples.put((self._copy(data), action, state, task, extras))
        self._ready.set()

    def discard(self) -> None:
        """Drop the unsaved take (in order with the queued samples)."""
        self._samples.put(None)
        self._ready.set()

    @property
    def backlog(self) -> int:
        return self._samples.qsize()

    def close(self) -> None:
        """Render the queued samples, then stop."""
        self._stop.set()
        self._ready.set()
        self._thread.join(timeout=30.)

    def show(self, stop, running, keys) -> None:
        """Display frames until the window is closed or `running()` is false; `keys`: {char: callback}."""
        import cv2

        cv2.namedWindow(self.TITLE, cv2.WINDOW_NORMAL)
        shown = 0
        try:
            while not stop.is_set() and running():
                with self._lock:
                    frame, frame_id = self._frame, self._frame_id
                if frame is not None and frame_id != shown:
                    scale = self.DISPLAY_HEIGHT / frame.shape[0]
                    image = cv2.resize(frame[:, :, ::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    if self.status:
                        cv2.putText(image, self.status, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (40, 40, 230), 4)
                    cv2.imshow(self.TITLE, image)
                    shown = frame_id
                key = cv2.waitKey(5) & 0xFF
                if chr(key) in keys:
                    keys[chr(key)]()
                if shown and cv2.getWindowProperty(self.TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break   # window closed
        finally:
            cv2.destroyAllWindows()

    def _run(self) -> None:
        import queue
        import mujoco

        data = mujoco.MjData(self.model)
        renderers = {name: mujoco.Renderer(self.model, *size) for name, size in self.sizes.items()}
        full, light = mujoco.MjvOption(), mujoco.MjvOption()
        full.geomgroup[self.HEAVY_GROUP] = 1
        try:
            while True:
                try:
                    item = self._samples.get_nowait()
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    self._ready.wait(.1)
                    self._ready.clear()
                    with self._lock:
                        state, self._state = self._state, None
                    if state is None:
                        continue
                    item = (state, None, None, None, None)
                else:
                    if item is None:   # discard
                        self.recorder.cancel_recording()
                        continue
                state, action, recorded, task, extras = item
                data.qpos[:], data.qvel[:], data.act[:], data.mocap_pos[:], data.mocap_quat[:] = state
                mujoco.mj_forward(self.model, data)
                images = {name: self._render(r, data, name, full if name == "ego" else light)
                          for name, r in renderers.items()}
                if action is not None and self.recorder.tick(action, recorded, images, task=task, extras=extras) and self.on_saved:
                    self.on_saved()
                ego, left, right = (images[name] for name in self.CAMERAS)
                bottom = np.hstack([left, right])
                top = cv2_resize(ego, bottom.shape[1])
                with self._lock:
                    self._frame, self._frame_id = np.vstack([top, bottom]), self._frame_id + 1
        finally:
            for renderer in renderers.values():
                renderer.close()

    @staticmethod
    def _render(renderer, data, name, option):
        import mujoco

        renderer.update_scene(data, name, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        return renderer.render()


def cv2_resize(image, width):
    """`image` scaled to `width` px wide, aspect kept."""
    import cv2

    height = round(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
