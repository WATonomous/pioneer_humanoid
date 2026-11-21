"""Wrist camera on a thread: every frame host-stamped, rectified to the sim wrist cameras' 640x480 pinhole.

The same camera model and mount as the robot's wrist camera, and the same ``wrist_cam.rectify()`` as the
``wrist_camera`` ROS node, so recorded frames look like what the policy sees on the arm.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from umi_gripper._paths import CAMERA_CALIBRATION, PIONEER_HUMANOID, add

OUT_SIZE = (640, 480)


def _device(value):
    return int(value) if isinstance(value, int) or str(value).isdigit() else str(value)


class CameraStream:
    """drain() -> [(host_s, rgb)] since the last drain, while ``recording``; ``latest`` always."""

    def __init__(self, camera_cfg: dict, capture=None, rectifier=None):
        import cv2
        self.cv2 = cv2
        self.rectifier = rectifier
        if capture is None:
            if camera_cfg.get("rectify", True):
                add(CAMERA_CALIBRATION, PIONEER_HUMANOID)
                from wrist_cam import WristCam, open_camera
                self.rectifier = WristCam(camera_cfg["calib_dir"]) if camera_cfg.get("calib_dir") else WristCam()
                capture = open_camera(_device(camera_cfg["device"]), self.rectifier, int(camera_cfg.get("fps", 30)))
            else:
                print("[UMI] camera.rectify is off: frames are only resized, so they won't match the sim pinhole")
                capture = cv2.VideoCapture(_device(camera_cfg["device"]))
                if not capture.isOpened():
                    raise RuntimeError(f"could not open camera {camera_cfg['device']}")
        self.capture = capture
        self.recording = False
        self.frames: deque[tuple[float, np.ndarray]] = deque()
        self.latest: np.ndarray | None = None
        self.count = 0
        self.error: BaseException | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="wrist-camera", daemon=True)

    def start(self) -> "CameraStream":
        self._thread.start()
        return self

    def _convert(self, bgr: np.ndarray) -> np.ndarray:
        if self.rectifier is not None:
            bgr = self.rectifier.rectify(bgr)
        elif (bgr.shape[1], bgr.shape[0]) != OUT_SIZE:
            bgr = self.cv2.resize(bgr, OUT_SIZE, interpolation=self.cv2.INTER_AREA)
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                ok, bgr = self.capture.read()
                stamp = time.monotonic()   # read() returns once the frame is in; the exposure was earlier: latency.camera_s
                if not ok:
                    raise RuntimeError("camera stopped delivering frames")
                rgb = self._convert(bgr)
                with self._lock:
                    self.latest = rgb
                    self.count += 1
                    if self.recording:
                        self.frames.append((stamp, rgb))
        except BaseException as exc:
            self.error = exc

    def set_recording(self, on: bool) -> None:
        with self._lock:
            self.recording = on
            self.frames.clear()

    def drain(self) -> list[tuple[float, np.ndarray]]:
        with self._lock:
            out = list(self.frames)
            self.frames.clear()
        return out

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.capture.release()
