#!/usr/bin/env python3
"""Real wrist camera frames -> the 640x480 pinhole image the sim renders. See README.md.

    cam = WristCam()
    cap = open_camera(0, cam)
    ok, raw = cap.read()
    img = cam.rectify(raw)

`python3 wrist_cam.py --camera 0` shows raw | rectified live.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pioneer_humanoid.camera_info import CALIB_DIR, read_camera_info

DEFAULT_CALIB = CALIB_DIR  # src/pioneer_humanoid/pioneer_humanoid/calib


def _load(path: Path):
    size, K, dist = read_camera_info(path)
    return size, np.array(K), np.array(dist)


class WristCam:
    """Remaps raw frames to the pinhole camera in <calib_dir>/<name>_pinhole_<W>x<H>.yaml."""

    def __init__(self, calib_dir: Path | str = DEFAULT_CALIB, name: str = "wrist_cam",
                 out_size: tuple[int, int] = (640, 480)):
        calib_dir = Path(calib_dir)
        self.raw_size, self.raw_K, self.raw_dist = _load(calib_dir / f"{name}.yaml")
        pinhole = calib_dir / f"{name}_pinhole_{out_size[0]}x{out_size[1]}.yaml"
        if not pinhole.exists():
            raise FileNotFoundError(f"{pinhole} missing; run: calibrate_intrinsics.py pinhole "
                                    f"--size {out_size[0]} {out_size[1]}")
        self.size, self.K, _ = _load(pinhole)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.raw_K, self.raw_dist, None, self.K, self.size, cv2.CV_16SC2)

    def rectify(self, raw: np.ndarray) -> np.ndarray:
        """Raw camera frame (calibrated resolution) -> undistorted, centred pinhole image."""
        h, w = raw.shape[:2]
        if (w, h) != self.raw_size:
            # A resized or cropped frame would remap without error but to the wrong place.
            raise ValueError(f"frame is {w}x{h}, calibration is for {self.raw_size[0]}x"
                             f"{self.raw_size[1]}; capture at that resolution")
        return cv2.remap(raw, self._map1, self._map2, cv2.INTER_LINEAR)


def open_camera(index: int, cam: WristCam) -> cv2.VideoCapture:
    """VideoCapture at the calibrated resolution (full sensor, so not cropped on macOS)."""
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.raw_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.raw_size[1])
    return cap


def main() -> int:
    ap = argparse.ArgumentParser(description="live raw | rectified wrist camera preview")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calib", default=str(DEFAULT_CALIB))
    args = ap.parse_args()

    cam = WristCam(args.calib)
    cap = open_camera(args.camera, cam)
    print(f"raw {cam.raw_size[0]}x{cam.raw_size[1]} -> pinhole {cam.size[0]}x{cam.size[1]}, "
          f"f = {cam.K[0, 0]:g} px. q quits.")
    last_frame = time.time()
    while True:
        ok, raw = cap.read()
        if not ok:
            # macOS cameras can take a moment to start; a camera that stops entirely is an error.
            if time.time() - last_frame > 3.0:
                print("no frames from the camera for 3 s (unplugged, or in use by another app?)")
                break
            if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                break
            continue
        last_frame = time.time()
        rect = cam.rectify(raw)
        h = rect.shape[0]
        left = cv2.resize(raw, (round(raw.shape[1] * h / raw.shape[0]), h))
        cv2.imshow("raw | rectified (what the policy sees)", np.hstack([left, rect]))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
