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
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CALIB = Path(__file__).resolve().parent / "wrist_cam_calib_1080"


def read_camera_info(path: Path):
    """(size, K, dist) from a camera_info YAML written by calibrate_intrinsics.py."""
    text = Path(path).read_text()

    def data(key):
        block = text.split(key + ":", 1)[1]
        inside = block.split("data: [", 1)[1].split("]", 1)[0]
        return np.array([float(v) for v in inside.split(",")])

    def scalar(key):
        return int(text.split(key + ":", 1)[1].split()[0])

    return ((scalar("image_width"), scalar("image_height")), data("camera_matrix").reshape(3, 3),
            data("distortion_coefficients"))


class WristCam:
    """Remaps raw frames to the pinhole camera in <calib_dir>/<name>_pinhole_<W>x<H>.yaml."""

    def __init__(self, calib_dir: Path | str = DEFAULT_CALIB, name: str = "wrist_cam",
                 out_size: tuple[int, int] = (640, 480)):
        calib_dir = Path(calib_dir)
        self.raw_size, self.raw_K, self.raw_dist = read_camera_info(calib_dir / f"{name}.yaml")
        pinhole = calib_dir / f"{name}_pinhole_{out_size[0]}x{out_size[1]}.yaml"
        if not pinhole.exists():
            raise FileNotFoundError(f"{pinhole} missing; run: calibrate_intrinsics.py pinhole "
                                    f"--out {calib_dir} --size {out_size[0]} {out_size[1]}")
        self.size, self.K, _ = read_camera_info(pinhole)
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
    while True:
        ok, raw = cap.read()
        if not ok:
            continue  # macOS cameras can take a moment to send the first frame
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
