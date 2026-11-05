"""Read ROS camera_info YAMLs (as written by perception/camera_calibration). Plain Python, no cv2 or numpy.

The measured wrist camera and the pinhole the sim renders live in calib/ (see the README in
src/perception/camera_calibration for how they are made).
"""
from pathlib import Path

CALIB_DIR = Path(__file__).resolve().parent / "calib"


def read_camera_info(path):
    """((width, height), K as 3 rows of 3, distortion coefficients) from a camera_info YAML."""
    text = Path(path).read_text()

    def data(key):
        block = text.split(key + ":", 1)[1]
        inside = block.split("data: [", 1)[1].split("]", 1)[0]
        return [float(v) for v in inside.split(",")]

    def scalar(key):
        return int(text.split(key + ":", 1)[1].split()[0])

    k = data("camera_matrix")
    return (scalar("image_width"), scalar("image_height")), [k[0:3], k[3:6], k[6:9]], data("distortion_coefficients")
