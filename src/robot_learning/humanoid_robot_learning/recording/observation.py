"""ROS image messages → uint8 HWC arrays for LeRobot."""

from __future__ import annotations

import numpy as np


def image_msg_to_hwc(msg) -> np.ndarray:
    """Convert sensor_msgs/Image to uint8 (H, W, 3)."""
    try:
        from cv_bridge import CvBridge

        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if cv_image.ndim == 2:
            cv_image = np.stack([cv_image] * 3, axis=-1)
        return np.ascontiguousarray(cv_image, dtype=np.uint8)
    except Exception:
        pass

    height = int(msg.height)
    width = int(msg.width)
    encoding = (msg.encoding or "").lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("rgb8", "bgr8"):
        channels = 3
        image = data.reshape((height, width, channels))
        if encoding == "bgr8":
            image = image[..., ::-1]
        return np.ascontiguousarray(image, dtype=np.uint8)

    if encoding in ("mono8", "8uc1"):
        gray = data.reshape((height, width))
        return np.ascontiguousarray(
            np.stack([gray, gray, gray], axis=-1), dtype=np.uint8
        )

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def fit_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Center-crop to height:width aspect, then resize (no stretching)."""
    h, w = image.shape[:2]
    if (h, w) == (height, width):
        return image
    import cv2

    if w * height > h * width:
        crop_w = h * width // height
        x0 = (w - crop_w) // 2
        image = image[:, x0: x0 + crop_w]
    else:
        crop_h = w * height // width
        y0 = (h - crop_h) // 2
        image = image[y0: y0 + crop_h]
    return np.ascontiguousarray(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
