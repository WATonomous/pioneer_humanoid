"""Read the shuttle's depth from a depth image (no ROS).

A shuttle covers only a few pixels, and TrackNet's pixel can be a few pixels off, so most of a small
patch around it shows what is behind it. Take the closest reading in the patch, and keep it only if it
is clearly in front of the usual depth at that pixel (the camera is fixed, so the median over earlier
depth images is the wall or floor behind). Otherwise the sensor missed the shuttle: no reading.
"""
import numpy as np


def patch_depths(depth_img, u, v, radius=2):
    """Valid readings (m) in the (2 * radius + 1)^2 patch around pixel (u, v)."""
    to_m = 0.001 if depth_img.dtype == np.uint16 else 1.  # RealSense: uint16 mm; simulators: float m
    ui, vi = int(round(u)), int(round(v))
    patch = depth_img[max(vi - radius, 0):vi + radius + 1, max(ui - radius, 0):ui + radius + 1].astype(float)
    return patch[(patch > 0) & np.isfinite(patch)] * to_m


def depth_reader(depth_img, earlier_imgs, margin=0.3):
    """depth_at(u, v) for ShuttleEKF.update: the shuttle's depth in m at that pixel, or None.
    earlier_imgs: recent depth images from the same (fixed) camera, for the background."""
    def depth_at(u, v):
        now = patch_depths(depth_img, u, v)
        if not now.size:
            return None
        behind = [np.median(d) for d in (patch_depths(img, u, v) for img in earlier_imgs) if d.size]
        closest = float(now.min())
        if behind and closest > np.median(behind) - margin:
            return None
        return closest
    return depth_at
