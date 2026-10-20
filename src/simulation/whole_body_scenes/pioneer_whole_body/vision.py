"""What the robot sees: its head camera's image (``head_cam``, rendered from the simulation) and colour blobs found in
it, put back into the world by intersecting each blob's ray with a known plane (the parts' tops on the bench).

Plans that use this don't read part poses from the simulator: where a part is and what colour it is come from the
image alone (the camera's own pose is the robot's, from its joint angles).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import ndimage

# hue ranges (degrees) of the colour classes; pixels must also be saturated and bright enough
HUES = {"red": ((345, 360), (0, 12)), "green": ((95, 160),), "blue": ((200, 245),), "yellow": ((42, 66),)}
MIN_SAT, MIN_VAL = 0.45, 0.2


@dataclass
class Blob:
    colour: str
    px: tuple            # centroid (u, v) in the image
    area: int            # pixels
    world: np.ndarray    # the centroid's ray met with the plane


def hsv(img: np.ndarray):
    """Hue (degrees), saturation, value of an RGB uint8 image."""
    x = img.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    c = mx - mn
    h = np.zeros_like(mx)
    nz = c > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / c[rm]) % 6
    h[gm] = (b - r)[gm] / c[gm] + 2
    h[bm] = (r - g)[bm] / c[bm] + 4
    return h * 60.0, np.where(mx > 0, c / np.maximum(mx, 1e-6), 0.0), mx


class Eye:
    """The head camera. ``look`` renders what it sees now; ``find`` the colour blobs on a plane z = ``plane_z``."""

    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480, camera: str = "head_cam"):
        self.m = model
        self.w, self.h = width, height
        self.cam = model.camera(camera).id
        self.renderer = mujoco.Renderer(model, height, width)
        self.f = (height / 2) / math.tan(math.radians(model.cam_fovy[self.cam]) / 2)
        self.last = None

    def look(self, data: mujoco.MjData) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.cam)
        self.last = self.renderer.render()
        return self.last

    def ray(self, data: mujoco.MjData, u: float, v: float):
        """Origin and direction (world) of pixel (u, v)'s ray."""
        d_cam = np.array([(u - self.w / 2) / self.f, -(v - self.h / 2) / self.f, -1.0])
        r = data.cam_xmat[self.cam].reshape(3, 3)
        d = r @ d_cam
        return data.cam_xpos[self.cam].copy(), d / np.linalg.norm(d)

    def on_plane(self, data, u, v, z: float) -> np.ndarray:
        o, d = self.ray(data, u, v)
        t = (z - o[2]) / d[2]
        return o + t * d

    def find(self, data: mujoco.MjData, plane_z: float, min_area: int = 30, image: np.ndarray | None = None) -> list:
        img = self.look(data) if image is None else image
        h, s, v = hsv(img)
        ok = (s > MIN_SAT) & (v > MIN_VAL)
        blobs = []
        for colour, ranges in HUES.items():
            mask = ok & np.any([(h >= a) & (h < b) for a, b in ranges], axis=0)
            lab, n = ndimage.label(mask)
            for i in range(1, n + 1):
                ys, xs = np.nonzero(lab == i)
                if len(xs) < min_area:
                    continue
                u, w = float(xs.mean()) + 0.5, float(ys.mean()) + 0.5     # pixel centres
                blobs.append(Blob(colour, (u, w), len(xs), self.on_plane(data, u, w, plane_z)))
        return blobs
