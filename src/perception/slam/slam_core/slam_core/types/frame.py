"""Frame is one synchronized image + depth snapshot, the input to VO.

Every source (D455 live or recorded, TUM, EuRoC, our ROS 2 bags, sim) is turned
into Frames by a loader in slam_core.io, so VO never knows where data came from.
Loaders do the unit and format conversions; Frame checks they got them right.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# One camera frame. depth is aligned to image pixel for pixel (on the D455 the
# left IR camera is the depth reference frame, so no alignment step is needed).
@dataclass
class Frame:
    stamp: float                        # seconds
    image: np.ndarray                   # (H, W) uint8 grayscale (left IR on the D455)
    depth: Optional[np.ndarray]         # (H, W) float32 meters, 0 = no depth. None if the source has none (EuRoC)
    K: np.ndarray                       # 3x3 intrinsics [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))  # OpenCV distortion (k1 k2 p1 p2 k3 ...)
    seq: int = -1                       # frame index in the source, -1 if unknown
    mask: Optional[np.ndarray] = None   # (H, W) bool, True = usable pixel (e.g. robot arm masked out)

    # Validate on construction. A wrong unit (mm instead of m) or a swapped axis here
    # would otherwise show up much later as a wrong scale or a bad pose.
    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64)
        self.dist = np.asarray(self.dist, dtype=np.float64).ravel()

        if not np.isfinite(self.stamp):
            raise ValueError(f"frame {self.seq}: stamp must be finite")

        img = self.image
        if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
            raise ValueError(f"frame {self.seq}: image must be (H, W) uint8 grayscale, "
                             f"got {getattr(img, 'shape', None)} {getattr(img, 'dtype', type(img))}")
        h, w = img.shape

        d = self.depth
        if d is not None:
            if not isinstance(d, np.ndarray) or not np.issubdtype(d.dtype, np.floating):
                # catches raw sensor depth (uint16 mm, or TUM's 1/5000 m) passed without conversion
                raise ValueError(f"frame {self.seq}: depth must be float meters, got "
                                 f"{getattr(d, 'dtype', type(d))}; divide by the depth scale first")
            if d.shape != (h, w):
                raise ValueError(f"frame {self.seq}: depth shape {d.shape} != image shape {(h, w)}")
            if not np.all(np.isfinite(d)) or np.any(d < 0):
                raise ValueError(f"frame {self.seq}: depth must be finite and >= 0 (use 0 for no depth)")
            self.depth = d.astype(np.float32, copy=False)

        K = self.K
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            raise ValueError(f"frame {self.seq}: K must be a finite 3x3 matrix")
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
            raise ValueError(f"frame {self.seq}: K must have fx, fy > 0 and last row 0 0 1")

        if self.mask is not None:
            m = np.asarray(self.mask)
            if m.shape != (h, w) or m.dtype != bool:
                raise ValueError(f"frame {self.seq}: mask must be (H, W) bool matching the image")
            self.mask = m

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def has_depth(self):
        return self.depth is not None
