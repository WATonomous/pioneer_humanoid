"""TUM RGB-D loader (depth scale 5000, RGB/depth association)."""
import os

import cv2
import numpy as np

from slam_core.types.frame import Frame

# fx fy cx cy, then k1 k2 p1 p2 k3, from TUM's file_formats page
CAMERAS = {
    "freiburg1": ([517.3, 516.5, 318.6, 255.3], [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]),
    "freiburg3": ([535.4, 539.2, 320.1, 247.6], [0, 0, 0, 0, 0]),
}
DEPTH_SCALE = 5000.0  # TUM depth PNG units per meter

# Reads rgb.txt or depth.txt and returns the timestamps and filenames


def _read_list(path):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    return np.array([float(r[0]) for r in rows]), [r[1] for r in rows]

# returns the right camera from the folder name


def _camera_for(root):
    name = os.path.basename(os.path.normpath(root))
    for cam, (k, d) in CAMERAS.items():
        if cam in name:
            fx, fy, cx, cy = k
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.array(d)
    raise ValueError(f"unknown TUM camera for {name}")


class TUMDataset:
    """Iterate a TUM sequence as Frames. RGB frames with no depth within max_dt are dropped."""

    # pairs every RGB image with its closest depth image in time and keeps the pair if they are withing 15 ms
    def __init__(self, root, max_dt=0.02):
        self.root = root
        self.K, self.dist = _camera_for(root)
        rgb_t, rgb_f = _read_list(os.path.join(root, "rgb.txt"))
        dep_t, dep_f = _read_list(os.path.join(root, "depth.txt"))
        self.pairs = []  # (stamp, rgb file, depth file)
        for t, f in zip(rgb_t, rgb_f):
            j = int(np.argmin(np.abs(dep_t - t)))
            if abs(dep_t[j] - t) <= max_dt:
                self.pairs.append((t, f, dep_f[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        t, f_rgb, f_dep = self.pairs[i]
        img = cv2.imread(os.path.join(self.root, f_rgb), cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(os.path.join(self.root, f_dep), cv2.IMREAD_UNCHANGED)
        depth = raw.astype(np.float32) / DEPTH_SCALE
        return Frame(stamp=t, image=img, depth=depth, K=self.K, dist=self.dist, seq=i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

# reads the TIM true camera poeses for scoring later. It is a seperate function because groundtructh isn't par to f the frame.


def load_groundtruth(root):
    """(N, 8) array: stamp tx ty tz qx qy qz qw, camera pose in the world."""
    return np.loadtxt(os.path.join(root, "groundtruth.txt"))
