"""Load a TUM RGB-D sequence as slam_core Frames and ground-truth poses (the only TUM-specific code)."""
from pathlib import Path

import numpy as np
from PIL import Image
from slam_core.types import Frame
from slam_core.types.pose import from_tum

# Where the prototypes and tests look by default: terrain_core/data/ (parents[1] = the outer terrain_core/ folder)
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "rgbd_dataset_freiburg1_desk"

# TUM freiburg1 camera intrinsics, from the dataset's website
K_FR1 = np.array([[517.3, 0.0, 318.6],
                  [0.0, 516.5, 255.3],
                  [0.0, 0.0, 1.0]])


class TumDataset:
    """One TUM RGB-D sequence folder. Reads depth.txt and groundtruth.txt once, when created."""

    def __init__(self, path, K=K_FR1):
        self.path = Path(path)
        self.K = K
        # depth.txt: one line per depth image, "timestamp filename"; lines starting with # are comments
        self.depth_files = [line.split() for line in (self.path / "depth.txt").read_text().splitlines()
                            if line.strip() and not line.startswith("#")]
        # groundtruth.txt: one pose per line (100 a second, from a motion-capture system):
        # timestamp tx ty tz qx qy qz qw. ndmin=2 keeps it a table even if there is only one line.
        self.groundtruth = np.loadtxt(self.path / "groundtruth.txt", ndmin=2)

    def __len__(self):
        return len(self.depth_files)

    def frame(self, index):
        """Depth image number `index` as a slam_core Frame (depth in metres)."""
        stamp, filename = self.depth_files[index]
        depth = np.array(Image.open(self.path / filename)).astype(np.float32) / 5000.0   # TUM: 5000 = 1 m
        # TR-4 only needs depth; Frame also wants a grayscale image, so give it a blank one.
        return Frame(stamp=float(stamp), image=np.zeros(depth.shape, np.uint8),
                     depth=depth, K=self.K, seq=index)

    def pose_at(self, stamp, max_gap=0.02):
        """Ground-truth camera pose T_world_cam (4x4) at time `stamp` (seconds): the closest line in time."""
        rows = self.groundtruth
        i = np.argmin(np.abs(rows[:, 0] - stamp))            # line with the closest timestamp
        gap = abs(rows[i, 0] - stamp)
        if gap > max_gap:
            raise ValueError(f"no ground-truth pose within {max_gap} s of {stamp} (closest is {gap:.3f} s away)")
        return from_tum(rows[i, 1:8])                        # (tx ty tz qx qy qz qw) -> 4x4
