"""Load TUM RGB-D frames as slam_core Frames (the only TUM-specific code)."""
from pathlib import Path

import numpy as np
from PIL import Image
from slam_core.types import Frame

# terrain/data/..., found relative to this file (parents[1] = the terrain/ folder)
DATA = Path(__file__).resolve().parents[1] / "data" / "rgbd_dataset_freiburg1_desk"

# TUM freiburg1 camera intrinsics, from the dataset's website
K_FR1 = np.array([[517.3, 0.0, 318.6],
                  [0.0, 516.5, 255.3],
                  [0.0, 0.0, 1.0]])


def load_tum_frame(index):
    """Load depth image number `index` from TUM as a slam_core Frame."""
    lines = [line.split() for line in (DATA / "depth.txt").read_text().splitlines()
             if not line.startswith("#")]
    stamp, filename = lines[index]
    depth = np.array(Image.open(DATA / filename)).astype(np.float32) / 5000.0   # TUM: 5000 = 1 m
    # TR-4 only needs depth; Frame also wants a grayscale image, so give it a blank one.
    return Frame(stamp=float(stamp), image=np.zeros(depth.shape, np.uint8),
                 depth=depth, K=K_FR1, seq=index)
