"""TR-4 in one call: a depth frame in, a cleaned point cloud in odom out."""
from dataclasses import dataclass

import numpy as np

from .crop import in_depth_range
from .depth import depth_to_points
from .downsample import voxel_downsample
from .transform import transform_points


@dataclass
class PreprocessConfig:
    """Settings for preprocess(). The team's values live in config/terrain.yaml."""
    z_min: float = 0.3          # m: closer than this the D455 can't measure depth
    z_max: float = 3.0          # m: farther than this its depth is too noisy
    voxel_size: float = 0.02    # m: keep at most one point per cube of this size

    @classmethod
    def from_yaml(cls, path):
        """Read the `preprocess:` section of a yaml file (missing keys keep the defaults)."""
        import yaml
        with open(path) as fh:
            settings = yaml.safe_load(fh) or {}
        return cls(**(settings.get("preprocess") or {}))


@dataclass
class Cloud:
    """A point cloud and the time it was measured."""
    stamp: float          # seconds: the depth image's timestamp
    points: np.ndarray    # (N, 3) float32 points in metres, in odom


def preprocess(frame, T_odom_cam, config=None):
    """Depth image -> 3D points -> crop -> voxel downsample -> odom frame.

    frame:      slam_core Frame with depth in metres and intrinsics K.
    T_odom_cam: 4x4 pose of the camera in odom, at frame.stamp.
    config:     PreprocessConfig; None uses the defaults.
    """
    config = config or PreprocessConfig()
    points = depth_to_points(frame.depth, frame.K)
    points = points[in_depth_range(points, config.z_min, config.z_max)]
    points = voxel_downsample(points, config.voxel_size)
    points_odom = transform_points(T_odom_cam, points).astype(np.float32)
    return Cloud(stamp=frame.stamp, points=points_odom)
