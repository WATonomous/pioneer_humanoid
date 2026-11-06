"""ROS-free backend components for the WATO SLAM library.

The package currently exports an incremental GTSAM pose graph that accepts
keyframes and loop constraints. ROS 2 message handling and node lifecycle
belong in the ROS integration layer. IMU factors and graph persistence are
future extensions.
"""

from .pose_graph import PoseGraphBackend

__all__ = ["PoseGraphBackend"]
