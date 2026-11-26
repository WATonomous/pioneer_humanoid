"""ROS-free backend components for the WATO SLAM library.

PoseGraphBackend is an incremental GTSAM (iSAM2) pose graph that takes keyframes, loop constraints and,
optionally, IMU data (preintegrated IMU factors and gravity leveling). Feed IMU either as a whole recording
per add_keyframe call (replay) or live with add_imu. ROS 2 message handling and node lifecycle belong in the
ROS integration layer. Graph persistence is a future extension.
"""

from .pose_graph import ImuNotReadyError, PoseGraphBackend

__all__ = ["ImuNotReadyError", "PoseGraphBackend"]
