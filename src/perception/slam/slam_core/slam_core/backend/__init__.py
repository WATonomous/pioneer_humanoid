"""Backend package for the WATO SLAM library.

Planned contents:
    - Export the reusable backend classes implemented in this package.
    - Keep ROS 2 message handling and node lifecycle outside this ROS-free
      library package.

The first backend class will be the BE-3 incremental pose graph. Later work
can add loop constraints, IMU factors, graph persistence, and other backend
components without changing the public import path.
"""

from .pose_graph import PoseGraphBackend

__all__ = ["PoseGraphBackend"]
