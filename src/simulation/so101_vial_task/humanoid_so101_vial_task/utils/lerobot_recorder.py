"""Re-export SimLeRobotRecorder from humanoid_robot_learning."""
from humanoid_robot_learning.sim.sim_recorder import SimLeRobotRecorder as LeRobotRecorder  # noqa: F401

__all__ = ["LeRobotRecorder"]
