"""arm_pose_io without importing the whole of humanoid_robot_learning (core image: no LeRobot)."""
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[3] / "robot_learning" / "humanoid_robot_learning" / "recording" / "arm_pose_io.py"
_spec = importlib.util.spec_from_file_location("arm_pose_io", _PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
vector_to_arm_pose = _module.vector_to_arm_pose
