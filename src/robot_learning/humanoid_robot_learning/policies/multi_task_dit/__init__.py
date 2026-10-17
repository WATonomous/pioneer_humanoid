"""Multi-Task DiT (Bryson Jones; LeRobot 0.6 ``policies/multi_task_dit``) for LeRobot 0.4.x.

Configuration and model are upstream with imports changed, plus ``vision_spatial_keypoints`` (off by default). The
processor is rebuilt from 0.4.x steps.
"""
from .configuration_multi_task_dit import MultiTaskDiTConfig

__all__ = ["MultiTaskDiTConfig"]
