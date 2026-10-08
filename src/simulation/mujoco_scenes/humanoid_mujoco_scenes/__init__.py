"""Plain-MuJoCo scenes for the Pioneer arm — one folder per scene.

See humanoid_mujoco_scenes/_register.py for the @scene decorator and how discovery works.
"""
from ._register import (  # noqa: F401
    add_floor, list_scenes, make_model, scene, scene_camera, scene_condition, scene_progress, scene_reset, scene_step,
)
