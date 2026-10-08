"""Plain-MuJoCo scenes for the Pioneer arm — one folder per scene.

See humanoid_mujoco_scenes/_register.py for the @scene decorator and how discovery works.
"""
from ._register import add_floor, list_scenes, make_model, scene, scene_camera, scene_progress, scene_reset, scene_step  # noqa: F401
from .workcell import (  # noqa: F401
    ROBOT_BASE_POS,
    TABLE_TOP_Z,
    TABLE_X_MAX,
    TABLE_X_MIN,
    TABLE_Y_HALF,
    WORKCELL_CAMERA,
    add_lightbox_workcell,
)
