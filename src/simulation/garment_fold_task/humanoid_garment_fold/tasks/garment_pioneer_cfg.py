"""Garment-fold env config for the WATonomous pioneer_bimanual_arm_8dof.

NEW (WATonomous) -- not vendored. Subclasses the robot-agnostic `GarmentEnvCfg`
and plugs in a single `pioneer_bimanual_arm_8dof` articulation in place of upstream's
two SO101 follower arms.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from humanoid_garment_fold.tasks.garment_env_cfg import GarmentEnvCfg
from pioneer_humanoid.arm_8dof_params import (
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    RIGHT_GRIPPER_JOINTS,
)
from pioneer_humanoid.bimanual_arm_8dof import BIMANUAL_ARM_8DOF_CFG
# CAD-sourced wrist-camera mount (arm_8dof_params.py), not the SO101 offset this
# file originally carried (aimed left_wrist into empty space, buried
# right_wrist in the gripper housing -- see scripts/smoke_test.py).
from pioneer_humanoid.arm_8dof_params import CAMERAS as _PIONEER_CAMERAS
from pioneer_humanoid.arm_params import vertical_aperture as _pioneer_vertical_aperture

__all__ = [
    "GarmentPioneerEnvCfg",
    "LEFT_ARM_JOINTS", "RIGHT_ARM_JOINTS",
    "LEFT_GRIPPER_JOINTS", "RIGHT_GRIPPER_JOINTS",
]

# Fallback Table038 (1.1 x 1.0 m, top at 0.52 m): its near edge 0.2 m in front of the arm base, under the garment.
TABLE038_POS = (0.0, 0.30, 0.0)

_WRIST_RES = (480, 640)  # height, width -- matches the rest of this task's cameras


def _wrist_optics(name: str) -> sim_utils.PinholeCameraCfg:
    cam = _PIONEER_CAMERAS[name]
    h, w = _WRIST_RES
    return sim_utils.PinholeCameraCfg(
        focal_length=cam.focal, horizontal_aperture=cam.h_aperture,
        vertical_aperture=_pioneer_vertical_aperture(name, h, w),
        clipping_range=cam.clip,
    )


_TOP_OPTICS = sim_utils.PinholeCameraCfg(
    focal_length=28.7, focus_distance=400.0, horizontal_aperture=38.11,
    clipping_range=(0.01, 50.0), lock_camera=True,
)


@configclass
class GarmentPioneerEnvCfg(GarmentEnvCfg):
    # 16-dim action, the pioneer_8dof dataset layout: [left 7 arm joints, left gripper,
    # right 7 arm joints, right gripper]; joint-position targets (rad), grippers 0 = open, 1 = closed.
    action_space = 16
    observation_space = 16

    # one articulation, replaces upstream left_robot + right_robot
    robot: ArticulationCfg = BIMANUAL_ARM_8DOF_CFG.replace(prim_path="/World/Robot")

    # Front axis is +X (see bimanual_vial_rack.sh); rotated +90deg about Z to
    # face the garment at world ~(0,0,0.73).
    #
    # Y=-0.40, Z=0.95: reach-tested, gets within ~1cm of the garment vs.
    # ~10-25cm short at the prior (0,-0.63,1.1997). Trade-off: Z=1.1997 is the
    # stand's true floor-standing height but was reach-infeasible from there,
    # so this re-sinks the stand ~25cm into the floor -- accepted, reach over
    # visual placement. (The apartment scene's floor+table are one baked
    # prim, not independently raisable; the Table038 fallback could be, at
    # the cost of the photoreal backdrop -- not done here.)
    robot_base_pos: tuple = (0.0, -0.40, 0.95)
    robot_base_rot: tuple = (0.7071068, 0.0, 0.0, 0.7071068)  # wxyz, +90 deg Z

    # Optional photoreal NuRec backdrop (visual only, no collision). Off by
    # default -- needs the NuRec renderer + manual alignment.
    backdrop_usd_path: str | None = None
    backdrop_pos: tuple = (0.0, 0.0, 0.0)
    backdrop_scale: float = 1.0

    # --- cameras: wrist cams on the pioneer wrist links, top cam world-fixed ---
    left_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/left_link7/left_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=_PIONEER_CAMERAS["wrist_left"].pos,
            rot=_PIONEER_CAMERAS["wrist_left"].rot,
            convention="opengl",
        ),
        data_types=["rgb"], spawn=_wrist_optics("wrist_left"),
        width=_WRIST_RES[1], height=_WRIST_RES[0], update_period=1 / 30.0,
    )
    right_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/right_link7/right_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=_PIONEER_CAMERAS["wrist_right"].pos,
            rot=_PIONEER_CAMERAS["wrist_right"].rot,
            convention="opengl",
        ),
        data_types=["rgb"], spawn=_wrist_optics("wrist_right"),
        width=_WRIST_RES[1], height=_WRIST_RES[0], update_period=1 / 30.0,
    )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/TopCam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 1.1), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"
        ),
        data_types=["rgb", "depth"], spawn=_TOP_OPTICS,
        width=640, height=480,
    )
    # extra fixed camera for an external "is the arm placed right" view (dev only)
    scene_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/SceneCam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.8, -2.0, 1.3), rot=(1.0, 0.0, 0.0, 0.0), convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0, focus_distance=400.0, horizontal_aperture=30.0,
            clipping_range=(0.05, 60.0),
        ),
        width=960, height=640,
    )
