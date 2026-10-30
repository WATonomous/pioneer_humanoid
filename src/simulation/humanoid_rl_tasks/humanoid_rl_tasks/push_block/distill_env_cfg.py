"""Distillation env: privileged teacher obs + camera student obs for push-block."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .push_env_cfg import PushBlockEnvCfg, PushBlockEnvCfg_PLAY
from .push_env_cfg import (
    BOX_EXCLUSION,
    DR_JOINT_POS_NOISE,
    DR_JOINT_VEL_NOISE,
    EventCfg,
    FLOOR_TARGET,
    FLOOR_Z,
    RAMP_BASE_X,
    RAMP_TOP_X,
    SPAWN_RAMP_TARGET,
)
from pioneer_humanoid.arm_params import CAMERAS, vertical_aperture  # noqa: E402

# Student cameras are the robot's own ego and left-wrist cameras (arm_params.CAMERAS), rendered at this size.
# 4:3 is required by the sensor calibration. Block is ~9 px wide in the ego view at 160x120.
STUDENT_IMG_WIDTH = 160
STUDENT_IMG_HEIGHT = 120
# Noise on the student's end-effector position (m): it is computed from the joint encoders, so it inherits their error (~0.01 rad x ~0.4 m).
STUDENT_EE_POS_NOISE = 0.005


# Colour ranges around each object's real colour: table white plastic, cube red, tray blue.
VISION_MATERIAL_TARGETS = {
    "/TableCollision": dict(tint=((0.85, 0.85, 0.85), (1.0, 1.0, 1.0)), roughness=(0.1, 0.5), metallic=(0.0, 0.3)),  # the asset's own 0.5 metallic renders gray
    "/Object": dict(tint=((0.75, 0.05, 0.05), (0.95, 0.20, 0.20)), roughness=(0.3, 0.8), metallic=(0.0, 0.05)),
    "/Box": dict(tint=((0.05, 0.25, 0.70), (0.25, 0.45, 0.95)), roughness=(0.3, 0.8), metallic=(0.0, 0.05)),
}


def _tiled_camera_cfg(name: str) -> TiledCameraCfg:
    """TiledCameraCfg for robot camera ``name``, with the same mount and lens as bimanual_arm.make_camera_cfg.

    Tiled: one batched render for all envs. Per-env CameraCfg cameras exhaust GPU render resources at 64 envs."""
    cam = CAMERAS[name]
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + f"{cam.body}/{cam.prim}",
        offset=TiledCameraCfg.OffsetCfg(pos=cam.pos, rot=cam.rot, convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cam.focal,
            horizontal_aperture=cam.h_aperture,
            vertical_aperture=vertical_aperture(name, STUDENT_IMG_HEIGHT, STUDENT_IMG_WIDTH),
            clipping_range=cam.clip,
        ),
        width=STUDENT_IMG_WIDTH,
        height=STUDENT_IMG_HEIGHT,
        data_types=["rgb"],
        update_period=0.0,
    )


def _add_student_cameras(scene) -> None:
    scene.ego_cam = _tiled_camera_cfg("ego")
    scene.wrist_cam = _tiled_camera_cfg("wrist_left")


@configclass
class DistillObservationsCfg:
    """Teacher = privileged policy obs; student = proprio + RGB (no block state)."""

    @configclass
    class TeacherCfg(ObsGroup):
        """Privileged observations matching the trained PPO teacher (37-D)."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        block_position = ObsTerm(func=mdp.block_center_in_robot_root_frame)
        block_orientation = ObsTerm(func=mdp.block_orientation)
        block_lin_vel = ObsTerm(func=mdp.block_lin_vel)
        ramp_geometry = ObsTerm(
            func=mdp.ramp_geometry,
            params={
                "ramp_base_x": RAMP_BASE_X,
                "ramp_top_x": RAMP_TOP_X,
                "floor_z": FLOOR_Z,
                "target": FLOOR_TARGET,
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StudentCfg(ObsGroup):
        """Proprioceptive student terms (no privileged block state), with sensor noise. The teacher's obs stay clean."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-DR_JOINT_POS_NOISE, n_max=DR_JOINT_POS_NOISE))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-DR_JOINT_VEL_NOISE, n_max=DR_JOINT_VEL_NOISE))
        ee_position = ObsTerm(
            func=mdp.ee_position_in_robot_root_frame,
            noise=Unoise(n_min=-STUDENT_EE_POS_NOISE, n_max=STUDENT_EE_POS_NOISE),
        )
        ramp_geometry = ObsTerm(
            func=mdp.ramp_geometry,
            params={
                "ramp_base_x": RAMP_BASE_X,
                "ramp_top_x": RAMP_TOP_X,
                "floor_z": FLOOR_Z,
                "target": FLOOR_TARGET,
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class StudentRgbCfg(ObsGroup):
        """RGB images (ego + left wrist) for the student vision encoders."""

        ego_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ego_cam"), "data_type": "rgb", "normalize": True},
        )
        wrist_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": True},
        )

        def __post_init__(self):
            self.enable_corruption = False
            # Keep spatial dims — do not flatten the images into a vector.
            self.concatenate_terms = False

    # Keep a `policy` alias so tools that expect it still see teacher-sized obs.
    policy: TeacherCfg = TeacherCfg()
    teacher: TeacherCfg = TeacherCfg()
    student: StudentCfg = StudentCfg()
    student_rgb: StudentRgbCfg = StudentRgbCfg()


@configclass
class DistillEventCfg(EventCfg):
    """Push reset events + vision DR (camera / light / materials+textures)."""

    # Mounting-tolerance jitter around the real camera mounts (guesses until the real tolerance is known).
    vision_camera_ego = EventTerm(
        func=mdp.randomize_robot_camera_pose,
        mode="reset",
        params={
            "camera_cfg": SceneEntityCfg("ego_cam"),
            "mount": "ego",
            "pos_range": (-0.01, 0.01),
            "rot_deg_range": (-2.0, 2.0),
        },
    )

    vision_camera_wrist = EventTerm(
        func=mdp.randomize_robot_camera_pose,
        mode="reset",
        params={
            "camera_cfg": SceneEntityCfg("wrist_cam"),
            "mount": "wrist_left",
            "pos_range": (-0.001, 0.001),
            "rot_deg_range": (-2.0, 2.0),
        },
    )

    vision_light = EventTerm(
        func=mdp.randomize_dome_light,
        mode="reset",
        params={
            "intensity_range": (1500.0, 4500.0),
            "color_scale_range": (0.85, 1.15),
        },
    )

    vision_materials = EventTerm(
        func=mdp.randomize_visual_materials,
        mode="reset",
        params={"targets": VISION_MATERIAL_TARGETS},
    )


@configclass
class PushBlockDistillEnvCfg(PushBlockEnvCfg):
    """Push-block env with the robot's ego and left-wrist cameras for vision distillation (bimanual arm)."""

    observations: DistillObservationsCfg = DistillObservationsCfg()
    events: DistillEventCfg = DistillEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Cameras are expensive; keep this small for the minimal distill loop.
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5
        self.sim.render_interval = self.decimation
        _add_student_cameras(self.scene)
        # Distillation runs at full difficulty from iteration 0: no DR / spawn ramps, final spawn range.
        self.curriculum.dr_ramp = None
        self.curriculum.spawn_ramp = None
        self.events.reset_object_position.params["pose_range"] = dict(SPAWN_RAMP_TARGET)
        self.events.reset_object_position.params["box_exclusion"] = BOX_EXCLUSION


@configclass
class PushBlockDistillEnvCfg_PLAY(PushBlockEnvCfg_PLAY):
    """Play/eval variant of the distillation env (vision DR off for clean eval)."""

    observations: DistillObservationsCfg = DistillObservationsCfg()
    # Keep base EventCfg (no vision DR) for deterministic play.
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.sim.render_interval = self.decimation
        self.observations.student.enable_corruption = False  # clean proprio for eval
        # the EE frame marker would be drawn into the student's camera images
        self.scene.ee_frame.debug_vis = False
        _add_student_cameras(self.scene)
