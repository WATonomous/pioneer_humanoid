"""Push-block environment: the arm pushes a block along the table, up a ramp,
onto the flat elevated interior floor of an open box.

Box geometry (from ``assets/box/box.obj``, corner-origin, Z-up):
254 x 254 x 31.75 mm shell, interior floor at z = 6.3 mm, walled on three
sides; the fourth side is open with a full-width ramp (table level up to the
floor over a ~29 mm run). The box is spawned with yaw -90 deg so its
up-the-ramp direction (box-local +y) is the env-frame ``+x`` axis.
"""

from __future__ import annotations

import os
import sys
from dataclasses import MISSING
from pathlib import Path

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .scene import (  # noqa: F401  (BOX_* / FLOOR_TARGET re-exported for distill_env_cfg)
    BLOCK_DROP_MIN_Z,
    BLOCK_HALF,
    BLOCK_INIT_POS,
    BOX_EXCLUSION,
    BOX_POS,
    BOX_QUAT,
    FLOOR_TARGET,
    FLOOR_X_MAX,
    FLOOR_Y_HALF,
    FLOOR_Z,
    FULL_YAW,
    PUSH_DIR,
    RAMP_BASE_X,
    RAMP_BASE_Z,
    RAMP_TOP_X,
    REPOSITION_START_STAGE,
    ROBOT_BASE_POS,
    SPAWN_STAGES,
    TABLE_TOP_Z,
    PushBlockLabSceneCfg,
    PushBlockSceneCfg,
)
from ..workcell import LAB_WORKCELL_CAMERA, RENDER_CARB_SETTINGS

# The canonical pioneer bimanual-arm config (editable-installed in the isaac_lab
# image; the sys.path entry is a fallback for a bare bind-mounted checkout).
_PIONEER = Path(__file__).resolve().parents[4] / "pioneer_humanoid"
if str(_PIONEER) not in sys.path:
    sys.path.insert(0, str(_PIONEER))
# Uses canonical URDF-side names directly (formerly aliased LEFT_* as RIGHT_* from an old, since-fixed URDF flip).
from pioneer_humanoid.arm_params import LEFT_GRIPPER_TIP_OFFSET  # noqa: E402
from pioneer_humanoid.urdf_joint_limits import JOINT_POS_LIMITS  # noqa: E402
from pioneer_humanoid.bimanual_arm import (  # noqa: E402
    BIMANUAL_ARM_CFG,
    LEFT_ARM_JOINTS,
    LEFT_EE_BODY,
    LEFT_FINGER_TIP_BODIES,
)


##
# MDP settings
##


# DR knobs read by the events below: scale ranges multiply the nominal value, friction is absolute (first guesses, nominal unchecked).
DR_BLOCK_STATIC_FRICTION = (0.4, 1.0)
DR_BLOCK_DYNAMIC_FRICTION = (0.3, 0.8)  # kept <= static
DR_FINGER_STATIC_FRICTION = (0.5, 1.2)
DR_FINGER_DYNAMIC_FRICTION = (0.4, 1.0)
DR_RESTITUTION = (0.0, 0.05)
DR_BLOCK_MASS_SCALE = (0.8, 1.2)
DR_ARM_LINK_MASS_SCALE = (0.9, 1.2)
DR_ARM_STIFFNESS_SCALE = (0.6, 1.2)  # real MIT gains are untuned bench defaults
DR_ARM_DAMPING_SCALE = (0.8, 1.2)
DR_JOINT_POS_NOISE = 0.01  # rad
DR_JOINT_VEL_NOISE = 0.2  # rad/s
# Mass, gains and obs noise ramp from nominal over these iterations; friction is sampled once at startup, so it is not ramped.
DR_RAMP_START_ITER = 100
DR_RAMP_END_ITER = 200
PPO_STEPS_PER_ITER = 24  # = num_steps_per_env in agents/rsl_rl_ppo_cfg.py
# Resuming restarts the env step counter; RESUME_ITER=<iteration> keeps the ramps in place.
RESUME_ITER = int(os.environ.get("RESUME_ITER", "0"))
# Spawn range widens from SPAWN_STAGES[0] to SPAWN_RAMP_TARGET after the DR ramp.
SPAWN_RAMP_START_ITER = 200
SPAWN_RAMP_END_ITER = 300
SPAWN_RAMP_TARGET = SPAWN_STAGES[1]


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Only the arm is actuated. The gripper joint is intentionally excluded from
    the action space: with no action term writing targets, the PD actuator
    holds it at its default (closed) position, so the policy physically cannot
    grasp the block - the closed gripper is the pushing tool.
    """

    arm_action: mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-DR_JOINT_POS_NOISE, n_max=DR_JOINT_POS_NOISE))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-DR_JOINT_VEL_NOISE, n_max=DR_JOINT_VEL_NOISE))
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
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    reset_object_position = EventTerm(
        func=mdp.reset_block_center_uniform,
        mode="reset",
        params={
            # Position + full yaw randomization of the block CENTER, as offsets
            # around the default center (0.21, 0.0). The spawn_curriculum term
            # overwrites this range in place, widening it stage by stage; the
            # value here is stage 0 (current moderate range) and is what a run
            # starts from. Box footprint is excluded via reject-and-resample so
            # widened offsets never spawn the block inside the walls/floor.
            "pose_range": dict(SPAWN_STAGES[0]),
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
            "box_exclusion": BOX_EXCLUSION,
        },
    )

    # domain randomization
    dr_block_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": DR_BLOCK_STATIC_FRICTION,
            "dynamic_friction_range": DR_BLOCK_DYNAMIC_FRICTION,
            "restitution_range": DR_RESTITUTION,
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    dr_finger_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(LEFT_FINGER_TIP_BODIES)),
            "static_friction_range": DR_FINGER_STATIC_FRICTION,
            "dynamic_friction_range": DR_FINGER_DYNAMIC_FRICTION,
            "restitution_range": DR_RESTITUTION,
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    dr_block_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": DR_BLOCK_MASS_SCALE,
            "operation": "scale",
        },
    )

    dr_arm_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": DR_ARM_LINK_MASS_SCALE,
            "operation": "scale",
        },
    )

    # arm joints only: the gripper keeps its nominal hold stiffness
    dr_arm_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS),
            "stiffness_distribution_params": DR_ARM_STIFFNESS_SCALE,
            "damping_distribution_params": DR_ARM_DAMPING_SCALE,
            "operation": "scale",
        },
    )

    check_gripper_tip_offset = EventTerm(func=mdp.check_gripper_tip_offset, mode="startup")

    # every 10 PPO iterations: 10 * 24 steps * 0.02 s
    recheck_gripper_tip_offset = EventTerm(
        func=mdp.check_gripper_tip_offset,
        mode="interval",
        interval_range_s=(4.8, 4.8),
        is_global_time=True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP: reach behind the block, block to goal, smooth actions."""

    ee_behind_block = RewTerm(
        func=mdp.ee_behind_block,
        params={"temp": 15.0, "push_offset": 0.065, "target": FLOOR_TARGET, "angle_gain": 2.0, "max_angle_deg": 80.0},
        weight=1,
    )

    block_to_target = RewTerm(
        func=mdp.block_to_target,
        params={"temp": 5.0, "target": FLOOR_TARGET},
        weight=2,
    )

    action_smoothness = RewTerm(func=mdp.action_smoothness, params={"temp": 0.5}, weight=-0.2)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": BLOCK_DROP_MIN_Z, "asset_cfg": SceneEntityCfg("object")},
    )

    block_off_course = DoneTerm(
        func=mdp.block_off_course,
        params={"x_min": 0.05, "x_max": 0.8, "y_limit": 0.30},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP: the domain randomization ramp, then the spawn-range ramp."""

    dr_ramp = CurrTerm(
        func=mdp.dr_ramp,
        params={
            "start_iter": DR_RAMP_START_ITER,
            "end_iter": DR_RAMP_END_ITER,
            "steps_per_iter": PPO_STEPS_PER_ITER,
            "iter_offset": RESUME_ITER,
            "event_ranges": {
                "dr_block_mass": {"mass_distribution_params": (DR_BLOCK_MASS_SCALE, 1.0)},
                "dr_arm_mass": {"mass_distribution_params": (DR_ARM_LINK_MASS_SCALE, 1.0)},
                "dr_arm_gains": {
                    "stiffness_distribution_params": (DR_ARM_STIFFNESS_SCALE, 1.0),
                    "damping_distribution_params": (DR_ARM_DAMPING_SCALE, 1.0),
                },
            },
            "noise_terms": {"joint_pos": DR_JOINT_POS_NOISE, "joint_vel": DR_JOINT_VEL_NOISE},
        },
    )

    success = CurrTerm(
        func=mdp.success_rate_logger,
        params={
            "window": 2000,
            "x_min": RAMP_TOP_X + BLOCK_HALF,
            "x_max": FLOOR_X_MAX,
            "y_half": FLOOR_Y_HALF - BLOCK_HALF,
            "floor_z": FLOOR_Z,
            "z_tol": 0.006,
            "rest_speed": 0.05,
        },
    )

    spawn_ramp = CurrTerm(
        func=mdp.spawn_ramp,
        params={
            "start_iter": SPAWN_RAMP_START_ITER,
            "end_iter": SPAWN_RAMP_END_ITER,
            "steps_per_iter": PPO_STEPS_PER_ITER,
            "iter_offset": RESUME_ITER,
            "event_term_name": "reset_object_position",
            "start_range": SPAWN_STAGES[0],
            "end_range": SPAWN_RAMP_TARGET,
        },
    )


##
# Environment configuration
##


def _ee_frame_cfg(*, debug_vis: bool) -> FrameTransformerCfg:
    """EE point: the fingertip midpoint, as a fixed offset from the left wrist link (LEFT_EE_BODY = link6l)."""
    marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer/ee_tcp")
    marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=debug_vis,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + LEFT_EE_BODY,
                name="end_effector",
                offset=OffsetCfg(pos=LEFT_GRIPPER_TIP_OFFSET, rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        ],
    )


@configclass
class PushBlockEnvCfg(ManagerBasedRLEnvCfg):
    """Push-block env: pioneer bimanual arm (left arm) drives a block up the ramp.

    Scene, geometry and grounding come from ``scene.PushBlockSceneCfg`` (the
    teleop-verified placement, shared with ``humanoid_isaac_scenes``).
    Only the arm is actuated; the closed gripper is the pushing tool.
    """

    # Full lab (room + overhead lighting). PushBlockSceneCfg is the simple one: table, lightbox and robot only.
    scene: PushBlockLabSceneCfg = PushBlockLabSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # GUI view from inside the lab (the room's back wall is at x = 1.10; env_0 is the one enclosed by the room)
        self.viewer.eye, self.viewer.lookat = LAB_WORKCELL_CAMERA
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        self.sim.render.carb_settings = dict(RENDER_CARB_SETTINGS)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # PhysX warned it needed ~16400 for 4096 envs and dropped interactions;
        # give real headroom since randomized block spawns/yaw create varied contacts.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 128 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # the exact-mesh ramp makes block-vs-ramp contacts overflow the 2**26 default at 8192 envs (needed ~69M, so ~138M at 16384)
        self.sim.physx.gpu_collision_stack_size = 2**28
        # contact patches overflow the 5 * 2**15 default at 16384 envs (needed ~166k)
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**16

        # ── bimanual arm ────────────────────────────────────────────────────
        self.scene.replicate_physics = True
        self.scene.robot = BIMANUAL_ARM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=BIMANUAL_ARM_CFG.init_state.replace(
                pos=ROBOT_BASE_POS,
                joint_pos={
                    **BIMANUAL_ARM_CFG.init_state.joint_pos,
                    "joint7l": JOINT_POS_LIMITS["joint7l"][0],
                    "joint8l": JOINT_POS_LIMITS["joint8l"][1],
                },
            ),
        )
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_ARM_JOINTS,
            scale=0.5,
            use_default_offset=True,
        )
        self.scene.ee_frame = _ee_frame_cfg(debug_vis=False)


@configclass
class PushBlockEnvCfg_PLAY(PushBlockEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        # Full detail in the viewer: ray-traced reflections and bounce lighting (the student play env goes back to the
        # training render mode, see distill_env_cfg).
        self.sim.render.rendering_mode = "quality"
        # Full training difficulty: all DR, obs noise, final spawn range, no ramps.
        self.curriculum.dr_ramp = None
        self.curriculum.spawn_ramp = None
        self.events.reset_object_position.params["pose_range"] = dict(SPAWN_RAMP_TARGET)
        self.scene.ee_frame = _ee_frame_cfg(debug_vis=True)
