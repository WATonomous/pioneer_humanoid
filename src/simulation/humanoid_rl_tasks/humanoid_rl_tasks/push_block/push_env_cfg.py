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
    FLOOR_Z_COLLISION,
    FULL_YAW,
    PUSH_DIR,
    RAMP_BASE_X,
    RAMP_BASE_Z,
    RAMP_TOP_X,
    REPOSITION_START_STAGE,
    ROBOT_BASE_POS,
    SPAWN_STAGES,
    TABLE_TOP_Z,
    PushBlockSceneCfg,
)

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


# ══ DOMAIN RANDOMIZATION KNOBS (turn these; the events below read them) ══════════════════════
# "scale" ranges multiply the nominal value (1.0 = unchanged). Friction is absolute (coefficient).
# Nominal friction of the block, fingers and table was not checked, so the friction ranges are a first guess.
DR_BLOCK_STATIC_FRICTION = (0.4, 1.0)
DR_BLOCK_DYNAMIC_FRICTION = (0.3, 0.8)  # kept <= static by make_consistent
DR_FINGER_STATIC_FRICTION = (0.5, 1.2)
DR_FINGER_DYNAMIC_FRICTION = (0.4, 1.0)
DR_RESTITUTION = (0.0, 0.05)
DR_BLOCK_MASS_SCALE = (0.8, 1.2)
DR_ARM_LINK_MASS_SCALE = (0.9, 1.2)  # heavier links -> more gravity load -> more sag
DR_ARM_STIFFNESS_SCALE = (0.6, 1.2)  # softer than nominal -> more sag (real MIT gains are untuned bench defaults)
DR_ARM_DAMPING_SCALE = (0.8, 1.2)
DR_JOINT_POS_NOISE = 0.01  # rad, observation noise (training only)
DR_JOINT_VEL_NOISE = 0.2  # rad/s, observation noise (training only)
# Mass, actuator gains and observation noise ramp linearly from none (nominal) at DR_RAMP_START_ITER to the full
# ranges above at DR_RAMP_END_ITER. Friction is NOT ramped: it is sampled once at startup, so it is full from iteration 0.
DR_RAMP_START_ITER = 100
DR_RAMP_END_ITER = 200
PPO_STEPS_PER_ITER = 24  # = num_steps_per_env in agents/rsl_rl_ppo_cfg.py
# The env step counter restarts at 0 when a run is resumed. RESUME_ITER=<iteration> keeps the ramps where they were.
RESUME_ITER = int(os.environ.get("RESUME_ITER", "0"))
# Block spawn range widens linearly from SPAWN_STAGES[0] to SPAWN_RAMP_TARGET (scene.py), after the DR ramp has finished.
SPAWN_RAMP_START_ITER = 200
SPAWN_RAMP_END_ITER = 300
SPAWN_RAMP_TARGET = SPAWN_STAGES[1]
# Event names the play configs switch off so you watch the policy on nominal physics.
DR_EVENT_NAMES = ("dr_block_material", "dr_finger_material", "dr_block_mass", "dr_arm_mass", "dr_arm_gains")
# ═════════════════════════════════════════════════════════════════════════════════════════════


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

    # ── domain randomization (values come from the DR_* knobs above) ──────────────
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
        params={"temp": 15.0, "push_offset": 0.05},
        weight=1,
    )

    block_to_target = RewTerm(
        func=mdp.block_to_target,
        params={"temp": 5.0, "target": FLOOR_TARGET},
        weight=2,
    )

    action_smoothness = RewTerm(func=mdp.action_smoothness, params={"temp": 0.5}, weight=-0.1)


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
            "floor_z": FLOOR_Z_COLLISION,
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

    scene: PushBlockSceneCfg = PushBlockSceneCfg(num_envs=1024, env_spacing=2.5)
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
        self.viewer.eye = (2.6, 2.6, 2.3)
        self.viewer.lookat = (0.4, 0.0, TABLE_TOP_Z)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # PhysX warned it needed ~16400 for 4096 envs and dropped interactions;
        # give real headroom since randomized block spawns/yaw create varied contacts.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 128 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

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
    # GUI CRASH WORKAROUND (interactive play/inspection only; does NOT affect
    # headless training): playing this task in a live GUI window aborts ~29 s
    # after the scene renders with a native Kit assertion
    #   carb::thread::detail::BaseMutex::unlock(): "unlock() called by
    #   non-owning thread" (carb/delegate Mutex.h:158)
    # -- a cross-thread race in Kit's tasking plugin triggered by something
    # bimanual-arm-specific. Root-cause fix is still TODO; until then launch
    # play.py with Kit forced to a single tasking thread, which serializes the
    # race away (measured cost: startup ~16 s vs ~11 s, no runtime impact):
    #
    #   ./isaaclab.sh -p .../rsl_rl_scripts/play.py \
    #       --task Isaac-Bimanual-Push-Block-Play-v0 --num_envs 4 \
    #       --checkpoint <run>/model_<N>.pt \
    #       "--kit_args=--/plugins/carb.tasking.plugin/threadCount=1"
    #
    # NOTE the "=" form ("--kit_args=--/..."): with a space, argparse mistakes
    # the "--/..." value for a new flag and errors. It is a Kit *launch* arg, so
    # it cannot live in the env cfg -- it must be passed on the CLI.
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        for name in DR_EVENT_NAMES:
            setattr(self.events, name, None)
        self.curriculum.dr_ramp = None
        self.curriculum.spawn_ramp = None
        self.scene.ee_frame = _ee_frame_cfg(debug_vis=True)
