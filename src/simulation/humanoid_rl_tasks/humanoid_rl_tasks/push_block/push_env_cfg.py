"""Push-block environment: the arm pushes a block along the table, up a ramp,
onto the flat elevated interior floor of an open box.

Box geometry (from ``assets/box/box.obj``, corner-origin, Z-up):
254 x 254 x 31.75 mm shell, interior floor at z = 6.3 mm, walled on three
sides; the fourth side is open with a full-width ramp (table level up to the
floor over a ~29 mm run). The box is spawned with yaw -90 deg so its
up-the-ramp direction (box-local +y) is the env-frame ``+x`` axis.
"""

from __future__ import annotations

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
from pioneer_humanoid.bimanual_arm import (  # noqa: E402
    BIMANUAL_ARM_CFG,
    LEFT_ARM_JOINTS,
    LEFT_EE_BODY,
)


##
# MDP settings
##


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
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # reach + stay behind the block relative to the push direction.
    # Coarse + fine pair (mirrors block_to_target below): the SO101 default
    # pose sits ~0.4-0.5 m from the push point, and a std=0.06-only kernel
    # underflows to exact 0.0 reward (float32) at that range, giving zero
    # gradient to ever learn reaching in the first place.
    ee_behind_block = RewTerm(
        func=mdp.ee_behind_block,
        params={"std": 0.25, "push_offset": 0.05},
        weight=1.5,
    )

    ee_behind_block_fine = RewTerm(
        func=mdp.ee_behind_block,
        params={"std": 0.06, "push_offset": 0.05},
        weight=1.5,
    )

    # dense progress along the up-the-ramp axis (signed velocity)
    push_progress = RewTerm(func=mdp.block_push_progress, params={"max_speed": 0.5}, weight=5.0)

    # extra asymmetric penalty for sliding back down the ramp
    backslide = RewTerm(func=mdp.block_backslide, weight=-8.0)

    # distance to the target point on the interior floor
    block_to_target = RewTerm(
        func=mdp.block_to_target,
        params={"std": 0.15, "target": FLOOR_TARGET},
        weight=8.0,
    )

    block_to_target_fine = RewTerm(
        func=mdp.block_to_target,
        params={"std": 0.05, "target": FLOOR_TARGET},
        weight=5.0,
    )

    # success: settled on the elevated interior floor
    block_on_floor = RewTerm(
        func=mdp.block_on_floor,
        params={
            "x_min": RAMP_TOP_X + BLOCK_HALF,
            "x_max": FLOOR_X_MAX - 0.01,
            "y_half": FLOOR_Y_HALF - 0.025,
            # use the effective collision floor: a settled block rests at center
            # z ~= FLOOR_Z_COLLISION + BLOCK_HALF = 0.0369, ~5 mm above the visual
            # floor, so keying on FLOOR_Z here made success unreachable.
            "floor_z": FLOOR_Z_COLLISION,
            "z_tol": 0.006,
            "rest_speed": 0.05,
        },
        weight=30.0,
    )

    # location-aware anti-scoop: no lifting in open table space
    # (~zero on the ramp and interior floor, where height gain is required)
    scoop_penalty = RewTerm(
        func=mdp.block_scoop_penalty,
        params={
            "ramp_base_x": RAMP_BASE_X,
            "ramp_top_x": RAMP_TOP_X,
            "ramp_base_z": RAMP_BASE_Z,
            "floor_z": FLOOR_Z,
            "box_x_max": FLOOR_X_MAX,
            "box_y_half": FLOOR_Y_HALF,
            "height_tol": 0.01,
            "dist_std": 0.05,
        },
        weight=-40.0,
    )

    # sideways drift off the ramp approach corridor
    lateral_deviation = RewTerm(func=mdp.block_lateral_deviation, params={"y_tol": 0.06}, weight=-10.0)

    # repositioning shaping: route the EE behind an off-corridor block so it can
    # be pushed toward the ramp mouth. Starts inactive (weight 0); the spawn
    # curriculum turns it on only once spawns can land beside/behind the box
    # (stage >= REPOSITION_START_STAGE).
    ee_reposition = RewTerm(
        func=mdp.ee_reposition_behind_block,
        params={
            "std": 0.06,
            "push_offset": 0.05,
            "ramp_mouth": (RAMP_BASE_X, 0.0),
            "corridor_x_max": RAMP_BASE_X,
            "corridor_y": 0.06,
            "gate_std": 0.08,
        },
        weight=0.0,
    )

    # action penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


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
        params={"x_min": 0.05, "x_max": 0.55, "y_limit": 0.20},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 10000}
    )

    # Performance-gated widening of the block spawn offsets. Advances a stage
    # once the rolling success rate (block_on_floor) over the last `window`
    # completed episodes clears `threshold`; also switches on the ee_reposition
    # reward at REPOSITION_START_STAGE. Logs Curriculum/spawn/{stage,
    # success_rate, x_off_hi, reposition_active}.
    spawn = CurrTerm(
        func=mdp.spawn_offset_curriculum,
        params={
            "stages": SPAWN_STAGES,
            "event_term_name": "reset_object_position",
            "success_reward_term": "block_on_floor",
            "window": 100,
            "threshold": 0.70,
            # minimum training before a stage may advance. At num_steps_per_env=24
            # the env step counter grows 24/iteration, so 1200 steps ~= 50 PPO
            # iterations per stage. Without this the 100-episode window refills
            # many times per iteration and the curriculum cascades straight to the
            # last stage before the policy ever trains on the wider spawns.
            "min_stage_steps": 1200,
            "min_episode_len": 2,
            "reposition_reward_term": "ee_reposition",
            "reposition_weight": 1.5,
            "reposition_start_stage": REPOSITION_START_STAGE,
        },
    )


##
# Environment configuration
##


def _ee_frame_cfg(*, debug_vis: bool) -> FrameTransformerCfg:
    """EE proxy at the left wrist link (LEFT_EE_BODY = link6l), zero offset.

    No verified fingertip-center offset from link6l exists as a static frame (the
    teleop code computes it dynamically from both finger tips at runtime, see
    ``pioneer_humanoid.bimanual_arm.compute_gripper_tip_pose_b``). Zero offset is a
    reasonable proxy for the push reward terms; refine with a measured offset if
    pushing behavior looks visibly off from the wrist position.
    """
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
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
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
            init_state=BIMANUAL_ARM_CFG.init_state.replace(pos=ROBOT_BASE_POS),
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
        self.scene.ee_frame = _ee_frame_cfg(debug_vis=True)
