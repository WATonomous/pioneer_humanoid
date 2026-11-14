from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from pioneer_humanoid.whole_body import WHOLE_BODY_HUMANOID_CFG

from ... import mdp
from ...locomotion_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, TerminationsCfg


BASE_BODY = "base"
FOOT_BODIES = ["Foot_L", "Foot_R"]
LEG_JOINTS = ["Hip_.*", "Knee_.*", "Ankle_.*"]
ANKLE_JOINTS = ["Ankle_R_.*", "Ankle_P_.*"]


@configclass
class PioneerHumanoidRewards(RewardsCfg):
    """Reward terms for the Pioneer humanoid V1 locomotion MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINTS)},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Hip_A_.*", "Hip_R_.*"])},
    )
    # Shared across rough and flat: with self_collision disabled, nothing physically
    # stops the legs from passing through each other -- this is a Wato-specific
    # morphology issue (G1 doesn't need an equivalent term), not something tied to
    # flat terrain specifically. See flat_env_cfg.py history for the weight/margin
    # tuning (linear clamp -> squared penalty, to avoid a stutter from the sharp
    # gradient at a hard margin boundary).
    # preserve_order=True: feet_crossing_l2 assumes body_ids[0]=left foot,
    # body_ids[1]=right foot -- SceneEntityCfg defaults to asset body index order
    # otherwise, which isn't guaranteed to match FOOT_BODIES = ["Foot_L", "Foot_R"].
    feet_crossing_penalty = RewTerm(
        func=mdp.feet_crossing_l2,
        weight=-40.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
            "margin": 0.08,
        },
    )
    # Rough-only (disabled on flat, which has no height_scanner -- see
    # PioneerHumanoidFlatEnvCfg.__post_init__): rough_terrain_fresh_001 plateaued with
    # the robot spawning and immediately sitting/crouching down and staying there for
    # the whole episode -- it turns out the shared TerminationsCfg (locomotion_env_cfg.py)
    # only has time_out and base_contact (a contact-force check on the base link), with
    # no height-based termination at all on rough (unlike flat's extra `low_base`
    # override), so a seated pose that doesn't put contact force through the base link
    # itself survives indefinitely with zero risk. mdp.base_height_l2 penalizes
    # deviation from a target standing height (matching _SPAWN_HEIGHT=0.75 in
    # pioneer_humanoid_v1.py) using the height scanner to stay terrain-relative -- a hard
    # world-Z threshold (like flat's low_base) isn't safe on uneven terrain, per
    # Isaac Lab's own root_height_below_minimum docstring ("currently only supported
    # for flat terrains").
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={"target_height": 0.75, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


@configclass
class PioneerHumanoidRoughTerminations(TerminationsCfg):
    # base_height_l2 (reward-only) didn't change the sitting/collapsed behavior after
    # ~200 iterations -- live play confirmed every env still ends up down, and since
    # it's only a per-step cost (not a termination), a fallen robot can just lie there
    # for the full ~1000-step episode with no urgency to recover. base_tilt_over_limit
    # measures body-frame projected-gravity tilt directly, so it works regardless of
    # terrain unevenness (unlike a height-based check) and catches "lying on its side"
    # specifically. This same function was tried on flat once (in combination with a
    # tight low_base threshold) and reverted for causing overly short episodes -- using
    # a generous limit here to start, since this is a different terrain/task and the
    # goal is only to catch genuine falls, not mild sway.
    base_tilt = DoneTerm(func=mdp.base_tilt_over_limit, params={"limit": 0.8})


@configclass
class PioneerHumanoidRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: PioneerHumanoidRewards = PioneerHumanoidRewards()
    terminations: PioneerHumanoidRoughTerminations = PioneerHumanoidRoughTerminations()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = WHOLE_BODY_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{BASE_BODY}"

        # Ported from flat_env_cfg.py: G1 never customizes per-joint action scale (it
        # uses the framework's uniform 0.5 default in both rough and flat), but these
        # values reflect Wato's actual actuator authority (undertuned ankle PD gains,
        # Hip_R turning range) discovered during flat-terrain tuning -- that's a
        # hardware property, not something specific to flat terrain, so it applies here
        # too.
        self.actions.joint_pos.scale = {
            "Hip_F_.*": 0.25,
            "Knee_.*": 0.30,
            "Hip_A_.*": 0.06,
            "Hip_R_.*": 0.15,
            "Ankle_R_.*": 0.08,
            "Ankle_P_.*": 0.12,
        }

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [BASE_BODY]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)

        # rough_terrain_fresh_001 plateaued on this exact symptom flat terrain hit
        # early this session: survival solid (94% time_out), turning tracks well, but
        # feet_air_time stuck near-zero (~0.001-0.003) and track_lin_vel_xy_exp flat
        # around 0.36-0.42 for 30+ min -- standing/turning-in-place satisficing rather
        # than real stepping. G1's own rough value (weight 0.25, threshold 0.4) wasn't
        # enough to make the first exploratory foot-lift worth the risk here, same as
        # it wasn't on flat. Raised to flat's proven values.
        #
        # 2.0/0.2 (matching flat) solved falling (base_tilt termination added
        # separately fixed that) and got turning tracking strong (~0.8-0.95), but
        # after ~1hr feet_air_time was still only ~0.013-0.017 -- an order of
        # magnitude below flat's ~0.25 once it walked confidently -- and live play
        # showed the robot attempting a step without committing to sustained walking.
        # Rough terrain makes stepping riskier than flat (uneven footing), so the
        # policy may need a stronger nudge than flat did to make that risk worth it.
        # Doubled again, same escalation pattern used for feet_crossing_penalty.
        #
        # 4.0/0.2: feet_air_time jumped ~3x (0.013-0.017 -> ~0.04-0.06) and episode
        # length/survival kept improving, but track_lin_vel_xy_exp plateaued around
        # 0.28-0.30 for 3+ consecutive checks (~15 min) while track_ang_vel_z_exp,
        # time_out, and base_tilt all kept setting new bests in the same window --
        # the extra stepping isn't yet converting into better forward tracking.
        # Doubled again rather than tightening track_lin_vel_xy_exp's std (tightening
        # tolerance on flat earlier this session overcorrected badly, killing the
        # gradient and stalling episode length at 45-51) or raising its weight
        # directly (a broader, less targeted lever than fixing the stepping deficit
        # this metric points to).
        #
        # 8.0/0.2: crashed. Mean action noise std was frozen at exactly 2.17 for the
        # entire resume (iteration 7251 onward, ~600 iterations) instead of adapting
        # normally -- the weight jump destabilized exploration from the very start,
        # not just right before the crash. It eventually hit
        # "RuntimeError: normal expects all elements of std >= 0.0" (a genuine PPO
        # divergence, not a GPU/driver fault). Rolled back to model_7249.pt (the last
        # checkpoint before this weight change) rather than resuming from the
        # crashed run's own last checkpoint (model_7800.pt), since the instability
        # was present throughout that whole resume, not a late-onset event -- that
        # checkpoint is likely in the same unstable regime. Trying a smaller step
        # (6.0, not another full doubling) this time.
        #
        # 6.0/0.2: crashed again with the identical error (~450 iterations in), even
        # with empirical_normalization=True also fixed and value_function loss
        # staying bounded (~0.10-0.13, never exploding) right up to the crash -- so
        # unbounded value targets were not the (sole) cause. Stepping back across all
        # attempts: 2.0 and 4.0 both ran stable for a full 7250-iteration run with
        # zero crashes; only 6.0 and 8.0 have crashed, both applied via resume from
        # the same completed model_7249.pt. That's a real stability cliff somewhere
        # between 4.0 and 6.0, not a value worth splitting further by guesswork.
        # Reverted to 4.0, the last confirmed-stable value, rather than continuing to
        # probe the 6-8 range blind.
        self.rewards.feet_air_time.weight = 4.0
        self.rewards.feet_air_time.params["threshold"] = 0.2

        # Live play at multiple checkpoints across every weight/entropy combination
        # tried above kept showing the same thing: the deterministic policy just
        # stands still, even at the best-yet reward numbers (noise annealed to 1.65,
        # tracking 0.40, feet_air_time 0.095). termination_penalty was still at the
        # shared base value of -200 for rough -- flat halves this to -100 in its own
        # __post_init__, an asymmetry never revisited for rough. Combined with rough
        # terrain's inherent extra fall risk (uneven footing makes a real stepping
        # attempt more likely to trip than flat ground), a harsher fall penalty here
        # makes "stand perfectly still and never risk it" a more strongly reinforced
        # local optimum than it ever was on flat. Halved to match flat, to make the
        # risk/reward tradeoff of attempting real steps less lopsided toward inaction.
        self.rewards.termination_penalty.weight = -100.0

        # Widened to match G1's actual rough config exactly (G1's own rough source
        # uses lin_vel_x up to 1.0 and ang_vel_z up to +-1.0 -- Wato's had been
        # sitting narrower, unexamined). lin_vel_y stays zero: G1 deliberately doesn't
        # allow lateral velocity commands on rough terrain either, only on flat.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = [BASE_BODY]


@configclass
class PioneerHumanoidRoughEnvCfg_PLAY(PioneerHumanoidRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Mirror the actual training ranges (PioneerHumanoidRoughEnvCfg.__post_init__ above) --
        # this was previously hardcoded to lin_vel_x=(0.5,0.5)/ang_vel_z=(0,0), silently
        # never updated when training's ranges changed, so every play view was forcing
        # zero turning + fixed forward speed regardless of what the policy was trained on.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
