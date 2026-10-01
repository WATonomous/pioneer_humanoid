"""ManagerBasedRlEnvCfg for the badminton receive task.

Physics matches the CPU scene: 1 ms timestep, 50 Hz control (decimation 20),
contacts only through the scene's explicit pairs (shuttle vs both racket
faces, floor and net; inter-arm racket/forearm clashes), shuttle drag and
cork orientation applied per substep by BadmintonAction.

Bimanual: both arms are actuated (12 joint targets, right arm first) with a
racket in each hand. Each shot is assigned to one arm by a fixed rule (side
of the predicted strike-plane crossing, see perception_command.py); the
other arm is held at its ready pose, the approach reward grades only the
assigned face, and both observation groups carry the assignment one-hot.
assign_source picks the prediction: "true" state for the teacher, "ekf"
for the student tasks.

Observation groups:
  student  proprioception + EKF-tracked shuttle and noisy trajectory prior
  teacher  proprioception + true shuttle state, true trajectory prior, and
           the privileged intercept target (p*, time-to-t*)

The teacher PPO phase trains on "teacher"; distillation maps "teacher" ->
"student" with the same action space.
"""

from __future__ import annotations

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import observations as obs_mdp
from mjlab.envs.mdp import rewards as rew_mdp
from mjlab.envs.mdp import terminations as term_mdp
from mjlab.envs.mdp import events as event_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (ObservationGroupCfg,
                                                ObservationTermCfg)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import aero
from humanoid_badminton import assets, mdp
from humanoid_badminton.feasibility import FeasibilityCommandCfg
from humanoid_badminton.perception_command import PerceptionCommandCfg
from humanoid_badminton.shuttle_action import BadmintonActionCfg

CLASH_SENSORS = ("arm_clash_l", "arm_clash_face_l")
DECIMATION = 20          # 1 ms physics, 50 Hz control — same as the CPU env
EPISODE_LENGTH_S = 3.0

FACE_SITES = SceneEntityCfg("robot", site_names=assets.FACE_SITES,
                            preserve_order=True)


def make_env_cfg(play: bool = False,
                 assign_source: str = "true") -> ManagerBasedRlEnvCfg:
    params = aero.load_params()

    # The scene XML already places arm_base_link (pos + mount yaw); passing
    # the same pose through InitialStateCfg COMPOSES with it (measured: face
    # 1.2 m too high, run 6 rhithgnl) rather than overwriting.
    robot = EntityCfg(
        spec_fn=assets.robot_spec,
        init_state=EntityCfg.InitialStateCfg(
            # keys are regexes matched with re.match (a prefix match):
            # anchor them, or "arm_joint1" also sets "arm_joint1L"
            joint_pos={j + "$": q for j, q in assets.READY_JOINT_POS.items()}),
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=("arm_joint.*",)),)),
    )
    shuttle = EntityCfg(
        spec_fn=assets.shuttle_spec,
        # parked out of play until the reset event launches an episode
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 5.0, 0.1)),
    )

    sensors = (
        # one primary per racket face (mjlab secondaries are single names)
        ContactSensorCfg(
            name="face_hit",
            primary=ContactMatch(mode="geom",
                                 pattern=("racket_face", "racket_face_l"),
                                 entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="shuttle_cork",
                                   entity="shuttle"),
            fields=("found", "force"),
            reduce="netforce",
            history_length=DECIMATION,
        ),
        ContactSensorCfg(
            name="shuttle_floor",
            primary=ContactMatch(mode="geom",
                                 pattern=("shuttle_col", "shuttle_cork"),
                                 entity="shuttle"),
            secondary=ContactMatch(mode="geom", pattern="court/floor"),
            fields=("found",),
            reduce="netforce",
        ),
        ContactSensorCfg(
            name="shuttle_net",
            primary=ContactMatch(mode="geom", pattern="shuttle_col",
                                 entity="shuttle"),
            secondary=ContactMatch(mode="geom", pattern="court/net"),
            fields=("found",),
            reduce="netforce",
        ),
        # inter-arm clash, covering every pair of build_scene.clash_pairs
        # with two sensors (a secondary must be one name): the left handle
        # and forearm have only clash pairs, so any contact on them counts;
        # the left face also meets the shuttle, so it is matched only
        # against the right-side geoms.
        ContactSensorCfg(
            name="arm_clash_l",
            primary=ContactMatch(
                mode="geom", entity="robot",
                pattern=("racket_handle_l", "arm_col_link4l",
                         "arm_col_link5l", "arm_col_link6l")),
            secondary_policy="any",
            fields=("found", "force"),
            reduce="netforce",
            history_length=DECIMATION,
        ),
        ContactSensorCfg(
            name="arm_clash_face_l",
            primary=ContactMatch(
                mode="geom", entity="robot",
                pattern=("racket_face", "racket_handle", "arm_col_link4",
                         "arm_col_link5", "arm_col_link6")),
            secondary=ContactMatch(mode="geom", pattern="racket_face_l",
                                   entity="robot"),
            fields=("found", "force"),
            reduce="netforce",
            history_length=DECIMATION,
        ),
    )

    proprio = {
        "joint_pos": ObservationTermCfg(
            func=obs_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObservationTermCfg(
            func=obs_mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5)),
        "face_state": ObservationTermCfg(
            func=mdp.face_state, params={"asset_cfg": FACE_SITES}),
        "actions": ObservationTermCfg(func=obs_mdp.last_action),
        "arm_assign": ObservationTermCfg(func=mdp.arm_assign),
    }
    student_terms = {
        **proprio,
        "shuttle": ObservationTermCfg(
            func=mdp.student_perception, params={"command_name": "perception"}),
        # distillation 1 missed the left/high (early-commit) shots: the
        # student had no way to tell a settling estimate from a converged
        # one. EKF std + prior jump give it that (a real filter has both).
        "shuttle_uncertainty": ObservationTermCfg(
            func=mdp.student_uncertainty,
            params={"command_name": "perception"}),
    }
    teacher_terms = {
        **proprio,
        "shuttle": ObservationTermCfg(
            func=mdp.teacher_perception, params={"command_name": "perception"}),
        "intercept": ObservationTermCfg(func=mdp.intercept_target),
    }
    observations = {
        # student proprioception is corrupted (sim encoders are exact);
        # its shuttle features carry EKF noise already
        "student": ObservationGroupCfg(student_terms, enable_corruption=True),
        "teacher": ObservationGroupCfg(teacher_terms, enable_corruption=False),
    }

    actions = {
        "joint_pos": BadmintonActionCfg(
            entity_name="robot",
            shuttle_entity_name="shuttle",
            actuator_names=(".*",),
            scale=1.0,
            use_default_offset=True,
            clip={j + "$": r for j, r in assets.joint_ranges().items()},
        ),
    }

    commands = {
        "perception": PerceptionCommandCfg(entity_name="shuttle",
                                           assign_source=assign_source),
        # logs Metrics/feasibility/* (peak joint vel / torque, rated-torque
        # duty cycle) so every run carries its own sim2real feasibility data
        "feasibility": FeasibilityCommandCfg(),
    }

    events = {
        "reset_arm": EventTermCfg(
            func=event_mdp.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (0.0, 0.0),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
        "launch_shuttle": EventTermCfg(
            func=mdp.reset_badminton_episode, mode="reset", params={}),
    }

    rewards = {
        # face_contact latches the has-hit flag; keep it first.
        # weight 100 = effective 2.0/hit after dt scaling: run 3 (j2h0aizn)
        # showed exploration finding hits at weight 10 (effective 0.2, only
        # ~2x the per-episode approach stream) without reinforcing them
        "face_contact": RewardTermCfg(
            func=mdp.face_contact, weight=100.0,
            params={"sensor_name": "face_hit"}),
        # std 0.8 ~ the median face->p* distance at the ready pose (0.69 m);
        # run 1 (a1vagllq) plateaued with std 0.4: near-zero gradient at that
        # distance, policy held still, exploration std collapsed
        "approach": RewardTermCfg(
            func=mdp.approach_intercept, weight=1.0,
            params={"std": 0.8, "asset_cfg": FACE_SITES}),
        # two-scale shaping: sigma 0.8 pulls from anywhere in the workspace
        # but is flat over the last 30 cm; runs 3-4 showed the policy parks
        # ~outside that band and only hits by luck (~0.1%), regardless of the
        # contact bonus scale. p* lies on the flight path, so precision
        # parking at p* converts directly into contacts.
        "approach_fine": RewardTermCfg(
            func=mdp.approach_intercept, weight=5.0,
            params={"std": 0.15, "asset_cfg": FACE_SITES}),
        "return_flight": RewardTermCfg(
            func=mdp.return_flight, weight=2.0, params={}),
        # run 12: land the return on the episode's launch origin. Sparse,
        # once per hit; weight 100 = effective 2.0 for a perfect landing,
        # matching the contact bonus. sigma 1.5 m: wide enough to grade the
        # whole far court, so the warm-started policy gets gradient from its
        # existing returns. Zero if the predicted flight misses the net
        # window (physical validity, not a style preference).
        "return_landing": RewardTermCfg(
            func=mdp.return_landing, weight=100.0, params={"sigma": 1.5}),
        # bimanual: rackets/forearms touching. Per tick in contact; weight
        # -50 = effective -1.0 per tick, so a clash costs about a hit's
        # bonus within two ticks. The episode also terminates (see below).
        "arm_clash": RewardTermCfg(
            func=mdp.arm_clash, weight=-50.0,
            params={"sensor_names": CLASH_SENSORS}),
        # smoothness (2026-10 envelope pass): energy-style costs, not a
        # posture preference. -0.01/-1e-4 were token values that let PPO
        # bang-bang the arm inside the (then huge) velocity envelope.
        "action_rate": RewardTermCfg(func=rew_mdp.action_rate_l2, weight=-0.05),
        "joint_limits": RewardTermCfg(
            func=rew_mdp.joint_pos_limits, weight=-5.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
        "joint_vel": RewardTermCfg(
            func=rew_mdp.joint_vel_l2, weight=-2e-3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
        # mdp.torque_over_rated (thermal penalty) is available but off: run 9
        # showed an instantaneous per-tick penalty suppresses the transient
        # swing peaks the strong joints are rated for and cost 6x in return
        # quality; the wrist overload it targeted was a servo-gain artifact
        # (see params control.kp). Feasibility stays a logged metric.
    }

    terminations = {
        "time_out": TerminationTermCfg(func=term_mdp.time_out, time_out=True),
        "grounded": TerminationTermCfg(
            func=mdp.shuttle_grounded, params={"sensor_name": "shuttle_floor"}),
        "net": TerminationTermCfg(
            func=mdp.shuttle_net, params={"sensor_name": "shuttle_net"}),
        # a clash ends the episode: the hardware would e-stop on it
        "arm_clash": TerminationTermCfg(
            func=mdp.arm_clash, params={"sensor_names": CLASH_SENSORS}),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1024,
            env_spacing=1.0,
            entities={"robot": robot, "shuttle": shuttle},
            sensors=sensors,
            spec_fn=assets.court_fn,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot", body_name="arm_base_link",
            distance=3.5, elevation=-15.0, azimuth=180.0),
        sim=SimulationCfg(
            # 30 explicit pairs (4 shuttle, 26 clash); clash contacts are rare
            # and end the episode
            nconmax=32,
            njmax=128,
            mujoco=MujocoCfg(timestep=params["integrator"]["dt"]),
        ),
        decimation=DECIMATION,
        episode_length_s=EPISODE_LENGTH_S,
    )
    if play:
        cfg.scene.num_envs = 16
    return cfg
