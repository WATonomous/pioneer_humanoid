"""Env config for the shared-arm tasks (humanoid_badminton/shared.py).

Same scene, physics, perception, terminations and rewards as the bimanual
env (env_cfg.make_env_cfg), with these changes:

  action        SharedArmAction: 6 targets for the active arm, in its frame
  observations  the single-arm layout in the active arm's frame
                (teacher: joint_pos, joint_vel, face_state, actions, shuttle,
                intercept = 58 dims; student: ... shuttle,
                shuttle_uncertainty), so single-arm checkpoints load as is
  approach      graded on the active arm's face from the first tick
  reset         with control.shared_start_pose "rest", both arms start at
                arm.rest_joint_pos instead of the ready pose

With chooser=True the action waits for an external lock (ChooserPolicy)
and every group gets <group>_right and <group>_left copies in the fixed
arm frames, which the chooser feeds to the critic.
"""

from __future__ import annotations

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (ObservationGroupCfg,
                                                ObservationTermCfg)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import aero
from humanoid_badminton import assets, shared
from humanoid_badminton.env_cfg import FACE_SITES, make_env_cfg


def _groups(frame: str) -> dict[str, ObservationGroupCfg]:
    def proprio():
        return {
            "joint_pos": ObservationTermCfg(
                func=shared.joint_pos_frame, params={"frame": frame},
                noise=Unoise(n_min=-0.01, n_max=0.01)),
            "joint_vel": ObservationTermCfg(
                func=shared.joint_vel_frame, params={"frame": frame},
                noise=Unoise(n_min=-0.5, n_max=0.5)),
            "face_state": ObservationTermCfg(
                func=shared.face_state_frame,
                params={"frame": frame, "asset_cfg": FACE_SITES}),
            "actions": ObservationTermCfg(
                func=shared.last_action_frame, params={"frame": frame}),
        }
    from humanoid_badminton import mdp
    student = {
        **proprio(),
        "shuttle": ObservationTermCfg(
            func=shared.student_perception_frame,
            params={"command_name": "perception", "frame": frame}),
        "shuttle_uncertainty": ObservationTermCfg(
            func=mdp.student_uncertainty,
            params={"command_name": "perception"}),
    }
    teacher = {
        **proprio(),
        "shuttle": ObservationTermCfg(
            func=shared.teacher_perception_frame,
            params={"command_name": "perception", "frame": frame}),
        "intercept": ObservationTermCfg(
            func=shared.intercept_frame, params={"frame": frame}),
    }
    return {
        "student": ObservationGroupCfg(student, enable_corruption=True),
        "teacher": ObservationGroupCfg(teacher, enable_corruption=False),
    }


def make_shared_env_cfg(play: bool = False, chooser: bool = False):
    cfg = make_env_cfg(play=play)
    obs = _groups("active")
    if chooser:
        for frame in ("right", "left"):
            obs.update({f"{g}_{frame}": c for g, c in _groups(frame).items()})
    cfg.observations = obs
    joint = cfg.actions["joint_pos"]
    cfg.actions = {
        "joint_pos": shared.SharedArmActionCfg(
            entity_name=joint.entity_name,
            shuttle_entity_name=joint.shuttle_entity_name,
            actuator_names=joint.actuator_names,
            scale=joint.scale,
            use_default_offset=joint.use_default_offset,
            clip=joint.clip,
            assign_mode="external" if chooser else "random"),
    }
    for name in ("approach", "approach_fine"):
        term = cfg.rewards[name]
        cfg.rewards[name] = RewardTermCfg(
            func=shared.approach_assigned, weight=term.weight,
            params=dict(term.params))
    if aero.load_params()["control"]["shared_start_pose"] == "rest":
        cfg.events["reset_arm"] = EventTermCfg(
            func=shared.reset_arms_to_rest, mode="reset", params={})
    assert assets.FACE_SITES[0] == "face_center"   # right face first
    return cfg
