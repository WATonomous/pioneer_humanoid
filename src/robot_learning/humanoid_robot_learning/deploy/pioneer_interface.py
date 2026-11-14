"""Adapter between Isaac Sim's pioneer_bimanual_arm_8dof (left arm) and RTCDrivenPolicy.

Mirrors humanoid_so101_vial_task.utils.lerobot_interface.LeRobotSO101Interface's
role, but simplified for this robot:

- No lerobot.robots/lerobot.teleoperators Robot class exists for this arm (it's not
  a LeRobot-registered robot), so we never call make_robot_from_config. Instead we
  build the dataset feature dict directly from the recording schema YAML via
  humanoid_robot_learning.recording.schema.build_features -- the same function the recorder
  itself uses to create the dataset, so this is guaranteed to match what the policy
  was trained on rather than being a hand-maintained duplicate.

- The recorded frame format (humanoid_robot_learning.recording.frame.build_lerobot_frame) is already a
  flat vector -- observation.state / action are (8,) float32 arrays, not SO101's
  per-joint named dict ("shoulder_pan.pos", etc). So sim_obs_to_policy_processor
  builds the frame directly instead of going through build_dataset_frame +
  robot_observation_processor.

- Joint limits come from pioneer_humanoid.arm_8dof_params.urdf_limits(), parsed
  from the URDF (radians) -- no degree/normalized-unit conversion needed, since
  both the sim and the dataset already speak radians directly. This is simpler
  than SO101's -100..100 leader convention for that reason.

- RTCDrivenPolicy owns pre/postprocessing (see humanoid_robot_learning.deploy.rtc_driver._replan) --
  this class only ever produces/consumes RAW, unprocessed frames and actions.

Joint order note: the 8 recorded values follow pioneer_8dof_dataset_schema.yaml:
left_shoulder_pitch .. left_wrist_yaw (the sim joints, rad), then left_gripper as a
0-1 closure fraction (0 open, 1 closed). The sim side reads and commands the
left_gripper joint for it; left_gripper_mimic follows it in the sim, never commanded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import os

import torch

from humanoid_robot_learning.recording.schema import build_features, load_yaml, select_arms
from pioneer_humanoid.arm_8dof_params import (
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_CLOSED,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRIPPER_OPEN,
    urdf_limits,
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

# Sim joints read for observation.state, in schema order: the seven arm joints, then left_gripper
# (converted to the 0-1 gripper closure).
SIM_STATE_JOINTS = [*LEFT_ARM_JOINTS, LEFT_GRIPPER_JOINTS[0]]
# Sim articulation action order.
SIM_ACTION_JOINT_ORDER = [*LEFT_ARM_JOINTS, *LEFT_GRIPPER_JOINTS]


class DummyDatasetMeta:
    """Minimal stand-in for a LeRobotDatasetMetadata, same pattern as
    humanoid_so101_vial_task.utils.lerobot_interface.DummyDatasetMeta -- make_policy
    only reads .features/.stats/.robot_type off this, it doesn't need a real dataset.
    """

    def __init__(self, features: dict[str, Any], robot_type: str):
        self.features = features
        self.stats: dict[str, Any] = {}
        self.robot_type = robot_type


class PioneerLeftArmInterface:
    """Sim-state <-> LeRobot-frame adapter for the pioneer_bimanual_arm_8dof left arm.

    Args:
        device: inference device (e.g. the sim's own torch device).
        cameras: {camera_key: {"height": int, "width": int}}, matching the schema's
            `images` block keys ("ego", "wrist_left", ...), which are also the keys of
            the visual_obs dict passed in, so no rename_map is needed here, unlike SO101.
        schema_path: path to pioneer_8dof_dataset_schema.yaml (the schema this
            checkpoint was trained against). Read once, at construction, to build the
            exact feature dict the recorder used.
        task_description: language instruction string passed to the policy.
    """

    def __init__(
        self,
        device: str,
        cameras: dict[str, dict[str, int]],
        schema_path: str | Path,
        task_description: str,
    ) -> None:
        self.device = device
        self.cameras = cameras
        self.task_description = task_description

        cfg = select_arms(load_yaml(Path(schema_path)), ["left"])
        self.joint_names: list[str] = list(cfg["joint_names"])  # 8: arm joints + left_gripper
        if len(self.joint_names) != len(SIM_STATE_JOINTS):
            raise ValueError(f"{schema_path}: left arm has {self.joint_names}, expected {len(SIM_STATE_JOINTS)} joints")
        self.robot_type: str = str(cfg["robot_id"])  # the recorders label the dataset with robot_id
        self.dataset_features: dict[str, Any] = build_features(cfg)

        # Clamp table for the 8 policy values: URDF limits for the arm, 0-1 for the gripper closure.
        limits = urdf_limits()
        mins = [limits[j][0] for j in LEFT_ARM_JOINTS] + [0.0]
        maxs = [limits[j][1] for j in LEFT_ARM_JOINTS] + [1.0]
        self.joint_mins = torch.tensor(mins, dtype=torch.float32, device=self.device)
        self.joint_maxs = torch.tensor(maxs, dtype=torch.float32, device=self.device)

        # Gripper joint positions at closure 0 and 1, in LEFT_GRIPPER_JOINTS order.
        self._grip_open = torch.tensor([LEFT_GRIPPER_OPEN[j] for j in LEFT_GRIPPER_JOINTS], device=self.device)
        self._grip_closed = torch.tensor([LEFT_GRIPPER_CLOSED[j] for j in LEFT_GRIPPER_JOINTS], device=self.device)

    def sim_state_to_policy(self, sim_state: torch.Tensor) -> torch.Tensor:
        """(8,) sim positions in SIM_STATE_JOINTS order -> (8,) policy values (left_gripper -> 0-1 closure)."""
        open_, closed = self._grip_open[0], self._grip_closed[0]
        closure = ((sim_state[-1] - open_) / (closed - open_)).clamp(0.0, 1.0)
        return torch.cat([sim_state[:-1], closure.reshape(1)])

    def make_policy(self, name_or_path: str) -> None:
        """Load a checkpoint + its matching pre/postprocessor pipeline.

        Call once, before any get_action calls. Mirrors
        LeRobotSO101Interface.make_policy, including its RTC_EXECUTION_HORIZON /
        RTC_MAX_GUIDANCE_WEIGHT env knobs, minus the Robot-derived feature-building
        (self.dataset_features is built in __init__).
        """
        policy_config = PreTrainedConfig.from_pretrained(name_or_path)
        action_shape = tuple(policy_config.output_features["action"].shape)
        if action_shape != (len(self.joint_names),):
            raise ValueError(f"{name_or_path}: policy action shape {action_shape}, the 8dof left arm takes "
                             f"({len(self.joint_names)},): an old 6-DoF arm checkpoint (7 values) can't run here")
        policy_config.pretrained_path = name_or_path
        policy_config.device = self.device

        # RTC for flow-matching policies (pi0/pi0.5/SmolVLA), off unless RTC_EXECUTION_HORIZON is set.
        rtc_execution_horizon = os.environ.get("RTC_EXECUTION_HORIZON")
        if rtc_execution_horizon is not None and hasattr(policy_config, "rtc_config"):
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            rtc_guidance_weight = float(os.environ.get("RTC_MAX_GUIDANCE_WEIGHT", "10.0"))
            policy_config.rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=int(rtc_execution_horizon),
                max_guidance_weight=rtc_guidance_weight,
            )
            print(f"[INFO]: Enabling RTC, execution_horizon={rtc_execution_horizon}, max_guidance_weight={rtc_guidance_weight}")

        dataset_meta = DummyDatasetMeta(self.dataset_features, self.robot_type)
        self.policy = make_policy(policy_config, ds_meta=dataset_meta)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=name_or_path,
            dataset_stats={},  # policy's own bundled stats are used
            preprocessor_overrides={
                "device_processor": {"device": policy_config.device},
            },
        )

    def sim_obs_to_policy_processor(
        self, sim_observation: torch.Tensor, visual_obs: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Sim joint positions (rad) + camera images -> a raw LeRobot obs_frame.

        Args:
            sim_observation: (8,) tensor, sim joint positions in SIM_STATE_JOINTS order
                (seven arm joints in radians, then left_gripper).
            visual_obs: sim's per-camera image dict, keyed the same way self.cameras
                is (e.g. visual_obs["rgb_ego"], visual_obs["rgb_wrist_left"] -- adjust the
                "rgb_" prefix here if your sim env names these differently).

        Returns: a dict matching humanoid_robot_learning.recording.frame.build_lerobot_frame's shape,
            minus "task" -- NOT yet preprocessed/normalized; RTCDrivenPolicy._replan does that.
            The task string is deliberately left out: _replan's
            prepare_observation_for_inference runs torch.from_numpy on every key (a string
            crashes it) and adds "task" itself from RTCDrivenPolicy.task_description.
        """
        state_np = self.sim_state_to_policy(sim_observation[:len(SIM_STATE_JOINTS)]).detach().cpu().numpy()

        obs_frame: dict[str, Any] = {
            "observation.state": state_np,
        }
        for camera_key in self.cameras:
            img = visual_obs[f"rgb_{camera_key}"][0]
            obs_frame[f"observation.images.{camera_key}"] = img.detach().cpu().numpy()

        return obs_frame

    def prediction_to_sim_processor(
        self, action_values: torch.Tensor, observation_frame: dict[str, Any], log: bool = False
    ) -> torch.Tensor:
        """Policy action (8,) -> sim action (8,), clamped, gripper closure mapped onto the left_gripper joint.

        Args:
            action_values: (8,) tensor from RTCDrivenPolicy.get_action -- already
                postprocessed (seven arm joints in radians, then the 0-1 gripper
                closure), in self.joint_names order.
            observation_frame: unused here (kept for signature symmetry with
                LeRobotSO101Interface.prediction_to_sim_processor / for future
                rerun-style logging); pass log=True + wire up logging if needed.

        Returns: (8,) tensor in SIM_ACTION_JOINT_ORDER, ready for env.step.
        """
        action_values = action_values.reshape(-1)  # (8,)

        clampedActions = torch.clamp(action_values, self.joint_mins, self.joint_maxs)

        gripper = self._grip_open + clampedActions[-1] * (self._grip_closed - self._grip_open)

        sim_action = torch.cat([clampedActions[:-1], gripper])  # (8,)
        return sim_action
