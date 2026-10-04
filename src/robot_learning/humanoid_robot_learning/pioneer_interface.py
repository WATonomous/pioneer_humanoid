"""Adapter between Isaac Sim's pioneer_bimanual_arm (left chain) and RTCDrivenPolicy.

Mirrors humanoid_so101_vial_task.utils.lerobot_interface.LeRobotSO101Interface's
role, but simplified for this robot:

- No lerobot.robots/lerobot.teleoperators Robot class exists for this arm (it's not
  a LeRobot-registered robot), so we never call make_robot_from_config. Instead we
  build the dataset feature dict directly from the recording schema YAML via
  humanoid_robot_learning.schema.build_features -- the same function the recorder
  itself uses to create the dataset, so this is guaranteed to match what the policy
  was trained on rather than being a hand-maintained duplicate.

- The recorded frame format (humanoid_robot_learning.frame.build_lerobot_frame) is already a
  flat vector -- observation.state / action are (7,) float32 arrays, not SO101's
  per-joint named dict ("shoulder_pan.pos", etc). So sim_obs_to_policy_processor
  builds the frame directly instead of going through build_dataset_frame +
  robot_observation_processor.

- Joint limits come from pioneer_humanoid.bimanual_arm.JOINT_POS_LIMITS, parsed
  from the URDF (radians) -- no degree/normalized-unit conversion needed, since
  both the sim and the dataset already speak radians directly. This is simpler
  than SO101's -100..100 leader convention for that reason.

- RTCDrivenPolicy owns pre/postprocessing (see humanoid_robot_learning.rtc_driver._replan) --
  this class only ever produces/consumes RAW, unprocessed frames and actions.

Joint order note: the 7 recorded joints are [joint1L, joint2l, joint3l, joint4l,
joint5l, joint6l, joint7l] (see dataset_schema_wato_arm_v2_push_box.yaml).
joint8l is NOT recorded -- it's driven as joint7l's synchronized mechanical
mirror (one gripper motor, two fingers; see bimanual_arm.py). prediction_to_sim
reconstructs joint8l = -joint7l before returning the 8-value sim action.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from humanoid_robot_learning.schema import build_features, load_yaml
from pioneer_humanoid.bimanual_arm import JOINT_POS_LIMITS

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

# Sim articulation action order. Note that joint8l is reconstructed from joint7l
SIM_ACTION_JOINT_ORDER = [
    "joint1L", "joint2l", "joint3l", "joint4l", "joint5l", "joint6l", "joint7l", "joint8l",
]


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
    """Sim-state <-> LeRobot-frame adapter for the pioneer_bimanual_arm left chain.

    Args:
        device: inference device (e.g. the sim's own torch device).
        cameras: {camera_key: {"height": int, "width": int}}, matching the schema's
            `images` block keys (this dataset already uses "ego"/"wrist" as both
            the sim's camera keys and the dataset's image keys, so no rename_map
            is needed here, unlike SO101).
        schema_path: path to the dataset_schema_wato_arm_v2_push_box.yaml (or
            whichever schema this checkpoint was trained against). Read once, at
            construction, to build the exact feature dict the recorder used.
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

        cfg = load_yaml(Path(schema_path))
        self.joint_names: list[str] = list(cfg["joint_names"])  # 7: arm joints + joint7l
        self.robot_type: str = cfg["robot_type"]
        self.dataset_features: dict[str, Any] = build_features(cfg)

        # Clamp table for the 7 recorded joints, in self.joint_names order. Pulled
        # straight from the URDF-derived table -- no hand-copied numbers.
        mins, maxs = [], []
        for name in self.joint_names:
            lo, hi = JOINT_POS_LIMITS[name]
            mins.append(lo)
            maxs.append(hi)
        self.joint_mins = torch.tensor(mins, dtype=torch.float32, device=self.device)
        self.joint_maxs = torch.tensor(maxs, dtype=torch.float32, device=self.device)

        # joint8l isn't in joint_names (see module docstring) -- its own limit is
        # looked up separately, at mirror time, since it moves opposite joint7l.
        self._joint8l_lo, self._joint8l_hi = JOINT_POS_LIMITS["joint8l"]

    def make_policy(self, name_or_path: str) -> None:
        """Load a checkpoint + its matching pre/postprocessor pipeline.

        Call once, before any get_action calls. Mirrors
        LeRobotSO101Interface.make_policy, minus the RTC/ACT research-knob env
        vars (set policy_config.rtc_config yourself before calling this, or add
        those knobs back in if you want them) and minus the Robot-derived
        feature-building (we already built self.dataset_features in __init__).
        """
        policy_config = PreTrainedConfig.from_pretrained(name_or_path)
        policy_config.pretrained_path = name_or_path
        policy_config.device = self.device

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
            sim_observation: (7,) tensor, sim joint positions in radians, in
                self.joint_names order (state uses only the 7 recorded joints --
                joint8l is not part of observation.state, same as at record time).
            visual_obs: sim's per-camera image dict, keyed the same way self.cameras
                is (e.g. visual_obs["rgb_ego"], visual_obs["rgb_wrist"] -- adjust the
                "rgb_" prefix here if your sim env names these differently).

        Returns: a dict matching humanoid_robot_learning.frame.build_lerobot_frame's shape --
            NOT yet preprocessed/normalized; RTCDrivenPolicy._replan does that.
        """
        state_np = sim_observation[:7].detach().cpu().numpy()

        obs_frame: dict[str, Any] = {
            "observation.state": state_np,
            "task": self.task_description,
        }
        for camera_key in self.cameras:
            img = visual_obs[f"rgb_{camera_key}"][0]
            obs_frame[f"observation.images.{camera_key}"] = img.detach().cpu().numpy()

        return obs_frame

    def prediction_to_sim_processor(
        self, action_values: torch.Tensor, observation_frame: dict[str, Any], log: bool = False
    ) -> torch.Tensor:
        """Policy action (7,) -> sim action (8,), clamped and gripper-mirrored.

        Args:
            action_values: (7,) tensor from RTCDrivenPolicy.get_action -- already
                postprocessed (real units/radians), in self.joint_names order.
            observation_frame: unused here (kept for signature symmetry with
                LeRobotSO101Interface.prediction_to_sim_processor / for future
                rerun-style logging); pass log=True + wire up logging if needed.

        Returns: (8,) tensor in SIM_ACTION_JOINT_ORDER, ready for env.step.
        """
        action_values = action_values.reshape(-1)  # (7,)

        clampedActions = torch.clamp(action_values, self.joint_mins, self.joint_maxs)

        joint7l = clampedActions[-1]
        # Mechanical mirror: one gripper motor drives both fingers with opposite
        # sign (see LEFT_GRIPPER_OPEN/CLOSED in bimanual_arm.py: joint7l closed is
        # -0.05, joint8l closed is +0.05 -- same magnitude, opposite sign).
        joint8l = torch.clamp(-joint7l, self._joint8l_lo, self._joint8l_hi)

        sim_action = torch.cat([clampedActions, joint8l.unsqueeze(0)])  # (8,)
        return sim_action
