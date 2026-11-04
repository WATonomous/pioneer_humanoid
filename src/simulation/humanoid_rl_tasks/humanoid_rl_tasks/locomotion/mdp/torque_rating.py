"""Rated-torque metric and penalty for the leg actuators.

Rated torque is a thermal (time-averaged) motor limit; short bursts up to peak are fine. The sim caps torque at
peak only (D13), so these terms watch how often and how far the policy goes above rated.

- TorqueRatingCommand: logs, per joint and episode, the fraction of env steps with |applied torque| > rated
  (TensorBoard key Metrics/<term name>/frac_above_rated_<joint>). It is a command term with an empty command
  because Isaac Lab logs command-term metrics every episode; a weight-0 reward term is skipped and logs 0
  (reward_manager.py:146).
- torque_above_rated_l1: penalty sum_j max(|tau_j| - rated_j, 0) / rated_j. Weight 0 until Task 17.

Both read asset.data.applied_torque, i.e. the actuator output after the DCMotor clip. At env-step rate that buffer
holds the last of the `decimation` physics substeps, so the metric samples 1 in 4 substeps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, RewardTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_rated(asset: Articulation, rated_torque: dict[str, float]) -> tuple[list[int], list[str], torch.Tensor]:
    """Joint ids, names and rated torques (in that order) for the exact joint names in `rated_torque`."""
    names = list(rated_torque)
    joint_ids, joint_names = asset.find_joints([f"^{name}$" for name in names], preserve_order=True)
    rated = torch.tensor([rated_torque[name] for name in joint_names], device=asset.device)
    if torch.any(rated <= 0):
        raise ValueError(f"rated torques must be > 0, got {rated_torque}")
    return joint_ids, joint_names, rated


class TorqueRatingCommand(CommandTerm):
    """Empty command whose metrics are the per-joint fraction of steps above rated torque."""

    cfg: TorqueRatingCommandCfg

    def __init__(self, cfg: TorqueRatingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.joint_ids, self.joint_names, self.rated_torque = _resolve_rated(self.robot, cfg.rated_torque)
        self._steps_above = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        self._steps = torch.zeros(self.num_envs, device=self.device)
        self._empty = torch.zeros(self.num_envs, 0, device=self.device)
        for name in self.joint_names:
            self.metrics[f"frac_above_rated_{name}"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._empty

    def _update_metrics(self):
        # CommandManager.compute runs after this step's resets, so envs that just reset (episode_length_buf 0)
        # still hold the old episode's last torque. Skip them so each count belongs to one episode.
        counted = (self._env.episode_length_buf > 0).float()
        above = (self.robot.data.applied_torque[:, self.joint_ids].abs() > self.rated_torque).float()
        self._steps_above += above * counted.unsqueeze(1)
        self._steps += counted
        frac = self._steps_above / self._steps.clamp_min(1.0).unsqueeze(1)
        for i, name in enumerate(self.joint_names):
            self.metrics[f"frac_above_rated_{name}"][:] = frac[:, i]

    def _resample_command(self, env_ids: Sequence[int]):
        # Called at episode reset (resampling_time_range is effectively infinite): start new counts.
        self._steps_above[env_ids] = 0.0
        self._steps[env_ids] = 0.0

    def _update_command(self):
        pass


@configclass
class TorqueRatingCommandCfg(CommandTermCfg):
    class_type: type = TorqueRatingCommand

    asset_name: str = "robot"
    rated_torque: dict[str, float] = {}
    """Exact joint name -> rated torque (Nm, joint side)."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)


class torque_above_rated_l1(ManagerTermBase):
    """Penalty: sum over joints of max(|tau| - rated, 0) / rated, shape (num_envs,)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params.get("asset_name", "robot")]
        self.joint_ids, _, self.rated_torque = _resolve_rated(self.robot, cfg.params["rated_torque"])

    def __call__(self, env: ManagerBasedRLEnv, rated_torque: dict[str, float], asset_name: str = "robot") -> torch.Tensor:
        tau = self.robot.data.applied_torque[:, self.joint_ids].abs()
        return torch.sum(torch.clamp(tau - self.rated_torque, min=0.0) / self.rated_torque, dim=1)
