"""Sim-to-real action terms for the Pioneer humanoid locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedJointPositionAction(JointPositionAction):
    """Joint position action whose targets reach the actuators a random number of policy steps late.

    Models the policy -> CAN -> motor driver latency on the real robot. Each env samples its own
    lag in [min_delay, max_delay] policy steps on every reset; the lag is held for the episode.
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if not 0 <= cfg.min_delay <= cfg.max_delay:
            raise ValueError(f"Need 0 <= min_delay <= max_delay, got {cfg.min_delay}, {cfg.max_delay}.")
        self._delay_buffer = DelayBuffer(cfg.max_delay, self.num_envs, device=self.device)
        self._resample_delays(slice(None), self.num_envs)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        # The buffer fills its whole history with the first target pushed after a reset,
        # so a freshly reset env holds its first target rather than a zero pose.
        self._processed_actions = self._delay_buffer.compute(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            num = self.num_envs
        else:
            num = len(env_ids)
        self._resample_delays(env_ids, num)
        self._delay_buffer.reset(env_ids)

    def _resample_delays(self, env_ids: Sequence[int] | slice, num: int) -> None:
        lags = torch.randint(
            self.cfg.min_delay, self.cfg.max_delay + 1, (num,), dtype=torch.int, device=self.device
        )
        self._delay_buffer.set_time_lag(lags, env_ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`DelayedJointPositionAction`. Delays are in policy steps."""

    class_type: type = DelayedJointPositionAction

    min_delay: int = 0
    max_delay: int = 0
