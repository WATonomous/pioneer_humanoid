"""A position action whose target cannot slew faster than a per-joint limit.

Measured on the spiral checkpoint (buzz_probe.py), the knee's *command* jumps
+-0.7 rad between consecutive 20 ms control steps -- about 35 rad/s of commanded
knee speed against a 3.67 rad/s motor -- and the joint answers by reversing
direction ~25 times a second while the foot taps the tread 2-3 times per step
instead of once. The straight deliverable, whose knee command walks smoothly
through its range, taps 0.4 times per step.

Filtering that command after the fact does not work: capping the target's rate
on the trained policy (slew_check.py, knee joints only) drops the taps but the
robot stops climbing and falls 12-26 times per 18 s, and the joint acceleration
*doubles* as it stumbles. The fast flick is load-bearing for that policy, so the
limit has to be part of the dynamics the policy trains against -- which is what
this term provides. It is also the right place for it on hardware: the same
limit then applies at deploy time, with no separate filter to drift out of sync.

Semantics: ``process_actions`` applies the usual affine transform and clip, then
clamps the change since the previous target to ``max_rate * step_dt``. Joints
without a matching pattern keep the stock behaviour (no limit).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class JointPositionActionRateLimited(JointPositionAction):
    """``JointPositionAction`` whose target may not move faster than ``max_rate``."""

    def __init__(self, cfg: JointPositionActionRateLimitedCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        rates = cfg.max_rate
        if rates is None:
            self._max_step = None
        else:
            if not isinstance(rates, dict):
                rates = {".*": float(rates)}
            self._max_step = torch.full((self.action_dim,), float("inf"), device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(rates, self._joint_names)
            self._max_step[index_list] = torch.tensor(value_list, device=self.device) * self._env.step_dt
        # Seed the previous target with the pose a reset would command, so the
        # first step of an episode is not clamped from an arbitrary zero.
        self._prev = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        if self._max_step is not None:
            step = (self._processed_actions - self._prev).clamp(-self._max_step, self._max_step)
            self._processed_actions = self._prev + step
        self._prev = self._processed_actions.clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._prev[env_ids] = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids]


@configclass
class JointPositionActionRateLimitedCfg(JointPositionActionCfg):
    """Configuration for :class:`JointPositionActionRateLimited`."""

    class_type: type = JointPositionActionRateLimited

    max_rate: dict[str, float] | float | None = None
    """Per-joint target slew limit, in rad/s. A float applies to every joint.

    Joints not matched by any pattern are unlimited. Defaults to None (the stock
    unclamped behaviour), so a task opts in.
    """
