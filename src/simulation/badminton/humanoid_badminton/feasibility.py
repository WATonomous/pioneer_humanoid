"""Sim-to-real feasibility metrics as a command term.

The sim caps torque at datasheet peaks but places no bound on joint
velocity and lets peak torque run continuously, so a policy can look great
in sim while demanding motions the real motors cannot deliver. This term
logs, per joint and per episode (wandb Metrics/feasibility/*):

  qvel_peak_jN   peak |joint velocity| (rad/s) — compare against the
                 motors' real speed limits
  tau_peak_jN    peak |actuator torque| (Nm)
  tau_duty_jN    fraction of ticks with |torque| above the datasheet RATED
                 (continuous) value from params arm.torque_rated — a proxy
                 for thermal load; transient swings are fine, a high duty
                 cycle is not

It publishes no observation (empty command) and changes nothing about
training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

import aero

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def joint_label(i: int) -> str:
    """Metric suffix for actuated joint i: j1..j6 right arm, j1l..j6l left."""
    return f"j{i % 6 + 1}" + ("l" if i >= 6 else "")


class FeasibilityCommand(CommandTerm):
    cfg: "FeasibilityCommandCfg"

    def __init__(self, cfg: "FeasibilityCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self._robot = env.scene[cfg.entity_name]
        # right arm j1-j6 then left arm j1-j6 (actuator order), same motors
        rated = aero.load_params()["arm"]["torque_rated"] * 2
        n, dev = self.num_envs, self.device
        self._rated = torch.tensor(rated, device=dev)
        self._nj = len(rated)
        self._qvel_peak = torch.zeros(n, self._nj, device=dev)
        self._tau_peak = torch.zeros(n, self._nj, device=dev)
        self._over = torch.zeros(n, self._nj, device=dev)
        self._ticks = torch.zeros(n, 1, device=dev)
        self._empty = torch.zeros(n, 0, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self._empty

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._qvel_peak[env_ids] = 0.0
        self._tau_peak[env_ids] = 0.0
        self._over[env_ids] = 0.0
        self._ticks[env_ids] = 0.0

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is not None:
            return
        qvel = self._robot.data.joint_vel.abs()
        # qfrc_actuator is the applied joint-space actuator force AFTER the
        # jnt_actfrcrange clamp (the datasheet peaks from build_scene.py);
        # data.actuator_force is the servo's unclamped request
        idx = self._robot.data.indexing.joint_v_adr
        tau = self._robot.data.data.qfrc_actuator[:, idx].abs()
        self._qvel_peak = torch.maximum(self._qvel_peak, qvel)
        self._tau_peak = torch.maximum(self._tau_peak, tau)
        self._over += (tau > self._rated).float()
        self._ticks += 1.0

    def _update_metrics(self) -> None:
        duty = self._over / self._ticks.clamp_min(1.0)
        for i in range(self._nj):
            j = joint_label(i)
            self.metrics[f"qvel_peak_{j}"] = self._qvel_peak[:, i]
            self.metrics[f"tau_peak_{j}"] = self._tau_peak[:, i]
            self.metrics[f"tau_duty_{j}"] = duty[:, i]


@dataclass(kw_only=True)
class FeasibilityCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: "ManagerBasedRlEnv") -> FeasibilityCommand:
        return FeasibilityCommand(self, env)
