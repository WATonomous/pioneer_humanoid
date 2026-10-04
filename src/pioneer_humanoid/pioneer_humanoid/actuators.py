"""Higher-fidelity actuator model for sim2real transfer.

``isaaclab.actuators.ImplicitActuatorCfg`` (used by ``whole_body.py`` today) solves the
joint drive implicitly and effectively instantaneously each physics step: the joint can
move toward any commanded target as fast as ``velocity_limit``/``effort_limit`` allow,
with no communication latency and no internal rate limiting beyond that. Real motor
controllers (the CubeMars/Robstride families this project uses) do not behave that way:
- Commands take one or more control-loop cycles to reach the motor (CAN bus round trip,
  firmware loop, RTOS scheduling jitter).
- Position-mode firmware ramps its own internal setpoint under an acceleration limit
  rather than snapping to a new target instantly.

``DelayedTrapezoidalPDActuator`` below reproduces both effects: it delays incoming
setpoints through a per-env randomizable circular buffer (inherited from
``DelayedPDActuator``), then tracks the (delayed) target with a trapezoidal
(accel/velocity-limited, kinematic-braking) motion profile before running the PD law
against that ramped reference instead of the raw target.

NOTE: ``acceleration_limit`` (and, ideally, the delay range) should be fit against real
step/sine command-response data per joint -- see the "Real hardware validation" section
of the PR this was introduced in. Without that fit, this is a structurally more
realistic actuator model than ``ImplicitActuatorCfg`` but not yet a *validated* one.
"""

from __future__ import annotations

import torch

from isaaclab.actuators import DelayedPDActuator
from isaaclab.actuators.actuator_pd_cfg import DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


@configclass
class DelayedTrapezoidalPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`DelayedTrapezoidalPDActuator`."""

    class_type: type = None  # set below, after the actuator class is defined

    acceleration_limit: dict[str, float] | float = 1.0e9
    """Max magnitude of joint acceleration (rad/s^2) the internal reference can ramp at.

    Defaults effectively unlimited (matches today's ``ImplicitActuatorCfg`` behavior,
    i.e. "opt in by setting a real value"). Must be fit against real step-response data
    per joint before this stops being a placeholder -- see the PR description.
    """

    dt: float = 0.005
    """Control-loop timestep (s) the internal trapezoidal reference is integrated at.

    Should match the real motor controller's internal loop period, not necessarily the
    outer RL policy's control rate. Defaults to 200 Hz (5 ms), a common motor-firmware
    loop rate for this project's actuators -- confirm against real firmware docs.
    """


class DelayedTrapezoidalPDActuator(DelayedPDActuator):
    r"""Delayed PD actuator with a trapezoidal (accel/velocity-limited) internal
    motion-profile reference, on top of :class:`DelayedPDActuator`'s command-delay
    modeling.

    Each ``compute()`` call:

    1. Applies the (per-env, randomizable) command delay to the incoming position,
       velocity, and effort setpoints -- inherited from ``DelayedPDActuator``.
    2. Ramps an internal position/velocity reference toward the delayed position
       target under ``acceleration_limit``, clamped by ``velocity_limit``, braking
       early enough (via :math:`v^2 = 2a \cdot d`) to land on target without overshoot.
    3. Runs the PD law (``stiffness``, ``damping``) against that ramped reference
       (not the raw target), plus any commanded feedforward effort, then clips to
       ``effort_limit``.
    """

    cfg: DelayedTrapezoidalPDActuatorCfg

    def __init__(self, cfg: DelayedTrapezoidalPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.acceleration_limit = self._parse_joint_parameter(cfg.acceleration_limit, torch.inf)
        self._dt = cfg.dt

        self.pos_joint_ref = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        self.vel_joint_ref = torch.zeros_like(self.pos_joint_ref)
        # deferred seeding: envs here get pos_joint_ref/vel_joint_ref re-initialized
        # from actual joint state on their next compute() call, instead of ramping
        # from a stale/zero reference right after a reset.
        self._needs_seed = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        super().reset(env_ids)
        self._needs_seed[env_ids] = True

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply the communication/control delay to all commanded setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        if torch.any(self._needs_seed):
            seed_mask = self._needs_seed
            self.pos_joint_ref[seed_mask] = joint_pos[seed_mask]
            self.vel_joint_ref[seed_mask] = joint_vel[seed_mask]
            self._needs_seed[seed_mask] = False

        # trapezoidal (accel/velocity-limited) ramp of the internal reference toward
        # the (delayed) commanded position target
        error_pos = control_action.joint_positions - self.pos_joint_ref
        dir_pos = torch.sign(error_pos + 1e-12)
        joint_vel_target = self.vel_joint_ref + dir_pos * self.acceleration_limit * self._dt
        joint_vel_target = torch.clamp(joint_vel_target, -self.velocity_limit, self.velocity_limit)
        # kinematic braking distance: start decelerating early enough to land on
        # target without overshoot, instead of ramping past it and correcting back
        stop_dist = 0.5 * joint_vel_target * joint_vel_target / self.acceleration_limit
        need_brake = torch.abs(error_pos) <= stop_dist
        joint_vel_target = torch.where(
            need_brake,
            dir_pos * torch.sqrt(2.0 * self.acceleration_limit * torch.abs(error_pos)),
            joint_vel_target,
        )
        self.pos_joint_ref = self.pos_joint_ref + joint_vel_target * self._dt
        self.vel_joint_ref = joint_vel_target

        # PD law against the ramped reference (not the raw target), plus feedforward
        error_pos = self.pos_joint_ref - joint_pos
        error_vel = self.vel_joint_ref - joint_vel
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action


DelayedTrapezoidalPDActuatorCfg.class_type = DelayedTrapezoidalPDActuator
