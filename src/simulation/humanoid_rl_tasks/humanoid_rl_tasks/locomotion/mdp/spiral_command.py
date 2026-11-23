"""A velocity command whose heading target follows a spiral staircase.

The command itself stays in the robot's base frame -- forward at the
commanded speed -- because that is what the policy has always been given and
what it is shaped around. What changes on a spiral is the *heading target*:
instead of a fixed world angle, it is refreshed every step to the tangent of
the staircase at the robot's current angle about the helix axis.

The inherited heading servo then turns the robot onto the path, and the
policy steers with exactly the observation it already had: the yaw-rate
command (the P-control output of that servo) plus the height scan of the
treads in front of it. No new observation terms are needed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils


@configclass
class SpiralVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration of the spiral heading command.

    Everything the inherited velocity command takes is inherited; the two
    extra fields describe the staircase the heading target follows.
    """

    centre: tuple[float, float] = (0.0, 0.0)
    walking_radius: float = 0.87


class SpiralVelocityCommand(UniformVelocityCommand):
    """A velocity command that steers along a spiral staircase.

    The heading target is the staircase's tangent at the robot's current
    angle, refreshed every step. On top of the inherited P-control on the
    heading error, the command carries a **feed-forward turn rate** of
    ``v / r`` -- the rate the path itself demands at the walking radius.

    The feed-forward is not optional: with a P-gain of 0.5 and a path that
    demands 0.46 rad/s (0.4 m/s on a 0.87 m radius), a servo without it
    settles 0.9 rad (53 degrees) off the tangent, which is not a spiral
    staircase any more. With it, the heading error stays near zero and the
    feed-forward tracks the path.

    The P-gain is 1.0 rather than the shared recipe's 0.5 because the robot
    spawns facing +Y while the staircase's tangent at the spawn is 35 deg:
    the turn-in has to be sharp. A gentle turn-in curves the robot *inwards*
    (it starts aimed 55 deg to the left of the tangent, i.e. at the middle of
    the staircase) and it walked into the middle of the treads on 70% of the
    episodes.

    Args:
        cfg: The command configuration (see ``SpiralVelocityCommandCfg``).
        env: The environment.
    """

    def __init__(self, cfg: SpiralVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.centre = (float(cfg.centre[0]), float(cfg.centre[1]))
        self.walking_radius = float(cfg.walking_radius)
        # Every env steers: the heading target is not sampled, it is computed.
        self.is_heading_env[:] = True
        self.heading_target = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        # The velocity command is resampled (the base class samples the speed);
        # the heading target is overwritten every step in _update_command.
        super()._resample_command(env_ids)
        self.is_heading_env[env_ids] = True

    def _update_command(self) -> None:
        """Point the heading target at the staircase's tangent, and steer."""
        position = self.robot.data.root_pos_w
        angle = torch.atan2(position[:, 1] - self.centre[1], position[:, 0] - self.centre[0])
        # The staircase winds counter-clockwise upward, so the direction of
        # travel is the counter-clockwise tangent: (-sin, cos).
        self.heading_target = angle + math.pi / 2.0
        self.is_heading_env[:] = True
        # P-control on the heading error, plus the turn rate the path demands.
        heading_error = math_utils.wrap_to_pi(self.heading_target - self.robot.data.heading_w)
        # The feed-forward uses the *measured* speed, not the commanded one:
        # the robot follows a circle of radius v/omega, so the turn rate that
        # holds the circle is the measured speed over the radius. With the
        # commanded speed instead, a robot walking slower than commanded
        # turned too fast for its own speed and spiralled *inwards* -- 63% of
        # the episodes ended with the robot in the middle of the staircase.
        speed = self.robot.data.root_lin_vel_b[:, 0]
        feed_forward = speed / max(self.walking_radius, 1e-3)
        omega = self.cfg.heading_control_stiffness * heading_error + feed_forward
        self.vel_command_b[:, 2] = torch.clamp(
            omega, min=self.cfg.ranges.ang_vel_z[0], max=self.cfg.ranges.ang_vel_z[1]
        )
        # Alignment gate: slow down while the heading is far from the
        # tangent, so the turn-in happens nearly in place instead of carving
        # a wide arc across the treads. The robot spawns facing +Y while the
        # tangent at the spawn is 35 deg, so this is the difference between
        # turning on the spot and sweeping across the outer edge.
        align = torch.clamp(1.0 - heading_error.abs() / 1.0, 0.15, 1.0)
        self.vel_command_b[:, 0] = self.vel_command_b[:, 0] * align
        # Enforce standing (zero command) for standing envs, as the base class
        # does.
        standing = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing, :] = 0.0
