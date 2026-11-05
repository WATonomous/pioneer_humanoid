"""Wide staging navigation for precise ball approach with drifting walking policies."""

from dataclasses import dataclass, fields

import numpy as np
import torch

from .approach import (
    ApproachCfg,
    ApproachController,
    plan_route,
    positioning_target,
    wrap_angle,
)


# Bound faster travel by measured braking room and retain the old speed close to the ball.
def travel_speed_limit(ball_gap, waypoint_distance, heading_error, maximum=0.5):
    far = (0.3 + 0.2 * ((ball_gap - 1.35) / 0.65).clamp(0, 1)).clamp(max=maximum)
    braking = torch.sqrt(2 * 0.35 * (waypoint_distance - 0.15).clamp(min=0))
    turning = torch.where(heading_error.abs() > 0.25, 0.3, maximum)
    return torch.minimum(torch.minimum(far, braking), turning)


# Turn to face the goal well outside the final stopping zone.
@dataclass(frozen=True)
class StagedNavigationCfg(ApproachCfg):
    max_forward: float = 0.30
    staging_distance: float = 1.65
    staging_clearance: float = 1.20
    staging_orbit: float = 1.80
    staging_tolerance: float = 0.18
    adaptive_speed: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.staging_tolerance < self.waypoint_tolerance:
            raise ValueError(
                "Staging tolerance must prevent skipping the goal-facing turn"
            )
        if (
            not self.standoff
            < self.staging_clearance
            < self.staging_distance
            < self.staging_orbit
        ):
            raise ValueError(
                "Require stopping point < stage clearance < stage distance < orbit"
            )


# Reuse 4A feedback, but plan the orbit and goal-facing turn before the straight final leg.
class StagedApproachController(ApproachController):
    def __init__(self, num_envs, device, cfg=None, dt=0.02):
        super().__init__(num_envs, device, cfg or StagedNavigationCfg(), dt)
        self.staged_hold = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids, positions, balls, goals):
        cfg = self.cfg
        planning_values = {
            item.name: getattr(cfg, item.name) for item in fields(ApproachCfg)
        }
        planning_values.update(
            standoff=cfg.staging_distance,
            planning_radius=cfg.staging_clearance,
            orbit_radius=cfg.staging_orbit,
        )
        planning = ApproachCfg(**planning_values)
        for i in ids.tolist():
            start, ball, goal = (
                positions[i].cpu().numpy(),
                balls[i].cpu().numpy(),
                goals[i].cpu().numpy(),
            )
            target = positioning_target(ball, goal, cfg.standoff, cfg.lateral_offset)
            gap = float(np.linalg.norm(start - ball))
            if gap < cfg.keepout:
                raise ValueError("Start is inside the ball safety clearance")
            if np.linalg.norm(start - target) < cfg.position_tolerance:
                route = np.array([target])
            else:
                # A close recovery first moves outward before joining the wide orbit.
                prefix = []
                if gap < cfg.staging_clearance:
                    start = ball + cfg.staging_orbit * (start - ball) / gap
                    prefix = [start]
                route = plan_route(start, ball, goal, planning)
                route = np.vstack((*prefix, route, target))
            self.routes[i] = torch.as_tensor(
                route, dtype=positions.dtype, device=self.device
            )
            self.cursor[i] = 0
        self.commands[ids] = 0
        self.aligning[ids] = False
        self.hold[ids] = 0
        self.escaping[ids] = False
        self.staged_hold[ids] = False

    def update(self, positions, headings, balls, goals):
        # Hold only heading at the staging node; do not demand unsupported backward correction.
        direction = goals - balls
        direction /= direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
        goal_heading = torch.atan2(direction[:, 1], direction[:, 0])
        replacements = []
        for i, route in enumerate(self.routes):
            if len(route) < 2 or self.cursor[i] != len(route) - 2:
                continue
            near = (
                positions[i] - route[self.cursor[i]]
            ).norm() < self.cfg.staging_tolerance
            if near or self.staged_hold[i]:
                if abs(float(wrap_angle(goal_heading[i] - headings[i]))) < 0.15:
                    self.cursor[i] += 1
                    self.staged_hold[i] = False
                    # Preserve goal-facing position correction after turn-induced drift near the target.
                    self.aligning[i] = (
                        positions[i] - route[-1]
                    ).norm() < self.cfg.align_exit
                else:
                    self.staged_hold[i] = True
                    replacements.append(
                        (i, self.cursor[i], route[self.cursor[i]].clone())
                    )
                    route[self.cursor[i]] = positions[i] + direction[i]
        previous_forward = self.commands[:, 0].clone()
        commands, metrics = super().update(positions, headings, balls, goals)
        # Apply the far-travel cap without bypassing the controller's acceleration limiter.
        if self.cfg.adaptive_speed:
            delta = metrics["waypoint"] - positions
            angle = wrap_angle(torch.atan2(delta[:, 1], delta[:, 0]) - headings)
            cap = travel_speed_limit(
                (balls - positions).norm(dim=1),
                delta.norm(dim=1),
                angle,
                self.cfg.max_forward,
            ).clamp(min=self.cfg.max_hold_speed)
            bounded = torch.minimum(commands[:, 0], cap)
            commands[:, 0] = torch.maximum(
                bounded,
                (previous_forward - self.cfg.linear_acceleration * self.dt).clamp(
                    min=0
                ),
            )
            self.commands[:, 0] = commands[:, 0]
        # Restore the real path for diagnostics and retain zero translation during the turn.
        for i, j, point in replacements:
            self.routes[i][j] = point
        self.commands[self.staged_hold, :2] = 0
        commands[self.staged_hold, :2] = 0
        metrics["staging"] = self.staged_hold.clone()
        return commands, metrics
