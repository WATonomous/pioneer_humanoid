"""Geometry and feedback navigation above the unchanged walking policy."""

import heapq
import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ApproachCfg:
    standoff: float = 0.85
    keepout: float = 0.60
    planning_radius: float = 0.82
    orbit_radius: float = 1.10
    waypoint_tolerance: float = 0.12
    align_enter: float = 0.15
    align_exit: float = 0.23
    position_tolerance: float = 0.20
    heading_tolerance: float = 0.20
    hold_seconds: float = 2.0
    ball_motion_tolerance: float = 0.05
    max_forward: float = 0.35
    forward_command_limit: float = 0.6
    max_lateral: float = 0.12
    max_yaw: float = 0.5
    linear_acceleration: float = 0.4
    angular_acceleration: float = 1.0
    hold_position_gain: float = 1.2
    max_hold_speed: float = 0.12
    hold_deadband: float = 0.04
    tracking_margin: float = 0.18
    lateral_offset: float = 0.0
    translation_heading_limit: float = 0.50

    def __post_init__(self):
        if not all(
            math.isfinite(x) and x > 0
            for name, x in vars(self).items()
            if name not in ("lateral_offset", "adaptive_speed")
        ):
            raise ValueError("Controller parameters must be finite and positive")
        if not math.isfinite(self.lateral_offset) or abs(self.lateral_offset) > 0.3:
            raise ValueError("Lateral setup offset must be finite and at most 0.3 m")
        if not self.keepout < self.planning_radius < self.standoff < self.orbit_radius:
            raise ValueError("Require keepout < planning radius < standoff < orbit")
        if not self.align_enter < self.position_tolerance < self.align_exit:
            raise ValueError("Alignment hysteresis must bracket position tolerance")
        if not 0 < self.translation_heading_limit <= math.pi / 2:
            raise ValueError("Translation heading limit must be within (0, pi/2]")
        if self.keepout + self.tracking_margin >= self.standoff:
            raise ValueError("Tracking guard must leave the stopping point reachable")
        if (
            self.max_forward > self.forward_command_limit
            or self.max_lateral > 0.2
            or self.max_yaw > 0.5
        ):
            raise ValueError("Commands must stay inside the trained velocity range")
        if self.max_hold_speed > min(self.max_forward, self.max_lateral):
            raise ValueError("Hold speed must respect the navigation speed bounds")
        if self.hold_deadband >= self.align_enter:
            raise ValueError("Hold deadband must be smaller than alignment entry")


DEFAULT_APPROACH_CFG = ApproachCfg()


def segment_clearance(a, b, center):
    a, b, center = np.asarray(a), np.asarray(b), np.asarray(center)
    ab = b - a
    t = np.clip(np.dot(center - a, ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return float(np.linalg.norm(a + t * ab - center))


def positioning_target(ball, goal, standoff=0.85, lateral_offset=0.0):
    ball, goal = np.asarray(ball, dtype=float), np.asarray(goal, dtype=float)
    direction = goal - ball
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        raise ValueError("Ball and goal must be distinct")
    direction /= norm
    sideways = np.array([-direction[1], direction[0]])
    return ball - standoff * direction - lateral_offset * sideways


def plan_route(start, ball, goal, cfg=DEFAULT_APPROACH_CFG):
    """Shortest visibility-graph polyline around a ball exclusion disk.

    The graph includes a ring of candidate waypoints; every selected segment
    clears the planning radius. This is a static-ball planner, not a field-wide
    obstacle planner or a recovery strategy for starting inside the disk.
    """
    start, ball = np.asarray(start, dtype=float), np.asarray(ball, dtype=float)
    target = positioning_target(ball, goal, cfg.standoff, cfg.lateral_offset)
    if np.linalg.norm(start - ball) < cfg.planning_radius:
        raise ValueError("Start is inside the ball planning clearance")
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring = ball + cfg.orbit_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    nodes = np.vstack((start, target, ring))
    edges = [[] for _ in nodes]
    for i in range(len(nodes)):
        for j in range(i):
            if (
                segment_clearance(nodes[i], nodes[j], ball)
                >= cfg.planning_radius - 1e-9
            ):
                length = float(np.linalg.norm(nodes[i] - nodes[j]))
                edges[i].append((j, length))
                edges[j].append((i, length))
    distances, previous = {0: 0.0}, {}
    queue = [(0.0, 0)]
    while queue:
        distance, i = heapq.heappop(queue)
        if distance > distances[i]:
            continue
        if i == 1:
            break
        for j, length in edges[i]:
            candidate = distance + length
            if candidate < distances.get(j, math.inf):
                distances[j], previous[j] = candidate, i
                heapq.heappush(queue, (candidate, j))
    if 1 not in distances:
        raise RuntimeError("No ball-avoiding route found")
    ids = [1]
    while ids[-1] != 0:
        ids.append(previous[ids[-1]])
    return nodes[list(reversed(ids))[1:]]


def yaw_from_quaternion(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class ApproachController:
    def __init__(self, num_envs, device, cfg=DEFAULT_APPROACH_CFG, dt=0.02):
        self.cfg, self.device, self.dt = cfg, device, dt
        self.routes = [[] for _ in range(num_envs)]
        self.cursor = [0] * num_envs
        self.commands = torch.zeros((num_envs, 3), device=device)
        self.aligning = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.hold = torch.zeros(num_envs, device=device)
        self.escaping = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids, positions, balls, goals):
        for i in ids.tolist():
            route = plan_route(
                positions[i].cpu().numpy(),
                balls[i].cpu().numpy(),
                goals[i].cpu().numpy(),
                self.cfg,
            )
            self.routes[i] = torch.as_tensor(
                route, dtype=positions.dtype, device=self.device
            )
            self.cursor[i] = 0
        self.commands[ids] = 0
        self.aligning[ids] = False
        self.hold[ids] = 0
        self.escaping[ids] = False

    def update(self, positions, headings, balls, goals):
        cfg = self.cfg
        # Restore clearance before tracking error can carry the robot into the ball.
        gap = (positions - balls).norm(dim=1)
        exit_distance = max(
            cfg.standoff + 0.1, cfg.keepout + cfg.tracking_margin + 0.15
        )
        previous_escape = self.escaping.clone()
        self.escaping = torch.where(
            self.escaping, gap < exit_distance, gap < cfg.keepout + cfg.tracking_margin
        )
        recovered = (previous_escape & ~self.escaping).nonzero().flatten()
        if len(recovered):
            self.reset(recovered, positions, balls, goals)
        direction = goals - balls
        direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sideways = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
        targets = balls - cfg.standoff * direction - cfg.lateral_offset * sideways
        waypoints = targets.clone()
        final_leg = torch.ones(len(positions), dtype=torch.bool, device=self.device)
        for i, route in enumerate(self.routes):
            while (
                self.cursor[i] < len(route) - 1
                and torch.linalg.vector_norm(positions[i] - route[self.cursor[i]])
                < cfg.waypoint_tolerance
            ):
                self.cursor[i] += 1
            if self.cursor[i] < len(route) - 1:
                waypoints[i] = route[self.cursor[i]]
                final_leg[i] = False
        # Escape follows an outward waypoint without asking for untrained reverse walking.
        radial = (positions - balls) / gap[:, None].clamp(min=1e-6)
        escape_target = balls + (exit_distance + 0.15) * radial
        waypoints = torch.where(self.escaping[:, None], escape_target, waypoints)
        final_leg &= ~self.escaping
        distance = (targets - positions).norm(dim=1)
        self.aligning = final_leg & torch.where(
            self.aligning, distance < cfg.align_exit, distance < cfg.align_enter
        )
        error = waypoints - positions
        cosine, sine = headings.cos(), headings.sin()
        ex = cosine * error[:, 0] + sine * error[:, 1]
        ey = -sine * error[:, 0] + cosine * error[:, 1]
        # The walking policy cannot reverse. If the target is too far behind
        # while facing the goal, return to navigation rather than idle forever.
        self.aligning &= ~(
            distance.gt(cfg.position_tolerance) & ex.lt(-cfg.position_tolerance)
        )
        bearing = torch.atan2(error[:, 1], error[:, 0])
        goal_heading = torch.atan2(direction[:, 1], direction[:, 0])
        desired_heading = torch.where(self.aligning, goal_heading, bearing)
        heading_error = wrap_angle(desired_heading - headings)
        alignment_error = wrap_angle(goal_heading - headings)
        desired = torch.zeros_like(self.commands)
        desired[:, 0] = (0.8 * ex).clamp(
            0, cfg.max_forward
        ) * heading_error.cos().clamp(min=0).square()
        # Follow route segments with forward/yaw commands rather than unstable turning strafes.
        desired[:, 1] = 0
        desired[:, :2] *= (heading_error.abs() < cfg.translation_heading_limit)[:, None]
        # Alignment must keep correcting position, not create a zero-command
        # band between position_tolerance and align_exit. First turn sufficiently
        # toward the goal, then use small forward/lateral corrections.
        hold_velocity = torch.stack(
            (
                (cfg.hold_position_gain * ex).clamp(0, cfg.max_hold_speed),
                (cfg.hold_position_gain * ey).clamp(
                    -cfg.max_hold_speed, cfg.max_hold_speed
                ),
            ),
            dim=1,
        )
        hold_velocity *= ((heading_error.abs() < 0.6) & (distance > cfg.hold_deadband))[
            :, None
        ]
        desired[:, :2] = torch.where(
            self.aligning[:, None], hold_velocity, desired[:, :2]
        )
        desired[:, 2] = (1.5 * heading_error).clamp(-cfg.max_yaw, cfg.max_yaw)
        desired[:, 2] = torch.where(heading_error.abs() < 0.04, 0.0, desired[:, 2])
        limits = (
            torch.tensor(
                [
                    cfg.linear_acceleration,
                    cfg.linear_acceleration,
                    cfg.angular_acceleration,
                ],
                device=self.device,
            )
            * self.dt
        )
        self.commands += (desired - self.commands).clamp(-limits, limits)
        positioned = (
            final_leg
            & (distance < cfg.position_tolerance)
            & (alignment_error.abs() < cfg.heading_tolerance)
        )
        self.hold = torch.where(positioned, self.hold + self.dt, 0.0)
        return self.commands.clone(), {
            "target": targets,
            "waypoint": waypoints,
            "position_error": distance,
            "heading_error": alignment_error.abs(),
            "ball_distance": (positions - balls).norm(dim=1),
            "positioned": positioned,
            "success": self.hold >= cfg.hold_seconds,
        }
