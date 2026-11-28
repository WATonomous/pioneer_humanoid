"""A private explicit-yaw navigation expert above the unchanged physical joint interface."""

import hashlib
from pathlib import Path

import torch

from .approach import yaw_from_quaternion
from .navigation import StagedApproachController, StagedNavigationCfg
from .navigation_frame import in_navigation_frame
from .twist_control import waypoint_twist


# Bind qualification to the helper code that actually turns targets into joint actions.
def source_hashes():
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "twist_navigation.py",
            "navigation_frame.py",
            "twist_control.py",
            "navigation.py",
            "approach.py",
            "strike_history.py",
            "runtime.py",
            "model.py",
            "ball.py",
            "pitch_scene.py",
            "match_rules.py",
            "match_audit.py",
        )
    }


class TwistNavigator:
    # Share measured state and applied actions; keep yaw-task commands separate from ball-skill history.
    def __init__(self, raw, robot, ball, actor, reference):
        from .demo import PolicyHistory

        self.raw, self.robot, self.ball = raw, robot, ball
        self.actor, self.reference = actor, reference
        self.history = PolicyHistory(robot, ball, raw.num_envs, raw.device)
        cfg = StagedNavigationCfg(max_lateral=0.1, max_hold_speed=0.08)
        self.controllers = [
            StagedApproachController(1, raw.device, cfg, raw.step_dt)
            for _ in range(raw.num_envs)
        ]
        self.reset()

    # A referee reset clears controller memory without creating any additional physical reset.
    def reset(self):
        n, device = self.raw.num_envs, self.raw.device
        self.history.reset()
        self.previous_active = torch.zeros(n, dtype=torch.bool, device=device)
        self.escaping = torch.zeros_like(self.previous_active)
        self.planned_ball = torch.full((n, 2), float("nan"), device=device)
        self.command = torch.zeros(n, 3, device=device)
        self.target = torch.zeros(n, 2, device=device)
        self.cached_action = torch.zeros(n, 16, device=device)
        self.sampled_step = None
        self.samples = 0

    # Plan only the active robot's route; close contact first needs a genuine outward recovery.
    def _plan(self, i, position, ball, goal):
        gap = (position - ball).norm()
        self.escaping[i] = gap < 0.65
        if not self.escaping[i]:
            self.controllers[i].reset(
                torch.zeros(1, dtype=torch.long, device=self.raw.device),
                position[None],
                ball[None],
                goal[None],
            )
        self.planned_ball[i] = ball

    # Sample each physical step once, including inactive steps, so GUI calls cannot advance a fake history.
    @torch.no_grad()
    def step(self, active, goals, applied_actions):
        from .demo import ball_independent_gait

        counter = int(self.raw.common_step_counter)
        if counter == self.sampled_step:
            return self.cached_action, self.command
        self.sampled_step = counter
        self.samples += 1
        self.history.last.copy_(applied_actions)
        self.command.zero_()
        position = self.robot.data.root_link_pos_w[:, :2]
        ball = self.ball.data.root_link_pos_w[:, :2]
        heading = yaw_from_quaternion(self.robot.data.root_link_quat_w)

        # Stage the goal-facing turn well behind the ball before the straight closing segment.
        for i in active.nonzero().flatten().tolist():
            moved = (ball[i] - self.planned_ball[i]).norm() > 0.25
            if not self.previous_active[i] or moved:
                self._plan(i, position[i], ball[i], goals[i])
            gap = (position[i] - ball[i]).norm()
            if gap < 0.6:
                self.escaping[i] = True
            if self.escaping[i] and gap >= 1.3:
                self._plan(i, position[i], ball[i], goals[i])
            if self.escaping[i]:
                radial = position[i] - ball[i]
                if gap < 0.05:
                    radial = ball[i] - goals[i]
                radial = radial / radial.norm().clamp(min=1e-6)
                self.target[i] = ball[i] + 1.4 * radial
                self.command[i] = waypoint_twist(
                    position[i : i + 1],
                    heading[i : i + 1],
                    self.target[i : i + 1],
                    maximum=0.2,
                )[0]
            else:
                body, metrics = self.controllers[i].update(
                    position[i : i + 1],
                    heading[i : i + 1],
                    ball[i : i + 1],
                    goals[i : i + 1],
                )
                c, s = heading[i].cos(), heading[i].sin()
                self.command[i, 0] = c * body[0, 0] - s * body[0, 1]
                self.command[i, 1] = s * body[0, 0] + c * body[0, 1]
                self.command[i, 2] = body[0, 2]
                self.target[i] = metrics["waypoint"][0]

        # Feed the actor real physical history in its trained frame and retain its actual yaw intent privately.
        self.previous_active.copy_(active)
        observations = self.history.sample(self.command)
        observations = in_navigation_frame(
            ball_independent_gait(observations), self.reference
        )
        self.cached_action = self.actor(observations).clamp(-3, 3)
        return self.cached_action, self.command
