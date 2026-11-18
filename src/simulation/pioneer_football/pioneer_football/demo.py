"""One physical CAD world for learned approach, dribbling, striking and optional one-on-one play."""

import argparse
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from tensordict import TensorDict

from .approach import ApproachCfg, plan_route, wrap_angle, yaw_from_quaternion
from .match_audit import MatchAudit
from .match_rules import MatchRules
from .runtime import (
    ALL_JOINTS,
    RECOVERY_BALL,
    ROOT,
    MovingBallCommand,
    MovingBallCommandCfg,
    contact_scene,
    directed_release,
    goal_crossing,
    load_actor,
    make_env_cfg,
    make_rl_cfg,
    place_trials,
    robot_spec,
)

PHASES = ("APPROACH", "DRIBBLE", "STRIKE", "RECOVER", "DONE", "FAILED")


# Recovery uses the striker already in control, so it must not restart action blending.
def skill_changes(phase, previous_phase):
    return phase.clamp(max=2) != previous_phase.clamp(max=2)


# Allow a direct policy switch when testing whether action blending suppresses a learned strike.
def blend_fraction(age, duration):
    if duration < 0:
        raise ValueError("Skill blend time cannot be negative")
    return torch.ones_like(age) if duration == 0 else (age / duration).clamp(0, 1)


# Test immediate task intent while retaining every measured physical history frame.
def immediate_strike_command(obs, command):
    result = obs.clone()
    result["actor"][:, 162:171] = command[:, None].expand(-1, 3, -1).flatten(1)
    return result


# Supply task progress only to the separate phase-aware striker, preserving shared physical history.
def with_release_phase(obs, history):
    return TensorDict(
        {"actor": torch.cat((obs["actor"], history), dim=1)}, batch_size=obs.batch_size
    )


# Ground-shooting models may use the otherwise-zero command-Z slots for an explicit speed cue.
def with_command_speed(obs, history):
    result = obs.clone()
    result["actor"][:, [164, 167, 170]] = history
    return result


# Isolate locomotion from irrelevant ball features while retaining actual robot state and shared history.
def ball_independent_gait(obs):
    result = obs.clone()
    values = result["actor"]
    values[:, 177:189].reshape(-1, 3, 4)[:] = values.new_tensor([0.7, 0.0, -0.7, 1.0])
    values[:, 189:207].reshape(-1, 3, 6)[:, :, 3:] = 0
    return result


# Recovery keeps the striker; the off-ball patrol is a separate expert rather than a new episode.
def policy_role(phase, defending):
    return torch.where(defending, 3, phase.clamp(max=2))


# Preserve the proven first approach; use the turning expert only after a dribbling attempt loses the ball.
def turning_navigation(phase, defending, had_dribbled, enabled):
    return bool(enabled) & (phase == 0) & ~defending & had_dribbled


# The turning expert was qualified at 0.4–0.7 m/s; retain bearing while enforcing that input range.
def turning_command(command, active):
    result = command.clone()
    speed = command[:, :2].norm(dim=1).clamp(min=1e-6)
    result[active, :2] *= (speed.clamp(0.4, 0.7) / speed)[active, None]
    return result


# Use stationary control after an approach or dribble stoppage while ongoing shots retain their recovery actor.
def stationary_restart(phase, defending, dead_ball, available):
    return (phase < 2) & ~defending & bool(dead_ball) & bool(available)


# Solo sequences fail when the ball leaves play; the match referee owns that event in two-player games.
def field_failure(ball_xy, scored, match):
    return (
        (not match)
        & (ball_xy.abs() > ball_xy.new_tensor([6.11, 4.11])).any(dim=1)
        & ~scored
    )


# The predecessor's last force sample cannot count as a strike before the new actor has acted.
def striker_contact(active, newly_started, strong):
    return active & ~newly_started & strong


# Switch to a kick only after the fastest forward motion and sideways body rocking have eased.
def stable_shot_motion(linear_velocity, angular_velocity):
    return (linear_velocity[:, :2].norm(dim=1) <= 0.8) & (
        angular_velocity[:, :2].norm(dim=1) <= 0.8
    )


# Keep the established shot target through recovery; only non-shooters need a stoppage fallback.
def recovery_command(command, heading, shooting, defending, speed):
    result = command.clone()
    fallback = ~shooting & ~defending
    result[fallback, :2] = (
        speed * torch.stack((heading.cos(), heading.sin()), dim=1)[fallback]
    )
    return result


# Once inside the goal mouth, aim through its volume instead of turning sideways at its front plane.
def goal_target(goals, ball):
    target = goals.clone()
    sign = torch.where(goals[:, 0] >= 0, 1.0, -1.0)
    forward = sign * (goals[:, 0] - ball[:, 0])
    inside = (ball[:, 1] - goals[:, 1]).abs() < 0.8
    target[:, 0] = torch.where(
        inside, ball[:, 0] + sign * forward.clamp(min=0.8), goals[:, 0]
    )
    return target


# Give the learned striker reaction time before contact instead of switching after a dribble tap.
def anticipatory_strike_window(delta, robot_velocity, ball_velocity, quiet):
    gap = delta.norm(dim=1)
    closing = ((robot_velocity - ball_velocity) * delta).sum(dim=1) / gap.clamp(
        min=1e-6
    )
    return (
        (gap >= 0.55)
        & (gap <= 0.95)
        & (closing >= 0.05)
        & (ball_velocity.norm(dim=1) <= 1.0)
        & (quiet >= 0.08)
    )


# Hand off genuine moving possession even during a touch; physical release is checked afterward.
def ready_for_strike(
    phase,
    progress,
    touches,
    distance,
    gap,
    angle,
    quiet,
    shot_distance,
    quiet_limit=1.0,
    alignment_limit=0.8,
):
    return (
        (phase == 1)
        & (progress >= 0.6)
        & (touches >= 2)
        & (distance <= shot_distance)
        & (gap < 1.0)
        & (angle < alignment_limit)
        & (quiet < quiet_limit)
    )


# Keep possession through ordinary stepping, and leave dribbling only after losing the ball or the turn.
def retain_dribbling(active, defending, delta, direction):
    return (
        active
        & ~defending
        & (delta.norm(dim=1) <= 1.6)
        & ((delta * direction).sum(dim=1) >= -0.2)
    )


# Recover a missed setup without erasing a released shot, an actual goal, or the retry budget.
def retry_unreleased_strike(phase, age, released, scored, ball_speed, retries, maximum):
    return (
        (phase == 2)
        & (age >= 2.5)
        & ~released
        & ~scored
        & (ball_speed < 0.15)
        & (retries < maximum)
    )


# Express every world-frame history entry in one common yaw frame; body-local features and actions stay intact.
def canonical_history(values):
    values = values.clone()
    sine, cosine = values[:, 175].clone(), values[:, 176].clone()

    def rotate(vectors):
        x, y = vectors[..., 0].clone(), vectors[..., 1].clone()
        vectors[..., 0] = cosine[:, None] * x + sine[:, None] * y
        vectors[..., 1] = -sine[:, None] * x + cosine[:, None] * y

    rotate(values[:, 162:171].reshape(-1, 3, 3))
    headings = values[:, 171:177].reshape(-1, 3, 2)
    old_sine, old_cosine = headings[..., 0].clone(), headings[..., 1].clone()
    headings[..., 0] = cosine[:, None] * old_sine - sine[:, None] * old_cosine
    headings[..., 1] = sine[:, None] * old_sine + cosine[:, None] * old_cosine
    velocities = values[:, 189:207].reshape(-1, 3, 6)
    rotate(velocities[..., :3])
    rotate(velocities[..., 3:])
    return values


class SoccerCommand(MovingBallCommand):
    # The coordinator owns commands; display its state instead of ineffective manual sliders.
    def create_gui(self, name, server, get_env_idx, **kwargs):
        server.gui.add_markdown(
            "Velocity targets are selected by the soccer coordinator."
        )


@dataclass(kw_only=True)
class SoccerCommandCfg(MovingBallCommandCfg):
    def build(self, env):
        return SoccerCommand(self, env)


class PolicyHistory:
    # Reproduce the trained term-major three-frame interface for either physical humanoid.
    def __init__(self, robot, ball, n, device, canonical=False):
        self.robot, self.ball = robot, ball
        self.canonical = canonical
        self.joints, _ = robot.find_joints(ALL_JOINTS, preserve_order=True)
        self.last = torch.zeros(n, 16, device=device)
        self.history = None

    def reset(self):
        self.last.zero_()
        self.history = None

    def sample(self, command):
        data, ball = self.robot.data, self.ball.data
        yaw = yaw_from_quaternion(data.root_link_quat_w)
        relative = quat_apply_inverse(
            data.root_link_quat_w, ball.root_link_pos_w - data.root_link_pos_w
        )
        terms = (
            data.root_link_ang_vel_b,
            data.projected_gravity_b,
            data.joint_pos[:, self.joints] - data.default_joint_pos[:, self.joints],
            data.joint_vel[:, self.joints] - data.default_joint_vel[:, self.joints],
            self.last,
            command,
            torch.stack((yaw.sin(), yaw.cos()), dim=1),
            torch.cat((relative, torch.ones_like(relative[:, :1])), dim=1),
            torch.cat((data.root_link_lin_vel_w, ball.root_link_lin_vel_w), dim=1),
        )
        self.history = (
            [value[:, None].repeat(1, 3, 1) for value in terms]
            if self.history is None
            else [
                torch.cat((old[:, 1:], value[:, None]), dim=1)
                for old, value in zip(self.history, terms, strict=True)
            ]
        )
        values = torch.cat([value.flatten(1) for value in self.history], dim=1)
        if values.shape[1] != 207:
            raise RuntimeError("Incorrect direct-policy observation layout")
        if self.canonical:
            values = canonical_history(values)
        return TensorDict({"actor": values}, batch_size=[len(values)])


# Ask the learned gait to turn gradually and slow near route nodes; never set physical velocity.
def travel_command(position, heading, target, maximum=0.7):
    delta = target - position
    error = wrap_angle(torch.atan2(delta[:, 1], delta[:, 0]) - heading)
    speed = torch.sqrt(0.8 * delta.norm(dim=1)).clamp(0.25, maximum)
    speed = torch.where(error.abs() > 0.45, speed.clamp(max=0.3), speed)
    angle = heading + error.clamp(-0.4, 0.4)
    return torch.stack(
        (speed * angle.cos(), speed * angle.sin(), torch.zeros_like(speed)), dim=1
    )


# Keep an unpossessed player moving around the ball instead of ordering an untrained zero-speed stop.
def shadow_target(position, ball):
    # Keep the whole loitering circle inside the pitch instead of clipping its next waypoint behind the robot.
    center = ball.clamp(ball.new_tensor([-3.0, -1.0]), ball.new_tensor([3.0, 1.0]))
    radial = position - center
    angle = torch.atan2(radial[:, 1], radial[:, 0]) + 0.5
    return center + 2.0 * torch.stack((angle.cos(), angle.sin()), dim=1)


# Follow from a separate lane; this initial opponent does not claim trained tackling ability.
def pursuit_target(ball, attack_direction):
    side = torch.stack((-attack_direction[:, 1], attack_direction[:, 0]), dim=1)
    target = ball - 2.0 * attack_direction + 1.2 * side
    return target.clamp(target.new_tensor([-5.0, -3.0]), target.new_tensor([5.0, 3.0]))


# Correct lateral drift early while commanding learned ball velocity, and aim through the goal plane.
def ball_path_command(ball, velocity, goals, speed, direct_goal=False):
    if direct_goal:
        direction = goals - ball
        return speed * direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
    sign = torch.where(goals[:, 0] >= 0, 1.0, -1.0)
    forward = (sign * (goals[:, 0] - ball[:, 0])).clamp(0.6, 1.2)
    lateral = goals[:, 1] - ball[:, 1] - 0.4 * velocity[:, 1]
    direction = torch.stack((sign * forward, lateral), dim=1)
    return speed * direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-6)


# Gradually request slower learned ball control before shooting, without prescribing joint motion.
def preparation_speed(distance, maximum, shot_distance, preparation_distance, minimum):
    if preparation_distance is None:
        return torch.full_like(distance[:, None], maximum)
    fraction = (
        (distance - shot_distance) / (preparation_distance - shot_distance)
    ).clamp(0, 1)
    return (minimum + (maximum - minimum) * fraction)[:, None]


class ApproachPath:
    # Replan around a moving ball instead of preserving a stale route after an opponent touches it.
    def __init__(self, n, device):
        self.routes = [None] * n
        self.balls = torch.full((n, 2), float("inf"), device=device)
        self.cfg = ApproachCfg(lateral_offset=0.15)

    def targets(self, positions, balls, goals, active):
        target = balls.clone()
        for i in active.nonzero().flatten().tolist():
            gap = float((positions[i] - balls[i]).norm())
            if gap < 0.86:
                radial = (positions[i] - balls[i]) / max(gap, 1e-6)
                target[i] = balls[i] + 1.2 * radial
                self.routes[i] = None
                continue
            if (
                self.routes[i] is None
                or float((balls[i] - self.balls[i]).norm()) > 0.25
            ):
                route = plan_route(
                    positions[i].cpu().numpy(),
                    balls[i].cpu().numpy(),
                    goals[i].cpu().numpy(),
                    self.cfg,
                )
                self.routes[i] = torch.as_tensor(
                    route, dtype=positions.dtype, device=positions.device
                )
                self.balls[i] = balls[i]
            route = self.routes[i]
            while len(route) > 1 and float((positions[i] - route[0]).norm()) < 0.2:
                route = route[1:]
            self.routes[i] = route
            target[i] = route[0]
        return target


# Separate the opponent's compiled spawn and use the same physical ball-foot material law.
def opponent_spec():
    spec = robot_spec()
    spec.body("base").pos = [1.8, -0.5, 0.9]
    for geom in spec.geoms:
        if geom.group == 2 and geom.parent.name == "base_link":
            geom.rgba = [0.2, 0.4, 0.9, 1.0]
    return spec


def two_team_scene(spec):
    contact_scene(spec)
    for i, side in enumerate(("l", "r")):
        spec.add_pair(
            name=f"opponent_ball_foot_{i}",
            geomname1="ball/ball_geom",
            geomname2=f"opponent/{side}_foot_collision_0",
            condim=3,
            friction=[RECOVERY_BALL.foot_friction] * 2 + [0, 0, 0],
            solref=[RECOVERY_BALL.contact_time, 0.14],
        )


def make_cfg(n, seconds, opponent, seed):
    cfg = make_env_cfg(play=True)
    original = cfg.commands["ball_velocity"]
    cfg.commands["ball_velocity"] = SoccerCommandCfg(
        **{field.name: getattr(original, field.name) for field in fields(original)}
    )
    cfg.scene.num_envs = n
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.episode_length_s = seconds + 5
    cfg.terminations.pop("possession_lost", None)
    cfg.terminations.pop("pitch_end", None)
    if opponent:
        # Match rules own falls, time and restarts; framework termination would demand an immediate reset.
        cfg.terminations.clear()
        cfg.scene.entities["opponent"] = deepcopy(cfg.scene.entities["robot"])
        cfg.scene.entities["opponent"].spec_fn = opponent_spec
        cfg.scene.entities["opponent"].init_state.pos = (1.8, -0.5, 0.841065)
        cfg.scene.spec_fn = two_team_scene
        cfg.sim.nconmax = 512
        cfg.sim.njmax = 2048
        cfg.scene.sensors += (
            ContactSensorCfg(
                name="opponent_ball",
                primary=ContactMatch(
                    mode="body", pattern=("l_foot", "r_foot"), entity="opponent"
                ),
                secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
                fields=("force",),
                reduce="netforce",
                history_length=cfg.decimation,
            ),
        )
    cfg.viewer.distance = 10.0
    cfg.viewer.elevation = -30.0
    cfg.viewer.max_extra_envs = 0
    return cfg


class Episode:
    # Share raw-action history across skills; all actors use the same source motors and arm mapping.
    def __init__(self, raw, env, args):
        self.raw, self.env, self.args = raw, env, args
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.robot, self.ball = raw.scene["robot"], raw.scene["ball"]
        self.gait = load_actor(args.gait_checkpoint, 207, make_rl_cfg(), raw.device)
        gait_saved = torch.load(
            args.gait_checkpoint, map_location="cpu", weights_only=False
        )
        self.gait_ball_independent = bool(
            (gait_saved.get("infos") or {}).get("navigation_ball_independent")
        )
        self.dribble = load_actor(
            args.dribble_checkpoint, 207, make_rl_cfg(), raw.device
        )
        strike_saved = torch.load(
            args.strike_checkpoint, map_location="cpu", weights_only=False
        )
        self.strike_inputs = strike_saved["actor_state_dict"]["mlp.0.weight"].shape[1]
        self.strike_speed_features = (strike_saved.get("infos") or {}).get(
            "strike_speed_features"
        )
        if self.strike_inputs not in (207, 210):
            raise ValueError(
                "Striker must use the shared history or the phase-aware history"
            )
        self.strike = load_actor(
            args.strike_checkpoint, self.strike_inputs, make_rl_cfg(), raw.device
        )
        # Use the qualified patrol expert only off the ball; solo approach keeps the original gait.
        self.patrol = None
        if args.opponent and getattr(args, "patrol_checkpoint", None):
            saved = torch.load(
                args.patrol_checkpoint, map_location="cpu", weights_only=False
            )
            if not (saved.get("infos") or {}).get("navigation_ball_independent"):
                raise ValueError(
                    "Patrol requires a ball-independent navigation checkpoint"
                )
            self.patrol = load_actor(
                args.patrol_checkpoint, 207, make_rl_cfg(), raw.device
            )
        self.turn = None
        self.turn_twist_reference = None
        if args.opponent and getattr(args, "turn_checkpoint", None):
            saved = torch.load(
                args.turn_checkpoint, map_location="cpu", weights_only=False
            )
            if not (saved.get("infos") or {}).get("navigation_ball_independent"):
                raise ValueError(
                    "Turning requires a ball-independent navigation checkpoint"
                )
            if (saved.get("infos") or {}).get("navigation_twist"):
                self.turn_twist_reference = saved["infos"]["navigation_yaw_reference"]
            # Loading an extra expert must not change the environment's sampled starting states.
            device = torch.device(raw.device)
            with torch.random.fork_rng(
                devices=[device.index or 0] if device.type == "cuda" else []
            ):
                self.turn = load_actor(
                    args.turn_checkpoint, 207, make_rl_cfg(), raw.device
                )
        canonical = getattr(args, "canonical_heading", False)
        self.history = PolicyHistory(
            self.robot, self.ball, raw.num_envs, raw.device, canonical
        )
        self.opponent_history = (
            PolicyHistory(
                raw.scene["opponent"], self.ball, raw.num_envs, raw.device, canonical
            )
            if args.opponent
            else None
        )
        self.twist_navigation = self.opponent_twist_navigation = None
        self.controller_dependencies = {}
        if self.turn_twist_reference is not None:
            from .twist_navigation import TwistNavigator, source_hashes

            self.controller_dependencies = source_hashes()
            self.twist_navigation = TwistNavigator(
                raw, self.robot, self.ball, self.turn, self.turn_twist_reference
            )
            self.opponent_twist_navigation = TwistNavigator(
                raw,
                raw.scene["opponent"],
                self.ball,
                self.turn,
                self.turn_twist_reference,
            )
        self.native_names = raw.action_manager.get_term("joint_pos").target_names
        # Record physical foot approach during transition diagnostics without changing policy inputs.
        self.foot_ids = self.robot.find_bodies(
            ("l_foot", "r_foot"), preserve_order=True
        )[0]
        self.opponent_ids = (
            raw.scene["opponent"].find_joints(self.native_names, preserve_order=True)[0]
            if args.opponent
            else None
        )
        self.match_touch_counts = [0, 0]
        self.match_was_touch = [False, False]
        self.rng = np.random.default_rng(args.seed)
        self.handoff_states = []
        self.reset(kickoff=0 if args.opponent else None)

    # Physical placement occurs only at an episode boundary or a declared match kickoff.
    def reset(self, kickoff=None):
        self.raw.reset()
        n = self.raw.num_envs
        device = self.raw.device
        layouts = []
        for i in range(n):
            lateral = (0.0, -0.3, 0.3)[i % 3]
            position = (
                np.array([-2.8, lateral])
                if kickoff is None
                else np.array([-0.95 if kickoff == 0 else -1.8, -0.15])
            )
            heading = 0.0
            if kickoff == 1:
                if getattr(self.args, "defender_mode", "orbit") == "pursuit":
                    position = np.array([2.4, -1.2])
                angle = np.arctan2(position[1], position[0]) + 0.5
                target = 2 * np.array([np.cos(angle), np.sin(angle)])
                heading = float(np.arctan2(*(target - position)[::-1]))
                if getattr(self.args, "defender_mode", "orbit") == "pursuit":
                    heading = math.pi
            layouts.append(
                (
                    "joga_soccer",
                    position + self.rng.uniform(-0.08, 0.08, 2)
                    if kickoff is None
                    else position,
                    np.array([-0.5, 0.0]) + self.rng.uniform(-0.04, 0.04, 2)
                    if kickoff is None
                    else np.zeros(2),
                    np.array([6.0, 0.0]),
                    float(self.rng.uniform(-0.06, 0.06))
                    if kickoff is None
                    else heading,
                )
            )
        place_trials(self.raw, torch.arange(n, device=device), layouts)
        if self.args.opponent:
            opponent = self.raw.scene["opponent"]
            state = opponent.data.default_root_state.clone()
            state[:, :2] = state.new_tensor(
                [[0.95, 0.15] if kickoff == 1 else [1.8, -0.5]]
            )
            opponent_heading = math.pi
            if kickoff == 0:
                position = np.array([1.8, -0.5])
                angle = np.arctan2(position[1], position[0]) + 0.5
                target = 2 * np.array([np.cos(angle), np.sin(angle)])
                opponent_heading = float(np.arctan2(*(target - position)[::-1]))
                if getattr(self.args, "defender_mode", "orbit") == "pursuit":
                    state[:, :2] = state.new_tensor([[-2.4, 1.2]])
                    opponent_heading = 0.0
            state[:, 3:7] = state.new_tensor(
                [
                    [
                        math.cos(opponent_heading / 2),
                        0.0,
                        0.0,
                        math.sin(opponent_heading / 2),
                    ]
                ]
            )
            state[:, 7:] = 0
            opponent.write_root_state_to_sim(state)
            # Reset events initialize only the primary robot, so explicitly restore the opponent's bent-knee pose.
            opponent.write_joint_state_to_sim(
                opponent.data.default_joint_pos.clone(),
                torch.zeros_like(opponent.data.joint_vel),
            )
            opponent.set_joint_position_target(opponent.data.default_joint_pos)
            self.raw.sim.forward()
            self.raw.scene.update(self.raw.step_dt)
            self.opponent_history.reset()
        self.goals = torch.tensor([[6.0, 0.0]] * n, device=device)
        self.phase = torch.zeros(n, dtype=torch.long, device=device)
        self.defending = torch.zeros(n, dtype=torch.bool, device=device)
        self.possession_team = torch.full(
            (n,), kickoff or 0, dtype=torch.long, device=device
        )
        self.previous_phase = torch.full_like(self.phase, -1)
        self.previous_policy_role = torch.full_like(self.phase, -1)
        self.dead_ball = False
        self.had_dribbled = torch.zeros_like(self.phase, dtype=torch.bool)
        self.opponent_had_dribbled = torch.zeros_like(self.had_dribbled)
        self.elapsed = torch.zeros(n, device=device)
        self.phase_age = torch.zeros_like(self.elapsed)
        self.quiet = torch.full_like(self.elapsed, 10.0)
        self.settled_age = torch.zeros_like(self.elapsed)
        self.touches = torch.zeros_like(self.phase)
        self.dribble_touches = torch.zeros_like(self.phase)
        self.shot_touches = torch.zeros_like(self.phase)
        self.dribble_origin = self.ball.data.root_link_pos_w[:, :2].clone()
        self.dribble_distance = torch.zeros_like(self.elapsed)
        self.dribble_direction = self.goals - self.dribble_origin
        self.dribble_direction /= self.dribble_direction.norm(
            dim=1, keepdim=True
        ).clamp(min=1e-6)
        self.shot_direction = torch.zeros(n, 2, device=device)
        self.scored = torch.zeros(n, dtype=torch.bool, device=device)
        self.released = self.scored.clone()
        self.shot_scored = self.scored.clone()
        self.strike_contact_seen = self.scored.clone()
        self.strike_release_history = torch.zeros(n, 3, device=device)
        self.strike_speed_history = torch.zeros(n, 3, device=device)
        self.previous_strike_contact = self.scored.clone()
        self.post_release_recontacts = torch.zeros_like(self.phase)
        self.unreleased_retries = torch.zeros_like(self.phase)
        self.applied = torch.zeros(n, 16, device=device)
        self.blend_origin = self.applied.clone()
        self.blend_age = torch.zeros_like(self.elapsed)
        self.opponent_applied = self.applied.clone()
        self.opponent_blend_origin = self.applied.clone()
        self.opponent_blend_age = self.blend_age.clone()
        self.opponent_previous_skill = torch.full_like(self.phase, -1)
        self.opponent_shot = torch.zeros(n, dtype=torch.bool, device=device)
        self.opponent_released = self.opponent_shot.clone()
        self.opponent_strike_contact_seen = self.opponent_shot.clone()
        self.opponent_release_history = torch.zeros(n, 3, device=device)
        self.opponent_speed_history = torch.zeros(n, 3, device=device)
        self.opponent_shot_direction = torch.zeros(n, 2, device=device)
        self.opponent_shot_age = torch.zeros_like(self.elapsed)
        self.opponent_quiet = torch.full_like(self.elapsed, 10.0)
        self.opponent_touches = torch.zeros_like(self.phase)
        # Mirror the primary player's latched possession rather than switching policies on every tap.
        self.opponent_dribbling = self.opponent_shot.clone()
        self.opponent_dribble_origin = self.dribble_origin.clone()
        self.opponent_dribble_direction = -self.goals - self.opponent_dribble_origin
        self.opponent_dribble_direction /= self.opponent_dribble_direction.norm(
            dim=1, keepdim=True
        ).clamp(min=1e-6)
        self.opponent_dribble_touches = torch.zeros_like(self.phase)
        self.previous_ball = self.ball.data.root_link_pos_w.clone()
        self.reason = ["running"] * n
        self.transitions = [[] for _ in range(n)]
        self.live_resets = torch.zeros(n, dtype=torch.long, device=device)
        self.paths = ApproachPath(n, device)
        self.opponent_paths = ApproachPath(n, device)
        self.history.reset()
        for navigator in (self.twist_navigation, self.opponent_twist_navigation):
            if navigator is not None:
                navigator.reset()
        self.last_step = int(self.raw.episode_length_buf[0])
        self.last_episode_steps = self.raw.episode_length_buf.clone()

    @torch.no_grad()
    def actions(self, dead_ball=False):
        self.dead_ball = bool(dead_ball)
        data = self.robot.data
        ball = self.ball.data.root_link_pos_w[:, :2]
        position = data.root_link_pos_w[:, :2]
        heading = yaw_from_quaternion(data.root_link_quat_w)
        aiming = goal_target(self.goals, ball)
        direction = aiming - ball
        direction /= direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
        if self.args.opponent:
            # Give the closer player possession priority; the other may recover a genuinely loose ball.
            own_gap = (position - ball).norm(dim=1)
            opponent_gap = (
                self.raw.scene["opponent"].data.root_link_pos_w[:, :2] - ball
            ).norm(dim=1)
            holder = torch.where(self.possession_team == 0, own_gap, opponent_gap)
            challenger = torch.where(self.possession_team == 0, opponent_gap, own_gap)
            takeover = (challenger + 0.35 < holder) & (challenger < 0.8)
            self.possession_team[takeover] = 1 - self.possession_team[takeover]
            self.defending = (self.phase == 0) & (self.possession_team == 1)
        target = self.paths.targets(
            position, ball, aiming, (self.phase == 0) & ~self.defending
        )
        defender_target = shadow_target(
            position, torch.zeros_like(ball) if self.patrol is not None else ball
        )
        if (
            self.args.opponent
            and getattr(self.args, "defender_mode", "orbit") == "pursuit"
        ):
            attack = -self.goals - ball
            attack /= attack.norm(dim=1, keepdim=True).clamp(min=1e-6)
            defender_target = pursuit_target(ball, attack)
        target = torch.where(self.defending[:, None], defender_target, target)
        command = travel_command(position, heading, target)
        if self.patrol is not None:
            patrol_command = travel_command(
                position, heading, defender_target, maximum=0.5
            )
            command[self.defending] = patrol_command[self.defending]
        ball_command = ball_path_command(
            ball,
            self.ball.data.root_link_lin_vel_w[:, :2],
            aiming,
            preparation_speed(
                (self.goals - ball).norm(dim=1),
                self.args.dribble_speed,
                self.args.shot_distance,
                getattr(self.args, "shot_preparation_distance", None),
                getattr(self.args, "shot_preparation_speed", 0.35),
            ),
            direct_goal=getattr(self.args, "direct_goal_command", False),
        )
        command[self.phase == 1, :2] = ball_command[self.phase == 1]
        shooting = (self.phase == 2) | (self.phase == 3) | (self.phase >= 4)
        command[shooting, :2] = self.args.shot_speed * self.shot_direction[shooting]
        # During a referee stoppage, the active player uses learned shot recovery while defenders keep patrolling.
        if self.dead_ball:
            command = recovery_command(
                command, heading, shooting, self.defending, self.args.shot_speed
            )
        turning = turning_navigation(
            self.phase, self.defending, self.had_dribbled, self.turn is not None
        ) & (not self.dead_ball)
        command = turning_command(command, turning)
        standing = stationary_restart(
            self.phase,
            self.defending,
            self.dead_ball,
            self.twist_navigation is not None,
        )
        navigation_action = None
        if self.twist_navigation is not None:
            navigation_action, navigation_command = self.twist_navigation.step(
                turning, aiming, self.applied
            )
            command[turning, :2] = navigation_command[turning, :2]
            command[standing, :2] = 0
        term = self.raw.command_manager.get_term("ball_velocity")
        term.velocity_w[:] = command
        term.time_left.fill_(1e6)
        obs = self.history.sample(command)
        gait = self.gait(
            ball_independent_gait(obs) if self.gait_ball_independent else obs
        )
        dribble = self.dribble(obs)
        strike_obs = (
            immediate_strike_command(obs, command)
            if getattr(self.args, "instant_strike_command", False)
            else obs
        )
        if self.strike_inputs == 210:
            self.strike_release_history = torch.cat(
                (self.strike_release_history[:, 1:], self.released.float()[:, None]),
                dim=1,
            )
            strike_obs = with_release_phase(strike_obs, self.strike_release_history)
        if self.strike_speed_features:
            info = self.strike_speed_features
            speed = (command[:, :2].norm(dim=1) - info["center"]) / info["scale"]
            self.strike_speed_history = torch.cat(
                (
                    self.strike_speed_history[:, 1:],
                    torch.where(shooting, speed, 0.0)[:, None],
                ),
                dim=1,
            )
            strike_obs = with_command_speed(strike_obs, self.strike_speed_history)
        strike = self.strike(strike_obs)
        desired = torch.where(
            (self.phase == 0)[:, None],
            gait,
            torch.where((self.phase == 1)[:, None], dribble, strike),
        ).clamp(-3, 3)
        if self.turn is not None:
            desired = torch.where(
                turning[:, None],
                navigation_action
                if navigation_action is not None
                else self.turn(ball_independent_gait(obs)),
                desired,
            ).clamp(-3, 3)
        if self.patrol is not None:
            desired = torch.where(
                self.defending[:, None],
                self.patrol(ball_independent_gait(obs)),
                desired,
            ).clamp(-3, 3)
        if self.dead_ball:
            recovery = strike.clamp(-3, 3)
            if navigation_action is not None:
                recovery = torch.where(standing[:, None], navigation_action, recovery)
            desired = torch.where(self.defending[:, None], desired, recovery)
        role = policy_role(
            self.phase,
            self.defending
            if self.patrol is not None
            else torch.zeros_like(self.defending),
        )
        role = torch.where(turning, 4, role)
        if self.dead_ball:
            role = torch.where(self.defending, role, torch.where(standing, 5, 2))
        changed = role != self.previous_policy_role
        self.blend_origin[changed] = self.applied[changed]
        self.blend_age[changed] = 0
        self.blend_age += self.raw.step_dt
        fraction = blend_fraction(
            self.blend_age, getattr(self.args, "skill_blend_time", 0.08)
        )[:, None]
        # Test the trained direct strike switch without changing approach or dribbling transitions.
        if getattr(self.args, "strike_blend_time", None) is not None:
            fraction = torch.where(
                (self.phase >= 2)[:, None],
                blend_fraction(self.blend_age, self.args.strike_blend_time)[:, None],
                fraction,
            )
        self.applied = self.blend_origin.lerp(desired, fraction)
        self.previous_phase[:] = self.phase
        self.previous_policy_role[:] = role
        self.history.last.copy_(self.applied)
        if self.args.opponent:
            self.opponent_actions()
        return self.applied

    # Initially script only tactical pursuit; the opponent's physical joint actions are learned too.
    def opponent_actions(self):
        opponent = self.raw.scene["opponent"]
        data = opponent.data
        ball = self.ball.data.root_link_pos_w[:, :2]
        position = data.root_link_pos_w[:, :2]
        goals = -self.goals
        aiming = goal_target(goals, ball)
        direction = aiming - ball
        direction /= direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
        heading = yaw_from_quaternion(data.root_link_quat_w)
        alignment = wrap_angle(
            torch.atan2(direction[:, 1], direction[:, 0]) - heading
        ).abs()
        near = (
            ((ball - position).norm(dim=1) < 1.0)
            & (((ball - position) * direction).sum(dim=1) > 0.15)
            & (alignment < 0.4)
        )
        defending = self.possession_team == 0
        near &= ~defending
        # The opponent uses the same learned striker after genuine dribbling progress and contacts.
        contact = (
            self.raw.scene["opponent_ball"]
            .data.force_history.norm(dim=-1)
            .amax(dim=(1, 2))
            > 0.1
        )
        self.opponent_touches += (contact & (self.opponent_quiet >= 0.12)).long()
        self.opponent_quiet = torch.where(
            contact, 0.0, self.opponent_quiet + self.raw.step_dt
        )
        entered = near & ~self.opponent_dribbling & ~self.opponent_shot
        self.opponent_had_dribbled[entered] = True
        self.opponent_dribble_origin[entered] = ball[entered]
        self.opponent_dribble_direction[entered] = direction[entered]
        self.opponent_dribble_touches[entered] = self.opponent_touches[entered]
        self.opponent_dribbling = retain_dribbling(
            self.opponent_dribbling | entered, defending, ball - position, direction
        )
        near = self.opponent_dribbling
        progress = (
            (ball - self.opponent_dribble_origin) * self.opponent_dribble_direction
        ).sum(dim=1)
        start_shot = (
            ready_for_strike(
                near.long(),
                progress,
                self.opponent_touches - self.opponent_dribble_touches,
                (goals - ball).norm(dim=1),
                (ball - position).norm(dim=1),
                alignment,
                self.opponent_quiet,
                self.args.shot_distance,
                getattr(self.args, "handoff_quiet_limit", 1.0),
                getattr(self.args, "shot_alignment_limit", 0.8),
            )
            & ~self.opponent_shot
        )
        if getattr(self.args, "stable_shot_setup", False):
            start_shot &= stable_shot_motion(
                data.root_link_lin_vel_w, data.root_link_ang_vel_b
            )
        start_shot &= not self.dead_ball
        if getattr(self.args, "anticipatory_strike", False):
            start_shot &= anticipatory_strike_window(
                ball - position,
                data.root_link_lin_vel_w[:, :2],
                self.ball.data.root_link_lin_vel_w[:, :2],
                self.opponent_quiet,
            )
        self.opponent_shot_direction[start_shot] = direction[start_shot]
        self.opponent_released[start_shot] = False
        self.opponent_strike_contact_seen[start_shot] = False
        self.opponent_release_history[start_shot] = 0
        self.opponent_speed_history[start_shot] = 0
        self.opponent_shot_age[start_shot] = 0
        self.opponent_shot |= start_shot
        self.opponent_dribbling[start_shot] = False
        self.opponent_shot_age += self.raw.step_dt
        if not self.dead_ball:
            self.opponent_shot &= ~defending & (self.opponent_shot_age < 8.0)
        target = self.opponent_paths.targets(position, ball, aiming, ~near & ~defending)
        defender_target = shadow_target(
            position, torch.zeros_like(ball) if self.patrol is not None else ball
        )
        if getattr(self.args, "defender_mode", "orbit") == "pursuit":
            defender_target = pursuit_target(ball, -direction)
        target = torch.where(defending[:, None], defender_target, target)
        command = travel_command(position, heading, target, maximum=0.5)
        ball_command = ball_path_command(
            ball,
            self.ball.data.root_link_lin_vel_w[:, :2],
            aiming,
            preparation_speed(
                (goals - ball).norm(dim=1),
                self.args.opponent_speed,
                self.args.shot_distance,
                getattr(self.args, "shot_preparation_distance", None),
                getattr(self.args, "shot_preparation_speed", 0.35),
            ),
            direct_goal=getattr(self.args, "direct_goal_command", False),
        )
        command[near, :2] = ball_command[near]
        command[self.opponent_shot, :2] = (
            self.args.shot_speed * self.opponent_shot_direction[self.opponent_shot]
        )
        if self.dead_ball:
            command = recovery_command(
                command, heading, self.opponent_shot, defending, self.args.shot_speed
            )
        base_skill = torch.where(self.opponent_shot, 2, near.long())
        turning = turning_navigation(
            base_skill, defending, self.opponent_had_dribbled, self.turn is not None
        ) & (not self.dead_ball)
        command = turning_command(command, turning)
        standing = stationary_restart(
            base_skill,
            defending,
            self.dead_ball,
            self.opponent_twist_navigation is not None,
        )
        navigation_action = None
        if self.opponent_twist_navigation is not None:
            navigation_action, navigation_command = self.opponent_twist_navigation.step(
                turning, aiming, self.opponent_applied
            )
            command[turning, :2] = navigation_command[turning, :2]
            command[standing, :2] = 0
        self.opponent_command = command.clone()
        obs = self.opponent_history.sample(command)
        strike_obs = (
            immediate_strike_command(obs, command)
            if getattr(self.args, "instant_strike_command", False)
            else obs
        )
        # Every striker tracks physical release, including the original 207-input policy.
        strong = (
            self.raw.scene["opponent_ball"]
            .data.force_history.norm(dim=-1)
            .amax(dim=(1, 2))
            > 5
        )
        self.opponent_strike_contact_seen |= striker_contact(
            self.opponent_shot, start_shot, strong
        )
        self.opponent_released |= self.opponent_shot & directed_release(
            self.ball.data.root_link_lin_vel_w,
            self.args.shot_speed * self.opponent_shot_direction,
            self.opponent_strike_contact_seen,
            self.opponent_quiet,
        )
        if self.strike_inputs == 210:
            self.opponent_release_history = torch.cat(
                (
                    self.opponent_release_history[:, 1:],
                    (self.opponent_released & self.opponent_shot).float()[:, None],
                ),
                dim=1,
            )
            strike_obs = with_release_phase(strike_obs, self.opponent_release_history)
        if self.strike_speed_features:
            info = self.strike_speed_features
            speed = (command[:, :2].norm(dim=1) - info["center"]) / info["scale"]
            self.opponent_speed_history = torch.cat(
                (
                    self.opponent_speed_history[:, 1:],
                    torch.where(self.opponent_shot, speed, 0.0)[:, None],
                ),
                dim=1,
            )
            strike_obs = with_command_speed(strike_obs, self.opponent_speed_history)
        action = torch.where(
            self.opponent_shot[:, None],
            self.strike(strike_obs),
            torch.where(
                near[:, None],
                self.dribble(obs),
                self.gait(
                    ball_independent_gait(obs) if self.gait_ball_independent else obs
                ),
            ),
        ).clamp(-3, 3)
        if self.turn is not None:
            action = torch.where(
                turning[:, None],
                navigation_action
                if navigation_action is not None
                else self.turn(ball_independent_gait(obs)),
                action,
            ).clamp(-3, 3)
        if self.patrol is not None:
            action = torch.where(
                defending[:, None], self.patrol(ball_independent_gait(obs)), action
            ).clamp(-3, 3)
        if self.dead_ball:
            recovery = self.strike(strike_obs).clamp(-3, 3)
            if navigation_action is not None:
                recovery = torch.where(standing[:, None], navigation_action, recovery)
            action = torch.where(defending[:, None], action, recovery)
        # Blend only skill transitions and record the exact actions actually sent to the opponent's motors.
        skill = torch.where(turning, 4, base_skill)
        if self.patrol is not None:
            skill = torch.where(defending, 3, skill)
        if self.dead_ball:
            skill = torch.where(defending, skill, torch.where(standing, 5, 2))
        changed = skill != self.opponent_previous_skill
        self.opponent_blend_origin[changed] = self.opponent_applied[changed]
        self.opponent_blend_age[changed] = 0
        self.opponent_blend_age += self.raw.step_dt
        fraction = blend_fraction(
            self.opponent_blend_age, getattr(self.args, "skill_blend_time", 0.08)
        )[:, None]
        if getattr(self.args, "strike_blend_time", None) is not None:
            fraction = torch.where(
                self.opponent_shot[:, None],
                blend_fraction(self.opponent_blend_age, self.args.strike_blend_time)[
                    :, None
                ],
                fraction,
            )
        action = self.opponent_blend_origin.lerp(action, fraction)
        self.opponent_applied = action
        self.opponent_previous_skill[:] = skill
        self.opponent_history.last.copy_(action)
        term = self.raw.action_manager.get_term("joint_pos")
        targets = data.default_joint_pos[:, self.opponent_ids] + term._scale * action
        limits = data.joint_pos_limits[:, self.opponent_ids]
        opponent.set_joint_position_target(
            targets.clamp(limits[..., 0], limits[..., 1]), joint_ids=self.opponent_ids
        )

    @torch.no_grad()
    def observe(self):
        dt = self.raw.step_dt
        active = self.phase < 4
        # A skill switch must preserve the physical episode; only declared dead balls may reset it.
        unexpected = active & (self.raw.episode_length_buf <= self.last_episode_steps)
        self.live_resets += unexpected.long()
        if unexpected.any():
            raise RuntimeError("A physical episode reset during a live skill sequence")
        previous_phase = self.phase.clone()
        self.elapsed += dt
        self.phase_age += dt
        data = self.robot.data
        ball = self.ball.data.root_link_pos_w
        contact = (
            self.raw.scene["ball_touch"]
            .data.force_history.norm(dim=-1)
            .amax(dim=(1, 2))
            > 0.1
        )
        self.touches += (contact & (self.quiet >= 0.12)).long()
        self.quiet = torch.where(contact, 0.0, self.quiet + dt)
        delta = ball[:, :2] - data.root_link_pos_w[:, :2]
        goal_delta = self.goals - ball[:, :2]
        distance = goal_delta.norm(dim=1)
        direction = goal_target(self.goals, ball[:, :2]) - ball[:, :2]
        direction /= direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
        angle = wrap_angle(
            torch.atan2(direction[:, 1], direction[:, 0])
            - yaw_from_quaternion(data.root_link_quat_w)
        ).abs()
        # Enter learned dribbling only after the gait reaches the ball facing the attacking goal.
        entered = (
            (self.phase == 0)
            & (delta.norm(dim=1) < 0.95)
            & ((delta * direction).sum(dim=1) > 0.2)
            & (angle < 0.4)
            & ~self.defending
        )
        self.phase[entered] = 1
        self.had_dribbled[entered] = True
        self.phase_age[entered] = 0
        first_possession = entered & (self.unreleased_retries == 0)
        self.dribble_origin[first_possession] = ball[first_possession, :2]
        self.dribble_direction[first_possession] = direction[first_possession]
        self.dribble_touches[entered] = self.touches[entered]
        dribbling = self.phase == 1
        progress = ((ball[:, :2] - self.dribble_origin) * self.dribble_direction).sum(
            dim=1
        )
        self.dribble_distance[dribbling] = progress[dribbling]
        # Hand off from genuine moving possession at a reachable shooting distance.
        strike = ready_for_strike(
            self.phase,
            self.dribble_distance,
            self.touches - self.dribble_touches,
            distance,
            delta.norm(dim=1),
            angle,
            self.quiet,
            self.args.shot_distance,
            getattr(self.args, "handoff_quiet_limit", 1.0),
            getattr(self.args, "shot_alignment_limit", 0.8),
        )
        if getattr(self.args, "stable_shot_setup", False):
            strike &= stable_shot_motion(
                data.root_link_lin_vel_w, data.root_link_ang_vel_b
            )
        if getattr(self.args, "anticipatory_strike", False):
            strike &= anticipatory_strike_window(
                delta,
                data.root_link_lin_vel_w[:, :2],
                self.ball.data.root_link_lin_vel_w[:, :2],
                self.quiet,
            )
        self.phase[strike] = 2
        self.phase_age[strike] = 0
        self.shot_direction[strike] = direction[strike]
        self.shot_touches[strike] = self.touches[strike]
        if getattr(self.args, "handoff_bank", None) and strike.any():
            saved_history = self.history.history
            saved_frame = self.history.canonical
            self.history.canonical = False
            current = self.history.sample(
                self.raw.command_manager.get_term("ball_velocity").velocity_w
            )["actor"]
            self.history.history = saved_history
            self.history.canonical = saved_frame
            for i in strike.nonzero().flatten().tolist():
                if len(self.handoff_states) < 512:
                    self.handoff_states.append(
                        {
                            "qpos": self.raw.sim.data.qpos[i].cpu().clone(),
                            "qvel": self.raw.sim.data.qvel[i].cpu().clone(),
                            "actor_history": current[i].cpu().clone(),
                            "command": self.raw.command_manager.get_term(
                                "ball_velocity"
                            )
                            .velocity_w[i]
                            .cpu()
                            .clone(),
                            "trial": i,
                        }
                    )
        # Require a new physical touch to release a shot, then let the striker recover balance.
        strike_contact = (
            self.raw.scene["ball_touch"]
            .data.force_history.norm(dim=-1)
            .amax(dim=(1, 2))
            > 5
        )
        self.strike_contact_seen |= striker_contact(
            self.phase == 2, strike, strike_contact
        )
        launch = (self.phase == 2) & directed_release(
            self.ball.data.root_link_lin_vel_w,
            self.args.shot_speed * self.shot_direction,
            self.strike_contact_seen,
            self.quiet,
        )
        self.released |= launch
        self.post_release_recontacts += (
            self.released & strike_contact & ~self.previous_strike_contact
        ).long()
        self.previous_strike_contact.copy_(strike_contact)
        self.phase[launch] = 3
        self.phase_age[launch] = 0
        crossed, inside, point = goal_crossing(self.previous_ball, ball, self.goals)
        goal = (
            active
            & crossed
            & inside
            & (point[:, 1].abs() <= 0.85)
            & (point[:, 2] <= 0.85)
        )
        self.scored |= goal
        self.shot_scored |= goal & self.released & (self.post_release_recontacts == 0)
        settled = (
            (self.phase == 3)
            & (self.quiet >= 0.2)
            & (data.root_link_lin_vel_w[:, :2].norm(dim=1) < 0.25)
            & (data.projected_gravity_b[:, 2] < -0.85)
        )
        self.settled_age = torch.where(settled, self.settled_age + dt, 0)
        complete = (
            (self.phase == 3)
            & self.shot_scored
            & self.released
            & (self.post_release_recontacts == 0)
            & (self.settled_age >= 2.0)
        )
        self.phase[complete] = 4
        for i in complete.nonzero().flatten().tolist():
            self.reason[i] = "goal_and_balance"
        reacquire = (self.phase == 1) & (
            (delta.norm(dim=1) > 1.6) | ((delta * direction).sum(dim=1) < -0.2)
        )
        if self.args.opponent:
            reacquire |= (self.phase == 1) & (self.possession_team == 1)
        self.phase[reacquire] = 0
        self.phase_age[reacquire] = 0
        for i in reacquire.nonzero().flatten().tolist():
            self.paths.routes[i] = None
        # Retry only an unreleased, unscored stopped ball; keep the complete physical history and progress.
        retry = retry_unreleased_strike(
            self.phase,
            self.phase_age,
            self.released,
            self.scored,
            self.ball.data.root_link_lin_vel_w[:, :2].norm(dim=1),
            self.unreleased_retries,
            getattr(self.args, "unreleased_retries", 0),
        ) & (ball[:, :2].abs() < ball.new_tensor([5.8, 3.8])).all(dim=1)
        self.phase[retry] = 0
        self.phase_age[retry] = 0
        self.unreleased_retries[retry] += 1
        self.strike_contact_seen[retry] = False
        self.previous_strike_contact[retry] = False
        for i in retry.nonzero().flatten().tolist():
            self.paths.routes[i] = None
        physical_fall = (
            self.raw.reset_terminated
            | (data.root_link_pos_w[:, 2] < 0.5)
            | (data.projected_gravity_b[:, 2] > -0.5)
        )
        failure = active & (
            physical_fall
            | (self.elapsed >= self.args.seconds)
            | field_failure(ball[:, :2], self.scored, self.args.opponent)
        )
        failure |= ((self.phase == 2) & (self.phase_age > 5.0)) | (
            (self.phase == 3) & (self.phase_age > 6.0) & ~self.shot_scored
        )
        self.phase[failure] = 5
        for i in failure.nonzero().flatten().tolist():
            self.reason[i] = "fall" if physical_fall[i] else "timeout_or_missed_shot"
        self.previous_ball = ball.clone()
        self.last_step = int(self.raw.episode_length_buf[0])
        self.last_episode_steps.copy_(self.raw.episode_length_buf)
        for i in (previous_phase != self.phase).nonzero().flatten().tolist():
            self.transitions[i].append(
                {
                    "time_s": float(self.elapsed[i]),
                    "from": PHASES[int(previous_phase[i])],
                    "to": PHASES[int(self.phase[i])],
                }
            )

    def row(self, index):
        return {
            "trial": index,
            "phase": PHASES[int(self.phase[index])],
            "success": int(self.phase[index]) == 4,
            "reason": self.reason[index],
            "touches": int(self.touches[index]),
            "dribble_distance_m": float(self.dribble_distance[index]),
            "shot_released": bool(self.released[index]),
            "scored": bool(self.scored[index]),
            "scored_after_release": bool(self.shot_scored[index]),
            "post_release_recontacts": int(self.post_release_recontacts[index]),
            "unreleased_retries": int(self.unreleased_retries[index]),
            "duration_s": float(self.elapsed[index]),
            "settled_recovery_s": float(self.settled_age[index]),
            "unexpected_live_resets": int(self.live_resets[index]),
            "transitions": self.transitions[index],
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("evaluate", "play"), default="evaluate")
    p.add_argument("--bundle", type=Path)
    p.add_argument("--gait-checkpoint", type=Path)
    p.add_argument("--dribble-checkpoint", type=Path)
    p.add_argument("--strike-checkpoint", type=Path)
    p.add_argument("--patrol-checkpoint", type=Path)
    p.add_argument("--turn-checkpoint", type=Path)
    p.add_argument("--opponent", action="store_true")
    p.add_argument("--canonical-heading", action="store_true")
    p.add_argument("--defender-mode", choices=("pursuit", "orbit"), default="pursuit")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--num-envs", type=int, default=3)
    p.add_argument("--dribble-speed", type=float, default=0.7)
    p.add_argument("--shot-speed", type=float, default=1.8)
    p.add_argument("--shot-distance", type=float, default=3.0)
    p.add_argument("--handoff-quiet-limit", type=float, default=1.0)
    p.add_argument("--skill-blend-time", type=float, default=0.08)
    p.add_argument("--strike-blend-time", type=float)
    p.add_argument("--instant-strike-command", action="store_true")
    p.add_argument("--shot-preparation-distance", type=float)
    p.add_argument("--shot-preparation-speed", type=float, default=0.35)
    p.add_argument("--shot-alignment-limit", type=float, default=0.8)
    p.add_argument("--anticipatory-strike", action="store_true")
    p.add_argument("--direct-goal-command", action="store_true")
    p.add_argument("--stable-shot-setup", action="store_true")
    p.add_argument("--unreleased-retries", type=int, choices=range(4), default=0)
    p.add_argument("--opponent-speed", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=20801)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--port", type=int, default=8087)
    p.add_argument("--autoplay", action="store_true")
    p.add_argument("--viewer-steps", type=int)
    p.add_argument("--video", type=Path)
    p.add_argument("--trace", type=Path)
    p.add_argument("--handoff-bank", type=Path)
    p.add_argument(
        "--output", type=Path, default=ROOT / "outputs/joga-soccer/evaluation.json"
    )
    args = p.parse_args()
    if args.bundle:
        if not args.bundle.is_file() or any(
            getattr(args, role + "_checkpoint")
            for role in ("gait", "dribble", "strike", "patrol", "turn")
        ):
            p.error(
                "Use an existing candidate manifest or three explicit checkpoint paths"
            )
        manifest = json.loads(args.bundle.read_text())
        if manifest.get("status") not in {
            "qualified_candidate_pending_visual_review",
            "runtime_refactor_validated",
        }:
            p.error("The supplied manifest has not passed physical qualification")
        if manifest.get("controller_dependencies"):
            from .twist_navigation import source_hashes

            if (
                manifest.get("controller_source_sha256")
                != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or manifest["controller_dependencies"] != source_hashes()
            ):
                p.error(
                    "The controller code changed after qualification; use its preserved source or requalify"
                )
        for role in ("gait", "dribble", "strike", "patrol", "turn"):
            if role in ("patrol", "turn") and role not in manifest["roles"]:
                continue
            entry = manifest["roles"][role]
            path = Path(entry["path"])
            if not path.is_absolute():
                path = args.bundle.parent / path
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            ):
                p.error(
                    f"The {role} checkpoint is missing or does not match its manifest hash"
                )
            setattr(args, role + "_checkpoint", path)
        for key in (
            "canonical_heading",
            "defender_mode",
            "dribble_speed",
            "shot_speed",
            "shot_distance",
            "handoff_quiet_limit",
            "skill_blend_time",
            "strike_blend_time",
            "instant_strike_command",
            "shot_preparation_distance",
            "shot_preparation_speed",
            "shot_alignment_limit",
            "anticipatory_strike",
            "direct_goal_command",
            "stable_shot_setup",
            "unreleased_retries",
            "opponent_speed",
        ):
            if key in manifest.get("execution", {}):
                setattr(args, key, manifest["execution"][key])
    if args.skill_blend_time < 0:
        p.error("Skill blend time must be nonnegative")
    if args.patrol_checkpoint and (
        not args.patrol_checkpoint.is_file() or args.defender_mode != "orbit"
    ):
        p.error("A patrol checkpoint requires an existing file and orbit tactics")
    if args.turn_checkpoint and not args.turn_checkpoint.is_file():
        p.error("Turning requires an existing checkpoint")
    if args.strike_blend_time is not None and args.strike_blend_time < 0:
        p.error("Strike blend time must be nonnegative")
    if args.shot_preparation_distance is not None and (
        args.shot_preparation_distance <= args.shot_distance
        or not 0
        < args.shot_preparation_speed
        <= min(args.dribble_speed, args.opponent_speed)
    ):
        p.error(
            "Preparation needs a farther distance and a positive speed below both dribble speeds"
        )
    if not 0 < args.shot_alignment_limit <= math.pi:
        p.error("Shot alignment limit must be in (0, pi]")
    if (
        any(
            getattr(args, role + "_checkpoint") is None
            or not getattr(args, role + "_checkpoint").is_file()
            for role in ("gait", "dribble", "strike")
        )
        or min(
            args.seconds,
            args.num_envs,
            args.dribble_speed,
            args.shot_speed,
            args.shot_distance,
            args.handoff_quiet_limit,
            args.opponent_speed,
        )
        <= 0
    ):
        p.error("Existing role checkpoints and positive parameters required")
    if args.handoff_bank and args.opponent:
        p.error(
            "Single-humanoid handoff states cannot include an opponent's generalized coordinates"
        )
    if args.video and args.mode != "evaluate":
        p.error("Offscreen video is supported in headless evaluation")
    if args.viewer_steps is not None and (args.viewer_steps < 1 or not args.autoplay):
        p.error("Bounded viewer steps require a positive count and --autoplay")
    if args.mode == "evaluate":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    n = 1 if args.mode == "play" or args.opponent else args.num_envs
    raw = ManagerBasedRlEnv(
        make_cfg(n, args.seconds, args.opponent, args.seed),
        device=args.device,
        render_mode="rgb_array" if args.video else None,
    )
    env = RslRlVecEnvWrapper(raw, clip_actions=3.0)
    server = None
    writer = None
    trace = None
    try:
        if args.video:
            import imageio.v2 as imageio

            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=25)
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            trace = args.trace.open("w")
        episode = Episode(raw, env, args)
        rules = MatchRules(duration=args.seconds)
        audit = MatchAudit()
        resets = 0
        previous = episode.ball.data.root_link_pos_w[0].cpu().tolist()

        def observe():
            nonlocal previous, resets
            if not args.opponent or rules.phase == "PLAY":
                episode.observe()
            if args.opponent:
                ball = episode.ball.data.root_link_pos_w[0].cpu().tolist()
                opponent = raw.scene["opponent"].data
                new_falls = audit.observe(
                    [
                        [
                            *body.root_link_pos_w[0].cpu().tolist(),
                            float(body.projected_gravity_b[0, 2]),
                        ]
                        for body in (episode.robot.data, opponent)
                    ]
                )
                for team, sensor in enumerate(("ball_touch", "opponent_ball")):
                    touching = bool(
                        raw.scene[sensor].data.force_history.norm(dim=-1).amax() > 0.1
                    )
                    episode.match_touch_counts[team] += int(
                        rules.phase == "PLAY"
                        and touching
                        and not episode.match_was_touch[team]
                    )
                    episode.match_was_touch[team] = touching
                event = rules.observe(
                    previous,
                    ball,
                    raw.step_dt,
                    player_fall=float(episode.robot.data.root_link_pos_w[0, 2]) < 0.5
                    or float(episode.robot.data.projected_gravity_b[0, 2]) > -0.5,
                    opponent_fall=float(opponent.root_link_pos_w[0, 2]) < 0.5
                    or float(opponent.projected_gravity_b[0, 2]) > -0.5,
                    attempt_end=bool(episode.phase[0] >= 4),
                )
                previous = ball
                # Record physical falls even when a goal or dead ball takes priority in the rules.
                if any(new_falls) and (event is None or event["event"] != "fall"):
                    rules.event(
                        "fall", player=new_falls[0], opponent=new_falls[1], audit=True
                    )
                if event:
                    print(json.dumps(event), flush=True)
                    if event["event"] == "kickoff":
                        episode.reset(rules.kickoff)
                        audit.kickoff()
                        resets += 1
                        previous = episode.ball.data.root_link_pos_w[0].cpu().tolist()

        if args.mode == "play":
            import viser
            from mjlab.viewer import ViserPlayViewer

            server = viser.ViserServer(port=args.port, label="Pioneer — learned soccer")
            text = server.gui.add_markdown(
                "Learned gait, dribble and strike; exact simulation state."
            )

            class Adapter:
                def reset(self):
                    nonlocal rules, audit, previous, resets
                    rules = MatchRules(duration=args.seconds)
                    audit = MatchAudit()
                    episode.reset(kickoff=0 if args.opponent else None)
                    previous = episode.ball.data.root_link_pos_w[0].cpu().tolist()
                    resets = 0

                def __call__(self, _obs):
                    if int(raw.episode_length_buf[0]) > episode.last_step:
                        observe()
                    label = (
                        "RESTART"
                        if args.opponent and rules.phase != "PLAY"
                        else "PATROL"
                        if int(episode.previous_policy_role[0]) == 3
                        else PHASES[int(episode.phase[0])]
                    )
                    text.content = f"**{label}** · {float(episode.elapsed[0]):.1f}s · touches {int(episode.touches[0])}"
                    if args.opponent:
                        text.content += f" · score {rules.score[0]}–{rules.score[1]}"
                    if (args.opponent and rules.phase == "FULL_TIME") or (
                        not args.opponent and episode.phase[0] >= 4
                    ):
                        viewer.pause()
                    return episode.actions(
                        dead_ball=args.opponent and rules.phase != "PLAY"
                    )

            viewer = ViserPlayViewer(env, Adapter(), viser_server=server)
            if not args.autoplay:
                viewer.pause()
            viewer.run(num_steps=args.viewer_steps)
            if viewer.get_status().last_error:
                raise RuntimeError(viewer.get_status().last_error)
        else:
            rows = [None] * n
            for step in range(math.ceil(args.seconds / raw.step_dt)):
                action = episode.actions(
                    dead_ball=args.opponent and rules.phase != "PLAY"
                )
                _, reward, _, _ = env.step(action)
                if not torch.isfinite(action).all() or not torch.isfinite(reward).all():
                    raise RuntimeError("Nonfinite learned-soccer state")
                observe()
                if writer and step % 2 == 0:
                    writer.append_data(raw.render())
                if trace and step % 2 == 0:
                    trace.write(
                        json.dumps(
                            {
                                "time_s": (step + 1) * raw.step_dt,
                                "robot_xy": episode.robot.data.root_link_pos_w[
                                    :, :2
                                ].tolist(),
                                "ball_xy": episode.ball.data.root_link_pos_w[
                                    :, :2
                                ].tolist(),
                                "ball_velocity_xy": episode.ball.data.root_link_lin_vel_w[
                                    :, :2
                                ].tolist(),
                                "foot_xy": episode.robot.data.body_link_pos_w[
                                    :, episode.foot_ids, :2
                                ].tolist(),
                                "foot_velocity_xy": episode.robot.data.body_link_lin_vel_w[
                                    :, episode.foot_ids, :2
                                ].tolist(),
                                "phase": [PHASES[i] for i in episode.phase.tolist()],
                                "policy_role": episode.previous_policy_role.tolist(),
                                "referee_phase": rules.phase if args.opponent else None,
                                "touches": episode.touches.tolist(),
                                "robot_heading_rad": yaw_from_quaternion(
                                    episode.robot.data.root_link_quat_w
                                ).tolist(),
                                "ball_quiet_s": episode.quiet.tolist(),
                                "dribble_distance_m": episode.dribble_distance.tolist(),
                                "opponent_xy": raw.scene["opponent"]
                                .data.root_link_pos_w[:, :2]
                                .tolist()
                                if args.opponent
                                else None,
                                "opponent_skill": episode.opponent_previous_skill.tolist()
                                if args.opponent
                                else None,
                                "opponent_heading_rad": yaw_from_quaternion(
                                    raw.scene["opponent"].data.root_link_quat_w
                                ).tolist()
                                if args.opponent
                                else None,
                                "opponent_command": episode.opponent_command.tolist()
                                if args.opponent
                                else None,
                                "opponent_velocity": raw.scene[
                                    "opponent"
                                ].data.root_link_lin_vel_w.tolist()
                                if args.opponent
                                else None,
                                "opponent_gravity": raw.scene[
                                    "opponent"
                                ].data.projected_gravity_b.tolist()
                                if args.opponent
                                else None,
                            }
                        )
                        + "\n"
                    )
                for i in range(n):
                    if rows[i] is None and episode.phase[i] >= 4:
                        rows[i] = episode.row(i)
                if (args.opponent and rules.phase == "FULL_TIME") or (
                    not args.opponent and (episode.phase >= 4).all()
                ):
                    break
                if not args.opponent:
                    retired = (episode.phase >= 4) & (
                        raw.reset_terminated | raw.reset_time_outs
                    )
                    ids = retired.nonzero().flatten()
                    if len(ids):
                        raw.reset(env_ids=ids)
            report = {
                "purpose": "interface_smoke"
                if args.seconds < 1
                else "functional_evaluation_of_supplied_checkpoints",
                "controller_dependencies": episode.controller_dependencies,
                "canonical_heading": args.canonical_heading,
                "opponent_tactics": args.defender_mode if args.opponent else None,
                "execution": {
                    key: getattr(args, key)
                    for key in (
                        "canonical_heading",
                        "defender_mode",
                        "dribble_speed",
                        "shot_speed",
                        "shot_distance",
                        "handoff_quiet_limit",
                        "skill_blend_time",
                        "strike_blend_time",
                        "instant_strike_command",
                        "shot_preparation_distance",
                        "shot_preparation_speed",
                        "shot_alignment_limit",
                        "anticipatory_strike",
                        "direct_goal_command",
                        "stable_shot_setup",
                        "unreleased_retries",
                        "opponent_speed",
                    )
                },
                "episodes": [
                    row if row is not None else episode.row(i)
                    for i, row in enumerate(rows)
                ],
                "match": {
                    "score": rules.score,
                    "events": rules.events,
                    "dead_ball_resets": resets,
                    "touch_bouts": episode.match_touch_counts,
                    "elapsed_s": rules.elapsed,
                    "phase": rules.phase,
                    "unexpected_live_resets": int(episode.live_resets.sum()),
                    **audit.report(),
                }
                if args.opponent
                else None,
                "checkpoint_sha256": {
                    role: hashlib.sha256(
                        getattr(args, role + "_checkpoint").read_bytes()
                    ).hexdigest()
                    for role in ("gait", "dribble", "strike", "patrol", "turn")
                    if getattr(args, role + "_checkpoint", None) is not None
                },
                "source_sha256": episode.source_sha256,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            if args.handoff_bank:
                args.handoff_bank.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "states": episode.handoff_states,
                        "joint_names": episode.robot.joint_names,
                        "physics_timestep": raw.physics_dt,
                        "checkpoint_sha256": report["checkpoint_sha256"],
                        "source_sha256": report["source_sha256"],
                        "purpose": "actual live dribble-to-strike handoff states",
                        "history_frame": "world",
                    },
                    args.handoff_bank,
                )
    finally:
        if writer:
            writer.close()
        if trace:
            trace.close()
        env.close()
        if server:
            server.stop()


if __name__ == "__main__":
    main()
