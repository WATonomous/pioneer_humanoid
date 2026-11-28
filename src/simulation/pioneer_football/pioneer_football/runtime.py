"""Selected Pioneer soccer configuration, physical control and inference helpers."""

import math
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import mjlab.envs.mdp as builtin
import mujoco
import torch
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg, pd_torque
from mjlab.entity.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.envs.mdp.events import reset_joints_by_offset, reset_root_state_uniform
from mjlab.envs.mdp.observations import (
    base_ang_vel,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
)
from mjlab.envs.mdp.rewards import action_rate_l2, is_terminated, joint_pos_limits
from mjlab.envs.mdp.terminations import (
    bad_orientation,
    root_height_below_minimum,
    time_out,
)
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl.config import RslRlModelCfg
from mjlab.scene.scene import SceneCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sim.sim import SimulationCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply as quat_apply
from mjlab.utils.lab_api.math import quat_apply_inverse as quat_apply_inverse
from mjlab.viewer.viewer_config import ViewerConfig
from rsl_rl.models.mlp_model import MLPModel
from tensordict._td import TensorDict

from .approach import yaw_from_quaternion
from .ball import BallCfg, PitchCfg, ball_spec, configure_scene
from .model import robot_spec as source_robot_spec

JOINTS = (
    "Hip_F_L",
    "Hip_A_L",
    "Hip_R_L",
    "Knee_L",
    "Ankle_P_L",
    "Ankle_R_L",
    "Hip_F_R",
    "Hip_A_R",
    "Hip_R_R",
    "Knee_R",
    "Ankle_P_R",
    "Ankle_R_R",
)


ARM_JOINTS = ("l_shoulder_pitch", "r_shoulder_pitch", "l_elbow", "r_elbow")


def pd_robot_spec(visuals=True):
    spec = source_robot_spec(visuals=visuals, arm_swing=True)
    for name in JOINTS:
        kind = name.rsplit("_", 1)[0]
        kp = 200.0 if kind in ("Hip_F", "Knee") else 150.0
        kd = 5.0
        if kind.startswith("Ankle"):
            kp, kd = (40.0, 2.0)
        spec.actuator(name).set_to_position(kp=kp, kv=kd)
    for name in ARM_JOINTS:
        spec.actuator(name).set_to_position(kp=40.0, kv=2.0)
    return spec


def actuated_robot_spec(visuals=True):
    spec = pd_robot_spec(visuals=visuals)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


CAD_BODIES = ("base_link", "l_hip_roll_link", "r_hip_roll_link")


def robot_spec(visuals=True):
    spec = actuated_robot_spec(visuals=True)
    original = list(spec.geoms)
    for name in CAD_BODIES:
        proxies = [
            g
            for g in original
            if g.parent.name == name and g.name.startswith(name + "_collision_")
        ]
        components = [
            g
            for g in original
            if g.parent.name == name and g.type == mujoco.mjtGeom.mjGEOM_MESH
        ]
        if len(proxies) != 1 or not components:
            raise ValueError(
                f"Expected one aggregate box and source CAD components for {name}"
            )
        proxy = proxies[0]
        for i, visual in enumerate(components):
            spec.body(name).add_geom(
                name=f"{name}_collision_cad_{i}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=visual.meshname,
                pos=visual.pos,
                quat=visual.quat,
                contype=proxy.contype,
                conaffinity=proxy.conaffinity,
                condim=proxy.condim,
                friction=proxy.friction,
                solref=proxy.solref,
                solimp=proxy.solimp,
                margin=proxy.margin,
                gap=proxy.gap,
                priority=proxy.priority,
                group=3,
                mass=0,
                rgba=[0.4, 0.8, 0.3, 0.3],
            )
        proxy.contype = proxy.conaffinity = 0
    if not visuals:
        for geom in original:
            if geom.group == 2:
                geom.rgba[3] = 0
    return spec


def command_interval(iteration):
    if iteration < 10000:
        return (7.0, 7.0)
    if iteration < 14000:
        return (5.0, 7.0)
    if iteration < 18000:
        return (3.0, 7.0)
    return (2.0, 8.0)


def sole_clearance(poses, half_sizes, floor_z):
    up = torch.zeros_like(poses[..., :3])
    up[..., 2] = 1.0
    local_up = quat_apply_inverse(poses[..., 3:7], up)
    extent = (local_up.abs() * half_sizes).sum(dim=-1)
    return poses[..., 2] - extent - floor_z[:, None]


def foot_contacts(env):
    force = env.scene["feet"].data.force_history
    if not hasattr(env, "_running_sole_geometry"):
        robot = env.scene["robot"]
        ids, _ = robot.find_geoms(
            ("l_foot_collision_0", "r_foot_collision_0"), preserve_order=True
        )
        global_ids = robot.indexing.geom_ids[ids].cpu().numpy()
        sizes = torch.tensor(
            env.sim.mj_model.geom_size[global_ids].copy(),
            device=env.device,
            dtype=torch.float32,
        )
        env._running_sole_geometry = (ids, sizes)
    ids, sizes = env._running_sole_geometry
    height = sole_clearance(
        env.scene["robot"].data.geom_pose_w[:, ids], sizes, env.scene.env_origins[:, 2]
    )
    loaded = (force.norm(dim=-1).amax(dim=-1) > 1.0) | (
        env.scene["feet"].data.found > 0
    )
    return loaded | (height <= 0.003)


def foot_heights(env):
    foot_contacts(env)
    ids, sizes = env._running_sole_geometry
    return sole_clearance(
        env.scene["robot"].data.geom_pose_w[:, ids], sizes, env.scene.env_origins[:, 2]
    )


class BallCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.velocity_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_update = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.quiet = torch.full((self.num_envs,), 10.0, device=self.device)
        self.touches = torch.zeros_like(self.quiet)
        self.was_contact = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self.swing_peak = torch.zeros_like(self.was_contact, dtype=torch.float32)
        self.last_swing_peak = torch.zeros_like(self.swing_peak)
        self.landed = torch.zeros_like(self.was_contact)
        self.iteration = 0
        for key in (
            "ball_speed",
            "requested_speed",
            "velocity_error",
            "visible_fraction",
            "turn_interval_s",
        ):
            self.metrics[key] = torch.zeros_like(self.quiet)

    @property
    def command(self):
        return self.velocity_w

    def _resample_command(self, env_ids):
        if self.cfg.curriculum:
            self.cfg.resampling_time_range = command_interval(self.iteration)
        speed = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.speed_range
        )
        heading = torch.empty_like(speed).uniform_(-math.pi, math.pi)
        if self.cfg.fixed_heading is not None:
            heading.fill_(self.cfg.fixed_heading)
        self.velocity_w[env_ids] = 0
        self.velocity_w[env_ids, :2] = speed[:, None] * torch.stack(
            (heading.cos(), heading.sin()), dim=1
        )

    def _update_command(self, env_ids=None):
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids
        )
        ids = ids[self.last_update[ids] != self._env.common_step_counter]
        if not len(ids):
            return
        self.last_update[ids] = self._env.common_step_counter
        touch = (
            self._env.scene["ball_touch"]
            .data.force_history.norm(dim=-1)
            .amax(dim=(1, 2))
            > 0.1
        )
        self.touches[ids] += (touch[ids] & (self.quiet[ids] >= 0.12)).float()
        self.quiet[ids] = torch.where(
            touch[ids], 0, self.quiet[ids] + self._env.step_dt
        )
        contact = foot_contacts(self._env)
        height = foot_heights(self._env)
        self.landed[ids] = contact[ids] & ~self.was_contact[ids]
        self.last_swing_peak[ids] = torch.where(
            self.landed[ids], self.swing_peak[ids], self.last_swing_peak[ids]
        )
        self.swing_peak[ids] = torch.where(
            contact[ids], 0, torch.maximum(height[ids], self.swing_peak[ids])
        )
        self.was_contact[ids] = contact[ids]

    def _update_metrics(self):
        if hasattr(self, "runner"):
            self.iteration = self.runner.current_learning_iteration
            if self.cfg.curriculum:
                self.cfg.resampling_time_range = command_interval(self.iteration)
        actual = self._env.scene["ball"].data.root_link_lin_vel_w
        self.metrics["ball_speed"][:] = actual[:, :2].norm(dim=1)
        self.metrics["requested_speed"][:] = self.velocity_w[:, :2].norm(dim=1)
        self.metrics["velocity_error"][:] = (
            actual[:, :2] - self.velocity_w[:, :2]
        ).norm(dim=1)
        self.metrics["visible_fraction"][:] = 1.0
        self.metrics["turn_interval_s"][:] = sum(self.cfg.resampling_time_range) / 2

    def reset(self, env_ids):
        self.last_update[env_ids] = -1
        self.quiet[env_ids], self.touches[env_ids] = (10, 0)
        self.was_contact[env_ids], self.landed[env_ids] = (False, False)
        self.swing_peak[env_ids], self.last_swing_peak[env_ids] = (0, 0)
        return super().reset(env_ids)

    def create_gui(
        self, name, server, get_env_idx, on_change=None, request_action=None
    ):
        speed = server.gui.add_slider(
            "Ball speed (m/s)", min=0.0, max=1.5, step=0.1, initial_value=0.7
        )
        heading = server.gui.add_slider(
            "Ball direction (deg)", min=-180, max=180, step=5, initial_value=0
        )

        def update(_event):
            angle = math.radians(heading.value)
            i = get_env_idx()
            self.velocity_w[i] = self.velocity_w.new_tensor(
                [speed.value * math.cos(angle), speed.value * math.sin(angle), 0]
            )
            self.time_left[i] = 1000000.0

        speed.on_update(update)
        heading.on_update(update)


class MovingBallCommand(BallCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.stall_time = torch.zeros_like(self.quiet)

    def _resample_command(self, env_ids):
        speed = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.speed_range
        )
        if self.cfg.curriculum:
            maximum = (
                0.15 if self.iteration < 500 else 0.4 if self.iteration < 1500 else 0.8
            )
            if self.cfg.training_heading_limit is not None:
                maximum = self.cfg.training_heading_limit
            robot = self._env.scene["robot"]
            quat = self._env.sim.data.qpos[:, robot.indexing.free_joint_q_adr[3:7]]
            heading = yaw_from_quaternion(quat)[env_ids]
            heading += torch.empty_like(heading).uniform_(-maximum, maximum)
        else:
            heading = torch.full_like(speed, self.cfg.fixed_heading or 0.0)
        self.velocity_w[env_ids] = 0
        self.velocity_w[env_ids, :2] = speed[:, None] * torch.stack(
            (heading.cos(), heading.sin()), dim=1
        )

    def _update_command(self, env_ids=None):
        fresh = self.last_update != self._env.common_step_counter
        if env_ids is not None:
            scope = torch.zeros_like(fresh)
            scope[env_ids] = True
            fresh &= scope
        super()._update_command(env_ids)
        robot = self._env.scene["robot"].data.root_link_lin_vel_w[:, :2].norm(dim=1)
        ball = self._env.scene["ball"].data.root_link_lin_vel_w[:, :2].norm(dim=1)
        stalled = (
            (robot < 0.1) & (ball < 0.1) & (self.velocity_w[:, :2].norm(dim=1) > 0.2)
        )
        self.stall_time[fresh] = torch.where(
            stalled[fresh], self.stall_time[fresh] + self._env.step_dt, 0
        )

    def reset(self, env_ids):
        self.stall_time[env_ids] = 0
        return super().reset(env_ids)


@dataclass(kw_only=True)
class BallCommandCfg(CommandTermCfg):
    curriculum: bool = True
    speed_range: tuple[float, float] = (0.3, 1.5)
    fixed_heading: float | None = None

    def build(self, env):
        return BallCommand(self, env)


@dataclass(kw_only=True)
class MovingBallCommandCfg(BallCommandCfg):
    training_heading_limit: float | None = None

    def build(self, env):
        return MovingBallCommand(self, env)


RECOVERY_BALL = BallCfg(
    sliding_friction=0.75,
    foot_friction=0.3,
    torsional_friction=0.05,
    rolling_friction=0.02,
    contact_time=0.055,
    damping_ratio=0.18,
)


def contact_scene(spec, **kwargs):
    configure_scene(spec, ball_cfg=RECOVERY_BALL, **kwargs)
    for pair in spec.pairs:
        if pair.name == "ball_ground":
            pair.solimp[:2] = [0.85, 0.95]
        elif pair.name.startswith("ball_foot_"):
            pair.solref[1] = 0.14


def directed_release(velocity, command, contact_seen, quiet):
    speed = command[:, :2].norm(dim=1)
    forward = (velocity[:, :2] * command[:, :2]).sum(dim=1) / speed.clamp(min=1e-06)
    return contact_seen & (quiet >= 0.08) & (speed > 0.1) & (forward >= 0.7 * speed)


DEFAULT_PITCH_CFG = PitchCfg()


DEFAULT_BALL_CFG = BallCfg()


@dataclass(frozen=True)
class KickCfg:
    strike_speed: float = 0.55
    minimum_launch_speed: float = 1.0
    recovery_seconds: float = 2.0
    flight_seconds: float = 8.0
    goal_height: float = DEFAULT_PITCH_CFG.goal_height
    ball_radius: float = DEFAULT_BALL_CFG.radius
    goal_half_width: float = DEFAULT_PITCH_CFG.goal_width / 2

    def __post_init__(self):
        if not all((math.isfinite(v) and v > 0 for v in vars(self).values())):
            raise ValueError("Kick parameters must be finite and positive")
        if self.strike_speed > 0.6:
            raise ValueError("Run-up speed exceeds walking training distribution")
        if self.flight_seconds <= self.recovery_seconds:
            raise ValueError("Flight timeout must allow the recovery interval")
        if self.ball_radius >= min(self.goal_half_width, self.goal_height / 2):
            raise ValueError("Ball does not fit the virtual goal")


def goal_crossing(previous, current, goals, cfg=None):
    cfg = cfg or KickCfg()
    sign = torch.where(goals[:, 0] >= 0, 1.0, -1.0)
    plane = goals[:, 0] + sign * cfg.ball_radius
    before = sign * (previous[:, 0] - plane)
    after = sign * (current[:, 0] - plane)
    crossed = (before < 0) & (after >= 0)
    fraction = (-before / (after - before).clamp(min=1e-09)).clamp(0, 1)
    point = previous + fraction[:, None] * (current - previous)
    inside = (
        ((point[:, 1] - goals[:, 1]).abs() <= cfg.goal_half_width - cfg.ball_radius)
        & (point[:, 2] >= 0)
        & (point[:, 2] <= cfg.goal_height - cfg.ball_radius)
    )
    return (crossed, crossed & inside, point)


def place_trials(raw, ids, layouts):
    robot, ball = (raw.scene["robot"], raw.scene["ball"])
    rstate = robot.data.default_root_state[ids].clone()
    bstate = ball.data.default_root_state[ids].clone()
    for local, layout in enumerate(layouts):
        _, xy, bxy, _, heading = layout
        rstate[local, :2] = torch.as_tensor(xy, device=raw.device)
        rstate[local, 3:7] = torch.tensor(
            [math.cos(heading / 2), 0, 0, math.sin(heading / 2)], device=raw.device
        )
        rstate[local, 7:] = 0
        bstate[local, :2] = torch.as_tensor(bxy, device=raw.device)
        bstate[local, 7:] = 0
    rstate[:, :3] += raw.scene.env_origins[ids]
    bstate[:, :3] += raw.scene.env_origins[ids]
    robot.write_root_state_to_sim(rstate, env_ids=ids)
    ball.write_root_state_to_sim(bstate, env_ids=ids)
    raw.sim.forward()
    raw.scene.update(raw.step_dt)


def load_actor(path, inputs, config, device):
    options = asdict(config.actor)
    options = {
        k: options[k]
        for k in ("hidden_dims", "activation", "obs_normalization", "distribution_cfg")
    }
    obs = TensorDict({"actor": torch.zeros((1, inputs), device=device)}, batch_size=[1])
    actor = MLPModel(obs, {"actor": ["actor"]}, "actor", 16, **options).to(device)
    actor.load_state_dict(
        torch.load(path, map_location=device, weights_only=False)["actor_state_dict"],
        strict=True,
    )
    return actor.eval()


ALL_JOINTS = (
    "Hip_F_L",
    "Hip_A_L",
    "Hip_R_L",
    "Knee_L",
    "Ankle_P_L",
    "Ankle_R_L",
    "Hip_F_R",
    "Hip_A_R",
    "Hip_R_R",
    "Knee_R",
    "Ankle_P_R",
    "Ankle_R_R",
    "l_shoulder_pitch",
    "r_shoulder_pitch",
    "l_elbow",
    "r_elbow",
)


ROOT = Path(__file__).resolve().parents[1]


def torque_bounds(velocity, peak_effort, no_load_speed):
    ratio = velocity / no_load_speed
    upper = peak_effort * (1 - ratio.clamp(0, 1))
    lower = -peak_effort * (1 + ratio.clamp(-1, 0))
    return (lower, upper)


class Motor(IdealPdActuator):
    param_names = (*IdealPdActuator.param_names, "no_load_speed")

    @staticmethod
    def control_law(params, cmd):
        requested = pd_torque(params["stiffness"], params["damping"], cmd)
        low, high = torque_bounds(
            cmd.vel, params["force_limit"], params["no_load_speed"]
        )
        return requested.clamp(min=low, max=high)

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self.no_load_speed = torch.full_like(self.force_limit, self.cfg.no_load_speed)


@dataclass(kw_only=True)
class MotorCfg(IdealPdActuatorCfg):
    no_load_speed: float

    def build(self, entity, target_ids, target_names):
        return Motor(self, entity, target_ids, target_names)


def world_command(env):
    return env.command_manager.get_command("ball_velocity")


def world_heading(env):
    yaw = yaw_from_quaternion(env.scene["robot"].data.root_link_quat_w)
    return torch.stack((yaw.sin(), yaw.cos()), dim=1)


def ball_observation(env):
    r, b = (env.scene["robot"].data, env.scene["ball"].data)
    body = quat_apply_inverse(r.root_link_quat_w, b.root_link_pos_w - r.root_link_pos_w)
    return torch.cat((body, torch.ones_like(body[:, :1])), dim=1)


def velocities(env):
    return torch.cat(
        (
            env.scene["robot"].data.root_link_lin_vel_w,
            env.scene["ball"].data.root_link_lin_vel_w,
        ),
        dim=1,
    )


def privileged_state(env):
    r, b = (env.scene["robot"].data, env.scene["ball"].data)
    feet = env.scene["feet"].data
    return torch.cat(
        (
            r.root_link_lin_vel_b,
            foot_heights(env),
            foot_contacts(env).float(),
            feet.current_air_time,
            feet.force.flatten(1),
            quat_apply_inverse(
                r.root_link_quat_w, b.root_link_pos_w - r.root_link_pos_w
            ),
            quat_apply_inverse(r.root_link_quat_w, b.root_link_lin_vel_w),
        ),
        dim=1,
    )


class JointAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.requested = self._processed_actions.clone()
        self.written = False

    def process_actions(self, actions):
        super().process_actions(actions)
        self.requested = self._processed_actions.clone()
        limits = self._entity.data.joint_pos_limits[:, self.target_ids]
        self._processed_actions.clamp_(min=limits[..., 0], max=limits[..., 1])
        self.written = False

    def apply_actions(self):
        if not self.written:
            super().apply_actions()
            self.written = True


@dataclass(kw_only=True)
class JointActionCfg(JointPositionActionCfg):
    def build(self, env):
        return JointAction(self, env)


def reset_layout(env, env_ids):
    robot, ball = (env.scene["robot"], env.scene["ball"])
    n = len(env_ids)
    heading = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)
    r = robot.data.default_root_state[env_ids].clone()
    r[:, :2] = torch.empty((n, 2), device=env.device).uniform_(-0.5, 0.5)
    r[:, 3:7], r[:, 7:] = (0, 0)
    r[:, 3], r[:, 6] = ((heading / 2).cos(), (heading / 2).sin())
    direction = torch.stack((heading.cos(), heading.sin()), dim=1)
    gap = torch.empty(n, device=env.device).uniform_(0.55, 1.0)
    b = ball.data.default_root_state[env_ids].clone()
    b[:, :2] = r[:, :2] + gap[:, None] * direction
    b[:, 7:] = 0
    r[:, :3] += env.scene.env_origins[env_ids]
    b[:, :3] += env.scene.env_origins[env_ids]
    robot.write_root_state_to_sim(r, env_ids=env_ids)
    ball.write_root_state_to_sim(b, env_ids=env_ids)


def ball_delta(env):
    r, b = (env.scene["robot"].data, env.scene["ball"].data)
    yaw = yaw_from_quaternion(r.root_link_quat_w)
    delta = b.root_link_pos_w[:, :2] - r.root_link_pos_w[:, :2]
    return torch.stack(
        (
            yaw.cos() * delta[:, 0] + yaw.sin() * delta[:, 1],
            -yaw.sin() * delta[:, 0] + yaw.cos() * delta[:, 1],
        ),
        dim=1,
    )


def ball_zone(env):
    return torch.exp(
        -(ball_delta(env) - ball_delta(env).new_tensor([0.2, 0])).square().sum(dim=1)
    )


def geometry(env):
    r, b = (env.scene["robot"].data, env.scene["ball"].data)
    delta = b.root_link_pos_w[:, :2] - r.root_link_pos_w[:, :2]
    command = world_command(env)[:, :2]
    speed = command.norm(dim=1).clamp(min=0.1)
    direction = command / speed[:, None]
    return (r, b, delta, speed, direction)


def controlled(env):
    _, b, delta, _, _ = geometry(env)
    return (
        (env.command_manager.get_term("ball_velocity").quiet < 1.0)
        & (delta.norm(dim=1) < 1.5)
        & (b.root_link_pos_w[:, 2] < 0.25)
    )


def velocity_reward(env):
    command = world_command(env)[:, :2]
    actual = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
    baseline = torch.exp(-command.square().sum(dim=1))
    tracking = torch.exp(-(actual - command).square().sum(dim=1))
    return ((tracking - baseline) / (1 - baseline).clamp(min=0.05)).clamp(
        -1, 1
    ) * controlled(env)


def progress_reward(env):
    r, b, _, speed, direction = geometry(env)
    ball = (b.root_link_lin_vel_w[:, :2] * direction).sum(dim=1)
    robot = (r.root_link_lin_vel_w[:, :2] * direction).sum(dim=1)
    return torch.minimum(ball, robot).div(speed).clamp(0, 1.2) * controlled(env)


def face_ball(env):
    delta = ball_delta(env)
    return torch.exp(-2 * (1 - delta[:, 0] / delta.norm(dim=1).clamp(min=1e-06)))


def upright(env):
    return torch.exp(-builtin.projected_gravity(env)[:, :2].square().sum(dim=1) / 0.2)


def engagement(env):
    r, _, _, speed, direction = geometry(env)
    forward = (r.root_link_lin_vel_w[:, :2] * direction).sum(dim=1)
    return 0.1 + 0.9 * (forward / speed).clamp(0, 1)


def moving_upright(env):
    return upright(env) * engagement(env)


def pose(env):
    r = env.scene["robot"].data
    sigma = r.joint_pos.new_tensor(
        [
            0.8
            if "shoulder" in n
            else 0.5
            if "elbow" in n
            else 0.3
            if "Hip_A" in n or "Ankle_R" in n
            else 0.6
            for n in env.scene["robot"].joint_names
        ]
    )
    sigma = sigma[None] * (
        1 + 0.5 * world_command(env)[:, :2].norm(dim=1, keepdim=True)
    )
    return torch.exp(
        -((r.joint_pos - r.default_joint_pos) / sigma).square().mean(dim=1)
    )


def moving_pose(env):
    return pose(env) * engagement(env)


def foot_ids(env):
    if not hasattr(env, "_joga_feet"):
        env._joga_feet, _ = env.scene["robot"].find_bodies(
            ("l_foot", "r_foot"), preserve_order=True
        )
    return env._joga_feet


def moving(env):
    return world_command(env)[:, :2].norm(dim=1) > 0.05


def clearance(env):
    velocity = (
        env.scene["robot"].data.body_link_lin_vel_w[:, foot_ids(env), :2].norm(dim=-1)
    )
    return ((foot_heights(env) - 0.1).abs() * velocity).sum(dim=1) * moving(env)


def self_collision(env):
    return (
        (env.scene["self_contact"].data.force_history.norm(dim=-1).amax(dim=-1) > 10)
        .float()
        .sum(dim=1)
    )


def feet_close(env):
    xy = env.scene["robot"].data.body_link_pos_w[:, foot_ids(env), :2]
    return 1 - torch.exp(
        -200 * (0.1 - (xy[:, 0] - xy[:, 1]).norm(dim=1)).clamp(min=0).square()
    )


def swing_height(env):
    term = env.command_manager.get_term("ball_velocity")
    return (term.landed * (term.last_swing_peak / 0.1 - 1).square()).sum(
        dim=1
    ) * moving(env)


def slip(env):
    velocity = env.scene["robot"].data.body_link_lin_vel_w[:, foot_ids(env), :2]
    return (velocity.square().sum(dim=-1) * foot_contacts(env)).sum(dim=1) * moving(env)


def torso_angular_velocity(env):
    return env.scene["robot"].data.root_link_ang_vel_b[:, :2].square().sum(dim=1)


def angular_momentum(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_joga_inertias"):
        ids = robot.indexing.body_ids.cpu().numpy()
        env._joga_inertias = (
            torch.tensor(
                env.sim.mj_model.body_mass[ids].copy(),
                device=env.device,
                dtype=torch.float32,
            ),
            torch.tensor(
                env.sim.mj_model.body_inertia[ids].copy(),
                device=env.device,
                dtype=torch.float32,
            ),
        )
    masses, inertia = env._joga_inertias
    pose_w, velocity_w = (robot.data.body_com_pose_w, robot.data.body_com_vel_w)
    position, quat = (pose_w[..., :3], pose_w[..., 3:])
    linear, angular = (velocity_w[..., :3], velocity_w[..., 3:])
    centre = (masses[None, :, None] * position).sum(dim=1) / masses.sum()
    spin = quat_apply(quat, inertia[None] * quat_apply_inverse(quat, angular))
    orbital = torch.linalg.cross(
        position - centre[:, None], masses[None, :, None] * linear, dim=-1
    )
    return (spin + orbital).sum(dim=1).square().sum(dim=1).float()


def soft_landing(env):
    force = env.scene["feet"].data.force_history.norm(dim=-1).amax(dim=-1)
    return (force * foot_contacts(env)).sum(dim=1) * moving(env)


def relative_progress(env):
    robot, ball, delta, _, direction = geometry(env)
    lateral = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
    error = delta - 0.35 * direction
    relative_velocity = (
        ball.root_link_lin_vel_w[:, :2] - robot.root_link_lin_vel_w[:, :2]
    )
    basis = torch.stack((direction, lateral), dim=1)
    width = delta.new_tensor([0.4, 0.2])
    position = (basis * error[:, None]).sum(dim=-1) / width
    velocity = (basis * relative_velocity[:, None]).sum(dim=-1) / width
    potential = torch.exp(-position.square().sum(dim=1))
    return (-2 * potential * (position * velocity).sum(dim=1)).clamp(-3, 3)


def stall_cost(env):
    return (env.command_manager.get_term("ball_velocity").stall_time - 0.8).clamp(0, 2)


def overshoot(env):
    _, ball, _, speed, _ = geometry(env)
    return (
        (ball.root_link_lin_vel_w[:, :2].norm(dim=1) - speed - 0.3)
        .clamp(min=0)
        .square()
    )


def arm_posture_cost(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_dribble_arm_ids"):
        env._dribble_arm_ids, _ = robot.find_joints(
            ("l_elbow", "r_elbow", "l_shoulder_pitch", "r_shoulder_pitch"),
            preserve_order=True,
        )
    q = robot.data.joint_pos[:, env._dribble_arm_ids]
    cost = (
        0.25 * (q[:, :2] + 0.45).square().sum(dim=1)
        + 4 * (q[:, :2] + 0.1).clamp(min=0).square().sum(dim=1)
        + (q[:, 2:].abs() - 0.8).clamp(min=0).square().sum(dim=1)
    )
    return cost / (1 + cost)


def body_ground(env):
    return env.scene["body_ground"].data.found.any(dim=1)


def pitch_end(env):
    return (
        env.scene["ball"].data.root_link_pos_w[:, :2].abs()
        > env.scene["ball"].data.root_link_pos_w.new_tensor([5.3, 3.3])
    ).any(dim=1)


def possession_lost(env):
    _, _, delta, _, _ = geometry(env)
    return (delta.norm(dim=1) > 3.0) & (env.episode_length_buf * env.step_dt > 2)


def make_env_cfg(play=True):
    """Build the exact selected soccer environment directly, without experiment inheritance."""
    if not play:
        raise ValueError(
            "This configuration is for frozen-policy playback; use gait rewards for new training."
        )
    return ManagerBasedRlEnvCfg(
        decimation=10,
        scene=SceneCfg(
            env_spacing=0.0,
            terrain=TerrainEntityCfg(textures=(), materials=()),
            entities={
                "robot": EntityCfg(
                    init_state=EntityCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.841065),
                        joint_pos={
                            "Hip_F_L$": 0.25,
                            "Hip_A_L$": 0.0,
                            "Hip_R_L$": 0.0,
                            "Knee_L$": -0.5,
                            "Ankle_P_L$": -0.25,
                            "Ankle_R_L$": 0.0,
                            "Hip_F_R$": 0.25,
                            "Hip_A_R$": 0.0,
                            "Hip_R_R$": 0.0,
                            "Knee_R$": -0.5,
                            "Ankle_P_R$": -0.25,
                            "Ankle_R_R$": 0.0,
                            "l_shoulder_pitch$": 0.4,
                            "r_shoulder_pitch$": 0.4,
                            "l_elbow$": -0.65,
                            "r_elbow$": -0.65,
                        },
                    ),
                    spec_fn=robot_spec,
                    articulation=EntityArticulationInfoCfg(
                        actuators=(
                            MotorCfg(
                                target_names_expr=("Hip_F_L",),
                                armature=0.01,
                                stiffness=200.0,
                                damping=5.0,
                                effort_limit=222.0,
                                no_load_speed=3.6652,
                            ),
                            MotorCfg(
                                target_names_expr=("Hip_A_L",),
                                armature=0.01,
                                stiffness=150.0,
                                damping=5.0,
                                effort_limit=120.0,
                                no_load_speed=20.944,
                            ),
                            MotorCfg(
                                target_names_expr=("Hip_R_L",),
                                armature=0.01,
                                stiffness=150.0,
                                damping=5.0,
                                effort_limit=60.0,
                                no_load_speed=20.42,
                            ),
                            MotorCfg(
                                target_names_expr=("Knee_L",),
                                armature=0.01,
                                stiffness=200.0,
                                damping=5.0,
                                effort_limit=222.0,
                                no_load_speed=3.6652,
                            ),
                            MotorCfg(
                                target_names_expr=("Ankle_P_L",),
                                armature=0.01,
                                stiffness=40.0,
                                damping=2.0,
                                effort_limit=60.0,
                                no_load_speed=20.42,
                            ),
                            MotorCfg(
                                target_names_expr=("Ankle_R_L",),
                                armature=0.01,
                                stiffness=40.0,
                                damping=2.0,
                                effort_limit=60.0,
                                no_load_speed=20.42,
                            ),
                            MotorCfg(
                                target_names_expr=("Hip_F_R",),
                                armature=0.01,
                                stiffness=200.0,
                                damping=5.0,
                                effort_limit=222.0,
                                no_load_speed=3.6652,
                            ),
                            MotorCfg(
                                target_names_expr=("Hip_A_R",),
                                armature=0.01,
                                stiffness=150.0,
                                damping=5.0,
                                effort_limit=120.0,
                                no_load_speed=20.944,
                            ),
                            MotorCfg(
                                target_names_expr=("Hip_R_R",),
                                armature=0.01,
                                stiffness=150.0,
                                damping=5.0,
                                effort_limit=60.0,
                                no_load_speed=20.42,
                            ),
                            MotorCfg(
                                target_names_expr=("Knee_R",),
                                armature=0.01,
                                stiffness=200.0,
                                damping=5.0,
                                effort_limit=222.0,
                                no_load_speed=3.6652,
                            ),
                            MotorCfg(
                                target_names_expr=("Ankle_P_R",),
                                armature=0.01,
                                stiffness=40.0,
                                damping=2.0,
                                effort_limit=60.0,
                                no_load_speed=20.42,
                            ),
                            MotorCfg(
                                target_names_expr=("Ankle_R_R",),
                                armature=0.01,
                                stiffness=40.0,
                                damping=2.0,
                                effort_limit=60.0,
                                no_load_speed=20.42,
                            ),
                            MotorCfg(
                                target_names_expr=("l_shoulder_pitch",),
                                armature=0.01,
                                stiffness=40.0,
                                damping=2.0,
                                effort_limit=53.0,
                                no_load_speed=6.0,
                            ),
                            MotorCfg(
                                target_names_expr=("r_shoulder_pitch",),
                                armature=0.01,
                                stiffness=40.0,
                                damping=2.0,
                                effort_limit=53.0,
                                no_load_speed=6.0,
                            ),
                            MotorCfg(
                                target_names_expr=("l_elbow",),
                                armature=0.01,
                                stiffness=40.0,
                                damping=2.0,
                                effort_limit=22.0,
                                no_load_speed=6.0,
                            ),
                            MotorCfg(
                                target_names_expr=("r_elbow",),
                                armature=0.01,
                                stiffness=40.0,
                                damping=2.0,
                                effort_limit=22.0,
                                no_load_speed=6.0,
                            ),
                        ),
                        soft_joint_pos_limit_factor=0.95,
                    ),
                ),
                "ball": EntityCfg(
                    init_state=EntityCfg.InitialStateCfg(pos=(0.7, 0, 0.112)),
                    spec_fn=partial(
                        ball_spec,
                        BallCfg(
                            sliding_friction=0.75,
                            foot_friction=0.3,
                            torsional_friction=0.05,
                            rolling_friction=0.02,
                            contact_time=0.055,
                            damping_ratio=0.18,
                        ),
                    ),
                ),
            },
            sensors=(
                ContactSensorCfg(
                    name="feet",
                    primary=ContactMatch(
                        mode="body", pattern=("l_foot", "r_foot"), entity="robot"
                    ),
                    secondary=ContactMatch(mode="body", pattern="terrain"),
                    reduce="netforce",
                    track_air_time=True,
                    history_length=10,
                ),
                ContactSensorCfg(
                    name="body_ground",
                    primary=ContactMatch(
                        mode="body",
                        pattern=".*",
                        entity="robot",
                        exclude=("l_foot", "r_foot"),
                    ),
                    secondary=ContactMatch(mode="body", pattern="terrain"),
                    fields=("found",),
                    reduce="netforce",
                ),
                ContactSensorCfg(
                    name="sole_contacts",
                    primary=ContactMatch(
                        mode="body", pattern=("l_foot", "r_foot"), entity="robot"
                    ),
                    secondary=ContactMatch(mode="body", pattern="terrain"),
                    fields=("found", "pos"),
                    reduce="mindist",
                    num_slots=4,
                ),
                ContactSensorCfg(
                    name="ball_touch",
                    primary=ContactMatch(
                        mode="body", pattern=("l_foot", "r_foot"), entity="robot"
                    ),
                    secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
                    fields=("force",),
                    reduce="netforce",
                    history_length=10,
                ),
                ContactSensorCfg(
                    name="self_contact",
                    primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
                    secondary=ContactMatch(
                        mode="subtree", pattern="base", entity="robot"
                    ),
                    history_length=10,
                ),
            ),
            spec_fn=contact_scene,
        ),
        observations={
            "actor": ObservationGroupCfg(
                terms={
                    "angular_velocity": ObservationTermCfg(func=base_ang_vel),
                    "gravity": ObservationTermCfg(func=projected_gravity),
                    "joint_positions": ObservationTermCfg(
                        func=joint_pos_rel,
                        params={
                            "asset_cfg": SceneEntityCfg(
                                name="robot",
                                joint_names=(
                                    "Hip_F_L",
                                    "Hip_A_L",
                                    "Hip_R_L",
                                    "Knee_L",
                                    "Ankle_P_L",
                                    "Ankle_R_L",
                                    "Hip_F_R",
                                    "Hip_A_R",
                                    "Hip_R_R",
                                    "Knee_R",
                                    "Ankle_P_R",
                                    "Ankle_R_R",
                                    "l_shoulder_pitch",
                                    "r_shoulder_pitch",
                                    "l_elbow",
                                    "r_elbow",
                                ),
                                preserve_order=True,
                            )
                        },
                    ),
                    "joint_velocities": ObservationTermCfg(
                        func=joint_vel_rel,
                        params={
                            "asset_cfg": SceneEntityCfg(
                                name="robot",
                                joint_names=(
                                    "Hip_F_L",
                                    "Hip_A_L",
                                    "Hip_R_L",
                                    "Knee_L",
                                    "Ankle_P_L",
                                    "Ankle_R_L",
                                    "Hip_F_R",
                                    "Hip_A_R",
                                    "Hip_R_R",
                                    "Knee_R",
                                    "Ankle_P_R",
                                    "Ankle_R_R",
                                    "l_shoulder_pitch",
                                    "r_shoulder_pitch",
                                    "l_elbow",
                                    "r_elbow",
                                ),
                                preserve_order=True,
                            )
                        },
                    ),
                    "actions": ObservationTermCfg(func=last_action),
                    "ball_command": ObservationTermCfg(func=world_command),
                    "heading": ObservationTermCfg(func=world_heading),
                    "ball_observation": ObservationTermCfg(func=ball_observation),
                    "simulation_velocities": ObservationTermCfg(func=velocities),
                },
                history_length=3,
            ),
            "critic": ObservationGroupCfg(
                terms={
                    "angular_velocity": ObservationTermCfg(func=base_ang_vel),
                    "gravity": ObservationTermCfg(func=projected_gravity),
                    "joint_positions": ObservationTermCfg(
                        func=joint_pos_rel,
                        params={
                            "asset_cfg": SceneEntityCfg(
                                name="robot",
                                joint_names=(
                                    "Hip_F_L",
                                    "Hip_A_L",
                                    "Hip_R_L",
                                    "Knee_L",
                                    "Ankle_P_L",
                                    "Ankle_R_L",
                                    "Hip_F_R",
                                    "Hip_A_R",
                                    "Hip_R_R",
                                    "Knee_R",
                                    "Ankle_P_R",
                                    "Ankle_R_R",
                                    "l_shoulder_pitch",
                                    "r_shoulder_pitch",
                                    "l_elbow",
                                    "r_elbow",
                                ),
                                preserve_order=True,
                            )
                        },
                    ),
                    "joint_velocities": ObservationTermCfg(
                        func=joint_vel_rel,
                        params={
                            "asset_cfg": SceneEntityCfg(
                                name="robot",
                                joint_names=(
                                    "Hip_F_L",
                                    "Hip_A_L",
                                    "Hip_R_L",
                                    "Knee_L",
                                    "Ankle_P_L",
                                    "Ankle_R_L",
                                    "Hip_F_R",
                                    "Hip_A_R",
                                    "Hip_R_R",
                                    "Knee_R",
                                    "Ankle_P_R",
                                    "Ankle_R_R",
                                    "l_shoulder_pitch",
                                    "r_shoulder_pitch",
                                    "l_elbow",
                                    "r_elbow",
                                ),
                                preserve_order=True,
                            )
                        },
                    ),
                    "actions": ObservationTermCfg(func=last_action),
                    "ball_command": ObservationTermCfg(func=world_command),
                    "heading": ObservationTermCfg(func=world_heading),
                    "ball_observation": ObservationTermCfg(func=ball_observation),
                    "privileged": ObservationTermCfg(func=privileged_state),
                },
                history_length=3,
            ),
        },
        actions={
            "joint_pos": JointActionCfg(
                entity_name="robot",
                actuator_names=(
                    "Hip_F_L",
                    "Hip_A_L",
                    "Hip_R_L",
                    "Knee_L",
                    "Ankle_P_L",
                    "Ankle_R_L",
                    "Hip_F_R",
                    "Hip_A_R",
                    "Hip_R_R",
                    "Knee_R",
                    "Ankle_P_R",
                    "Ankle_R_R",
                    "l_shoulder_pitch",
                    "r_shoulder_pitch",
                    "l_elbow",
                    "r_elbow",
                ),
                scale={
                    "Hip_F_L": 0.45,
                    "Hip_A_L": 0.12,
                    "Hip_R_L": 0.25,
                    "Knee_L": 0.55,
                    "Ankle_P_L": 0.22,
                    "Ankle_R_L": 0.12,
                    "Hip_F_R": 0.45,
                    "Hip_A_R": 0.12,
                    "Hip_R_R": 0.25,
                    "Knee_R": 0.55,
                    "Ankle_P_R": 0.22,
                    "Ankle_R_R": 0.12,
                    "l_shoulder_pitch": 0.65,
                    "r_shoulder_pitch": 0.65,
                    "l_elbow": 0.1,
                    "r_elbow": 0.1,
                },
                preserve_order=True,
            )
        },
        events={
            "reset_base": EventTermCfg(
                func=reset_root_state_uniform,
                params={"pose_range": {"yaw": (0.0, 0.0)}, "velocity_range": {}},
                mode="reset",
            ),
            "reset_joints": EventTermCfg(
                func=reset_joints_by_offset,
                params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
                mode="reset",
            ),
            "joga_layout": EventTermCfg(func=reset_layout, mode="reset"),
        },
        sim=SimulationCfg(nconmax=128, njmax=512, contact_sensor_maxmatch=128),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            max_extra_envs=0,
            elevation=-25,
            entity_name="robot",
            body_name="base",
        ),
        episode_length_s=12,
        rewards={
            "ball_zone": RewardTermCfg(func=ball_zone, weight=0.15),
            "ball_velocity": RewardTermCfg(func=velocity_reward, weight=6),
            "ball_progress": RewardTermCfg(func=progress_reward, weight=3),
            "face_ball": RewardTermCfg(func=face_ball, weight=0.1),
            "upright": RewardTermCfg(func=moving_upright, weight=0.3),
            "pose": RewardTermCfg(func=moving_pose, weight=0.3),
            "clearance": RewardTermCfg(func=clearance, weight=-2.0),
            "joint_limits": RewardTermCfg(
                func=joint_pos_limits,
                params={
                    "asset_cfg": SceneEntityCfg(
                        name="robot",
                        joint_names=(
                            "Hip_F_L",
                            "Hip_A_L",
                            "Hip_R_L",
                            "Knee_L",
                            "Ankle_P_L",
                            "Ankle_R_L",
                            "Hip_F_R",
                            "Hip_A_R",
                            "Hip_R_R",
                            "Knee_R",
                            "Ankle_P_R",
                            "Ankle_R_R",
                            "l_shoulder_pitch",
                            "r_shoulder_pitch",
                            "l_elbow",
                            "r_elbow",
                        ),
                        preserve_order=True,
                    )
                },
                weight=-1.0,
            ),
            "self_collision": RewardTermCfg(func=self_collision, weight=-1.0),
            "feet_close": RewardTermCfg(func=feet_close, weight=-1.0),
            "swing_height": RewardTermCfg(func=swing_height, weight=-0.25),
            "slip": RewardTermCfg(func=slip, weight=-0.1),
            "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.02),
            "torso_angular_velocity": RewardTermCfg(
                func=torso_angular_velocity, weight=-0.05
            ),
            "angular_momentum": RewardTermCfg(func=angular_momentum, weight=-0.02),
            "soft_landing": RewardTermCfg(func=soft_landing, weight=-1e-05),
            "approach": RewardTermCfg(func=relative_progress, weight=1.5),
            "stall": RewardTermCfg(func=stall_cost, weight=-0.5),
            "overshoot": RewardTermCfg(func=overshoot, weight=-1),
            "fall": RewardTermCfg(func=is_terminated, weight=-750),
            "arms": RewardTermCfg(func=arm_posture_cost, weight=-1.5),
        },
        terminations={
            "timeout": TerminationTermCfg(func=time_out, time_out=True),
            "tilt": TerminationTermCfg(
                func=bad_orientation, params={"limit_angle": 0.8}
            ),
            "low_pelvis": TerminationTermCfg(
                func=root_height_below_minimum, params={"minimum_height": 0.55}
            ),
            "body_ground": TerminationTermCfg(func=body_ground),
            "pitch_end": TerminationTermCfg(func=pitch_end, time_out=True),
            "possession_lost": TerminationTermCfg(func=possession_lost),
        },
        commands={
            "ball_velocity": MovingBallCommandCfg(
                resampling_time_range=(1000000.0, 1000000.0),
                curriculum=False,
                speed_range=(0.7, 0.7),
                fixed_heading=0,
            )
        },
    )


def make_rl_cfg():
    """Inference architecture only; playback never builds a learner or optimizer."""
    return SimpleNamespace(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.05,
                "std_type": "log",
            },
        )
    )
