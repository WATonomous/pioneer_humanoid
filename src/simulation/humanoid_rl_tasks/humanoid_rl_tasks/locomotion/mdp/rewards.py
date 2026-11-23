from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_rotate_inverse, yaw_quat

from .capsule_clearance import segment_segment_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def base_height_l2_finite(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize terrain-relative base-height error without propagating missed rays.

    Isaac Lab represents a ray-cast miss with ``+inf`` coordinates.  Its stock
    ``base_height_l2`` averages those raw hit heights, so one missed ray makes
    the reward and PPO returns non-finite.  Average only valid hits here.  An
    all-missed scan receives a neutral, finite value for this one reward term;
    ``height_scan_invalid`` terminates and resets that environment in the same
    step.
    """
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is None:
        adjusted_target_height = torch.full_like(asset.data.root_pos_w[:, 2], target_height)
    else:
        sensor = env.scene.sensors[sensor_cfg.name]
        hit_height = sensor.data.ray_hits_w[..., 2]
        finite_hit = torch.isfinite(hit_height)
        finite_hit_count = finite_hit.sum(dim=1)
        mean_hit_height = torch.where(finite_hit, hit_height, 0.0).sum(dim=1) / finite_hit_count.clamp_min(1)
        adjusted_target_height = target_height + mean_hit_height
        # The invalid-scan termination is evaluated before rewards, but reward
        # terms still run on that terminal transition.  Make an all-missed scan
        # contribute zero here rather than manufacturing a ground height.
        adjusted_target_height = torch.where(
            finite_hit_count > 0,
            adjusted_target_height,
            asset.data.root_pos_w[:, 2],
        )
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def feet_crossing_l2(env, asset_cfg: SceneEntityCfg, margin: float = 0.0) -> torch.Tensor:
    """Penalize the left foot crossing to the right side of the right foot (or vice versa).

    asset_cfg.body_ids must resolve to exactly two bodies, ordered [left_foot, right_foot].
    Lateral (body-frame Y) separation is left_y - right_y; this should stay positive (left
    is left of right). The penalty is the *squared* magnitude by which that separation drops
    below `margin` -- squaring (rather than a linear clamp) gives a gentler gradient for small
    violations and a much steeper one for large violations, so a foot swinging past the midline
    into the other foot's side gets penalized even though self_collision is disabled and would
    otherwise let them interpenetrate freely. A linear clamp here previously produced a visible
    stutter: a hard, discontinuous gradient right at the margin boundary caused sharp corrective
    "flinches" every time a foot approached it.
    """
    asset = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)
    rel_pos_w = foot_pos_w - root_pos_w
    num_bodies = rel_pos_w.shape[1]
    yaw = yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)
    rel_pos_yaw = quat_rotate_inverse(yaw, rel_pos_w.reshape(-1, 3)).reshape(rel_pos_w.shape)
    left_y = rel_pos_yaw[:, 0, 1]
    right_y = rel_pos_yaw[:, 1, 1]
    separation = left_y - right_y
    return torch.clamp(margin - separation, min=0.0).square()


def feet_opposite_calf_gaps(
    env,
    feet_cfg: SceneEntityCfg,
    calves_cfg: SceneEntityCfg,
    foot_segment: tuple[tuple[float, float, float], tuple[float, float, float]],
    foot_radius: float,
    calf_segments: tuple,
) -> torch.Tensor:
    """Return conservative foot-to-opposite-calf surface gaps, ordered left/right.

    Both entity configurations must select two bodies in left/right order from
    the same asset. Geometry is expressed in each rigid body's link frame. The
    full heel-to-toe capsule and three calf capsules detect encounters missed
    by a foot-origin lateral-separation check. These are geometric proxies,
    independent of whether PhysX self-collisions are enabled.
    """
    if feet_cfg.name != calves_cfg.name:
        raise ValueError("Foot/calf clearance requires both selections to use the same asset.")
    asset = env.scene[feet_cfg.name]
    feet_pos = asset.data.body_link_pos_w[:, feet_cfg.body_ids, :]
    feet_quat = asset.data.body_link_quat_w[:, feet_cfg.body_ids, :]
    # Reverse the ordered calves: left foot vs right calf, and vice versa.
    calves_pos = asset.data.body_link_pos_w[:, calves_cfg.body_ids, :].flip(1)
    calves_quat = asset.data.body_link_quat_w[:, calves_cfg.body_ids, :].flip(1)
    if feet_pos.shape[1] != 2 or calves_pos.shape[1] != 2:
        raise ValueError("Foot/calf clearance requires exactly two ordered feet and calves.")
    if foot_radius <= 0.0 or not calf_segments or any(segment[2] <= 0.0 for segment in calf_segments):
        raise ValueError("Foot/calf capsules require positive radii and at least one calf segment.")

    def body_point(position, quaternion, local_point):
        point = position.new_tensor(local_point).expand_as(position)
        rotated = quat_apply(quaternion.reshape(-1, 4), point.reshape(-1, 3))
        return position + rotated.reshape_as(position)

    foot_start = body_point(feet_pos, feet_quat, foot_segment[0])
    foot_end = body_point(feet_pos, feet_quat, foot_segment[1])
    gaps = []
    for start, end, radius in calf_segments:
        calf_start = body_point(calves_pos, calves_quat, start)
        calf_end = body_point(calves_pos, calves_quat, end)
        gaps.append(segment_segment_distance(foot_start, foot_end, calf_start, calf_end) - foot_radius - radius)
    # Overlapping calf proxies must not count the same encounter multiple times.
    return torch.stack(gaps, dim=-1).amin(dim=-1)


def feet_opposite_calf_clearance_l2(
    env,
    feet_cfg: SceneEntityCfg,
    calves_cfg: SceneEntityCfg,
    foot_segment: tuple[tuple[float, float, float], tuple[float, float, float]],
    foot_radius: float,
    calf_segments: tuple,
    margin: float = 0.01,
) -> torch.Tensor:
    """Penalize each foot approaching or intersecting the opposite calf."""
    if margin < 0.0:
        raise ValueError("Foot/calf clearance margin cannot be negative.")
    gaps = feet_opposite_calf_gaps(env, feet_cfg, calves_cfg, foot_segment, foot_radius, calf_segments)
    return (margin - gaps).clamp_min(0.0).square().sum(dim=-1)


def joint_pair_symmetric_deviation_l2(
    env, asset_cfg: SceneEntityCfg, deadband: float = 0.0
) -> torch.Tensor:
    """Penalize excessive same-sign deviation of a mirrored joint pair.

    The Pioneer humanoid's left and right hip-yaw axes point in opposite physical
    directions. Equal joint-coordinate deviations therefore rotate both feet
    outward (or both inward), while opposite-sign deviations remain available for
    steering. The squared excess outside ``deadband`` targets severe toe splay
    without fighting small corrective yaw motions around the nominal pose.
    """
    asset = env.scene[asset_cfg.name]
    joint_deviation = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    if joint_deviation.shape[1] != 2:
        raise ValueError(
            "joint_pair_symmetric_deviation_l2 requires exactly two ordered joints, "
            f"got {joint_deviation.shape[1]}."
        )
    symmetric_deviation = 0.5 * (joint_deviation[:, 0] + joint_deviation[:, 1])
    return torch.clamp(torch.abs(symmetric_deviation) - deadband, min=0.0).square()


def joint_deviation_l2_when_commanded_straight(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    deadband: float = 0.0,
    yaw_rate_std: float = 0.25,
) -> torch.Tensor:
    """Penalize each selected joint outside a deadband during straight commands.

    A Gaussian gate makes the penalty strongest at zero commanded yaw and fades
    it smoothly as a turn is requested. This prevents an asymmetric policy from
    hiding a mirrored-pair deviation in just one joint, without taking away the
    hip-yaw motion needed for deliberate steering.
    """
    if yaw_rate_std <= 0.0:
        raise ValueError(f"yaw_rate_std must be positive, got {yaw_rate_std}.")
    asset = env.scene[asset_cfg.name]
    joint_deviation = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    deviation_excess = torch.clamp(torch.abs(joint_deviation) - deadband, min=0.0)
    yaw_command = env.command_manager.get_command(command_name)[:, 2]
    straight_command_gate = torch.exp(-torch.square(yaw_command / yaw_rate_std))
    return torch.sum(torch.square(deviation_excess), dim=1) * straight_command_gate


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def arm_swing_coupling(
    env,
    asset_cfg: SceneEntityCfg,
    hip_joint_names: tuple[str, str] = ("Hip_F_L", "Hip_F_R"),
    shoulder_joint_names: tuple[str, str] = ("l_shoulder_pitch", "r_shoulder_pitch"),
    gain: float = 0.5,
    deadband: float = 0.3,
) -> torch.Tensor:
    """Cost the shoulder pitch velocity not following the same-side hip velocity.

    Humanoid arm swing is driven by the legs: the arm goes back while the
    same-side thigh goes forward. The URDF axes are signed so that a positive hip
    flexion is the thigh moving forward and a positive shoulder pitch is the arm
    moving backward, so the natural pairing is ``v_shoulder ~ +gain*v_hip``
    (in phase in joint coordinates, anti-phase in world space).

    Velocity, not position: a position target ``q_shoulder ~ gain*q_hip`` carries
    the hip's mean with it (the stance hip sits ~0.39 rad from its default), so
    the arms would park at a constant forward offset and swing only a few degrees
    around it. Hip velocity is zero-mean by construction, so this asks for the
    swing itself and nothing else.

    Without a term like this the arms are free: the warm-started arm channels
    start at zero weight and nothing in the stair-course reward pays for moving
    them, so they stay parked at the sides (measured: 0.02 rad of total arm
    deviation after 500 iterations, and unchanged after another 230 with a
    position-coupled version of this term at weight -0.1 -- the hip's own offset
    dominated that error, so it was a near-constant cost the critic absorbed
    instead of a signal the arm channels could act on).

    ``deadband`` (rad/s) keeps the small wobble out of the cost so standing still
    is not penalised for micro-motion.
    """
    asset = env.scene[asset_cfg.name]
    hip_ids = [asset.joint_names.index(n) for n in hip_joint_names]
    shoulder_ids = [asset.joint_names.index(n) for n in shoulder_joint_names]
    hip_vel = asset.data.joint_vel[:, hip_ids]
    shoulder_vel = asset.data.joint_vel[:, shoulder_ids]
    error = torch.clamp(torch.abs(shoulder_vel - gain * hip_vel) - deadband, min=0.0)
    return torch.sum(error, dim=1)


# ---------------------------------------------------------------------------
# Gait symmetry: price the left/right *bias*, not the anti-phase swing.
#
# Measured on the 2026-11-10 stairs policy (model_1350, 24 s, pushes off): the
# right knee sat 54 deg more flexed than the left on average (hip pitch 32 deg),
# the left foot was "in contact" 81 % of the time while the right was 18 %, and
# the left foot slid at 0.62 m/s while touching. Nothing in the reward set
# compared the legs to each other, so a bias was free.
#
# A raw |q_L - q_R| term would fight every step, because the legs are meant to
# be in anti-phase. These compare *slow averages* instead: what a symmetric gait
# has in common is that the time-averaged joint angles and the time-averaged
# stance duty match, whatever the instantaneous phase is.
# ---------------------------------------------------------------------------


def _ema(env, key: str, value: torch.Tensor, tau_s: float) -> torch.Tensor:
    """Exponentially weighted average of ``value`` over roughly ``tau_s`` seconds."""
    store = getattr(env, "_gait_ema_store", None)
    if store is None:
        store = {}
        env._gait_ema_store = store
    state = store.get(key)
    if state is None or state.shape != value.shape:
        state = value.clone()
        store[key] = state
    alpha = min(1.0, env.step_dt / max(tau_s, env.step_dt))
    state.mul_(1.0 - alpha).add_(value, alpha=alpha)
    return state


def joint_bias_mirror(
    env,
    asset_cfg: SceneEntityCfg,
    tau_s: float = 1.0,
    deadband: float = 0.05,
) -> torch.Tensor:
    """Cost a persistent left/right bias in the leg joints.

    Pairs are compared in mirrored joint coordinates: sagittal joints share an
    axis on both sides (symmetric gait = equal time-averaged angles), while
    abduction, hip rotation and ankle roll have axis-mirrored signs (symmetric
    gait = opposite time-averaged angles). ``deadband`` (rad) keeps small
    unavoidable bias out of the cost.

    The average runs over ``tau_s`` seconds, so the alternating swing cancels
    and only the bias a limp would leave behind is priced.
    """
    asset = env.scene[asset_cfg.name]
    pairs = (
        ("Hip_F_L", "Hip_F_R", 1.0),
        ("Knee_L", "Knee_R", 1.0),
        ("Ankle_P_L", "Ankle_P_R", 1.0),
        ("Hip_A_L", "Hip_A_R", -1.0),
        ("Hip_R_L", "Hip_R_R", -1.0),
        ("Ankle_R_L", "Ankle_R_R", -1.0),
    )
    names = asset.joint_names
    left_ids = [names.index(p[0]) for p in pairs]
    right_ids = [names.index(p[1]) for p in pairs]
    signs = torch.tensor([p[2] for p in pairs], dtype=asset.data.joint_pos.dtype,
                         device=asset.data.joint_pos.device)

    left = _ema(env, "joint_bias_left", asset.data.joint_pos[:, left_ids], tau_s)
    right = _ema(env, "joint_bias_right", asset.data.joint_pos[:, right_ids], tau_s)
    excess = (left - signs * right).abs() - deadband
    return excess.clamp_min(0.0).sum(dim=1)


def feet_duty_symmetry(
    env,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    tau_s: float = 2.0,
) -> torch.Tensor:
    """Cost the difference between the two feet's time-averaged stance duty.

    A symmetric gait gives each foot the same share of ground contact; a hop or
    a drag leaves one foot dominant. Averaging over ``tau_s`` makes this
    independent of the instantaneous anti-phase contact pattern.
    """
    left = env.scene.sensors[left_sensor_cfg.name].data.net_forces_w.norm(dim=-1).amax(dim=-1)
    right = env.scene.sensors[right_sensor_cfg.name].data.net_forces_w.norm(dim=-1).amax(dim=-1)
    duty_left = _ema(env, "duty_left", (left > 1.0).to(left.dtype), tau_s)
    duty_right = _ema(env, "duty_right", (right > 1.0).to(right.dtype), tau_s)
    return (duty_left - duty_right).abs()
