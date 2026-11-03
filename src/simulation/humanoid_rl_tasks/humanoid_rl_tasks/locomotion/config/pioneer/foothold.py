# Foothold-conditioned perception after leggedrobotics/foothold-prediction-model ("Watch Your Step: Anticipated Perception
# for Humanoid Locomotion", Heyrman et al., ETH Zurich; MIT License, Copyright (c) 2026 ETH Zurich): foothold_scan and the
# pose/contact observations from fpm/mdp/observations.py, the gravity assistance from fpm/mdp/events.py, unmodified but for
# imports and show_patches. foothold_scan runs the foot predictor (retrained every 500 iterations by foot_pred.FootPredictionSupervisor) and
# returns, for each foot, its predicted next foothold (x, y in the base frame) and a 7 x 7 height patch around it.
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor, RayCaster

from .foot_pred import FootPredictionSupervisor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class _FootholdState:
    supervisor: FootPredictionSupervisor | None = None
    model: torch.jit.ScriptModule | None = None
    model_path: str | None = None
    model_mtime: float = 0.0
    xy_drift: torch.Tensor | None = None
    z_drift: torch.Tensor | None = None
    markers: VisualizationMarkers | None = None


_STATE = _FootholdState()


def _sample_uniform(shape, value_range, device) -> torch.Tensor:
    return torch.rand(shape, device=device) * (value_range[1] - value_range[0]) + value_range[0]


def _reload_model(env: ManagerBasedEnv):
    path = _STATE.supervisor.model_path
    if not path or not os.path.exists(path):
        return
    mtime = os.path.getmtime(path)
    if _STATE.model is None or path != _STATE.model_path or mtime > _STATE.model_mtime:
        print(f"[FPM] loading {path}")
        _STATE.model = torch.jit.load(path, map_location=env.device).eval()
        _STATE.model_path = path
        _STATE.model_mtime = mtime


def _predict_footholds(env: ManagerBasedEnv, use_model: bool, bootstrap_footholds) -> torch.Tensor:
    if use_model:
        cached = getattr(getattr(env, "observation_manager", None), "_obs_buffer", None)
        if cached is not None and "system_state" in cached:  # before the first observation: the bootstrap footholds
            with torch.inference_mode():
                return _STATE.model(cached["system_state"]).view(env.num_envs, 2, 2)
    return torch.tensor(bootstrap_footholds, device=env.device).expand(env.num_envs, 2, 2)


def _draw_patches(patches_w: torch.Tensor):
    """Draws what the policy sees: the height points around each predicted foothold (blue), the footholds (orange)."""
    if _STATE.markers is None:
        def ball(radius, color):
            return sim_utils.SphereCfg(radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
        _STATE.markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/footholds", markers={"patch": ball(0.014, (0.1, 0.8, 0.95)), "foothold": ball(0.04, (1.0, 0.45, 0.05))}))
    centres = patches_w[:, :, patches_w.shape[2] // 2].reshape(-1, 3)
    points = torch.cat([patches_w.reshape(-1, 3), centres])
    kinds = torch.cat([torch.zeros(len(points) - len(centres)), torch.ones(len(centres))]).long()
    _STATE.markers.visualize(translations=points + torch.tensor([0.0, 0.0, 0.012], device=points.device), marker_indices=kinds)


def foothold_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scan_resolution: float = 0.05,
    scan_size: tuple[float, float] = (1.6, 1.0),
    patch_size: int = 7,
    xy_drift_range: tuple[float, float] = (0.0, 0.0),
    z_drift_range: tuple[float, float] = (0.0, 0.0),
    bootstrap_footholds: tuple = ((0.0, 0.1), (0.0, -0.1)),
    train_fpm: bool = True,
    force_model: bool = False,
    fpm_kwargs: dict | None = None,
    show_patches: bool = False,
) -> torch.Tensor:
    if _STATE.supervisor is None:
        _STATE.supervisor = FootPredictionSupervisor(env, train=train_fpm, **(fpm_kwargs or {}))
    _STATE.supervisor.pre_step(env)
    _reload_model(env)

    use_model = _STATE.model is not None and (force_model or not _STATE.supervisor.use_bootstrap_footholds())
    preds_b = _predict_footholds(env, use_model, bootstrap_footholds)

    num_envs, device = env.num_envs, env.device
    reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if _STATE.xy_drift is None or _STATE.xy_drift.shape[0] != num_envs:
        _STATE.xy_drift = _sample_uniform((num_envs, 2), xy_drift_range, device)
    if len(reset_ids) > 0 and tuple(xy_drift_range) != (0.0, 0.0):
        _STATE.xy_drift[reset_ids] = _sample_uniform((len(reset_ids), 2), xy_drift_range, device)

    asset: Articulation = env.scene["robot"]
    sensor: RayCaster = env.scene[sensor_cfg.name]

    feet_b = torch.cat([preds_b, torch.zeros(num_envs, 2, 1, device=device)], dim=-1)
    root_quat = asset.data.root_quat_w.unsqueeze(1).repeat(1, 2, 1)
    feet_w = asset.data.root_pos_w.unsqueeze(1) + math_utils.quat_apply(root_quat, feet_b)
    sensor_quat = sensor.data.quat_w.unsqueeze(1).repeat(1, 2, 1)
    feet_s = math_utils.quat_apply(math_utils.quat_inv(sensor_quat), feet_w - sensor.data.pos_w.unsqueeze(1))
    feet_s[..., :2] += _STATE.xy_drift.unsqueeze(1)

    hit_z = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    height_map = sensor.data.pos_w[:, 2].unsqueeze(1) - hit_z - 0.5

    num_x = int(scan_size[0] / scan_resolution) + 1
    num_y = int(scan_size[1] / scan_resolution) + 1
    origin_x, origin_y = -scan_size[0] / 2.0, -scan_size[1] / 2.0
    half = patch_size // 2
    k_range = torch.arange(-half, half + (patch_size % 2), device=device)
    off_x, off_y = torch.meshgrid(k_range, k_range, indexing="ij")
    off_x, off_y = off_x.flatten(), off_y.flatten()

    outputs, patches_w = [], []
    for i in range(2):
        anchor_x = torch.round((feet_s[:, i, 0] - origin_x) / scan_resolution).long()
        anchor_y = torch.round((feet_s[:, i, 1] - origin_y) / scan_resolution).long()
        sample_x = torch.clamp(anchor_x.unsqueeze(1) + off_x, 0, num_x - 1)
        sample_y = torch.clamp(anchor_y.unsqueeze(1) + off_y, 0, num_y - 1)
        idx = sample_y * num_x + sample_x
        heights = torch.gather(height_map, 1, idx)
        outputs.append(torch.cat([preds_b[:, i, :], heights], dim=1))
        if show_patches:
            patches_w.append(torch.gather(sensor.data.ray_hits_w, 1, idx.unsqueeze(-1).expand(-1, -1, 3)))
    obs = torch.cat(outputs, dim=1)
    if show_patches:
        _draw_patches(torch.stack(patches_w, dim=1))

    z_on = tuple(z_drift_range) != (0.0, 0.0)
    if _STATE.z_drift is None or _STATE.z_drift.shape[0] != num_envs:
        _STATE.z_drift = (
            _sample_uniform((num_envs, 1), z_drift_range, device) if z_on else torch.zeros(num_envs, 1, device=device)
        )
    if len(reset_ids) > 0 and z_on:
        _STATE.z_drift[reset_ids] = _sample_uniform((len(reset_ids), 1), z_drift_range, device)
    if z_on:
        n = patch_size**2
        obs[:, 2 : 2 + n] += _STATE.z_drift
        obs[:, n + 4 : 2 * n + 4] += _STATE.z_drift
    return obs


def body_contact(env: ManagerBasedEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    return (torch.max(torch.norm(forces, dim=-1), dim=1)[0] > threshold).float()


def base_pose_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.cat([asset.data.root_pos_w, asset.data.root_quat_w], dim=-1)


def body_pose_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pose = torch.cat(
        [asset.data.body_pos_w[:, asset_cfg.body_ids], asset.data.body_quat_w[:, asset_cfg.body_ids]], dim=-1
    )
    return pose.view(pose.shape[0], -1)


def body_pos_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    rel = asset.data.body_pos_w[:, asset_cfg.body_ids] - asset.data.root_pos_w.unsqueeze(1)
    quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, rel.shape[1], -1)
    return math_utils.quat_apply_inverse(quat, rel).view(rel.shape[0], -1)


def assist_fraction(levels: torch.Tensor, init_offset: float, cutoff_level: float) -> torch.Tensor:
    max_assist = 1.0 - float(init_offset)
    alpha = max_assist * (1.0 - levels.float() / float(cutoff_level))
    return torch.clamp(alpha, min=0.0, max=max_assist)


def _terrain_levels(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    shared = env.curriculum_manager._curriculum_state.get("terrain_levels", None)
    if shared is not None:
        level = float(shared.item()) if torch.is_tensor(shared) else float(shared)
        return torch.full((len(env_ids),), level, device=env.device)
    return env.scene.terrain.terrain_levels[env_ids].to(env.device)


def apply_gravity_compensation(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    init_offset: float = 0.5,
    cutoff_level: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=asset.device)

    alpha = assist_fraction(_terrain_levels(env, env_ids), init_offset, cutoff_level)
    mass = asset.data.default_mass.to(asset.device)
    if mass.dim() == 3:
        mass = mass[..., 0]
    force = alpha * mass.sum(dim=1)[env_ids] * abs(float(env.sim.cfg.gravity[2]))

    if isinstance(asset_cfg.body_ids, (list, torch.Tensor)):
        num_bodies = len(asset_cfg.body_ids)
    else:
        num_bodies = asset.num_bodies
    up_w = torch.zeros((len(env_ids), 3), device=asset.device)
    up_w[:, 2] = force / float(num_bodies)
    up_b = math_utils.quat_apply_inverse(asset.data.root_quat_w[env_ids], up_w)
    forces = up_b.unsqueeze(1).expand(-1, num_bodies, -1).contiguous()
    asset.set_external_force_and_torque(
        forces=forces, torques=torch.zeros_like(forces), env_ids=env_ids, body_ids=asset_cfg.body_ids, is_global=False
    )


def gravity_offset_level(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, init_offset: float = 0.5, cutoff_level: float = 4.0
) -> torch.Tensor:
    ids = torch.arange(env.scene.num_envs, device=env.device)
    return torch.mean(assist_fraction(_terrain_levels(env, ids), init_offset, cutoff_level)).unsqueeze(0)
