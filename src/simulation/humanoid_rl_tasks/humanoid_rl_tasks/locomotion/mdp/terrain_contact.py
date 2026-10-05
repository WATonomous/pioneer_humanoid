"""Terrain-only foot sensing for tasks with physical robot self-contact."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


class TerrainContactSensor(ContactSensor):
    """Track force history and air/contact time only against configured terrain.

    Isaac Lab's ordinary contact sensor reports filtered force matrices, but its
    net force and stance timers still include all collision partners. This
    single-foot sensor retains the usual data/reset interface while deriving
    those quantities from the terrain force matrix instead.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        if self.num_bodies != 1 or not self.cfg.filter_prim_paths_expr:
            raise RuntimeError("TerrainContactSensor requires one foot and a nonempty terrain filter.")
        if self.contact_physx_view.filter_count < 1:
            raise RuntimeError("TerrainContactSensor's filter did not resolve to a terrain collider.")

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)
        if self.cfg.track_air_time:
            air_time = self._data.current_air_time[env_ids].clone()
            contact_time = self._data.current_contact_time[env_ids].clone()
            last_air_time = self._data.last_air_time[env_ids].clone()
            last_contact_time = self._data.last_contact_time[env_ids].clone()

        # The inherited updater owns force-matrix/history/pose handling. Pass
        # integer indices because the parent normalizes a full index list.
        parent_env_ids = range(self._num_envs) if isinstance(env_ids, slice) else env_ids
        super()._update_buffers_impl(parent_env_ids)
        terrain_force = self._data.force_matrix_w[env_ids].sum(dim=-2)
        self._data.net_forces_w[env_ids] = terrain_force
        self._data.net_forces_w_history[env_ids, 0] = terrain_force

        if self.cfg.track_air_time:
            elapsed = (self._timestamp[env_ids] - self._timestamp_last_update[env_ids]).unsqueeze(-1)
            in_contact = terrain_force.norm(dim=-1) > self.cfg.force_threshold
            first_contact = (air_time > 0.0) & in_contact
            first_detached = (contact_time > 0.0) & ~in_contact
            self._data.last_air_time[env_ids] = torch.where(first_contact, air_time + elapsed, last_air_time)
            self._data.last_contact_time[env_ids] = torch.where(
                first_detached, contact_time + elapsed, last_contact_time
            )
            self._data.current_air_time[env_ids] = torch.where(in_contact, 0.0, air_time + elapsed)
            self._data.current_contact_time[env_ids] = torch.where(in_contact, contact_time + elapsed, 0.0)


def feet_air_time_positive_biped_terrain(
    env,
    command_name: str,
    threshold: float,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Apply the existing biped air-time reward using terrain-only stance."""
    sensors = [env.scene.sensors[cfg.name] for cfg in (left_sensor_cfg, right_sensor_cfg)]
    air_time = torch.cat([sensor.data.current_air_time for sensor in sensors], dim=1)
    contact_time = torch.cat([sensor.data.current_contact_time for sensor in sensors], dim=1)
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = in_contact.int().sum(dim=1) == 1
    reward = torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0).amin(dim=1).clamp(max=threshold)
    reward *= env.command_manager.get_command(command_name)[:, :2].norm(dim=1) > 0.1
    return reward


def feet_slide_terrain(
    env,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Apply the existing sliding penalty only while each foot touches terrain."""
    sensors = [env.scene.sensors[cfg.name] for cfg in (left_sensor_cfg, right_sensor_cfg)]
    contacts = torch.cat(
        [sensor.data.net_forces_w_history.norm(dim=-1).amax(dim=1) > 1.0 for sensor in sensors], dim=1
    )
    body_velocity = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return (body_velocity.norm(dim=-1) * contacts).sum(dim=1)
