# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-only foot sensing for tasks with physical robot self-contact."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import convert_quat


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
        """Update from filtered terrain forces, without all-contact timer work.

        Buffer/history and timer operations follow Isaac Lab 2.3.2's contact
        sensor (BSD-3-Clause). The parent still owns allocation, reset and lazy
        update timestamps. Only the force source differs: self-contact must not
        advance the terrain stance clock.
        """
        if len(env_ids) == 0:
            return
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        num_filters = self.contact_physx_view.filter_count
        force_matrix = self.contact_physx_view.get_contact_force_matrix(dt=self._sim_physics_dt)
        force_matrix = force_matrix.view(-1, self._num_bodies, num_filters, 3)[env_ids]
        self._data.force_matrix_w[env_ids] = force_matrix
        terrain_force = force_matrix.sum(dim=-2)
        self._data.net_forces_w[env_ids] = terrain_force
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[env_ids].roll(1, dims=1)
            self._data.net_forces_w_history[env_ids, 0] = terrain_force
            self._data.force_matrix_w_history[env_ids] = self._data.force_matrix_w_history[env_ids].roll(1, dims=1)
            self._data.force_matrix_w_history[env_ids, 0] = force_matrix

        # Preserve optional ContactSensor features, although the walking recipe
        # only needs forces, history and timers.
        if self.cfg.track_pose:
            pose = self.body_physx_view.get_transforms().view(-1, self._num_bodies, 7)[env_ids]
            pose[..., 3:] = convert_quat(pose[..., 3:], to="wxyz")
            self._data.pos_w[env_ids], self._data.quat_w[env_ids] = pose.split([3, 4], dim=-1)
        if self.cfg.track_contact_points:
            _, points, _, _, counts, starts = self.contact_physx_view.get_contact_data(dt=self._sim_physics_dt)
            self._data.contact_pos_w[env_ids] = self._unpack_contact_buffer_data(points, counts, starts)[env_ids]
        if self.cfg.track_friction_forces:
            forces, _, counts, starts = self.contact_physx_view.get_friction_data(dt=self._sim_physics_dt)
            self._data.friction_forces_w[env_ids] = self._unpack_contact_buffer_data(
                forces, counts, starts, avg=False, default=0.0
            )[env_ids]

        if self.cfg.track_air_time:
            elapsed = (self._timestamp[env_ids] - self._timestamp_last_update[env_ids]).unsqueeze(-1)
            air_time = self._data.current_air_time[env_ids]
            contact_time = self._data.current_contact_time[env_ids]
            in_contact = terrain_force.norm(dim=-1) > self.cfg.force_threshold
            first_contact = (air_time > 0.0) & in_contact
            first_detached = (contact_time > 0.0) & ~in_contact
            self._data.last_air_time[env_ids] = torch.where(
                first_contact, air_time + elapsed, self._data.last_air_time[env_ids]
            )
            self._data.last_contact_time[env_ids] = torch.where(
                first_detached, contact_time + elapsed, self._data.last_contact_time[env_ids]
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
