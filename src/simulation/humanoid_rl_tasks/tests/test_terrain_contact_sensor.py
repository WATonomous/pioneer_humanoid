#!/usr/bin/env python3
"""CPU tests for terrain-only sensing using fake PhysX force buffers.

The small parent double mirrors Isaac Lab 2.3.2's allocation/reset/lazy-update
contract. It deliberately cannot update forces: every tested force and timer
calculation must run through the real TerrainContactSensor implementation.
These tests do not start SimulationApp and do not validate PhysX collision wiring.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


SOURCE = Path(__file__).resolve().parents[1] / "humanoid_rl_tasks/locomotion/mdp/terrain_contact.py"


class FakeContactView:
    def __init__(self, count, bodies=1, filters=1):
        self.filter_count = filters
        self.forces = torch.zeros(count, bodies, filters, 3)
        self.force_reads = 0
        self.contact_points = torch.full_like(self.forces, 2.0)
        self.friction = torch.full_like(self.forces, 3.0)

    def get_contact_force_matrix(self, dt):
        self.force_reads += 1
        return self.forces.reshape(-1, self.filter_count, 3)

    def get_net_contact_forces(self, dt):
        raise AssertionError("Terrain sensor must not read all-partner forces or invoke the parent's updater")

    def get_contact_data(self, dt):
        return None, self.contact_points, None, None, "counts", "starts"

    def get_friction_data(self, dt):
        return self.friction, None, "counts", "starts"


class FakeContactSensor:
    """Only the parent-owned lifecycle/interface; no force or timer updater."""

    def __init__(self, cfg, count=3, bodies=1, filters=1):
        self.cfg = cfg
        self._num_envs = count
        self._num_bodies = bodies
        self._sim_physics_dt = 0.005
        self.contact_physx_view = FakeContactView(count, bodies, filters)
        poses = torch.zeros(count, bodies, 7)
        poses[..., :3] = torch.tensor([1.0, 2.0, 3.0])
        poses[..., 6] = 1.0  # PhysX xyzw quaternion.
        self.body_physx_view = SimpleNamespace(get_transforms=lambda: poses.clone())
        self.unpack_calls = []

    @property
    def num_bodies(self):
        return self._num_bodies

    def _initialize_impl(self):
        count, bodies, filters = self._num_envs, self._num_bodies, self.contact_physx_view.filter_count
        self._timestamp = torch.zeros(count)
        self._timestamp_last_update = torch.zeros(count)
        self._is_outdated = torch.ones(count, dtype=torch.bool)
        self._data = SimpleNamespace(net_forces_w=torch.zeros(count, bodies, 3))
        data = self._data
        data.force_matrix_w = torch.zeros(count, bodies, filters, 3)
        if self.cfg.history_length > 0:
            data.net_forces_w_history = torch.zeros(count, self.cfg.history_length, bodies, 3)
            data.force_matrix_w_history = torch.zeros(count, self.cfg.history_length, bodies, filters, 3)
        else:
            data.net_forces_w_history = data.net_forces_w.unsqueeze(1)
            data.force_matrix_w_history = data.force_matrix_w.unsqueeze(1)
        for name in ("last_air_time", "current_air_time", "last_contact_time", "current_contact_time"):
            setattr(data, name, torch.zeros(count, bodies) if self.cfg.track_air_time else None)
        data.pos_w = torch.zeros(count, bodies, 3)
        data.quat_w = torch.zeros(count, bodies, 4)
        data.contact_pos_w = torch.zeros_like(data.force_matrix_w)
        data.friction_forces_w = torch.zeros_like(data.force_matrix_w)

    def _update_buffers_impl(self, env_ids):
        raise AssertionError("All-contact parent updater must not run")

    @property
    def data(self):
        ids = self._is_outdated.nonzero().squeeze(-1)
        if len(ids):
            self._update_buffers_impl(ids)
            self._timestamp_last_update[ids] = self._timestamp[ids]
            self._is_outdated[ids] = False
        return self._data

    def update(self, dt):
        self._timestamp += dt
        self._is_outdated |= self._timestamp - self._timestamp_last_update + 1.0e-6 >= self.cfg.update_period
        if self.cfg.history_length > 0:
            _ = self.data

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._timestamp[ids] = 0.0
        self._timestamp_last_update[ids] = 0.0
        self._is_outdated[ids] = True
        data = self._data
        for name in ("net_forces_w", "net_forces_w_history", "force_matrix_w", "force_matrix_w_history"):
            getattr(data, name)[ids] = 0.0
        if self.cfg.track_air_time:
            for name in ("current_air_time", "last_air_time", "current_contact_time", "last_contact_time"):
                getattr(data, name)[ids] = 0.0

    def _unpack_contact_buffer_data(self, values, counts, starts, **kwargs):
        self.unpack_calls.append((counts, starts, kwargs))
        return values


def load_sensor():
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = object
    sensors = ModuleType("isaaclab.sensors")
    sensors.ContactSensor = FakeContactSensor
    math = ModuleType("isaaclab.utils.math")
    math.convert_quat = lambda value, to: value.roll(1, dims=-1)
    modules = {
        "isaaclab": ModuleType("isaaclab"),
        "isaaclab.managers": managers,
        "isaaclab.sensors": sensors,
        "isaaclab.utils": ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": math,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("_terrain_contact_under_test", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


@unittest.skipIf(torch is None, "CPU Torch is supplied by the Isaac container")
class TerrainContactSensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_sensor()

    def make_sensor(self, count=3, bodies=1, filters=1, **kwargs):
        settings = {
            "filter_prim_paths_expr": ["/World/ground/terrain/mesh"],
            "history_length": 3,
            "track_air_time": True,
            "track_pose": False,
            "track_contact_points": False,
            "track_friction_forces": False,
            "force_threshold": 1.0,
            "update_period": 0.005,
        }
        settings.update(kwargs)
        sensor = self.module.TerrainContactSensor(SimpleNamespace(**settings), count, bodies, filters)
        sensor._initialize_impl()
        return sensor

    def assertClose(self, actual, expected):
        expected = torch.as_tensor(expected, dtype=actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_rejects_multiple_sensor_bodies_empty_filter_or_unresolved_filter(self):
        for kwargs in ({"bodies": 2}, {"filter_prim_paths_expr": []}, {"filters": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.make_sensor(**kwargs)

    def test_only_filtered_force_is_read_and_partner_forces_are_summed_before_contact_test(self):
        sensor = self.make_sensor(count=2, filters=2)
        sensor.contact_physx_view.forces[:, 0, :, 2] = torch.tensor([[10.0, -10.0], [2.0, 3.0]])
        sensor.update(0.1)
        self.assertClose(sensor.data.net_forces_w[:, 0, 2], [0.0, 5.0])
        self.assertClose(sensor.data.current_air_time[:, 0], [0.1, 0.0])
        self.assertClose(sensor.data.current_contact_time[:, 0], [0.0, 0.1])
        self.assertEqual(sensor.contact_physx_view.force_reads, 1)

    def test_timer_transitions_include_elapsed_time_exactly_once(self):
        sensor = self.make_sensor(count=1)
        sensor.update(0.1)
        sensor.update(0.1)
        self.assertClose(sensor.data.current_air_time, [[0.2]])
        sensor.contact_physx_view.forces[..., 2] = 5.0
        sensor.update(0.1)
        self.assertClose(sensor.data.last_air_time, [[0.3]])
        self.assertClose(sensor.data.current_air_time, [[0.0]])
        self.assertClose(sensor.data.current_contact_time, [[0.1]])
        sensor.update(0.2)
        sensor.contact_physx_view.forces.zero_()
        sensor.update(0.1)
        self.assertClose(sensor.data.last_contact_time, [[0.4]])
        self.assertClose(sensor.data.current_air_time, [[0.1]])
        self.assertClose(sensor.data.current_contact_time, [[0.0]])

    def test_contact_threshold_remains_strict(self):
        sensor = self.make_sensor(count=2)
        sensor.contact_physx_view.forces[:, 0, 0, 2] = torch.tensor([1.0, 1.001])
        sensor.update(0.02)
        self.assertClose(sensor.data.current_air_time[:, 0], [0.02, 0.0])
        self.assertClose(sensor.data.current_contact_time[:, 0], [0.0, 0.02])

    def test_subset_uses_individual_elapsed_time_without_touching_other_envs_or_timestamps(self):
        sensor = self.make_sensor()
        sensor._timestamp[:] = torch.tensor([0.1, 0.4, 0.7])
        sensor._timestamp_last_update[:] = torch.tensor([0.05, 0.1, 0.5])
        sensor._data.current_air_time[1] = 8.0
        sensor._data.net_forces_w_history[1] = 99.0
        sensor._update_buffers_impl(torch.tensor([0, 2]))
        self.assertClose(sensor._data.current_air_time[:, 0], [0.05, 8.0, 0.2])
        self.assertTrue((sensor._data.net_forces_w_history[1] == 99.0).all().item())
        self.assertClose(sensor._timestamp_last_update, [0.05, 0.1, 0.5])

    def test_empty_subset_is_a_noop(self):
        sensor = self.make_sensor()
        sensor._update_buffers_impl([])
        self.assertEqual(sensor.contact_physx_view.force_reads, 0)
        self.assertClose(sensor._data.current_air_time, [[0.0]] * 3)

    def test_history_contains_only_terrain_forces_and_rolls_once_per_update(self):
        sensor = self.make_sensor(count=1)
        for force in (1.0, 2.0, 3.0, 4.0):
            sensor.contact_physx_view.forces[..., 2] = force
            sensor.update(0.02)
        self.assertClose(sensor.data.net_forces_w_history[0, :, 0, 2], [4.0, 3.0, 2.0])
        self.assertClose(sensor.data.force_matrix_w_history[0, :, 0, 0, 2], [4.0, 3.0, 2.0])
        self.assertEqual(sensor.contact_physx_view.force_reads, 4)

    def test_zero_history_preserves_parent_latest_frame_aliases(self):
        sensor = self.make_sensor(count=1, history_length=0)
        sensor.contact_physx_view.forces[..., 2] = 5.0
        sensor.update(0.02)
        self.assertClose(sensor.data.net_forces_w_history[0, :, 0, 2], [5.0])
        self.assertClose(sensor.data.force_matrix_w_history[0, :, 0, 0, 2], [5.0])
        self.assertEqual(sensor.data.net_forces_w.data_ptr(), sensor.data.net_forces_w_history.data_ptr())
        self.assertEqual(sensor.data.force_matrix_w.data_ptr(), sensor.data.force_matrix_w_history.data_ptr())

    def test_reading_current_data_does_not_advance_timers_or_read_physx_twice(self):
        sensor = self.make_sensor(count=1, history_length=0)
        sensor.update(0.1)
        self.assertClose(sensor.data.current_air_time, [[0.1]])
        self.assertClose(sensor.data.current_air_time, [[0.1]])
        self.assertEqual(sensor.contact_physx_view.force_reads, 1)
        sensor.update(0.05)
        self.assertClose(sensor.data.current_air_time, [[0.15]])
        self.assertEqual(sensor.contact_physx_view.force_reads, 2)

    def test_inherited_subset_reset_clears_histories_and_timers_without_leaking_previous_episode(self):
        sensor = self.make_sensor(count=2)
        sensor.contact_physx_view.forces[..., 2] = 5.0
        sensor.update(0.2)
        sensor.reset([0])
        self.assertClose(sensor._data.current_contact_time[:, 0], [0.0, 0.2])
        self.assertClose(sensor._timestamp, [0.0, 0.2])
        self.assertTrue((sensor._data.net_forces_w_history[0] == 0.0).all().item())
        self.assertTrue((sensor._data.force_matrix_w_history[0] == 0.0).all().item())
        sensor.contact_physx_view.forces.zero_()
        sensor.update(0.02)
        self.assertClose(sensor.data.current_air_time[:, 0], [0.02, 0.02])
        self.assertClose(sensor.data.last_contact_time[:, 0], [0.0, 0.22])

    def test_disabling_air_time_does_not_require_timer_buffers(self):
        sensor = self.make_sensor(count=1, track_air_time=False)
        sensor.contact_physx_view.forces[..., 2] = 4.0
        sensor.update(0.02)
        self.assertClose(sensor.data.net_forces_w[:, 0, 2], [4.0])
        self.assertIsNone(sensor.data.current_air_time)

    def test_optional_pose_retains_xyzw_to_wxyz_conversion(self):
        sensor = self.make_sensor(count=1, track_pose=True)
        sensor.update(0.02)
        self.assertClose(sensor.data.pos_w, [[[1.0, 2.0, 3.0]]])
        self.assertClose(sensor.data.quat_w, [[[1.0, 0.0, 0.0, 0.0]]])

    def test_optional_contact_points_and_friction_keep_parent_unpack_contract(self):
        sensor = self.make_sensor(count=1, track_contact_points=True, track_friction_forces=True)
        sensor.update(0.02)
        self.assertTrue((sensor.data.contact_pos_w == 2.0).all().item())
        self.assertTrue((sensor.data.friction_forces_w == 3.0).all().item())
        self.assertEqual(sensor.unpack_calls, [
            ("counts", "starts", {}), ("counts", "starts", {"avg": False, "default": 0.0}),
        ])

    def test_randomized_subset_transitions_match_independent_scalar_reference(self):
        sensor = self.make_sensor()
        random = torch.Generator().manual_seed(324)
        names = ("current_air_time", "current_contact_time", "last_air_time", "last_contact_time")
        reference = [{name: 0.0 for name in names} for _ in range(3)]
        timestamps = [0.0] * 3
        for step in range(20):
            sensor._timestamp += 0.02
            sensor.contact_physx_view.forces.copy_(torch.randn(3, 1, 1, 3, generator=random))
            ids = [0, 2] if step % 2 else [0, 1, 2]
            sensor._update_buffers_impl(ids)
            for index in ids:
                elapsed = float(sensor._timestamp[index]) - timestamps[index]
                state = reference[index]
                contact = float(sensor.contact_physx_view.forces[index].norm()) > sensor.cfg.force_threshold
                if contact:
                    if state["current_air_time"] > 0.0:
                        state["last_air_time"] = state["current_air_time"] + elapsed
                    state["current_air_time"] = 0.0
                    state["current_contact_time"] += elapsed
                else:
                    if state["current_contact_time"] > 0.0:
                        state["last_contact_time"] = state["current_contact_time"] + elapsed
                    state["current_contact_time"] = 0.0
                    state["current_air_time"] += elapsed
                timestamps[index] = float(sensor._timestamp[index])
            sensor._timestamp_last_update[ids] = sensor._timestamp[ids]
            for name in names:
                self.assertClose(getattr(sensor._data, name)[:, 0], [state[name] for state in reference])

    def test_timer_outputs_match_pre_change_adapter_for_randomized_state_and_subset(self):
        """Keep the former adapter's cloned-state math as a regression oracle.

        The former parent update could produce arbitrary all-contact timers;
        its result was always overwritten from these pre-parent snapshots.
        Compare that exact replacement, including untouched environments.
        """
        random = torch.Generator().manual_seed(811)
        names = ("current_air_time", "current_contact_time", "last_air_time", "last_contact_time")
        for ids in (list(range(3)), [0, 2], [1]):
            sensor = self.make_sensor()
            previous = {}
            for name in names:
                values = torch.rand(3, 1, generator=random)
                values[values < 0.5] = 0.0
                setattr(sensor._data, name, values.clone())
                previous[name] = values.clone()
            sensor._timestamp_last_update[:] = torch.rand(3, generator=random)
            sensor._timestamp[:] = sensor._timestamp_last_update + torch.rand(3, generator=random)
            sensor.contact_physx_view.forces.copy_(torch.randn(3, 1, 1, 3, generator=random))
            sensor._update_buffers_impl(ids)

            # This is the pre-change override, not the optimized implementation.
            air_time = previous["current_air_time"][ids].clone()
            contact_time = previous["current_contact_time"][ids].clone()
            last_air_time = previous["last_air_time"][ids].clone()
            last_contact_time = previous["last_contact_time"][ids].clone()
            elapsed = (sensor._timestamp[ids] - sensor._timestamp_last_update[ids]).unsqueeze(-1)
            terrain_force = sensor.contact_physx_view.forces[ids].sum(dim=-2)
            in_contact = terrain_force.norm(dim=-1) > sensor.cfg.force_threshold
            first_contact = (air_time > 0.0) & in_contact
            first_detached = (contact_time > 0.0) & ~in_contact
            previous["last_air_time"][ids] = torch.where(first_contact, air_time + elapsed, last_air_time)
            previous["last_contact_time"][ids] = torch.where(
                first_detached, contact_time + elapsed, last_contact_time
            )
            previous["current_air_time"][ids] = torch.where(in_contact, 0.0, air_time + elapsed)
            previous["current_contact_time"][ids] = torch.where(in_contact, contact_time + elapsed, 0.0)
            for name in names:
                self.assertClose(getattr(sensor._data, name), previous[name])


if __name__ == "__main__":
    unittest.main(verbosity=2)
