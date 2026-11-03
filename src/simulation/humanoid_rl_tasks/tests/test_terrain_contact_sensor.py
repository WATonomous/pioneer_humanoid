#!/usr/bin/env python3
"""CPU checks for terrain forces, timing, history and independent resets.

The parent double supplies buffer allocation and the reset/lazy-update lifecycle;
the real TerrainContactSensor computes every tested force and timer value.
These tests do not start Isaac or validate PhysX collision wiring.
"""

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
    def __init__(self, count, filters):
        self.filter_count = filters
        self.forces = torch.zeros(count, 1, filters, 3)
        self.force_reads = 0

    def get_contact_force_matrix(self, dt):
        self.force_reads += 1
        return self.forces.reshape(-1, self.filter_count, 3)

    def get_net_contact_forces(self, dt):
        raise AssertionError("Terrain sensor must not read all-partner forces or invoke the parent's updater")


class FakeContactSensor:
    """Parent lifecycle for the recipe's single-foot, history-enabled sensor."""

    def __init__(self, cfg, count=3, filters=1):
        self.cfg = cfg
        self._num_envs = count
        self._num_bodies = 1
        self._sim_physics_dt = 0.005
        self.contact_physx_view = FakeContactView(count, filters)

    @property
    def num_bodies(self):
        return self._num_bodies

    def _initialize_impl(self):
        count, filters = self._num_envs, self.contact_physx_view.filter_count
        self._timestamp = torch.zeros(count)
        self._timestamp_last_update = torch.zeros(count)
        self._is_outdated = torch.ones(count, dtype=torch.bool)
        self._data = SimpleNamespace(
            net_forces_w=torch.zeros(count, 1, 3),
            force_matrix_w=torch.zeros(count, 1, filters, 3),
            net_forces_w_history=torch.zeros(count, self.cfg.history_length, 1, 3),
            force_matrix_w_history=torch.zeros(count, self.cfg.history_length, 1, filters, 3),
        )
        for name in ("last_air_time", "current_air_time", "last_contact_time", "current_contact_time"):
            setattr(self._data, name, torch.zeros(count, 1))

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
        _ = self.data

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._timestamp[ids] = 0.0
        self._timestamp_last_update[ids] = 0.0
        self._is_outdated[ids] = True
        for name in ("net_forces_w", "net_forces_w_history", "force_matrix_w", "force_matrix_w_history",
                     "current_air_time", "last_air_time", "current_contact_time", "last_contact_time"):
            getattr(self._data, name)[ids] = 0.0


def load_sensor():
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = object
    sensors = ModuleType("isaaclab.sensors")
    sensors.ContactSensor = FakeContactSensor
    math = ModuleType("isaaclab.utils.math")
    math.convert_quat = None  # The recipe disables pose tracking.
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

    def make_sensor(self, count=3, filters=1):
        cfg = SimpleNamespace(
            filter_prim_paths_expr=["/World/ground/terrain/mesh"],
            history_length=3, track_air_time=True, track_pose=False,
            track_contact_points=False, track_friction_forces=False,
            force_threshold=1.0, update_period=0.005,
        )
        sensor = self.module.TerrainContactSensor(cfg, count, filters)
        sensor._initialize_impl()
        return sensor

    def assertClose(self, actual, expected):
        expected = torch.as_tensor(expected, dtype=actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

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

    def test_history_contains_only_terrain_forces_and_rolls_once_per_update(self):
        sensor = self.make_sensor(count=1)
        for force in (1.0, 2.0, 3.0, 4.0):
            sensor.contact_physx_view.forces[..., 2] = force
            sensor.update(0.02)
        self.assertClose(sensor.data.net_forces_w_history[0, :, 0, 2], [4.0, 3.0, 2.0])
        self.assertClose(sensor.data.force_matrix_w_history[0, :, 0, 0, 2], [4.0, 3.0, 2.0])
        self.assertEqual(sensor.contact_physx_view.force_reads, 4)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
