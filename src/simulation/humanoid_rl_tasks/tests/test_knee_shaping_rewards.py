#!/usr/bin/env python3
"""CPU-only tests for knee shaping; no SimulationApp or GPU is started.

Run unittest discovery with the Isaac image's Python for Torch-backed tests.
The managers import is stubbed; robot, action, command, and contact data are fake.
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


SOURCE = Path(__file__).resolve().parents[1] / "humanoid_rl_tasks/locomotion/mdp/knee_shaping.py"


class SceneEntityCfg:
    def __init__(self, name, joint_ids=None, joint_names=None, body_ids=None):
        self.name = name
        self.joint_ids = joint_ids
        self.joint_names = joint_names
        self.body_ids = [0] if body_ids is None else body_ids


class ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def load_rewards():
    """Load only this module with a lightweight managers stub, restoring imports."""
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = SceneEntityCfg
    managers.ManagerTermBase = ManagerTermBase
    previous = {name: sys.modules.get(name) for name in ("isaaclab", "isaaclab.managers")}
    sys.modules["isaaclab"] = ModuleType("isaaclab")
    sys.modules["isaaclab.managers"] = managers
    try:
        spec = importlib.util.spec_from_file_location("_local_knee_shaping_under_test", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class FakeScene(dict):
    def __init__(self, asset, sensors):
        super().__init__(robot=asset)
        self.sensors = sensors


class FakeActionManager:
    def __init__(self, term):
        self.term = term

    def get_term(self, name):
        if name != "joint_pos":
            raise KeyError(name)
        return self.term


class FakeCommandManager:
    def __init__(self, command):
        self.command = command

    def get_command(self, name):
        if name != "base_velocity":
            raise KeyError(name)
        return self.command


def tensor(values):
    return torch.as_tensor(values, dtype=torch.float32)


def make_env(count=1, targets=None):
    """Use deliberately different articulation and action joint orders."""
    joint_names = ["Hip_R", "Knee_R", "Knee_L", "Ankle_L"]
    action_names = ["Ankle_L", "Knee_L", "Hip_R", "Knee_R"]
    scale = tensor([[0.3, 0.5, 0.7, 0.25]]).expand(count, -1).clone()
    offset = tensor([[0.0, 0.2, 0.0, -0.1]]).expand(count, -1).clone()
    clip = tensor([[[-10.0, 10.0], [-1.0, -0.05], [-10.0, 10.0], [-1.0, -0.05]]])
    clip = clip.expand(count, -1, -1).clone()
    if targets is None:
        targets = tensor([[0.0, -0.4, 0.0, -0.7]]).expand(count, -1).clone()
    else:
        targets = tensor(targets)
    raw = ((targets - offset) / scale).clone()
    term = SimpleNamespace(
        _joint_names=action_names,
        _scale=scale,
        _offset=offset,
        _clip=clip,
        raw_actions=raw,
    )
    asset = SimpleNamespace(
        joint_names=joint_names,
        data=SimpleNamespace(
            joint_pos=tensor([[0.0, -0.7, -0.4, 0.0]]).expand(count, -1).clone(),
        ),
    )
    sensors = {}
    for name in ("left_terrain", "right_terrain"):
        sensors[name] = SimpleNamespace(
            data=SimpleNamespace(
                current_air_time=torch.zeros(count, 1),
                current_contact_time=torch.full((count, 1), 0.2),
            )
        )
    command = tensor([[0.5, 0.0, 0.0]]).expand(count, -1).clone()
    env = SimpleNamespace(
        scene=FakeScene(asset, sensors),
        action_manager=FakeActionManager(term),
        command_manager=FakeCommandManager(command),
        num_envs=count,
    )
    return env


KNEES = SceneEntityCfg("robot", joint_ids=[2, 1], joint_names=["Knee_L", "Knee_R"])
LEFT = SceneEntityCfg("left_terrain", body_ids=[0])
RIGHT = SceneEntityCfg("right_terrain", body_ids=[0])


@unittest.skipIf(torch is None, "CPU Torch is supplied by the Isaac container")
class KneeRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rewards = load_rewards()

    def assertTensorEqual(self, actual, expected, atol=1.0e-6):
        expected = tensor(expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, expected, atol=atol, rtol=1.0e-6)

    def overshoot(self, env, **kwargs):
        params = {"asset_cfg": KNEES, **kwargs}
        term = self.rewards.knee_target_overshoot_smooth_l1(SimpleNamespace(params=params), env)
        return term(env, **params)

    def swing(self, env, **kwargs):
        params = {"asset_cfg": KNEES, "left_sensor_cfg": LEFT, "right_sensor_cfg": RIGHT, **kwargs}
        term = self.rewards.knee_swing_flexion_deficit_l2(SimpleNamespace(params=params), env)
        return term(env, **params)

    def set_phase(self, env, left_air, right_air, left_contact, right_contact):
        for name, air, contact in (
            ("left_terrain", left_air, left_contact),
            ("right_terrain", right_air, right_contact),
        ):
            data = env.scene.sensors[name].data
            data.current_air_time = tensor(air).reshape(-1, 1)
            data.current_contact_time = tensor(contact).reshape(-1, 1)

    def test_overshoot_inside_or_at_bounds_zero_and_other_joints_ignored(self):
        env = make_env(3, targets=[
            [99.0, -0.4, -99.0, -0.7],
            [0.0, -1.0, 0.0, -0.05],
            [0.0, -0.05, 0.0, -1.0],
        ])
        self.assertTensorEqual(self.overshoot(env), [0.0, 0.0, 0.0])

    def test_overshoot_both_bounds_and_quadratic_near_boundary(self):
        env = make_env(4, targets=[
            [0.0, -1.2, 0.0, -0.5],
            [0.0, -0.5, 0.0, 0.25],
            [0.0, -1.05, 0.0, -0.5],
            [0.0, -1.2, 0.0, 0.25],
        ])
        self.assertTensorEqual(self.overshoot(env, beta=0.1), [0.15, 0.25, 0.0125, 0.4])

    def test_overshoot_scale_offset_and_resolved_action_joint_mapping(self):
        env = make_env(targets=[[0.0, -1.2, 0.0, -0.03]])
        self.assertEqual(env.action_manager.term._joint_names.index("Knee_L"), 1)
        self.assertEqual(KNEES.joint_ids[0], 2)
        # Two distinct distances: lower-bound excess .2 and upper-bound excess .02.
        self.assertTensorEqual(self.overshoot(env, beta=0.1), [0.152])

    def test_overshoot_cost_symmetric_between_knees(self):
        env = make_env(2, targets=[
            [0.0, -1.3, 0.0, -0.5],
            [0.0, -0.5, 0.0, -1.3],
        ])
        self.assertTensorEqual(self.overshoot(env), [0.25, 0.25])

    def test_overshoot_linear_far_outside_is_not_capped(self):
        env = make_env(3, targets=[
            [0.0, -1.2, 0.0, -0.5],
            [0.0, -11.0, 0.0, -0.5],
            [0.0, -101.0, 0.0, -0.5],
        ])
        self.assertTensorEqual(self.overshoot(env), [0.15, 9.95, 99.95], atol=1.0e-5)

    def test_swing_uses_each_legs_own_phase_and_is_symmetric(self):
        env = make_env(2)
        # Articulation joint order is [Hip_R, Knee_R, Knee_L, Ankle_L].
        env.scene["robot"].data.joint_pos = tensor([[0.0, -1.0, -0.1, 0.0],
                                                  [0.0, -0.1, -1.0, 0.0]])
        self.set_phase(env, [0.2, 0.0], [0.0, 0.2], [0.0, 0.3], [0.3, 0.0])
        self.assertTensorEqual(self.swing(env), [0.0625, 0.0625])

    def test_swing_sufficient_flexion_zero(self):
        env = make_env()
        env.scene["robot"].data.joint_pos[:, 2] = -0.6
        self.set_phase(env, [0.2], [0.0], [0.0], [0.3])
        self.assertTensorEqual(self.swing(env), [0.0])

    def test_swing_both_air_both_stance_and_outside_window_disabled(self):
        env = make_env(5)
        env.scene["robot"].data.joint_pos.fill_(-0.1)
        self.set_phase(env,
                       [0.2, 0.0, 0.01, 0.5, 0.0],
                       [0.2, 0.0, 0.0, 0.0, 0.01],
                       [0.0, 0.3, 0.0, 0.0, 0.3],
                       [0.0, 0.3, 0.3, 0.3, 0.0])
        self.assertTensorEqual(self.swing(env), [0.0] * 5)

    def test_swing_standing_or_pure_yaw_commands_disabled(self):
        env = make_env(3)
        env.scene["robot"].data.joint_pos.fill_(-0.1)
        self.set_phase(env, [0.2] * 3, [0.0] * 3, [0.0] * 3, [0.3] * 3)
        env.command_manager.command = tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.05, 0.05, 0.0]])
        self.assertTensorEqual(self.swing(env), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
