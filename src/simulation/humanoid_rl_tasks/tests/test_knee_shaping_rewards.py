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


def load_rewards():
    """Load only this module with a lightweight managers stub, restoring imports."""
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = SceneEntityCfg
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


def make_env(count=1, targets=None, raw_requires_grad=False):
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
    raw = ((targets - offset) / scale).detach().clone().requires_grad_(raw_requires_grad)
    term = SimpleNamespace(
        _joint_names=action_names,
        _joint_ids=[3, 2, 0, 1],
        _scale=scale,
        _offset=offset,
        _clip=clip,
        raw_actions=raw,
        processed_actions=torch.clamp(targets, min=clip[..., 0], max=clip[..., 1]),
        action_dim=4,
    )
    asset = SimpleNamespace(
        joint_names=joint_names,
        num_joints=4,
        data=SimpleNamespace(
            joint_names=joint_names,
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
    # Self-contact/general forces should never be used to classify terrain stance.
    sensors["contact_forces"] = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=torch.full((count, 2, 3), 10000.0))
    )
    command = tensor([[0.5, 0.0, 0.0]]).expand(count, -1).clone()
    env = SimpleNamespace(
        scene=FakeScene(asset, sensors),
        action_manager=FakeActionManager(term),
        command_manager=FakeCommandManager(command),
        num_envs=count,
        device="cpu",
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
        return self.rewards.knee_target_overshoot_smooth_l1(env, KNEES, **kwargs)

    def swing(self, env, **kwargs):
        return self.rewards.knee_swing_flexion_deficit_l2(
            env, asset_cfg=KNEES, left_sensor_cfg=LEFT, right_sensor_cfg=RIGHT, **kwargs
        )

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

    def test_overshoot_scalar_scale_and_offset(self):
        env = make_env(targets=[[0.0, -1.2, 0.0, -0.03]])
        term = env.action_manager.term
        targets = term.raw_actions.detach() * term._scale + term._offset
        term._scale = 0.5
        term._offset = -0.1
        term.raw_actions = (targets - term._offset) / term._scale
        self.assertTensorEqual(self.overshoot(env, beta=0.1), [0.152])

    def test_overshoot_environment_specific_scale_and_offset(self):
        env = make_env(2, targets=[[0.0, -1.2, 0.0, -0.03], [0.0, -1.2, 0.0, -0.03]])
        term = env.action_manager.term
        targets = term.raw_actions.detach() * term._scale + term._offset
        term._scale[1] = tensor([0.9, 1.7, 0.8, 0.4])
        term._offset[1] = tensor([0.0, -0.3, 0.0, 0.2])
        term.raw_actions = (targets - term._offset) / term._scale
        self.assertTensorEqual(self.overshoot(env, beta=0.1), [0.152, 0.152])

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

    def test_overshoot_does_not_mutate_action_or_asset_buffers(self):
        env = make_env(targets=[[0.0, -1.2, 0.0, 0.25]])
        term = env.action_manager.term
        watched = [term.raw_actions, term.processed_actions, term._scale, term._offset, term._clip,
                   env.scene["robot"].data.joint_pos]
        originals = [value.detach().clone() for value in watched]
        self.overshoot(env)
        for actual, original in zip(watched, originals):
            self.assertTrue(torch.equal(actual, original))

    def test_overshoot_finite_differentiable_tensor(self):
        env = make_env(targets=[[0.0, -1.2, 0.0, 0.25]], raw_requires_grad=True)
        cost = self.overshoot(env)
        self.assertTrue(torch.isfinite(cost).all().item())
        cost.sum().backward()
        grad = env.action_manager.term.raw_actions.grad
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertTensorEqual(grad, [[0.0, -0.5, 0.0, 0.25]])

    def test_overshoot_rejects_invalid_beta(self):
        for beta in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(beta=beta), self.assertRaises((ValueError, RuntimeError)):
                self.overshoot(make_env(), beta=beta)

    def test_overshoot_rejects_missing_nonfinite_or_reversed_bounds(self):
        for mode in ("missing", "nonfinite", "reversed", "equal"):
            env = make_env()
            term = env.action_manager.term
            if mode == "missing":
                term._clip = None
            elif mode == "nonfinite":
                term._clip[:, 1, 0] = float("nan")
            elif mode == "reversed":
                term._clip[:, 1, :] = tensor([0.0, -1.0])
            else:
                term._clip[:, 1, :] = tensor([-1.0, -1.0])
            with self.subTest(mode=mode), self.assertRaises((ValueError, RuntimeError)):
                self.overshoot(env)

    def test_overshoot_rejects_missing_or_ambiguous_knee_action_mapping(self):
        for action_names in (["Ankle_L", "Ankle_R", "Hip_R", "Knee_R"],
                             ["Knee_L", "Knee_L", "Hip_R", "Knee_R"]):
            env = make_env()
            env.action_manager.term._joint_names = action_names
            with self.subTest(action_names=action_names), self.assertRaises((ValueError, RuntimeError)):
                self.overshoot(env)

    def test_overshoot_rejects_incompatible_shapes_or_different_action_asset(self):
        for mode in ("clip_shape", "raw_shape", "different_asset"):
            env = make_env()
            term = env.action_manager.term
            if mode == "clip_shape":
                term._clip = torch.zeros(1, 3, 2)
            elif mode == "raw_shape":
                term.raw_actions = torch.zeros(1, 3)
            else:
                term._asset = object()
            with self.subTest(mode=mode), self.assertRaises((ValueError, RuntimeError)):
                self.overshoot(env)

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

    def test_swing_deficit_clamps_wrong_direction_flexion_at_zero(self):
        env = make_env()
        env.scene["robot"].data.joint_pos[:, 2] = 0.1
        self.set_phase(env, [0.2], [0.0], [0.0], [0.3])
        self.assertTensorEqual(self.swing(env), [0.1225])

    def test_swing_air_time_ramp_reaches_full_strength_smoothly(self):
        env = make_env(3)
        env.scene["robot"].data.joint_pos.fill_(-0.1)
        self.set_phase(env, [0.02, 0.05, 0.08], [0.0] * 3, [0.0] * 3, [0.3] * 3)
        self.assertTensorEqual(
            self.swing(env, min_air_time=0.02, air_time_ramp=0.06), [0.0, 0.03125, 0.0625]
        )

    def test_swing_upper_time_boundary_is_inclusive_then_disabled(self):
        env = make_env(2)
        env.scene["robot"].data.joint_pos.fill_(-0.1)
        self.set_phase(env, [0.45, 0.451], [0.0, 0.0], [0.0, 0.0], [0.3, 0.3])
        self.assertTensorEqual(self.swing(env, max_air_time=0.45), [0.0625, 0.0])

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

    def test_swing_xy_norm_gate_accepts_lateral_motion(self):
        env = make_env()
        env.scene["robot"].data.joint_pos[:, 2] = -0.1
        self.set_phase(env, [0.2], [0.0], [0.0], [0.3])
        env.command_manager.command = tensor([[0.0, 0.5, 0.0]])
        self.assertTensorEqual(self.swing(env), [0.0625])

    def test_swing_self_contact_does_not_count_as_terrain_stance(self):
        env = make_env()
        env.scene["robot"].data.joint_pos.fill_(-0.1)
        self.set_phase(env, [0.2], [0.2], [0.0], [0.0])
        self.assertTrue((env.scene.sensors["contact_forces"].data.net_forces_w > 0.0).all().item())
        self.assertTensorEqual(self.swing(env), [0.0])

    def test_swing_finite_differentiable_tensor(self):
        env = make_env()
        env.scene["robot"].data.joint_pos = tensor([[0.0, -1.0, -0.1, 0.0]]).requires_grad_()
        self.set_phase(env, [0.2], [0.0], [0.0], [0.3])
        cost = self.swing(env)
        self.assertTrue(torch.isfinite(cost).all().item())
        cost.sum().backward()
        self.assertTensorEqual(env.scene["robot"].data.joint_pos.grad, [[0.0, 0.0, 0.5, 0.0]])

    def test_swing_rejects_invalid_parameters(self):
        for kwargs in (
            {"minimum_flexion": -0.1}, {"minimum_flexion": float("nan")},
            {"min_air_time": -0.1}, {"max_air_time": 0.0},
            {"min_air_time": 0.5, "max_air_time": 0.1},
            {"move_threshold": -0.1}, {"move_threshold": float("inf")},
            {"air_time_ramp": -0.1}, {"air_time_ramp": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, RuntimeError)):
                self.swing(make_env(), **kwargs)

    def test_swing_rejects_reversed_resolved_knee_order(self):
        wrong_order = SceneEntityCfg("robot", joint_ids=[1, 2], joint_names=["Knee_R", "Knee_L"])
        with self.assertRaises((ValueError, RuntimeError)):
            self.rewards.knee_swing_flexion_deficit_l2(
                make_env(), asset_cfg=wrong_order, left_sensor_cfg=LEFT, right_sensor_cfg=RIGHT
            )

    def test_swing_rejects_multi_body_terrain_sensor(self):
        env = make_env()
        env.scene.sensors["left_terrain"].data.current_air_time = torch.zeros(1, 2)
        env.scene.sensors["left_terrain"].data.current_contact_time = torch.zeros(1, 2)
        with self.assertRaises((ValueError, RuntimeError)):
            self.swing(env)


if __name__ == "__main__":
    unittest.main(verbosity=2)
