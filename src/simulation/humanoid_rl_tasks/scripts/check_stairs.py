#!/usr/bin/env python3
"""Check the real stairs recipe; optionally step two robots with zero actions.

This starts Isaac Sim even without --runtime. Run ONLY when no training/viewer
owns the GPU. It never trains, loads checkpoints, or changes source assets.
The separate tests/test_stairs_contract.py suite is safe during training.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runtime", action="store_true", help="Also run a two-env, 20-step physics smoke test.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import copy
import json

import gymnasium as gym
import numpy as np
import torch

import humanoid_rl_tasks  # noqa: F401 -- register project tasks
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


TRAIN_TASK = "Isaac-Velocity-Stairs-PioneerHumanoid-v0"
PLAY_TASK = "Isaac-Velocity-Stairs-PioneerHumanoid-Play-v0"
BASE_TASK = "Isaac-Velocity-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-v0"


def _copy_field(target, source, path):
    for name in path[:-1]:
        target, source = target[name], source[name]
    target[path[-1]] = copy.deepcopy(source[path[-1]])


def _surface_height(meshes, x, y):
    """CPU vertical ray through horizontal box tops, without optional rtree."""
    heights = []
    point = np.array([x, y])
    for mesh in meshes:
        triangles = mesh.triangles[mesh.face_normals[:, 2] > 0.999]
        a = triangles[:, 0, :2]
        b = triangles[:, 1, :2] - a
        c = triangles[:, 2, :2] - a
        p = point - a
        determinant = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
        valid = np.abs(determinant) > 1.0e-12
        u = np.zeros(len(triangles))
        v = np.zeros(len(triangles))
        u[valid] = (p[valid, 0] * c[valid, 1] - p[valid, 1] * c[valid, 0]) / determinant[valid]
        v[valid] = (b[valid, 0] * p[valid, 1] - b[valid, 1] * p[valid, 0]) / determinant[valid]
        inside = valid & (u >= -1.0e-8) & (v >= -1.0e-8) & (u + v <= 1.0 + 1.0e-8)
        heights.extend(triangles[inside, 0, 2].tolist())
    if not heights:
        raise AssertionError(f"No terrain surface below {(x, y)}")
    return max(heights)


def check_geometry(generator):
    """Verify up → level walkway → down → flat exit on actual mesh functions."""
    results = []
    for height in (0.0, 0.03, 0.08, 0.15):
        cfg = generator.sub_terrains["stair_course"].replace(size=generator.size, step_height_range=(height, height))
        meshes, origin = cfg.function(0.5, cfg)
        cx, cy, cz = origin.tolist()
        assert np.isfinite(origin).all() and abs(cz) < 1.0e-6
        for dx in (-0.2, 0.2):
            for dy in (-0.2, 0.2):
                assert abs(_surface_height(meshes, cx + dx, cy + dy) - cz) < 1.0e-6
        walkway_start = cfg.approach_length + cfg.num_steps * cfg.step_width
        descending_start = walkway_start + cfg.walkway_length
        goal_x = descending_start + cfg.num_steps * cfg.step_width + 1.0
        x_values = [cx]
        expected = [0.0]
        for index in range(cfg.num_steps):
            x_values.append(cfg.approach_length + (index + 0.5) * cfg.step_width)
            expected.append((index + 1) * height)
        for fraction in (0.1, 0.5, 0.9):
            x_values.append(walkway_start + fraction * cfg.walkway_length)
            expected.append(cfg.num_steps * height)
        for index in range(cfg.num_steps):
            x_values.append(descending_start + (index + 0.5) * cfg.step_width)
            expected.append((cfg.num_steps - index - 1) * height)
        x_values.append(goal_x)
        expected.append(0.0)
        actual = [_surface_height(meshes, x, cy) for x in x_values]
        np.testing.assert_allclose(actual, expected, atol=1.0e-6)
        results.append({"riser_m": height, "steps_per_flight": cfg.num_steps,
                        "walkway_length_m": cfg.walkway_length, "goal_x_m": goal_x,
                        "spawn": origin.tolist(), "surface_heights_m": actual})
    return results


def main():
    baseline = load_cfg_from_registry(BASE_TASK, "env_cfg_entry_point")
    train = load_cfg_from_registry(TRAIN_TASK, "env_cfg_entry_point")
    play = load_cfg_from_registry(PLAY_TASK, "env_cfg_entry_point")
    runner = load_cfg_from_registry(TRAIN_TASK, "rsl_rl_cfg_entry_point")
    base_runner = load_cfg_from_registry(BASE_TASK, "rsl_rl_cfg_entry_point")
    active_rewards = {
        name: term["weight"] for name, term in baseline.to_dict()["rewards"].items()
        if isinstance(term, dict) and term.get("weight", 0) != 0
    }
    assert len(active_rewards) == 19, "Frozen walking baseline must retain exactly 19 active rewards"
    assert base_runner.policy.actor_obs_normalization and base_runner.policy.critic_obs_normalization
    # Full config comparison, rather than just observation/action dimensions.
    expected = copy.deepcopy(baseline.to_dict())
    actual = train.to_dict()
    for path in (
        ("scene", "terrain", "terrain_generator"),
        ("scene", "terrain", "max_init_terrain_level"),
        ("commands", "base_velocity", "heading_command"),
        ("commands", "base_velocity", "rel_heading_envs"),
        ("commands", "base_velocity", "ranges", "lin_vel_x"),
        ("commands", "base_velocity", "ranges", "lin_vel_y"),
        ("commands", "base_velocity", "ranges", "ang_vel_z"),
        ("commands", "base_velocity", "ranges", "heading"),
        ("events", "reset_base", "params", "pose_range", "x"),
        ("events", "reset_base", "params", "pose_range", "y"),
        ("events", "reset_base", "params", "pose_range", "yaw"),
        ("episode_length_s",),
        ("curriculum", "terrain_levels"),
    ):
        _copy_field(expected, actual, path)
    for name in ("course_complete", "course_out_of_bounds"):
        expected["terminations"][name] = copy.deepcopy(actual["terminations"][name])
    assert expected == actual, "Stairs changed something beyond terrain/commands/resets/course episode logic"
    expected_runner = copy.deepcopy(base_runner.to_dict())
    expected_runner.update(max_iterations=1000, save_interval=25, resume=False, experiment_name="pioneer_humanoid_stairs")
    assert runner.to_dict() == expected_runner, "PPO defaults changed beyond isolated pilot metadata"
    for cfg in (train, play, runner):
        saved = copy.deepcopy(cfg.to_dict())
        cfg.from_dict(copy.deepcopy(saved))
        assert cfg.to_dict() == saved, "Hydra config round-trip changed values"
    for mode, task in (("", TRAIN_TASK), ("-Play", PLAY_TASK)):
        alias = f"Isaac-Locomotion-Stairs-PioneerHumanoid{mode}-v0"
        assert gym.spec(alias).kwargs == gym.spec(task).kwargs
    assert play.curriculum.terrain_levels is None
    assert play.scene.terrain.terrain_generator.curriculum
    assert not play.observations.policy.enable_corruption
    assert play.scene.terrain.terrain_generator.sub_terrains["stair_course"].step_height_range == (0.08, 0.08)
    assert play.scene.terrain.terrain_generator.sub_terrains["flat"].step_height_range == (0.0, 0.0)
    report = {"registered_aliases_verified": True, "baseline_preserved": True,
              "active_baseline_rewards": active_rewards, "observation_normalization_verified": True,
              "hydra_roundtrip_verified": True, "geometry": check_geometry(train.scene.terrain.terrain_generator),
              "runtime_smoke_verified": False, "training_started": False}
    if args.runtime:
        train.scene.num_envs = 2
        train.seed = 43
        train.sim.device = args.device
        env = gym.make(TRAIN_TASK, cfg=train)
        try:
            env.reset(seed=43)
            base = env.unwrapped
            assert base.action_manager.total_action_dim == 12
            actions = torch.zeros((2, 12), device=base.device)
            for _ in range(20):
                observations, rewards, _, _, _ = env.step(actions)
                assert observations["policy"].shape == (2, 235)
                assert torch.isfinite(observations["policy"]).all()
                assert torch.isfinite(rewards).all()
                for side in ("left", "right"):
                    contacts = base.scene[f"{side}_foot_terrain_contact"].data.force_matrix_w
                    assert contacts is not None and torch.isfinite(contacts).all()
            report.update(runtime_smoke_verified=True, runtime_envs=2, runtime_steps=20,
                          observations=235, actions=12)
        finally:
            env.close()
    print("[stairs_check] " + json.dumps(report, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
