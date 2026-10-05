"""CPU-only source contracts for the additive Pioneer stairs task.

Run from the repository root with::

    python3 -m unittest discover -s src/simulation/humanoid_rl_tasks/tests \
        -p 'test_stairs_contract.py'

The small doubles below exercise the new modules without importing Isaac Lab,
starting Kit, allocating CUDA memory, or loading the robot. They check the new
configuration logic and registration strings, not simulator/API compatibility
or whether the resulting policy can negotiate stairs. Real simulator validation
must run separately once the GPU is idle.
"""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest


TASK_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "humanoid_rl_tasks.locomotion.config.pioneer_humanoid_v1"
CONFIG_ROOT = TASK_ROOT / "humanoid_rl_tasks/locomotion/config/pioneer_humanoid_v1"
ENV_FILE = CONFIG_ROOT / "stairs_env_cfg.py"
RUNNER_FILE = CONFIG_ROOT / "agents/stairs_ppo_cfg.py"
REGISTRY_FILE = CONFIG_ROOT / "__init__.py"
BASE_ENV = "PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg"
BASE_RUNNER = "PioneerHumanoidRoughNoStairsSelectiveKneeShapePPORunnerCfg"


class _Cfg(SimpleNamespace):
    def copy(self):
        return deepcopy(self)

    def replace(self, **changes):
        copied = self.copy()
        copied.__dict__.update(changes)
        return copied


class _PlaneCfg(_Cfg):
    pass


class _StairCourseCfg(_Cfg):
    def __init__(self, **changes):
        values = dict(size=(16.0, 6.0), step_height_range=(0.03, 0.15),
                      step_width=0.35, num_steps=12, approach_length=2.0,
                      walkway_length=2.0, spawn_x=0.9)
        values.update(changes)
        super().__init__(**values)


def _plain(value):
    if isinstance(value, SimpleNamespace):
        return {key: _plain(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    return value


class _RoughEnv:
    """Independent sentinel state for the inherited rough configuration."""

    def __init__(self):
        self.scene = _Cfg(
            robot=_Cfg(asset="existing_robot", selective_collision=True),
            terrain=_Cfg(terrain_generator=_Cfg(sub_terrains={"rough": _Cfg(proportion=1.0)}),
                         max_init_terrain_level=1),
            height_scanner=_Cfg(pattern="existing_235_observation_layout"),
            left_foot_terrain_contact=_Cfg(filter="terrain_only"),
            right_foot_terrain_contact=_Cfg(filter="terrain_only"),
            num_envs=4096,
            env_spacing=2.5,
        )
        self.commands = _Cfg(base_velocity=_Cfg(
            ranges=_Cfg(lin_vel_x=(0.0, 1.0), lin_vel_y=(0.0, 0.0),
                        ang_vel_z=(-1.0, 1.0), heading=(-3.14, 3.14)),
            heading_command=True,
            rel_standing_envs=0.02,
        ))
        self.events = _Cfg(reset_base=_Cfg(params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
        }))
        self.curriculum = _Cfg(terrain_levels="existing_terrain_levels_vel")
        self.observations = _Cfg(policy=_Cfg(enable_corruption=True,
                                           fields="unchanged_observation_order"))
        self.actions = _Cfg(joint_pos=_Cfg(clip={"Knee_.*": (-0.95, -0.05)},
                                          scale="existing_action_scales"))
        self.rewards = _Cfg(knee_swing_flexion="existing_knee_reward",
                            feet_air_time="terrain_only_existing_reward")
        self.terminations = _Cfg(invalid_height_scan="existing_finite_scan_guard")
        self.sim = _Cfg(dt=0.005, physx="existing_gpu_settings")
        self.decimation = 4
        self.episode_length_s = 20.0
        self.__post_init__()

    def __post_init__(self):
        pass


class _RoughRunner:
    def __init__(self):
        self.num_steps_per_env = 24
        self.max_iterations = 6000
        self.save_interval = 25
        self.experiment_name = "pioneer_humanoid_rough_no_stairs_selective_knee_shape"
        self.resume = False
        self.empirical_normalization = None
        self.policy = _Cfg(actor_obs_normalization=True, critic_obs_normalization=True,
                           noise_std_type="log", actor_hidden_dims=[512, 256, 128])
        self.algorithm = _Cfg(learning_rate=1.0e-3, schedule="adaptive", entropy_coef=0.006)
        self.__post_init__()

    def __post_init__(self):
        pass


def _load_without_imports(path, namespace):
    """Execute actual new configuration statements with explicit lightweight doubles."""
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parsed.body = [node for node in parsed.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(parsed, str(path), "exec"), namespace)
    return namespace


def _env_namespace():
    terrain_gen = SimpleNamespace(
        TerrainGeneratorCfg=_Cfg,
        MeshPlaneTerrainCfg=_PlaneCfg,
    )
    course_mdp = SimpleNamespace(
        course_complete=object(),
        course_out_of_bounds=object(),
        complete_stair_course_levels=object(),
    )
    return _load_without_imports(ENV_FILE, {
        "__name__": PACKAGE + ".stairs_env_cfg",
        "configclass": lambda cls: cls,
        "terrain_gen": terrain_gen,
        "TerrainGeneratorCfg": _Cfg,
        "MeshStairCourseTerrainCfg": _StairCourseCfg,
        "CurrTerm": _Cfg,
        "DoneTerm": _Cfg,
        "CurriculumTermCfg": _Cfg,
        "TerminationTermCfg": _Cfg,
        "mdp": course_mdp,
        "course_mdp": course_mdp,
        "stair_course_mdp": course_mdp,
        "COURSE_SIZE": (16.0, 6.0),
        "COURSE_SPAWN_X": 0.9,
        "COURSE_GOAL_X": 13.4,
        "COURSE_GOAL_DISTANCE": 12.5,
        "course_goal_distance": lambda cfg: (cfg.approach_length + 2 * cfg.num_steps * cfg.step_width
                                              + cfg.walkway_length + 1.0 - cfg.spawn_x),
        BASE_ENV: _RoughEnv,
    })


class StairsEnvironmentContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = _env_namespace()
        cls.train_type = cls.namespace["PioneerHumanoidStairsEnvCfg"]
        cls.play_type = cls.namespace["PioneerHumanoidStairsEnvCfg_PLAY"]

    def test_complete_stair_course_precedes_flat_and_covers_the_map(self):
        terrains = self.train_type().scene.terrain.terrain_generator.sub_terrains
        self.assertEqual(list(terrains), ["stair_course", "flat"])
        self.assertIsInstance(terrains["stair_course"], _StairCourseCfg)
        self.assertIsInstance(terrains["flat"], _StairCourseCfg)
        self.assertEqual([cfg.proportion for cfg in terrains.values()], [0.8, 0.2])
        self.assertAlmostEqual(sum(cfg.proportion for cfg in terrains.values()), 1.0)
        # Flat rows must share the asymmetric course spawn, not the stock
        # centered plane origin (which would make a 12.5 m goal unreachable).
        self.assertEqual(terrains["flat"].step_height_range, (0.0, 0.0))
        self.assertEqual(terrains["flat"].spawn_x, terrains["stair_course"].spawn_x)

    def test_stair_course_has_up_flight_walkway_down_flight_and_ground_spawn(self):
        terrain = self.train_type().scene.terrain.terrain_generator
        cfg = terrain.sub_terrains["stair_course"]
        self.assertEqual(cfg.step_height_range, (0.03, 0.15))
        self.assertEqual(cfg.step_width, 0.35)
        self.assertEqual(cfg.num_steps, 12)
        self.assertEqual(cfg.approach_length, 2.0)
        self.assertEqual(cfg.walkway_length, 2.0)
        self.assertEqual(cfg.spawn_x, 0.9)
        self.assertLess(cfg.spawn_x + 0.2, cfg.approach_length)

    def test_training_grid_is_seeded_and_not_cached(self):
        terrain = self.train_type().scene.terrain.terrain_generator
        self.assertEqual(terrain.size, (16.0, 6.0))
        self.assertEqual((terrain.num_rows, terrain.num_cols), (10, 10))
        self.assertEqual(terrain.border_width, 20.0)
        self.assertEqual(terrain.seed, 42)
        self.assertFalse(terrain.use_cache)

    def test_training_starts_at_easiest_curriculum_level(self):
        cfg = self.train_type()
        self.assertEqual(cfg.scene.terrain.max_init_terrain_level, 0)
        self.assertTrue(cfg.scene.terrain.terrain_generator.curriculum)
        curriculum = cfg.curriculum.terrain_levels
        self.assertIs(curriculum.func, self.namespace["mdp"].complete_stair_course_levels)
        self.assertAlmostEqual(curriculum.params["goal_distance"], 12.5)
        self.assertEqual(curriculum.params["lateral_limit"], 2.0)
        self.assertEqual(curriculum.params["minimum_height"], 0.65)
        self.assertEqual(curriculum.params["tilt_limit"], 0.8)

    def test_training_commands_and_reset_favor_forward_traversal(self):
        cfg = self.train_type()
        command = cfg.commands.base_velocity
        self.assertEqual(command.ranges.lin_vel_x, (0.25, 0.6))
        self.assertEqual(command.ranges.lin_vel_y, (0.0, 0.0))
        self.assertEqual(command.ranges.ang_vel_z, (-0.5, 0.5))
        self.assertEqual(command.ranges.heading, (0.0, 0.0))
        self.assertTrue(command.heading_command)
        self.assertEqual(command.rel_heading_envs, 1.0)
        pose = cfg.events.reset_base.params["pose_range"]
        self.assertEqual(pose["x"], (-0.2, 0.2))
        self.assertEqual(pose["y"], (-0.2, 0.2))
        self.assertEqual(pose["yaw"], (-0.15, 0.15))

    def test_course_completion_and_bounds_are_explicit_and_guarded(self):
        cfg = self.train_type()
        self.assertEqual(cfg.episode_length_s, 75.0)
        completion = cfg.terminations.course_complete
        self.assertIs(completion.func, self.namespace["mdp"].course_complete)
        self.assertTrue(completion.time_out)
        self.assertAlmostEqual(completion.params["goal_distance"], 12.5)
        self.assertEqual(completion.params["lateral_limit"], 2.0)
        self.assertEqual(completion.params["minimum_height"], 0.65)
        self.assertEqual(completion.params["tilt_limit"], 0.8)
        bounds = cfg.terminations.course_out_of_bounds
        self.assertIs(bounds.func, self.namespace["mdp"].course_out_of_bounds)
        self.assertFalse(getattr(bounds, "time_out", False))
        self.assertEqual(bounds.params["backward_limit"], -0.7)
        self.assertEqual(bounds.params["lateral_limit"], 2.0)

    def test_episode_budget_accommodates_extended_route_at_slowest_command(self):
        cfg = self.train_type()
        goal = cfg.terminations.course_complete.params["goal_distance"]
        slowest_speed = cfg.commands.base_velocity.ranges.lin_vel_x[0]
        furthest_reset = -cfg.events.reset_base.params["pose_range"]["x"][0]
        self.assertAlmostEqual(goal / slowest_speed, 50.0)
        # Even the furthest allowed start retains substantial time for the
        # stairs, rather than timing out at the old six-step route's budget.
        self.assertGreaterEqual(cfg.episode_length_s, 1.4 * (goal + furthest_reset) / slowest_speed)

    def test_completion_and_curriculum_parameters_do_not_share_mutable_dicts(self):
        first, second, play = self.train_type(), self.train_type(), self.play_type()
        completion = first.terminations.course_complete.params
        curriculum = first.curriculum.terrain_levels.params
        self.assertIsNot(completion, curriculum)
        self.assertIsNot(completion, second.terminations.course_complete.params)
        self.assertIsNot(completion, play.terminations.course_complete.params)
        completion["lateral_limit"] = 99.0
        self.assertEqual(curriculum["lateral_limit"], 2.0)
        self.assertEqual(second.terminations.course_complete.params["lateral_limit"], 2.0)
        self.assertEqual(play.terminations.course_complete.params["lateral_limit"], 2.0)

    def test_train_retains_robot_actions_rewards_physics_and_sensor_layout(self):
        cfg, baseline = self.train_type(), _RoughEnv()
        for field in ("actions", "rewards", "sim", "observations"):
            with self.subTest(field=field):
                self.assertEqual(_plain(getattr(cfg, field)), _plain(getattr(baseline, field)))
        for name, value in vars(baseline.terminations).items():
            with self.subTest(termination=name):
                self.assertEqual(_plain(getattr(cfg.terminations, name)), _plain(value))
        for field in ("robot", "height_scanner", "left_foot_terrain_contact", "right_foot_terrain_contact"):
            with self.subTest(field=field):
                self.assertEqual(_plain(getattr(cfg.scene, field)), _plain(getattr(baseline.scene, field)))
        self.assertEqual(cfg.decimation, baseline.decimation)

    def test_generated_configs_do_not_share_mutable_terrain_state(self):
        first, second = self.train_type(), self.train_type()
        original = _plain(second.scene.terrain.terrain_generator)
        first.scene.terrain.terrain_generator.sub_terrains["stair_course"].step_height_range = (0.2, 0.3)
        self.assertEqual(_plain(second.scene.terrain.terrain_generator), original)
        self.assertEqual(_plain(self.train_type().scene.terrain.terrain_generator), original)

    def test_train_and_play_never_mutate_global_generator_template(self):
        template = self.namespace["PIONEER_STAIRS_TERRAINS_CFG"]
        before = _plain(template)
        train, play = self.train_type(), self.play_type()
        self.assertIsNot(train.scene.terrain.terrain_generator, template)
        self.assertIsNot(play.scene.terrain.terrain_generator, template)
        self.assertIsNot(train.scene.terrain.terrain_generator, play.scene.terrain.terrain_generator)
        for name in ("stair_course", "flat"):
            self.assertIsNot(train.scene.terrain.terrain_generator.sub_terrains[name], template.sub_terrains[name])
            self.assertIsNot(play.scene.terrain.terrain_generator.sub_terrains[name], template.sub_terrains[name])
        self.assertEqual(_plain(template), before)

    def test_play_has_fixed_easy_stairs_and_no_level_advancement(self):
        cfg = self.play_type()
        generator = cfg.scene.terrain.terrain_generator
        self.assertEqual(cfg.scene.num_envs, 50)
        self.assertEqual((generator.num_rows, generator.num_cols), (5, 10))
        self.assertIsNone(cfg.scene.terrain.max_init_terrain_level)
        self.assertIsNone(cfg.curriculum.terrain_levels)
        self.assertTrue(generator.curriculum)
        self.assertEqual(generator.sub_terrains["stair_course"].step_height_range, (0.08, 0.08))
        self.assertEqual(cfg.episode_length_s, 75.0)

    def test_play_is_noise_free_fixed_forward_and_does_not_modify_training(self):
        before = _plain(self.train_type().scene.terrain.terrain_generator)
        cfg = self.play_type()
        self.assertFalse(cfg.observations.policy.enable_corruption)
        self.assertEqual(cfg.commands.base_velocity.ranges.lin_vel_x, (0.4, 0.4))
        self.assertEqual(cfg.commands.base_velocity.rel_standing_envs, 0.0)
        self.assertEqual(cfg.events.reset_base.params["pose_range"]["yaw"], (0.0, 0.0))
        self.assertEqual(_plain(self.train_type().scene.terrain.terrain_generator), before)


class StairsRunnerContracts(unittest.TestCase):
    def test_runner_is_fresh_and_logs_into_a_separate_experiment(self):
        namespace = _load_without_imports(RUNNER_FILE, {
            "__name__": PACKAGE + ".agents.stairs_ppo_cfg",
            "configclass": lambda cls: cls,
            BASE_RUNNER: _RoughRunner,
        })
        runner = namespace["PioneerHumanoidStairsPPORunnerCfg"]()
        baseline = _RoughRunner()
        self.assertFalse(runner.resume)
        self.assertEqual(runner.max_iterations, 1000)
        self.assertEqual(runner.save_interval, 25)
        self.assertNotEqual(runner.experiment_name, baseline.experiment_name)
        self.assertIn("stairs", runner.experiment_name)
        self.assertEqual(_plain(runner.policy), _plain(baseline.policy))
        self.assertEqual(_plain(runner.algorithm), _plain(baseline.algorithm))
        self.assertEqual(runner.num_steps_per_env, baseline.num_steps_per_env)
        runner.policy.actor_hidden_dims.append(16)
        self.assertEqual(baseline.policy.actor_hidden_dims, [512, 256, 128])


class StairsRegistrationContracts(unittest.TestCase):
    def test_train_and_play_register_both_existing_prefix_aliases(self):
        registrations = []
        _load_without_imports(REGISTRY_FILE, {
            "__name__": PACKAGE,
            "gym": SimpleNamespace(register=lambda **kwargs: registrations.append(kwargs)),
            "agents": SimpleNamespace(__name__=PACKAGE + ".agents"),
        })
        stairs = {item["id"]: item for item in registrations if "-Stairs-" in item["id"]}
        expected = {f"{prefix}-Stairs-PioneerHumanoid{suffix}-v0"
                    for prefix in ("Isaac-Locomotion", "Isaac-Velocity") for suffix in ("", "-Play")}
        self.assertEqual(set(stairs), expected)
        self.assertEqual(len([item for item in registrations if "-Stairs-" in item["id"]]), 4)
        for task, item in stairs.items():
            with self.subTest(task=task):
                suffix = "_PLAY" if "-Play-" in task else ""
                self.assertEqual(item["entry_point"], "isaaclab.envs:ManagerBasedRLEnv")
                self.assertTrue(item["disable_env_checker"])
                self.assertEqual(item["kwargs"]["env_cfg_entry_point"],
                                 PACKAGE + ".stairs_env_cfg:PioneerHumanoidStairsEnvCfg" + suffix)
                self.assertEqual(item["kwargs"]["rsl_rl_cfg_entry_point"],
                                 PACKAGE + ".agents.stairs_ppo_cfg:PioneerHumanoidStairsPPORunnerCfg")

    def test_existing_rough_knee_shaping_aliases_keep_their_entry_points(self):
        registrations = []
        _load_without_imports(REGISTRY_FILE, {
            "__name__": PACKAGE,
            "gym": SimpleNamespace(register=lambda **kwargs: registrations.append(kwargs)),
            "agents": SimpleNamespace(__name__=PACKAGE + ".agents"),
        })
        by_id = {item["id"]: item for item in registrations}
        for prefix in ("Isaac-Locomotion", "Isaac-Velocity"):
            for suffix in ("", "-Play"):
                task = f"{prefix}-RoughNoStairsSelectiveKneeShape-PioneerHumanoid{suffix}-v0"
                env_suffix = "_PLAY" if suffix else ""
                with self.subTest(task=task):
                    self.assertEqual(by_id[task]["kwargs"], {
                        "env_cfg_entry_point": PACKAGE + ".rough_env_cfg:" + BASE_ENV + env_suffix,
                        "rsl_rl_cfg_entry_point": PACKAGE + ".agents.rsl_rl_ppo_cfg:" + BASE_RUNNER,
                    })

    def test_new_modules_depend_on_team_recipe_not_local_experiment_helpers(self):
        for path, classname, expected_base in (
            (ENV_FILE, "PioneerHumanoidStairsEnvCfg", BASE_ENV),
            (RUNNER_FILE, "PioneerHumanoidStairsPPORunnerCfg", BASE_RUNNER),
        ):
            parsed = ast.parse(path.read_text(encoding="utf-8"))
            classes = {node.name: node for node in parsed.body if isinstance(node, ast.ClassDef)}
            self.assertEqual([ast.unparse(node) for node in classes[classname].bases], [expected_base])
            imports = [node.module or "" for node in ast.walk(parsed) if isinstance(node, ast.ImportFrom)]
            self.assertFalse(any("swing_timing" in module or "outputs" in module for module in imports))
            self.assertNotIn("AppLauncher", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
