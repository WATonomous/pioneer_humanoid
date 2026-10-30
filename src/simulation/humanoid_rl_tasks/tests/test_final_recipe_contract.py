"""Freeze the final recipe through lightweight config doubles, without Isaac/Kit.

The fingerprint was captured from the pre-flattening recipe, excluding only the
two unused self-contact diagnostic sensors deliberately removed by this change.
Only the explicit versioned-asset relocation is normalized; physical settings
remain part of the fingerprint. This does not validate installed Isaac physics.
"""

import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[4]
TASKS = ROOT / "src/simulation/humanoid_rl_tasks/humanoid_rl_tasks/locomotion"
CONFIG = TASKS / "config/pioneer_humanoid_v1"
ENV_CLASS = "PioneerHumanoidRoughNoStairsSelectiveKneeShapeEnvCfg"
RUNNER_CLASS = "PioneerHumanoidRoughNoStairsSelectiveKneeShapePPORunnerCfg"


class _Cfg(SimpleNamespace):
    def __init__(self, **changes):
        for cls in reversed(type(self).__mro__):
            for name, value in vars(cls).items():
                if not name.startswith("_") and not callable(value) and not isinstance(value, property):
                    setattr(self, name, deepcopy(value))
        vars(self).update(changes)
        self.__post_init__()

    def __post_init__(self):
        pass

    def copy(self):
        return deepcopy(self)

    def replace(self, **changes):
        result = self.copy()
        vars(result).update(changes)
        return result


_Cfg.InitialStateCfg = _Cfg
_Cfg.JointDriveCfg = _Cfg
_Cfg.PDGainsCfg = _Cfg
_Cfg.Ranges = _Cfg
_Cfg.OffsetCfg = _Cfg


def _configclass(cls):
    cls.__init__ = _Cfg.__init__
    if not hasattr(cls, "__post_init__"):
        cls.__post_init__ = _Cfg.__post_init__
    cls.copy = _Cfg.copy
    cls.replace = _Cfg.replace
    return cls


class _Entity(_Cfg):
    def __init__(self, name=None, **changes):
        super().__init__(name=name, **changes)


class _Mdp(SimpleNamespace):
    def __getattr__(self, name):
        return name


class _FactoryNamespace:
    def __getattr__(self, name):
        return _Cfg


class _BaseEnv(_Cfg):
    sim = _Cfg(dt=0.0, physx=_Cfg())


class _Term(_Cfg):
    params = {}


def _execute(path, namespace, source=None):
    parsed = ast.parse(path.read_text(encoding="utf-8") if source is None else source)
    parsed.body = [node for node in parsed.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace.update(__name__="_recipe_contract", __file__=str(path))
    exec(compile(parsed, str(path), "exec"), namespace)
    return namespace


def _namespace(rough_source=None, runner_source=None):
    """Evaluate the actual source declarations with explicit framework substitutes."""
    rough_terrains = _Cfg(
        num_rows=10, num_cols=20, curriculum=True,
        sub_terrains={name: _Cfg(proportion=0.2, definition=name) for name in (
            "random_rough", "boxes", "hf_pyramid_slope", "hf_pyramid_slope_inv")},
    )

    def selective_collision(robot):
        robot.selective_self_collision = True

    namespace = {
        "configclass": _configclass, "math": math, "os": os, "MISSING": "<missing>",
        "mdp": _Mdp(UniformVelocityCommandCfg=_Cfg, JointPositionActionCfg=_Cfg),
        "sim_utils": _FactoryNamespace(), "terrain_gen": _FactoryNamespace(),
        "patterns": _FactoryNamespace(), "ROUGH_TERRAINS_CFG": rough_terrains,
        "ISAAC_NUCLEUS_DIR": "<nucleus>", "ISAACLAB_NUCLEUS_DIR": "<lab_nucleus>",
        "enable_selective_self_collision": selective_collision,
        "TerrainContactSensor": "TerrainContactSensor",
        "feet_air_time_positive_biped_terrain": "feet_air_time_positive_biped_terrain",
        "feet_slide_terrain": "feet_slide_terrain",
        "ManagerBasedRLEnvCfg": _BaseEnv, "SceneEntityCfg": _Entity,
        "RslRlOnPolicyRunnerCfg": _Cfg,
    }
    for name in ("ArticulationCfg", "AssetBaseCfg", "ImplicitActuatorCfg", "CurrTerm", "EventTerm",
                 "ObsGroup", "ObsTerm", "RewTerm", "DoneTerm", "InteractiveSceneCfg", "ContactSensorCfg",
                 "RayCasterCfg", "TerrainImporterCfg", "Unoise", "RslRlPpoActorCriticCfg",
                 "RslRlPpoAlgorithmCfg"):
        namespace[name] = _Cfg
    for name in ("CurrTerm", "EventTerm", "ObsTerm", "RewTerm", "DoneTerm"):
        namespace[name] = _Term
    _execute(ROOT / "src/pioneer_humanoid/pioneer_humanoid/whole_body.py", namespace)
    _execute(ROOT / "src/pioneer_humanoid/pioneer_humanoid/rough_baseline.py", namespace)
    _execute(TASKS / "locomotion_env_cfg.py", namespace)
    _execute(CONFIG / "rough_env_cfg.py", namespace, rough_source)
    _execute(CONFIG / "agents/rsl_rl_ppo_cfg.py", namespace, runner_source)
    return namespace


def _plain(value):
    if isinstance(value, SimpleNamespace) or hasattr(value, "__dict__"):
        return {name: _plain(item) for name, item in vars(value).items() if not name.startswith("_")}
    if isinstance(value, dict):
        return {name: _plain(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, str):
        # whole_body.py joins source asset paths using the host path dialect.
        # A Windows CPU run must produce the same recipe hash as Linux.
        value = value.replace("\\", "/").replace(str(ROOT).replace("\\", "/"), "<repo>")
        # Exact path-only aliases preserve the original physical recipe digest.
        # Do not normalize the upstream shared asset: it is a different robot.
        return value.replace(
            "assets/old_leg_only_humanoid/rl_baseline_v1.urdf",
            "assets/whole_body_humanoid/whole_body_humanoid.urdf",
        ).replace(
            "assets/old_leg_only_humanoid/usd/rl_baseline_v1",
            "assets/whole_body_humanoid/usd",
        )
    return value


def _fingerprint(config):
    data = _plain(config)
    for name in ("left_foot_right_calf_contact", "right_foot_left_calf_contact"):
        data.get("scene", {}).pop(name, None)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


# Original effective configurations before removal of historical tuning classes.
ENV_FINGERPRINT = "1ab6c2f1d436b938143df441906027c09115a1b38eab92cf200113c092723c52"
PLAY_FINGERPRINT = "fcd2321e9c1143042fafdf2f7248b21cd3f784a2d0655854b4f335b5dccbc105"
RUNNER_FINGERPRINT = "96ec2750a32602bf58e3e7e189aa6d0e3ff8c52787d87c112a538fb0a586afb3"
ROBOT_FINGERPRINT = "6f82889fad2810d1abcddac3136bc14964a51b7e8709b296c1700a746310918f"
ACTIVE_REWARDS = {
    "track_lin_vel_xy_exp": ("track_lin_vel_xy_yaw_frame_exp", 1.0),
    "track_ang_vel_z_exp": ("track_ang_vel_z_world_exp", 2.0),
    "ang_vel_xy_l2": ("ang_vel_xy_l2", -0.05),
    "dof_torques_l2": ("joint_torques_l2", -1.5e-7),
    "dof_acc_l2": ("joint_acc_l2", -1.25e-7),
    "action_rate_l2": ("action_rate_l2", -0.005),
    "feet_air_time": ("feet_air_time_positive_biped_terrain", 4.0),
    "flat_orientation_l2": ("flat_orientation_l2", -1.0),
    "dof_pos_limits": ("joint_pos_limits", -1.0),
    "termination_penalty": ("is_terminated", -100.0),
    "feet_slide": ("feet_slide_terrain", -0.1),
    "joint_deviation_hip": ("joint_deviation_l1", -0.1),
    "feet_crossing_penalty": ("feet_crossing_l2", -40.0),
    "base_height_l2": ("base_height_l2_finite", -5.0),
    "hip_yaw_splay": ("joint_pair_symmetric_deviation_l2", -0.5),
    "hip_yaw_individual_straight": ("joint_deviation_l2_when_commanded_straight", -0.5),
    "foot_opposite_calf_clearance": ("feet_opposite_calf_clearance_l2", -40.0),
    "knee_target_overshoot": ("knee_target_overshoot_smooth_l1", -0.025),
    "knee_swing_flexion": ("knee_swing_flexion_deficit_l2", -1.5),
}


class FinalRecipeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = _namespace()

    def test_fingerprint_normalizes_source_path_separators(self):
        relative = "assets/whole_body_humanoid/whole_body_humanoid.urdf"
        posix = str(ROOT).replace("\\", "/") + "/" + relative
        windows = str(ROOT).replace("/", "\\") + "\\" + relative.replace("/", "\\")
        self.assertEqual(_plain(posix), "<repo>/" + relative)
        self.assertEqual(_plain(windows), "<repo>/" + relative)

    def test_frozen_environment_and_play_source_settings_are_unchanged(self):
        self.assertEqual(_fingerprint(self.namespace[ENV_CLASS]()), ENV_FINGERPRINT)
        self.assertEqual(_fingerprint(self.namespace[ENV_CLASS + "_PLAY"]()), PLAY_FINGERPRINT)

    def test_exact_nineteen_reward_functions_and_weights(self):
        for classname in (ENV_CLASS, ENV_CLASS + "_PLAY"):
            rewards = vars(self.namespace[classname]().rewards)
            actual = {name: (term.func, term.weight) for name, term in rewards.items()
                      if isinstance(term, _Cfg) and getattr(term, "weight", 0.0) != 0.0}
            self.assertEqual(actual, ACTIVE_REWARDS)

    def test_final_runner_preserves_the_saved_normalized_ppo_recipe(self):
        self.assertEqual(_fingerprint(self.namespace[RUNNER_CLASS]()), RUNNER_FINGERPRINT)

    def test_legacy_tasks_retain_upstream_normalization_and_log_names(self):
        for terrain in ("Rough", "Flat"):
            runner = self.namespace[f"PioneerHumanoid{terrain}PPORunnerCfg"]()
            self.assertTrue(runner.empirical_normalization)
            self.assertTrue(runner.policy.actor_obs_normalization)
            self.assertTrue(runner.policy.critic_obs_normalization)
            self.assertEqual(runner.experiment_name, f"pioneer_humanoid_{terrain.lower()}")

    def test_baseline_preserves_all_original_robot_settings_except_asset_location(self):
        baseline = self.namespace["ROUGH_NO_STAIRS_BASELINE_CFG"]
        self.assertEqual(_fingerprint(baseline), ROBOT_FINGERPRINT)
        self.assertTrue(baseline.spawn.asset_path.replace("\\", "/").endswith(
            "/assets/old_leg_only_humanoid/rl_baseline_v1.urdf"))
        self.assertTrue(baseline.spawn.usd_dir.replace("\\", "/").endswith(
            "/assets/old_leg_only_humanoid/usd/rl_baseline_v1"))

    def test_final_baseline_does_not_change_legacy_robot_or_height_target(self):
        shared = self.namespace["WHOLE_BODY_HUMANOID_CFG"]
        before = _fingerprint(shared)
        final = self.namespace[ENV_CLASS]()
        legacy = self.namespace["PioneerHumanoidRoughEnvCfg"]()
        self.assertEqual(_fingerprint(shared), before)
        self.assertEqual(_fingerprint(legacy.scene.robot), _fingerprint(
            shared.replace(prim_path="{ENV_REGEX_NS}/Robot")))
        self.assertEqual(legacy.scene.robot.init_state.pos, (0.0, 0.0, 0.75))
        self.assertEqual(legacy.scene.robot.init_state.joint_pos["Ankle_P_R"], -0.10)
        self.assertEqual(legacy.scene.robot.init_state.joint_pos["Ankle_R_R"], 0.04)
        self.assertEqual(legacy.rewards.base_height_l2.params["target_height"], 0.75)
        self.assertEqual(final.scene.robot.init_state.pos, (0.0, 0.0, 0.84))
        self.assertEqual(final.scene.robot.init_state.joint_pos["Ankle_P_R"], 0.10)
        self.assertEqual(final.scene.robot.init_state.joint_pos["Ankle_R_R"], 0.0)
        self.assertEqual(final.rewards.base_height_l2.params["target_height"], 0.84)

    def test_final_recipe_has_one_reward_and_train_play_pair_not_historical_lineage(self):
        parsed = ast.parse((CONFIG / "rough_env_cfg.py").read_text(encoding="utf-8"))
        classes = {node.name: node for node in parsed.body if isinstance(node, ast.ClassDef)}
        self.assertEqual({name for name in classes if "NoStairs" in name}, {
            ENV_CLASS, ENV_CLASS + "_PLAY", "PioneerHumanoidRoughNoStairsSelectiveKneeShapeRewards"})
        self.assertEqual(ast.unparse(classes[ENV_CLASS].bases[0]), "PioneerHumanoidRoughEnvCfg")
        self.assertEqual(ast.unparse(classes[ENV_CLASS + "_PLAY"].bases[0]), ENV_CLASS)

    def test_final_runner_has_no_intermediate_experiment_classes(self):
        parsed = ast.parse((CONFIG / "agents/rsl_rl_ppo_cfg.py").read_text(encoding="utf-8"))
        classes = {node.name: node for node in parsed.body if isinstance(node, ast.ClassDef)}
        self.assertEqual(set(classes), {
            "PioneerHumanoidRoughPPORunnerCfg", "PioneerHumanoidFlatPPORunnerCfg", RUNNER_CLASS})
        self.assertEqual(ast.unparse(classes[RUNNER_CLASS].bases[0]), "PioneerHumanoidRoughPPORunnerCfg")

    def test_train_play_and_legacy_configs_do_not_share_mutable_settings(self):
        train = self.namespace[ENV_CLASS]()
        original = _fingerprint(train)
        play = self.namespace[ENV_CLASS + "_PLAY"]()
        play.rewards.knee_swing_flexion.params["minimum_flexion"] = 99.0
        play.scene.terrain.terrain_generator.sub_terrains["plane"].proportion = 99.0
        self.assertEqual(_fingerprint(train), original)
        self.assertEqual(_fingerprint(self.namespace[ENV_CLASS]()), original)
        final = self.namespace[RUNNER_CLASS]()
        final.policy.actor_hidden_dims.append(99)
        legacy = self.namespace["PioneerHumanoidRoughPPORunnerCfg"]()
        self.assertEqual(legacy.policy.actor_hidden_dims, [512, 256, 128])

    def test_no_unused_pair_sensors_and_required_terrain_sensors_retained(self):
        for classname in (ENV_CLASS, ENV_CLASS + "_PLAY"):
            scene = self.namespace[classname]().scene
            self.assertFalse(hasattr(scene, "left_foot_right_calf_contact"))
            self.assertFalse(hasattr(scene, "right_foot_left_calf_contact"))
            for side in ("left", "right"):
                sensor = getattr(scene, f"{side}_foot_terrain_contact")
                self.assertEqual(sensor.class_type, "TerrainContactSensor")
                self.assertEqual(sensor.filter_prim_paths_expr, ["/World/ground/terrain/mesh"])
                self.assertEqual(sensor.history_length, 3)
                self.assertTrue(sensor.track_air_time)


if __name__ == "__main__":
    unittest.main()
