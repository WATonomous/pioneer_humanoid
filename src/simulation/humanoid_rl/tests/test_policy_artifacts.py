"""CPU/stdlib contracts for portable playback; never import Isaac or Torch."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_rl.policy_artifacts import (  # noqa: E402
    ACTION_CONTRACT,
    ASSET_DIRECTORY_RELATIVE,
    ASSET_USD_FILE,
    DEFAULT_TASK,
    EXPECTED_JOINT_NAMES,
    EXPECTED_OBSERVATION_TERMS,
    SUPPORTED_TASKS,
    asset_hashes,
    canonical_task,
    configure_flat_terrain,
    load_metadata,
    sha256,
    validate_assets,
    validate_env_config,
    validate_runtime_layout,
)


def manifest():
    return {
        "format": "wato_torchscript_policy",
        "format_version": 3,
        "task": DEFAULT_TASK,
        "policy_sha256": "a" * 64,
        "observation_dim": 235,
        "critic_observation_dim": 235,
        "action_dim": 12,
        "observation_terms": [{"name": name, "dimension": size} for name, size in EXPECTED_OBSERVATION_TERMS],
        "action_joint_names": list(EXPECTED_JOINT_NAMES),
        "actor_observation_normalized": True,
        "critic_observation_normalized": True,
        "actor_observation_normalizer": "EmpiricalNormalization",
        "critic_observation_normalizer": "EmpiricalNormalization",
        "clip_actions": None,
        "step_dt": 0.02,
        "action_contract": deepcopy(ACTION_CONTRACT),
        "asset_directory_relative": ASSET_DIRECTORY_RELATIVE.as_posix(),
        "asset_usd_file": ASSET_USD_FILE,
        "asset_files_sha256": {ASSET_USD_FILE: "b" * 64},
        "probe": {
            "observation": [0.0] * 235,
            "expected_action": [0.0] * 12,
            "rtol": 1.0e-5,
            "atol": 1.0e-5,
        },
    }


def env_config():
    action = SimpleNamespace(
        scale=deepcopy(ACTION_CONTRACT["scale"]),
        use_default_offset=True,
        clip={"Knee_.*": (-0.95, -0.05)},
        asset_name="robot",
        joint_names=[".*"],
    )
    terrains = {
        name: SimpleNamespace(proportion=0.2, marker=name)
        for name in ("plane", "random_rough", "boxes", "hf_pyramid_slope", "hf_pyramid_slope_inv")
    }
    return SimpleNamespace(
        actions=SimpleNamespace(joint_pos=action),
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
        scene=SimpleNamespace(terrain=SimpleNamespace(terrain_generator=SimpleNamespace(sub_terrains=terrains))),
    )


def runtime_env():
    term = SimpleNamespace(_joint_names=list(EXPECTED_JOINT_NAMES))
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(joint_names=list(EXPECTED_JOINT_NAMES))},
        action_manager=SimpleNamespace(get_term=lambda name: term),
        observation_manager=SimpleNamespace(
            active_terms={"policy": [name for name, _ in EXPECTED_OBSERVATION_TERMS]},
            group_obs_term_dim={"policy": [(size,) for _, size in EXPECTED_OBSERVATION_TERMS]},
        ),
        step_dt=0.02,
    )


class PolicyMetadataTests(unittest.TestCase):
    def load(self, value, task=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return load_metadata(path, task)

    def test_accepts_v3_without_policy_file_or_simulator(self):
        expected = manifest()
        expected["checkpoint"] = "/another/machine/model_4000.pt"
        self.assertEqual(self.load(expected), expected)

    def test_play_aliases_and_separate_stairs_task(self):
        for task, canonical in SUPPORTED_TASKS.items():
            with self.subTest(task=task):
                self.assertEqual(canonical_task(task), canonical)
                value = manifest()
                value["task"] = task
                self.assertEqual(self.load(value, canonical)["task"], task)
        with self.assertRaisesRegex(ValueError, "Task mismatch"):
            self.load(manifest(), "Isaac-Locomotion-Stairs-PioneerHumanoid-Play-v0")

    def test_rejects_retired_training_and_flat_tasks(self):
        for task in (
            "Isaac-Locomotion-RoughNoStairs-PioneerHumanoid-Play-v0",
            DEFAULT_TASK.replace("-Play", ""),
            "Isaac-Locomotion-Flat-PioneerHumanoid-Play-v0",
            None,
        ):
            with self.subTest(task=task), self.assertRaises(ValueError):
                canonical_task(task)

    def test_requires_all_contract_fields(self):
        for key in manifest():
            value = manifest()
            del value[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(value)

    def test_rejects_version_dimensions_normalization_and_order(self):
        changes = (
            ("format_version", 2), ("format_version", True),
            ("observation_dim", 234), ("observation_dim", 235.0),
            ("critic_observation_dim", 234), ("action_dim", 11),
            ("actor_observation_normalized", False), ("critic_observation_normalized", 1),
            ("actor_observation_normalizer", "Identity"),
            ("critic_observation_normalizer", "Identity"),
            ("action_joint_names", list(reversed(EXPECTED_JOINT_NAMES))),
            ("observation_terms", list(reversed(manifest()["observation_terms"]))),
            ("policy_sha256", "g" * 64), ("policy_sha256", "a" * 63),
            ("clip_actions", 1.0), ("step_dt", 0.025),
        )
        for key, changed in changes:
            value = manifest()
            value[key] = changed
            with self.subTest(key=key, changed=changed), self.assertRaises(ValueError):
                self.load(value)

    def test_rejects_action_contract_drift_and_extra_keys(self):
        for field, value in (
            ("scale", {"Knee_.*": 0.3}),
            ("clip", {"Knee_.*": [-1.0, 0.0]}),
            ("use_default_offset", 1),
            ("clip_actions", 1.0),
            ("step_dt", float("nan")),
            ("unexpected", "field"),
        ):
            changed = manifest()
            changed["action_contract"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(changed)

    def test_rejects_nonfinite_malformed_and_loose_probe(self):
        changes = (
            ("observation", [0.0] * 234),
            ("expected_action", [0.0] * 11),
            ("observation", [float("nan")] + [0.0] * 234),
            ("expected_action", [float("inf")] + [0.0] * 11),
            ("observation", [False] + [0.0] * 234),
            ("rtol", float("nan")), ("rtol", 0.01), ("atol", -1.0),
            ("atol", True),
        )
        for field, value in changes:
            changed = manifest()
            changed["probe"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.load(changed)

    def test_rejects_asset_traversal_and_foreign_paths(self):
        for field in ("asset_directory_relative", "asset_usd_file"):
            for value in ("../robot.usd", "/robot.usd", "C:/robot.usd", "folder\\robot.usd", "./robot.usd"):
                changed = manifest()
                changed[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.load(changed)
        for key in ("../layer.usd", "/layer.usd", "C:/layer.usd", "sub/part:stream.usd", "config.yaml"):
            changed = manifest()
            changed["asset_files_sha256"][key] = "c" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(changed)


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / ASSET_DIRECTORY_RELATIVE
        self.directory.mkdir(parents=True)
        self.usd = self.directory / ASSET_USD_FILE
        self.usd.write_bytes(b"synthetic cached USD")
        (self.directory / "configuration").mkdir()
        (self.directory / "configuration/robot.usda").write_bytes(b"synthetic payload")
        self.metadata = manifest()
        self.metadata["asset_files_sha256"] = asset_hashes(self.directory)

    def test_checks_hashes_and_ignores_converter_configuration(self):
        self.assertEqual(sha256(self.usd), hashlib.sha256(self.usd.read_bytes()).hexdigest())
        (self.directory / "config.yaml").write_text("asset_path: /another/machine\n", encoding="utf-8")
        self.assertEqual(validate_assets(self.root, self.metadata), self.usd)
        self.assertEqual(len(asset_hashes(self.directory)), 2)

    def test_reports_changed_extra_and_missing_usd_layers(self):
        self.usd.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed=.*whole_body_humanoid"):
            validate_assets(self.root, self.metadata)
        self.usd.write_bytes(b"synthetic cached USD")
        extra = self.directory / "extra.usdc"
        extra.write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "extra=.*extra.usdc"):
            validate_assets(self.root, self.metadata)
        extra.unlink()
        (self.directory / "configuration/robot.usda").unlink()
        with self.assertRaisesRegex(ValueError, "missing=.*robot.usda"):
            validate_assets(self.root, self.metadata)

    def test_missing_canonical_asset_is_explicit(self):
        self.usd.unlink()
        with self.assertRaises(FileNotFoundError):
            validate_assets(self.root, self.metadata)

    def test_preserves_symlink_root_spelling(self):
        alias = self.root / "mapped_repo"
        try:
            alias.symlink_to(self.root, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Platform cannot create directory symlinks: {error}")
        result = validate_assets(alias, self.metadata)
        self.assertEqual(result, alias / ASSET_DIRECTORY_RELATIVE / ASSET_USD_FILE)
        self.assertNotEqual(result, result.resolve())

    def test_validates_assets_without_resolving_root_spelling(self):
        with patch.object(Path, "resolve", side_effect=AssertionError("Mapped paths must not be resolved")):
            self.assertEqual(validate_assets(self.root, self.metadata), self.usd)


class EnvironmentContractTests(unittest.TestCase):
    def test_matching_config_and_runtime_layout(self):
        validate_env_config(env_config(), manifest())
        validate_runtime_layout(runtime_env(), manifest())

    def test_config_rejects_changed_scale_offset_clip_period_and_mapping(self):
        for field, value in (
            ("scale", {"Knee_.*": 0.3}),
            ("use_default_offset", False),
            ("clip", {"Knee_.*": (-1.0, 0.0)}),
            ("asset_name", "another_robot"),
            ("joint_names", ["Knee_.*"]),
        ):
            cfg = env_config()
            setattr(cfg.actions.joint_pos, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_env_config(cfg, manifest())
        for field, value in (("dt", 0.01), ("dt", float("inf")), ("dt", -0.005)):
            cfg = env_config()
            setattr(cfg.sim, field, value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_env_config(cfg, manifest())
        cfg = env_config()
        cfg.decimation = True
        with self.assertRaises(ValueError):
            validate_env_config(cfg, manifest())

    def test_runtime_rejects_unavailable_or_drifted_layout(self):
        for change in ("robot", "action", "names", "dimensions", "missing_names", "missing_dimensions", "dt"):
            env = runtime_env()
            if change == "robot":
                env.scene["robot"].joint_names.reverse()
            elif change == "action":
                env.action_manager.get_term("joint_pos")._joint_names.reverse()
            elif change == "names":
                env.observation_manager.active_terms["policy"].reverse()
            elif change == "dimensions":
                env.observation_manager.group_obs_term_dim["policy"][-1] = (186,)
            elif change == "missing_names":
                del env.observation_manager.active_terms
            elif change == "missing_dimensions":
                del env.observation_manager.group_obs_term_dim
            else:
                env.step_dt = 0.025
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_runtime_layout(env, manifest())

    def test_flat_changes_only_five_proportions_and_is_idempotent(self):
        cfg = env_config()
        expected = deepcopy(cfg)
        terrains = expected.scene.terrain.terrain_generator.sub_terrains
        for name, terrain in terrains.items():
            terrain.proportion = 1.0 if name == "plane" else 0.0
        self.assertIs(configure_flat_terrain(cfg), cfg)
        self.assertEqual(cfg, expected)
        configure_flat_terrain(cfg)
        self.assertEqual(cfg, expected)

    def test_flat_rejects_other_generators_without_partial_changes(self):
        for change in ("extra", "missing", "no_proportion", "stairs"):
            cfg = env_config()
            terrains = cfg.scene.terrain.terrain_generator.sub_terrains
            if change == "extra":
                terrains["another"] = SimpleNamespace(proportion=0.2)
            elif change == "missing":
                del terrains["boxes"]
            elif change == "no_proportion":
                del terrains["boxes"].proportion
            else:
                cfg.scene.terrain.terrain_generator.sub_terrains = {"stair_course": SimpleNamespace(proportion=1.0)}
            before = deepcopy(cfg)
            with self.subTest(change=change), self.assertRaises(ValueError):
                configure_flat_terrain(cfg)
            self.assertEqual(cfg, before)

    def test_flat_supports_saved_dictionary_configuration(self):
        names = env_config().scene.terrain.terrain_generator.sub_terrains
        cfg = {"scene": {"terrain": {"terrain_generator": {"sub_terrains": {
            name: {"proportion": 0.2, "marker": name} for name in names
        }}}}}
        configure_flat_terrain(cfg)
        terrains = cfg["scene"]["terrain"]["terrain_generator"]["sub_terrains"]
        self.assertEqual({name: term["proportion"] for name, term in terrains.items()}, {
            name: 1.0 if name == "plane" else 0.0 for name in names
        })
        self.assertEqual({name: term["marker"] for name, term in terrains.items()}, {name: name for name in names})


if __name__ == "__main__":
    unittest.main()
