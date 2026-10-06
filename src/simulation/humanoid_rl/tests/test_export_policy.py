"""CPU synthetic-checkpoint tests; no Isaac Sim, GPU, or trained weights."""

import copy
from contextlib import redirect_stderr
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
    import rsl_rl
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch and RSL-RL are supplied by the Isaac container")
class ExportPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        package_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(package_root))
        try:
            cls.exporter = importlib.import_module("humanoid_rl.scripts.export_policy")
            cls.contract = importlib.import_module("humanoid_rl.policy_artifacts")
        finally:
            sys.path.pop(0)
        torch.manual_seed(812)
        cls.state = {"log_std": torch.zeros(12)}
        for prefix, output in (("actor", 12), ("critic", 1)):
            dimensions = (235, 512, 256, 128, output)
            for index, input_dim, output_dim in zip((0, 2, 4, 6), dimensions, dimensions[1:]):
                layer = torch.nn.Linear(input_dim, output_dim)
                cls.state.update({
                    f"{prefix}.{index}.{name}": value.clone() for name, value in layer.state_dict().items()
                })
            normalizer = cls.exporter.EmpiricalNormalization(235)
            with torch.no_grad():
                normalizer.update(torch.randn(64, 235) * 2.0 + 0.7)
            cls.state.update({
                f"{prefix}_obs_normalizer.{name}": value.clone() for name, value in normalizer.state_dict().items()
            })

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pioneer_export_test_")
        self.root = Path(self.temporary.name)
        assets = self.root / self.contract.ASSET_DIRECTORY_RELATIVE
        assets.mkdir(parents=True)
        (assets / self.contract.ASSET_USD_FILE).write_text("#usda 1.0\n", encoding="utf-8")
        self.checkpoint, self.policy, self.metadata = (
            self.root / name for name in ("model_7.pt", "policy.pt", "policy.json")
        )
        self.saved = {"model_state_dict": copy.deepcopy(self.state), "iter": 7, "infos": None}
        torch.save(self.saved, self.checkpoint)

    def tearDown(self):
        self.temporary.cleanup()

    def export(self, **kwargs):
        return self.exporter.export_checkpoint(
            self.checkpoint, self.policy, self.metadata, self.contract.DEFAULT_TASK,
            repository_root=self.root, **kwargs,
        )

    def reject_changed_state(self, change, pattern):
        change(self.saved["model_state_dict"])
        torch.save(self.saved, self.checkpoint)
        with self.assertRaisesRegex(ValueError, pattern):
            self.export()
        self.assertFalse(self.policy.exists())
        self.assertFalse(self.metadata.exists())

    def test_roundtrip_preserves_normalizer_exactly_once_and_manifest(self):
        metadata = self.export()
        self.assertEqual(self.contract.load_metadata(self.metadata), metadata)
        self.assertEqual(metadata["checkpoint"], "model_7.pt")
        self.assertEqual(metadata["checkpoint_iteration"], 7)
        self.assertEqual(metadata["policy_sha256"], self.contract.sha256(self.policy))
        actor = self.exporter._build_mlp(self.state, "actor")
        normalizer = self.exporter._build_normalizer(self.state, "actor_obs_normalizer")
        observations = torch.randn(8, 235)
        with torch.inference_mode():
            expected = actor(normalizer(observations))
            twice = actor(normalizer(normalizer(observations)))
            actual = torch.jit.load(str(self.policy))(observations)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, twice))

    def test_missing_normalizer_is_rejected(self):
        self.reject_changed_state(lambda state: state.pop("actor_obs_normalizer._mean"), "missing")

    def test_extra_actor_key_is_not_silently_ignored(self):
        self.reject_changed_state(lambda state: state.update({"actor.unused": torch.ones(1)}), "unhandled")

    def test_wrong_architecture_is_rejected(self):
        self.reject_changed_state(
            lambda state: state.update({"actor.0.weight": torch.zeros(512, 234)}), "architecture"
        )

    def test_nonfinite_weight_is_rejected(self):
        self.reject_changed_state(lambda state: state["actor.0.weight"].fill_(float("nan")), "NaN")

    def test_bad_normalizer_shape_is_rejected(self):
        self.reject_changed_state(
            lambda state: state.update({"actor_obs_normalizer._mean": torch.zeros(1, 234)}), "shape/dtype"
        )

    def test_inconsistent_normalizer_is_rejected(self):
        self.reject_changed_state(lambda state: state["actor_obs_normalizer._std"].add_(1.0), "Inconsistent")

    def test_non_tensor_policy_state_is_rejected(self):
        self.reject_changed_state(lambda state: state.update({"log_std": None}), "not a tensor")

    def test_output_collision_requires_explicit_overwrite(self):
        self.policy.write_bytes(b"existing policy")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(self.policy.read_bytes(), b"existing policy")
        self.assertFalse(self.metadata.exists())
        self.export(overwrite=True)
        self.assertEqual(self.contract.sha256(self.policy), json.loads(self.metadata.read_text())["policy_sha256"])

    def test_metadata_collision_does_not_create_policy(self):
        self.metadata.write_text("existing metadata", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertFalse(self.policy.exists())
        self.assertEqual(self.metadata.read_text(), "existing metadata")

    def test_checkpoint_cannot_be_overwritten(self):
        before = self.contract.sha256(self.checkpoint)
        with self.assertRaisesRegex(ValueError, "source checkpoint"):
            self.exporter.export_checkpoint(
                self.checkpoint, self.checkpoint, self.metadata, self.contract.DEFAULT_TASK,
                overwrite=True, repository_root=self.root,
            )
        self.assertEqual(self.contract.sha256(self.checkpoint), before)

    def test_missing_assets_does_not_publish_outputs(self):
        (self.root / self.contract.ASSET_DIRECTORY_RELATIVE / self.contract.ASSET_USD_FILE).unlink()
        with self.assertRaises(FileNotFoundError):
            self.export()
        self.assertFalse(self.policy.exists())
        self.assertFalse(self.metadata.exists())

    def test_task_is_explicit_and_legacy_task_rejected(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.exporter._parse_args(["--checkpoint", "source.pt", "--output", "policy.pt"])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.exporter.export_checkpoint(
                self.checkpoint, self.policy, self.metadata,
                "Isaac-Locomotion-RoughNoStairs-PioneerHumanoid-Play-v0", repository_root=self.root,
            )


if __name__ == "__main__":
    unittest.main()
