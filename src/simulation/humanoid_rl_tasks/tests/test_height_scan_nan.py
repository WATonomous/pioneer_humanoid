"""CPU-only test for the NaN-safe height-scan observation term."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "humanoid_rl_tasks/locomotion/mdp/observations.py"
)


@unittest.skipIf(torch is None, "CPU Torch is supplied by the Isaac container")
class HeightScanNaNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Stub Isaac Lab so this test runs without Kit/pxr, like the other
        # CPU-only tests in this directory.
        names = ["isaaclab", "isaaclab.envs", "isaaclab.envs.mdp", "isaaclab.envs.mdp.observations"]
        stubs = {name: ModuleType(name) for name in names}
        stubs["isaaclab"].envs = stubs["isaaclab.envs"]
        stubs["isaaclab.envs"].mdp = stubs["isaaclab.envs.mdp"]
        stubs["isaaclab.envs.mdp"].observations = stubs["isaaclab.envs.mdp.observations"]
        stubs["isaaclab.envs.mdp.observations"].height_scan = lambda *args, **kwargs: None
        saved = {name: sys.modules.get(name) for name in names}
        sys.modules.update(stubs)
        try:
            spec = importlib.util.spec_from_file_location("observations_under_test", SOURCE)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        finally:
            for name, previous in saved.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

    def test_nan_becomes_zero_and_other_values_are_unchanged(self):
        self.module._isaac_height_scan = lambda env, sensor_cfg, offset: torch.tensor(
            [[[float("nan"), 0.4, -1.2, float("inf")]]]
        )

        scan = self.module.height_scan(SimpleNamespace(), SimpleNamespace(), 0.5)

        torch.testing.assert_close(
            scan,
            torch.tensor([[[0.0, 0.4, -1.2, float("inf")]]]),
            equal_nan=False,
        )


if __name__ == "__main__":
    unittest.main()
