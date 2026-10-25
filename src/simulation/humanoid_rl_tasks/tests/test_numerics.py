"""CPU-only tests for the optional launcher guard; no Isaac imports."""

import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU Torch is supplied by the Isaac container")
class StrictGradientClippingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[2] / "humanoid_rl/humanoid_rl/numerics.py"
        spec = importlib.util.spec_from_file_location("numerics_under_test", source)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_finite_gradient_matches_default_and_restores(self):
        original = torch.nn.utils.clip_grad_norm_
        expected = torch.nn.Parameter(torch.zeros(2))
        actual = torch.nn.Parameter(torch.zeros(2))
        expected.grad = torch.tensor([3.0, 4.0])
        actual.grad = expected.grad.clone()
        norm = original([expected], 1.0)
        with self.module.strict_gradient_clipping():
            observed = torch.nn.utils.clip_grad_norm_([actual], 1.0)
        torch.testing.assert_close(observed, norm)
        torch.testing.assert_close(actual.grad, expected.grad)
        self.assertIs(torch.nn.utils.clip_grad_norm_, original)

    def test_nonfinite_raises_without_corrupting_gradient_and_restores(self):
        original = torch.nn.utils.clip_grad_norm_
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                parameter = torch.nn.Parameter(torch.zeros(1))
                parameter.grad = torch.tensor([value])
                with self.assertRaises(RuntimeError):
                    with self.module.strict_gradient_clipping():
                        # Even an explicit positional False cannot bypass the guard.
                        torch.nn.utils.clip_grad_norm_([parameter], 1.0, 2.0, False)
                self.assertIs(torch.nn.utils.clip_grad_norm_, original)
                if value == float("inf"):
                    self.assertTrue(torch.isinf(parameter.grad).all())
                else:
                    self.assertTrue(torch.isnan(parameter.grad).all())

    def test_disabled_leaves_original_unchanged(self):
        original = torch.nn.utils.clip_grad_norm_
        with self.module.strict_gradient_clipping(enabled=False):
            self.assertIs(torch.nn.utils.clip_grad_norm_, original)

    def test_nested_guards_restore_the_outer_guard(self):
        original = torch.nn.utils.clip_grad_norm_
        with self.module.strict_gradient_clipping():
            outer = torch.nn.utils.clip_grad_norm_
            with self.module.strict_gradient_clipping():
                self.assertIsNot(torch.nn.utils.clip_grad_norm_, outer)
            self.assertIs(torch.nn.utils.clip_grad_norm_, outer)
        self.assertIs(torch.nn.utils.clip_grad_norm_, original)


if __name__ == "__main__":
    unittest.main()
