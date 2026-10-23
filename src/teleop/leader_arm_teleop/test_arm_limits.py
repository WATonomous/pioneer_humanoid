import math
import unittest

from arm_limits import joint_limits_deg, limited_target


class LimitsTest(unittest.TestCase):
    def test_overshoot_does_not_shift_zero(self):
        for bounds in joint_limits_deg():
            lo, hi = bounds
            for sign in (1, -1):
                for endpoint in (lo-5, hi+5, lo-45, hi+45):
                    for _ in range(3):
                        result = limited_target(math.radians(endpoint/sign), sign, 1, bounds)
                        self.assertAlmostEqual(math.degrees(result), max(lo, min(hi, endpoint)))
                        self.assertEqual(limited_target(0, sign, 1, bounds), 0)

    def test_user_example(self):
        self.assertEqual(
            [round(math.degrees(limited_target(math.radians(a), 1, 1, (-60, 170))))
             for a in (0, 170, 175, 174, 170, 100, 5, 0)],
            [0, 170, 170, 170, 170, 100, 5, 0],
        )

    def test_d_remains_inverted(self):
        self.assertAlmostEqual(math.degrees(limited_target(math.radians(155), -1, 1, (-150, 0))), -150)
        self.assertEqual(limited_target(0, -1, 1, (-150, 0)), 0)


if __name__ == '__main__':
    unittest.main()
