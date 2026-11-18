"""Approximate human-style ROM for this model's straight-down neutral pose.

Independent joint limits approximate anatomy, not coupled shoulder mechanics.
Shoulder lift 170 deg and elbow flexion 150 deg are rounded starting values
informed by https://archive.cdc.gov/www_cdc_gov/ncbddd/jointrom/index.html .
Extension/adduction and symmetric axial rotations are tuning choices.
"""
import math

LEFT_LIMITS_DEG = ((-60, 170), (-30, 170), (-90, 90), (-150, 0), (-90, 90))


def joint_limits_deg(arm="left"):
    if arm == "left":
        return LEFT_LIMITS_DEG
    return tuple((-hi, -lo) for lo, hi in LEFT_LIMITS_DEG)


def limited_target(angle, sign, scale, bounds_deg):
    """Clamp an absolute zero-relative measurement; never modify its baseline."""
    lo, hi = (math.radians(v) for v in bounds_deg)
    return max(lo, min(hi, angle * sign * scale))
