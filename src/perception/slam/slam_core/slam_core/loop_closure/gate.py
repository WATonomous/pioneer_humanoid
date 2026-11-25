"""Plausibility gate: does a loop agree with what odometry says, given how much drift could have built up?

Odometry gives an estimate of the pose between the two keyframes along with a covariance that grows with the
distance travelled. A loop that disagrees by far more than that covariance allows (plus the loop's own
uncertainty) is rejected, however good its appearance and inlier count look. This is the check that stops
identical-looking hallways from being accepted.

Limits: heavy unmodelled drift can make a true loop look implausible, so the covariance can be inflated
(`inflation`), and the optimizer should still use a robust loss as a second line of defence.
"""
import numpy as np

from .se3 import mahalanobis2, pose_error

# 99.9th percentile of chi-squared with 6 degrees of freedom
CHI2_6_999 = 22.46


# Returns (d2, ok). d2 is the squared Mahalanobis distance between the loop and odometry's prediction
def plausibility(graph, old_id, new_id, T_loop, S_loop, inflation=1.0, chi2=CHI2_6_999):
    rel = graph.relative(old_id, new_id)
    if rel is None:
        return float("inf"), False
    T_odom, S_odom = rel
    r = pose_error(T_odom, T_loop)
    d2 = mahalanobis2(r, inflation * S_odom + S_loop)
    return d2, d2 <= chi2
