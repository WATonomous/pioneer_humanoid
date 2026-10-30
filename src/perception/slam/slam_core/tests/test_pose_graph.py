"""Small BE-3 test using three fake keyframes."""

import unittest

import numpy as np

from slam_core.backend import PoseGraphBackend
from slam_core.types import Keyframe
from slam_core.types.pose import covariance_from_sigmas, make_T


def fake_keyframe(index):
    return Keyframe(
        id=index,
        prev_id=-1 if index == 0 else index - 1,
        stamp=float(index),
        T_prev_kf=make_T(t=None if index == 0 else [1.0, 0.0, 0.0]),
        covariance=covariance_from_sigmas(0.01, 0.02),
        T_odom_kf=make_T(t=[0.8 * index, 0.0, 0.0]),
    )


class PoseGraphTest(unittest.TestCase):
    def test_fake_keyframes(self):
        backend = PoseGraphBackend()

        for index in range(3):
            backend.add_keyframe(fake_keyframe(index))

        poses = backend.optimized_poses()
        positions = [poses[index].translation() for index in range(3)]
        np.testing.assert_allclose(
            positions,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            atol=1e-5,
        )
        print(positions)


if __name__ == "__main__":
    unittest.main()
