import unittest
import numpy as np
from humanoid_robot_learning.sim.cargo_randomization import (
    sample_reset, tray_expert_yaw, WIDE_SUPPORT_CENTER, WIDE_SUPPORT_SIZE)
from humanoid_robot_learning.sim.cargo_finetune import preserve_normalization


class RandomizationTests(unittest.TestCase):
    def test_baseline_preserves_reset_and_expert_pace_rng(self):
        reference = np.random.default_rng(17)
        xy = [reference.uniform(-.005, .005), reference.uniform(-.005, .005)]
        yaw = reference.uniform(-.025, .025)
        rng, params = sample_reset(17)
        np.testing.assert_array_equal(params["pickup_offset_xy_m"], xy)
        self.assertEqual(params["yaw_rad"], yaw)
        self.assertEqual(rng.uniform(.92, 1.08), reference.uniform(.92, 1.08))
        self.assertEqual(params["arm_joint_offset_rad"], [0.] * 6)

    def test_varied_is_reproducible_bounded_and_changes_all_requested_factors(self):
        samples = [sample_reset(i, "varied")[1] for i in range(100)]
        self.assertEqual(samples[31], sample_reset(31, "varied")[1])
        xy = np.array([x["pickup_offset_xy_m"] for x in samples])
        joints = np.array([x["arm_joint_offset_rad"] for x in samples])
        rgb = np.array([x["body_color_rgb"] for x in samples])
        self.assertTrue((np.abs(xy) <= .015).all() and (np.abs(xy) > .005).any())
        self.assertTrue((np.abs(joints) <= .06).all() and (joints.std(0) > .02).all())
        self.assertTrue(((rgb >= 0) & (rgb <= 1)).all())
        self.assertGreater(len(np.unique(rgb, axis=0)), 50)

    def test_tray_moves_visibly_while_pickup_pose_stays_fixed(self):
        samples = [sample_reset(i, "tray")[1] for i in range(1000)]
        self.assertEqual(samples[17], sample_reset(17, "tray")[1])
        targets = np.array([s["target_xy_m"] for s in samples])
        self.assertGreater(np.ptp(targets[:, 0]), .195)
        self.assertGreater(np.ptp(targets[:, 1]), .039)
        # A turned bag still fits entirely within the original unrotated rim.
        yaw = max(s["expert_place_yaw_rad"] for s in samples)
        self.assertEqual({s["expert_place_yaw_rad"] for s in samples}, {0., yaw})
        extent = [.2032*np.cos(yaw)+.11451235*np.sin(yaw),
                  .2032*np.sin(yaw)+.11451235*np.cos(yaw)]
        self.assertTrue((np.array(extent) < [.28, .20]).all())
        self.assertTrue((targets >= [.25, .49]).all())
        self.assertTrue((targets <= [.45, .53]).all())
        for sample in samples:
            self.assertEqual(sample["pickup_offset_xy_m"], [0., 0.])
            self.assertEqual(sample["yaw_rad"], 0.)
            # Actual near rim clears the fixed full-size bag by >1 cm.
            self.assertGreater(sample["target_xy_m"][1] - .22, .14 + .11451235 + .009)

    def test_tray_placement_and_carry_height_within_joint_limited_workspace(self):
        from scipy.spatial.transform import Rotation
        from humanoid_robot_learning.sim.cargo_expert import ArmKinematics
        kin = ArmKinematics()
        for x in np.linspace(.25, .45, 5):
            for y in (.49, .53):
                q = np.array([0., 0., 0., -2.1, 0., 0.])
                for z in (1.1098, 1.2298, 1.2348):
                    rotation = Rotation.from_euler("z", tray_expert_yaw([x, y]))
                    point = np.array([x, y, z-.363-1.1997]) + rotation.apply([0, .08002, .363])
                    q, error, angle = kin.ik(point, rotation, q)
                    self.assertLess(error, .013, (x, y, z))
                    self.assertLess(angle, .11, (x, y, z))
                    self.assertTrue(((q >= kin.lo) & (q <= kin.hi)).all())

    def test_finetune_retains_saved_stats_without_mutating_caller_overrides(self):
        kwargs = {"dataset_stats": "new stats", "pretrained_path": "v1",
                  "preprocessor_overrides": {"normalizer_processor": {"stats": "new"},
                                             "device_processor": {"device": "cuda"}},
                  "postprocessor_overrides": {"unnormalizer_processor": {"stats": "new"}}}
        result = preserve_normalization(kwargs)
        self.assertNotIn("normalizer_processor", result["preprocessor_overrides"])
        self.assertNotIn("unnormalizer_processor", result["postprocessor_overrides"])
        self.assertNotIn("dataset_stats", result)
        self.assertEqual(result["preprocessor_overrides"]["device_processor"], {"device": "cuda"})
        self.assertIn("normalizer_processor", kwargs["preprocessor_overrides"])

    def test_wide_samples_cover_visible_range_and_fit_the_pickup_surface(self):
        samples = [sample_reset(i, "wide")[1] for i in range(1000)]
        positions = np.array([p["pickup_offset_xy_m"] for p in samples]) + [.5, .14]
        self.assertTrue((positions >= [.475, .115]).all())
        self.assertTrue((positions <= [.525, .15]).all())
        self.assertGreater(np.ptp(positions[:, 0]), .047)
        self.assertEqual(samples[9], sample_reset(9, "wide")[1])
        yaw = np.array([p["yaw_rad"] for p in samples])
        extent = np.stack((.2032*np.cos(yaw) + .11451235*np.abs(np.sin(yaw)),
                           .11451235*np.cos(yaw) + .2032*np.abs(np.sin(yaw))), axis=1)
        self.assertTrue((np.abs(positions-WIDE_SUPPORT_CENTER) + extent
                         <= np.array(WIDE_SUPPORT_SIZE)/2 - .005).all())
        self.assertTrue((positions[:, 1] + extent[:, 1] < .285).all())  # tray rim clearance

    def test_wide_boundary_approach_grasp_and_lift_are_inside_joint_limited_reach(self):
        from scipy.spatial.transform import Rotation
        from humanoid_robot_learning.sim.cargo_expert import ArmKinematics
        kin = ArmKinematics()
        for x in np.linspace(.475, .525, 5):
            for y in np.linspace(.115, .15, 3):
                for yaw in (-.04, 0, .04):
                    q = np.array([0., 0., 0., -2.1, 0., 0.])
                    rotation = Rotation.from_euler("z", yaw)
                    grasp = np.array([x, y, .7468]) + rotation.apply([0., .08002, .363]) - [0, 0, 1.1997]
                    for height in (.105, 0, .12):
                        q, error, angle = kin.ik(grasp + [0, 0, height], rotation, q)
                        self.assertLess(error, .003 if height == 0 else .008, (x, y, yaw, height))
                        self.assertLess(angle, .07, (x, y, yaw, height))
                        self.assertTrue(((q >= kin.lo) & (q <= kin.hi)).all())


if __name__ == "__main__":
    unittest.main()
