"""Wall light-switch scenes (toggle, rocker, lever; left and right arm): limits, start pose, success, layout jitter."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "pioneer_humanoid")]

import mujoco  # noqa: E402
from humanoid_mujoco_scenes import make_model, scene_reset, scene_step, scene_success  # noqa: E402
from pioneer_humanoid.mujoco_bimanual_arm_8dof import set_home  # noqa: E402

# scene -> (hinge joint, travel each way in rad, "on" threshold in rad)
SCENES = {
    "toggle_switch": ("switch", 0.5, 0.35),
    "toggle_switch_right": ("switch", 0.5, 0.35),
    "rocker_switch": ("switch", 0.15, 0.1),
    "rocker_switch_right": ("switch", 0.15, 0.1),
}


def start(name, seed=0):
    model = make_model(name)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    set_home(model, data)
    scene_reset(name)(model, data, np.random.default_rng(seed))
    mujoco.mj_forward(model, data)
    return model, data


class LightSwitchSceneTests(unittest.TestCase):
    def test_hinge_range_is_in_radians(self):
        # The scene compiler is in radians: a range written in degrees reads as +-28 rad and the lever flops around.
        for name, (joint, travel, _) in SCENES.items():
            with self.subTest(scene=name):
                model, _ = start(name)
                np.testing.assert_allclose(model.joint(joint).range, (-travel, travel), atol=1e-9)

    def test_starts_off_with_room_around_the_home_pose(self):
        for name, (joint, travel, _) in SCENES.items():
            with self.subTest(scene=name):
                model, data = start(name)
                mujoco.mj_step(model, data, nstep=50)
                self.assertEqual(data.ncon, 0, "the fingers must not start touching the wall, plate or lever")
                self.assertLess(float(data.qpos[model.joint(joint).qposadr[0]]), -travel * 0.5, "starts on the OFF side")
                self.assertFalse(scene_success(name)(model, data))

    def test_success_is_the_switch_past_its_threshold(self):
        for name, (joint, travel, threshold) in SCENES.items():
            with self.subTest(scene=name):
                model, data = start(name)
                adr = model.joint(joint).qposadr[0]
                data.qpos[adr] = threshold - 0.02
                mujoco.mj_forward(model, data)
                self.assertFalse(scene_success(name)(model, data))
                data.qpos[adr] = threshold + 0.02
                mujoco.mj_forward(model, data)
                self.assertTrue(scene_success(name)(model, data))

    def test_detent_holds_the_switch_at_the_end_it_is_nearest(self):
        for name, (joint, travel, _) in SCENES.items():
            with self.subTest(scene=name):
                model, data = start(name)
                hook, adr = scene_step(name), model.joint(joint).qposadr[0]
                data.qpos[adr] = travel * 0.6
                mujoco.mj_forward(model, data)
                for _ in range(300):
                    hook(model, data)
                    mujoco.mj_step(model, data)
                self.assertGreater(float(data.qpos[adr]), travel * 0.9, "it snaps to the ON end and stays")

    def test_each_reset_moves_the_plate_a_little_and_the_same_seed_repeats(self):
        plate = []
        for seed in (0, 1, 0):
            model, _ = start("toggle_switch", seed)
            plate.append(model.body_pos[model.body("plate").id].copy())
        self.assertFalse(np.allclose(plate[0], plate[1]))
        np.testing.assert_allclose(plate[0], plate[2])
        self.assertLess(np.abs(plate[0][1:] - plate[1][1:]).max(), 0.07)

    def test_right_scenes_are_the_mirror_of_the_left_ones(self):
        left, _ = start("toggle_switch")
        right, _ = start("toggle_switch_right")
        a, b = left.body_pos[left.body("plate").id], right.body_pos[right.body("plate").id]
        self.assertAlmostEqual(a[0], b[0])
        self.assertGreater(a[1], 0)
        self.assertLess(b[1], 0)


if __name__ == "__main__":
    unittest.main()
