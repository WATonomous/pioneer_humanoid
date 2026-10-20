"""Placement checks use real contacts and dynamics, including optional stacks."""
import unittest

import mujoco
import numpy as np

from humanoid_mujoco_scenes import make_model, scene_success
from humanoid_mujoco_scenes.bimanual_sorting import scene as sorting
from pioneer_humanoid.mujoco_bimanual_arm import set_home


class SortingPlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_model("bimanual_sorting")

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        set_home(self.model, self.data)
        for color, (x, y) in sorting.TRAYS.items():
            for i in range(2):
                self.place(f"{color}_{i}", (x + (-0.035 if i == 0 else 0.035), y,
                           sorting.TABLE_TOP_Z + 0.006 + sorting.BLOCK_HALF))
        mujoco.mj_forward(self.model, self.data)

    def place(self, name, position, yaw=0):
        adr = self.model.joint(name).qposadr[0]
        self.data.qpos[adr:adr + 3] = position
        self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def advance(self, seconds):
        for _ in range(round(seconds / (10 * self.model.opt.timestep))):
            sorting.step(self.model, self.data)
            mujoco.mj_step(self.model, self.data, nstep=10)
        mujoco.mj_forward(self.model, self.data)

    def test_separate_blocks_must_settle_before_success(self):
        self.advance(0.1)
        self.assertFalse(scene_success("bimanual_sorting")(self.model, self.data))
        self.advance(0.7)
        self.assertEqual(sorting.progress(self.model, self.data)[0], 4)

    def test_stacks_are_optional_and_accepted(self):
        for color, (x, y) in sorting.TRAYS.items():
            for i in range(2):
                self.place(f"{color}_{i}", (x, y, sorting.TABLE_TOP_Z + 0.006
                           + sorting.BLOCK_HALF * (1 + 2 * i)), yaw=np.pi / 8)
        mujoco.mj_forward(self.model, self.data)
        self.advance(0.8)
        self.assertEqual(sorting.progress(self.model, self.data)[0], 4)

    def test_airborne_stationary_block_is_not_supported(self):
        self.advance(0.8)
        adr = self.model.joint("blue_1").qposadr[0]
        self.data.qpos[adr + 2] += 0.009
        mujoco.mj_forward(self.model, self.data)
        self.assertNotIn("blue_1", sorting._placed_blocks(self.model, self.data))
        self.assertFalse(scene_success("bimanual_sorting")(self.model, self.data))

    def test_upward_moving_block_is_not_placed(self):
        self.advance(0.8)
        joint = self.model.joint("blue_1")
        self.data.qpos[joint.qposadr[0] + 2] += 0.009
        self.data.qvel[joint.dofadr[0] + 2] = 2.0
        mujoco.mj_forward(self.model, self.data)
        self.assertFalse(scene_success("bimanual_sorting")(self.model, self.data))

    def test_rotated_corners_must_clear_inner_walls(self):
        x, y = sorting.TRAYS["red"]
        self.place("red_0", (x + 0.06, y, sorting.TABLE_TOP_Z + 0.026), yaw=np.pi / 4)
        mujoco.mj_forward(self.model, self.data)
        self.assertNotIn("red_0", sorting._placed_blocks(self.model, self.data))

    def test_wrong_tray_does_not_count(self):
        x, y = sorting.TRAYS["blue"]
        self.place("red_0", (x, y, sorting.TABLE_TOP_Z + 0.026))
        mujoco.mj_forward(self.model, self.data)
        self.advance(0.8)
        self.assertNotIn("red_0", sorting._placed_blocks(self.model, self.data))

    def test_block_touching_gripper_is_not_released(self):
        self.advance(0.8)
        body = self.model.body("link7l").id
        geom = next(g for g in range(self.model.ngeom)
                    if self.model.geom_bodyid[g] == body and self.model.geom_contype[g])
        old_position = self.model.geom_pos[geom].copy()
        old_size = self.model.geom_size[geom].copy()
        try:
            # Move a finger collision box beside a supported cube to create a real robot contact.
            target = self.data.xpos[self.model.body("red_0").id] + [0.025, 0, 0]
            rotation = self.data.xmat[body].reshape(3, 3)
            self.model.geom_pos[geom] = rotation.T @ (target - self.data.xpos[body])
            self.model.geom_size[geom] = [0.01, 0.01, 0.01]
            mujoco.mj_forward(self.model, self.data)
            self.assertNotIn("red_0", sorting._placed_blocks(self.model, self.data))
            self.assertFalse(scene_success("bimanual_sorting")(self.model, self.data))
        finally:
            self.model.geom_pos[geom] = old_position
            self.model.geom_size[geom] = old_size

    def test_new_placement_restarts_settling_timer(self):
        self.advance(0.8)
        joint = self.model.joint("red_0")
        original = self.data.qpos[joint.qposadr[0]:joint.qposadr[0] + 3].copy()
        self.data.qpos[joint.qposadr[0] + 2] += 0.01
        mujoco.mj_forward(self.model, self.data)
        sorting.step(self.model, self.data)
        self.place("red_0", original)
        mujoco.mj_forward(self.model, self.data)
        self.assertFalse(scene_success("bimanual_sorting")(self.model, self.data))
        self.advance(0.5)
        self.assertTrue(scene_success("bimanual_sorting")(self.model, self.data))


if __name__ == "__main__":
    unittest.main()
