"""make_model builds each scene into the rig's spec: no empty-prefix attach warning, and the spec's radians hold."""
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "pioneer_humanoid")]

import mujoco  # noqa: E402
from humanoid_mujoco_scenes import list_scenes, make_model  # noqa: E402


class MakeModelTests(unittest.TestCase):
    def test_every_scene_builds_without_mujoco_warnings(self):
        seen = []
        mujoco.set_mju_user_warning(seen.append)
        self.addCleanup(mujoco.set_mju_user_warning, None)
        for robot in ("8dof", "armv2"):
            for name in list_scenes():
                with self.subTest(robot=robot, scene=name), warnings.catch_warnings(record=True) as raised:
                    warnings.simplefilter("always")
                    make_model(name, robot=robot)
                    self.assertEqual([str(w.message) for w in raised], [])
        self.assertEqual(seen, [])

    def test_joint_limits_are_the_urdf_radians_in_every_scene(self):
        # Compiling the rig's spec in degrees (MuJoCo's default) would read these radians as degrees.
        from pioneer_humanoid.arm_8dof_params import urdf_limits
        limits = urdf_limits()
        for name in list_scenes():
            model = make_model(name)
            for joint, (lo, hi) in limits.items():
                with self.subTest(scene=name, joint=joint):
                    np.testing.assert_allclose(model.joint(joint).range, (lo, hi), atol=1e-9)

    def test_zip_tie_strap_springs_rest_straight(self):
        # springref is given in radians: each bend spring relaxes at minus its threaded bend (a straight strap).
        from humanoid_mujoco_scenes.zip_tie.scene import _PATH, _wrap
        model = make_model("zip_tie")
        for i in range(1, len(_PATH)):
            joint = model.joint(f"strap{i}_bend")
            self.assertAlmostEqual(model.qpos_spring[joint.qposadr[0]], -_wrap(_PATH[i][2] - _PATH[i - 1][2]), places=9)


    def test_gripper_holds_like_a_position_servo(self):
        # AK40-10 gripper (arm_8dof_params._GRIPPER): settles without ringing and a 10 N push barely opens it.
        from pioneer_humanoid.arm_8dof_params import GRIPPER_STROKE
        from pioneer_humanoid.mujoco_bimanual_arm_8dof import set_home
        model = make_model("bare")
        data = mujoco.MjData(model)
        set_home(model, data)
        joint, act = model.joint("right_gripper"), model.actuator("right_gripper").id
        data.ctrl[act] = GRIPPER_STROKE / 2
        mujoco.mj_step(model, data, nstep=300)
        self.assertLess(abs(data.qpos[joint.qposadr[0]] - GRIPPER_STROKE / 2), 2e-4)
        data.qfrc_applied[joint.dofadr[0]] = -10.0
        mujoco.mj_step(model, data, nstep=500)
        self.assertLess(GRIPPER_STROKE / 2 - data.qpos[joint.qposadr[0]], .003)   # was 21 mm at 400 N/m


if __name__ == "__main__":
    unittest.main()
