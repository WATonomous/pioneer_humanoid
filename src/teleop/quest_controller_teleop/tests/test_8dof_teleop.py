"""Quest teleop on pioneer_bimanual_arm_8dof (the rig with the 7-DoF + 1-DoF-gripper arms): limits, rules, IK, driver."""
from pathlib import Path
import math
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[2]/"src/pioneer_humanoid")]
try:
    import mink  # noqa: F401
    import mujoco  # noqa: F401
except ImportError:  # pragma: no cover
    mink = None

from pioneer_humanoid import arm_8dof_params as P

CHART_DEG = {"shoulder_pitch": (-90, 180), "shoulder_roll": (-20, 160), "shoulder_yaw": (-120, 120),
             "elbow": (-5, 130), "forearm_roll": (-110, 110), "wrist_pitch": (-60, 60), "wrist_yaw": (-90, 90)}


class Arm8dofModelTests(unittest.TestCase):
    def test_urdf_limits_are_the_joint_chart_for_both_arms(self):
        limits = P.urdf_limits()
        for side in ("left", "right"):
            for joint, (lo, hi) in CHART_DEG.items():
                np.testing.assert_allclose(np.degrees(limits[f"{side}_{joint}"]), (lo, hi), atol=1e-4)
            self.assertEqual(limits[f"{side}_gripper"], (0.0, P.GRIPPER_STROKE))

    def test_seven_arm_joints_and_one_gripper_per_side(self):
        self.assertEqual(len(P.LEFT_ARM_JOINTS), 7)
        self.assertEqual(len(P.RIGHT_ARM_JOINTS), 7)
        self.assertEqual(P.LEFT_GRIPPER_JOINTS, ["left_gripper"])


@unittest.skipIf(mink is None, "mink / mujoco not installed")
class Arm8dofTeleopTests(unittest.TestCase):
    def driver(self):
        from scene_driver import ControllerSceneDriver
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return ControllerSceneDriver({"left": P.LEFT_ARM_JOINTS, "right": P.RIGHT_ARM_JOINTS}, 1.5, .9,
                                     Path(folder.name)/"pivot.json")

    def test_limits_are_urdf_plus_elbow_rule_no_hand_picked_ranges(self):
        d = self.driver()
        for s in d.solvers.values():
            expected = s.physical_limits.copy()
            expected[3, 0] = math.radians(10.)        # elbow >= 10 deg bent
            np.testing.assert_allclose(s.limits, expected)
            self.assertIsNone(s.keepout)               # no torso on the stand: keep-out opt-in (--body-keepout)

    def test_kinematics_mirror_between_arms(self):
        d = self.driver()
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.uniform(*d.solvers["left"].limits.T)
            pl, _, _, _ = d.solvers["left"].kinematics(q)
            pr, _, _, _ = d.solvers["right"].kinematics(q)
            np.testing.assert_allclose(pr, pl * [1, -1, 1], atol=1e-9)

    def test_tracks_a_reachable_pose_within_the_rules(self):
        d = self.driver()
        s = d.solvers["left"]
        goal = s.rest + np.radians([30, 15, 20, 10, 30, 20, -20])
        p, r, _, _ = s.kinematics(goal)
        t = SimpleNamespace(active=True, position=p, rotation=r, saturation=0.)
        for _ in range(500):
            q = s.solve(None, t, .01)
            self.assertTrue(np.all(q >= s.limits[:, 0] - 1e-9) and np.all(q <= s.limits[:, 1] + 1e-9))
        self.assertLess(np.linalg.norm(s.kinematics(q)[0] - p), .005)

    def test_sim_feedforward_removes_gravity_sag_and_tracking_lag(self):
        sys.path[:0] = [str(HERE/"mujoco"), str(HERE.parents[2]/"src/simulation/mujoco_scenes")]
        import mujoco
        import mujoco_sim as ms
        from humanoid_mujoco_scenes import make_model
        from pioneer_humanoid.mujoco_bimanual_arm_8dof import set_home
        model = make_model("bimanual_sorting", robot="8dof")
        errors = {}
        for feedforward in (False, True):
            data = mujoco.MjData(model)
            set_home(model, data)
            mujoco.mj_forward(model, data)
            ids = ms.arm_ids(model, "right")
            q0, worst = data.qpos[ids.qpos].copy(), 0.
            for k in range(100):   # 1 s: the shoulder sweeps at 0.5 rad/s
                command = q0 + np.array([.5, 0, 0, 0, 0, 0, 0]) * k * .01
                velocity = np.array([.5, 0, 0, 0, 0, 0, 0])
                data.ctrl[ids.arm_acts] = ms.arm_ctrl(model, data, ids, command, velocity) if feedforward else command
                mujoco.mj_step(model, data, nstep=round(.01 / model.opt.timestep))
                worst = max(worst, np.abs(data.qpos[ids.qpos] - command).max()) if k > 50 else worst
            errors[feedforward] = worst
        self.assertGreater(errors[False], .03)     # plain PD trails ~kv/kp x speed
        self.assertLess(errors[True], .01)


if __name__ == "__main__":
    unittest.main()
