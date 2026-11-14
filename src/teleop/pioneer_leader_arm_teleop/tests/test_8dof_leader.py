"""The 8-servo leader (A..H: G = wrist yaw, H = gripper) on the pioneer_bimanual_arm_8dof arms: layout, mapping, boards file, schema."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
SRC = HERE.parents[1]
sys.path[:0] = [str(HERE), str(SRC / "pioneer_humanoid"), str(SRC / "simulation" / "mujoco_scenes"),
                str(SRC / "robot_learning")]

import leader_mapping as lm  # noqa: E402
from servo_leader import LAYOUTS, calibration_entry, parse_signs  # noqa: E402

EIGHT = LAYOUTS["8dof"]
HOME = [0.0, 0.0, 0.0, math.radians(90), 0.0, 0.0, 0.0]
LIMITS = [(-1.57, 3.14), (-0.35, 2.79), (-2.09, 2.09), (-0.087, 2.27), (-1.92, 1.92), (-1.05, 1.05), (-1.57, 1.57)]


def leader_at(joints, grip=0.0, signs=(1.0,) * 8):
    """Leader readings (A..H) that map to `joints` and gripper closure `grip` (0 = open)."""
    arm = [q / s for q, s in zip(joints, signs)]
    return (*arm, -math.radians(grip * 41.5) / signs[-1])


class Layout8dofTests(unittest.TestCase):

    def test_layout_adds_wrist_yaw_servo_on_a_free_bus_id(self):
        self.assertEqual(tuple(EIGHT.servo_ids), tuple("ABCDEFGH"))
        self.assertEqual(EIGHT.arm_servos, tuple("ABCDEFG"))
        self.assertEqual(EIGHT.gripper_servo, "H")
        ids = list(EIGHT.servo_ids.values())
        self.assertEqual(len(ids), len(set(ids)))
        old = LAYOUTS["armv2"].servo_ids
        self.assertEqual({k: EIGHT.servo_ids[k] for k in "ABCDEF"}, {k: old[k] for k in "ABCDEF"})
        self.assertEqual(EIGHT.servo_ids["H"], old["G"])   # same gripper servo
        self.assertEqual(EIGHT.servo_ids["G"], 8)          # the new wrist-yaw servo

    def test_default_signs_parse_for_each_side(self):
        for side in lm.SIDES:
            self.assertEqual(len(parse_signs(lm.SIDE_SIGNS_8DOF[side], EIGHT.servo_ids)), 8)
            with self.assertRaises(ValueError):
                parse_signs(lm.SIDE_SIGNS[side], EIGHT.servo_ids)   # the 7-servo signs are refused

    def test_mapping_engages_at_home_then_follows_all_seven_joints_clamped(self):
        m = lm.LeaderMapping((1.0,) * 8, 1.0, HOME, LIMITS, layout=EIGHT)
        target, _ = m.update(leader_at([0.5] * 7))          # far from home: holds home
        self.assertFalse(m.engaged)
        self.assertEqual(target, HOME)
        m.update(leader_at(HOME))
        self.assertTrue(m.engaged)
        q = [0.3, 0.2, -0.4, 1.2, 0.5, -0.3, 0.7]
        target, grip = m.update(leader_at(q, grip=0.5))
        for a, b in zip(target, q):
            self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(grip, 0.5)
        target, _ = m.update(leader_at([9.0] * 7))          # past every limit: clamped
        self.assertEqual(target, [hi for _, hi in LIMITS])

    def test_lock_wrist_holds_both_wrist_joints(self):
        m = lm.LeaderMapping((1.0,) * 8, 1.0, HOME, LIMITS, locked=EIGHT.wrist_servos, layout=EIGHT)
        m.update(leader_at(HOME))
        target, _ = m.update(leader_at([0.1, 0.1, 0.1, 1.6, 0.1, 0.8, -0.8]))
        self.assertEqual(target[5:], [0.0, 0.0])
        self.assertAlmostEqual(target[0], 0.1)

    def test_wrong_sizes_are_refused(self):
        with self.assertRaises(ValueError):
            lm.LeaderMapping((1.0,) * 7, 1.0, HOME, LIMITS, layout=EIGHT)
        with self.assertRaises(ValueError):
            lm.LeaderMapping((1.0,) * 8, 1.0, HOME[:6], LIMITS[:6], layout=EIGHT)

    def test_boards_file_round_trip_with_the_new_servo(self):
        positions = {sid: 2000 + sid for sid in EIGHT.servo_ids.values()}
        data = {"right": {"enabled": True, "usb_serial": "ABC", "signs": lm.SIDE_SIGNS_8DOF["right"],
                          "servos": calibration_entry(positions, EIGHT.servo_ids)}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leader_arm_8dof.json"
            path.write_text(json.dumps(data))
            with mock.patch.object(lm, "port_of", return_value="/dev/null"):
                board = lm.load_leader_boards(path, ["right"], EIGHT)["right"]
                self.assertEqual(board.layout, EIGHT)
                self.assertEqual(board.zeros[8], 2008)
                with self.assertRaises(ValueError):        # a 7-servo file is not an 8dof leader
                    data["right"]["servos"].pop("G")
                    path.write_text(json.dumps(data))
                    lm.load_leader_boards(path, ["right"], EIGHT)

    def test_robot_defaults_per_target(self):
        import argparse
        for target, robot, file in (("mujoco", "8dof", "leader_arm_8dof.json"), ("isaac", "8dof", "leader_arm_8dof.json"),
                                    ("real", "armv2", "leader_arm.json")):
            args = argparse.Namespace(filter_alpha=0.35, robot=None, target=target, leader_arm=None)
            with mock.patch.object(lm, "load_leader_boards", return_value={}) as load:
                lm.check_leader_args(argparse.ArgumentParser(), args, ["right"])
            self.assertEqual(args.robot, robot)
            self.assertEqual(args.leader_arm.name, file)
            self.assertEqual(load.call_args.args[2], LAYOUTS[robot])
        for target, robot in (("real", "8dof"), ("isaac", "armv2")):   # the real arm is the old arm, Isaac the 8dof
            args = argparse.Namespace(filter_alpha=0.35, robot=robot, target=target, leader_arm=None)
            with self.assertRaises(SystemExit):
                lm.check_leader_args(argparse.ArgumentParser(), args, ["right"])


class Schema8dofTests(unittest.TestCase):

    def test_8dof_schema_records_seven_joints_and_gripper_per_arm(self):
        from humanoid_robot_learning.recording.schema import load_yaml, select_arms
        from humanoid_robot_learning.sim.sim_teleop_record import DEFAULT_SCHEMAS
        from pioneer_humanoid import arm_8dof_params as P
        cfg = select_arms(load_yaml(DEFAULT_SCHEMAS["8dof"]), ["left", "right"])
        self.assertEqual(cfg["joint_names"], P.LEFT_ARM_JOINTS + ["left_gripper"] + P.RIGHT_ARM_JOINTS + ["right_gripper"])
        self.assertEqual(cfg["robot_id"], "pioneer_8dof_bimanual")
        old = load_yaml(DEFAULT_SCHEMAS["armv2"])
        self.assertNotEqual(cfg["record"]["root"].split("_bimanual")[0], old["record"]["root"])


class Signs8dofTests(unittest.TestCase):

    def test_default_signs_follow_the_joint_axes_of_both_models(self):
        """SIDE_SIGNS_8DOF = SIDE_SIGNS x (8dof joint axis . old joint axis) for A..F, at the shared hanging zero."""
        try:
            import mujoco
            import numpy as np
            from humanoid_mujoco_scenes import make_model
            from pioneer_humanoid import arm_8dof_params as B, arm_params as A
        except ImportError as exc:
            self.skipTest(f"needs mujoco and the scenes: {exc}")
        axes = {}
        for robot, P in (("armv2", A), ("8dof", B)):
            m = make_model("peg_insert", cameras={}, robot=robot)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            axes[robot] = {s: [d.xaxis[m.joint(j).id].copy() for j in J]
                           for s, J in (("left", P.LEFT_ARM_JOINTS), ("right", P.RIGHT_ARM_JOINTS))}
        for side in lm.SIDES:
            old = parse_signs(lm.SIDE_SIGNS[side])
            new = parse_signs(lm.SIDE_SIGNS_8DOF[side], EIGHT.servo_ids)
            flips = [float(np.sign(a @ b)) for a, b in zip(axes["armv2"][side], axes["8dof"][side][:6])]
            self.assertEqual(list(new[:6]), [o * f for o, f in zip(old[:6], flips)], side)
            self.assertEqual(new[-1], old[-1], side)                           # gripper unchanged


if __name__ == "__main__":
    unittest.main()
