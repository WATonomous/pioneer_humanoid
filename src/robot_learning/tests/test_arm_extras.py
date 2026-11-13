"""Extra dataset fields on the 8dof rig (hand pose, joint velocity / effort), modality.json, and appending to a dataset
made before the extras existed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SRC / "robot_learning"), str(SRC / "pioneer_humanoid"), str(SRC / "teleop/quest_controller_teleop")]

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None
try:
    import lerobot  # noqa: F401
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    lerobot = None

from humanoid_robot_learning.recording.modality import build_modality, write_modality  # noqa: E402

LEFT = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_forearm_roll",
        "left_wrist_pitch", "left_wrist_yaw", "left_gripper"]
RIGHT = [n.replace("left_", "right_") for n in LEFT]


def arm_model():
    from pioneer_humanoid.mujoco_bimanual_arm_8dof import arm_spec, set_home
    model = arm_spec().compile()
    data = mujoco.MjData(model)
    set_home(model, data)
    mujoco.mj_forward(model, data)
    return model, data


@unittest.skipIf(mujoco is None, "mujoco not installed")
class ArmExtrasTests(unittest.TestCase):
    def setUp(self):
        from humanoid_robot_learning.sim.mujoco_arm_extras import ArmExtras, arm_8dof_spec
        self.model, self.data = arm_model()
        self.arms = [arm_8dof_spec(s) for s in ("left", "right")]
        self.extras = ArmExtras(self.model, self.arms)

    def commands_at(self, data):
        return {a.side: data.qpos[[self.model.joint(j).qposadr[0] for j in a.joints]].copy() for a in self.arms}

    def test_feature_sizes_and_names(self):
        f = self.extras.features
        self.assertEqual(len(f["observation.ee_pose"]), 18)
        self.assertEqual(f["observation.ee_pose"][:3], ["left_hand_x", "left_hand_y", "left_hand_z"])
        self.assertEqual(f["observation.velocity"], LEFT + RIGHT)
        self.assertEqual(f["observation.effort"], LEFT + RIGHT)
        values = self.extras.values(self.data, self.commands_at(self.data))
        for name, comps in f.items():
            self.assertEqual(values[name].shape, (len(comps),), name)
            self.assertEqual(values[name].dtype, np.float32)
            self.assertTrue(np.isfinite(values[name]).all(), name)

    def test_hand_pose_matches_the_teleop_kinematics(self):
        from control import PioneerKinematics
        rng = np.random.default_rng(0)
        for i, arm in enumerate(self.arms):
            kin = PioneerKinematics(arm.joints)
            for _ in range(5):
                q = rng.uniform(*kin.physical_limits.T)
                self.data.qpos[[self.model.joint(j).qposadr[0] for j in arm.joints]] = q
                mujoco.mj_forward(self.model, self.data)
                pose = self.extras.values(self.data, self.commands_at(self.data))["observation.ee_pose"][9 * i:9 * i + 9]
                _, r, _, _ = kin.kinematics(q)
                np.testing.assert_allclose(pose[3:6], r[:, 0], atol=1e-5)
                np.testing.assert_allclose(pose[6:9], r[:, 1], atol=1e-5)
                self.assertAlmostEqual(float(np.dot(pose[3:6], pose[6:9])), 0., places=5)

    def test_commanded_pose_is_fk_of_the_command_not_the_measured_joints(self):
        commands = self.commands_at(self.data)
        v = self.extras.values(self.data, commands)
        np.testing.assert_allclose(v["action.ee_pose"], v["observation.ee_pose"], atol=1e-6)
        commands["left"][3] -= 0.5                     # left elbow commanded 0.5 rad straighter
        v = self.extras.values(self.data, commands)
        self.assertGreater(np.linalg.norm(v["action.ee_pose"][:3] - v["observation.ee_pose"][:3]), 0.02)
        np.testing.assert_allclose(v["action.ee_pose"][9:], v["observation.ee_pose"][9:], atol=1e-6)   # right arm

    def test_effort_holds_gravity_and_velocity_follows_motion(self):
        d = self.data
        d.ctrl[:] = 0.
        for a in self.arms:
            for j in a.joints:
                d.ctrl[self.model.actuator(j).id] = d.qpos[self.model.joint(j).qposadr[0]]
        mujoco.mj_step(self.model, d, nstep=200)        # settle holding home against gravity
        mujoco.mj_forward(self.model, d)
        v = self.extras.values(d, self.commands_at(d))
        self.assertGreater(abs(v["observation.effort"][0]), 0.1)   # left shoulder pitch holds the arm up
        d.qvel[self.model.joint("left_elbow").dofadr[0]] = 1.5
        v = self.extras.values(d, self.commands_at(d))
        self.assertAlmostEqual(float(v["observation.velocity"][3]), 1.5, places=5)


class ModalityTests(unittest.TestCase):
    def test_bimanual_8dof_slices(self):
        extras = {"observation.ee_pose": [f"{s}_hand_{c}" for s in ("left", "right") for c in "x y z a b c d e f".split()]}
        m = build_modality(LEFT + RIGHT, ["ego", "wrist_left"], extras)
        self.assertEqual(m["state"]["left_arm"], {"start": 0, "end": 7})
        self.assertEqual(m["state"]["left_gripper"], {"start": 7, "end": 8})
        self.assertEqual(m["state"]["right_arm"], {"start": 8, "end": 15})
        self.assertEqual(m["action"]["right_gripper"], {"start": 15, "end": 16})
        self.assertEqual(m["state"]["right_eef_rotation_6d"],
                         {"original_key": "observation.ee_pose", "start": 12, "end": 18})
        self.assertNotIn("left_eef_position", m["action"])   # no action.ee_pose given
        self.assertEqual(m["video"]["wrist_left"], {"original_key": "observation.images.wrist_left"})

    def test_single_arm_and_old_arm_names(self):
        self.assertEqual(build_modality(RIGHT, [])["state"], {"right_arm": {"start": 0, "end": 7},
                                                              "right_gripper": {"start": 7, "end": 8}})
        self.assertIsNone(build_modality(["joint1L", "joint2l", "joint7l"], []))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(write_modality(Path(tmp), ["joint1L"], []))
            path = write_modality(Path(tmp), LEFT, ["ego"])
            self.assertEqual(json.loads(path.read_text())["state"]["left_arm"]["end"], 7)


@unittest.skipIf(lerobot is None or mujoco is None, "lerobot / torch / mujoco not installed")
class RecorderTests(unittest.TestCase):
    def recorder(self, root, extra_features):
        from humanoid_robot_learning.sim.sim_recorder import SimLeRobotRecorder
        r = SimLeRobotRecorder(task_name="t", repo_id="test/x", dataset_root=root, fps=25, device="cpu",
                               joint_names=LEFT, cameras={}, robot_type="pioneer_8dof_left_arm",
                               extra_features=extra_features, rate_limit=False, buffer_capacity_s=1.)
        r.init_dataset()
        return r

    def record_take(self, r, extras):
        flags = r.start_flags()
        flags.start = True
        for _ in range(3):
            r.tick(np.zeros(8, np.float32), np.zeros(8, np.float32), {}, extras=extras)
        flags.success = True
        r.tick(np.zeros(8, np.float32), np.zeros(8, np.float32), {}, extras=extras)
        r.finalize()

    def test_extras_and_modality_written_then_old_dataset_appends_without_them(self):
        from humanoid_robot_learning.sim.mujoco_arm_extras import ArmExtras, arm_8dof_spec
        model, data = arm_model()
        ex = ArmExtras(model, [arm_8dof_spec("left")])
        values = ex.values(data, {"left": data.qpos[[model.joint(j).qposadr[0] for j in LEFT[:7]]].copy()})
        with tempfile.TemporaryDirectory() as tmp:
            new = Path(tmp) / "new"
            self.record_take(self.recorder(new, ex.features), values)
            info = json.loads((new / "meta/info.json").read_text())
            self.assertEqual(info["features"]["observation.ee_pose"]["shape"], [9])
            self.assertEqual(info["features"]["observation.effort"]["names"], LEFT)
            modality = json.loads((new / "meta/modality.json").read_text())
            self.assertEqual(modality["state"]["left_eef_position"]["original_key"], "observation.ee_pose")

            old = Path(tmp) / "old"
            self.record_take(self.recorder(old, None), None)          # a dataset from before the extras
            r = self.recorder(old, ex.features)                       # appending with them: dropped, not an error
            self.assertEqual(r.extra_features, {})
            self.record_take(r, values)
            info = json.loads((old / "meta/info.json").read_text())
            self.assertEqual(info["total_episodes"], 2)
            self.assertNotIn("observation.ee_pose", info["features"])


if __name__ == "__main__":
    unittest.main()
