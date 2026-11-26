"""make_model builds each scene into the rig's spec: no empty-prefix attach warning, and the spec's radians hold."""
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "pioneer_humanoid")]

import mujoco  # noqa: E402
from humanoid_mujoco_scenes import (  # noqa: E402
    list_scenes,
    make_model,
    scene_progress,
    scene_reset,
    scene_step,
    scene_success,
)


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

    def test_ak45_wrist_motors_are_visual_only(self):
        model = make_model("door_open")
        mesh_ids = {i for i in range(model.nmesh) if model.mesh(i).name == "ak45_10_visual"}
        self.assertEqual(len(mesh_ids), 1)  # one shared asset, instanced at four wrist axes
        wrist_motor_geoms = [
            model.geom(i) for i in range(model.ngeom)
            if model.geom(i).type == mujoco.mjtGeom.mjGEOM_MESH
            and model.geom(i).dataid[0] in mesh_ids
        ]
        self.assertEqual(
            {model.body(geom.bodyid[0]).name for geom in wrist_motor_geoms},
            {"left_link5", "left_link6", "right_link5", "right_link6"},
        )
        self.assertTrue(all(geom.contype[0] == 0 and geom.conaffinity[0] == 0 for geom in wrist_motor_geoms))

    def test_8dof_finger_contacts_follow_visible_jaw_faces(self):
        # The old combined convex hull extended about 9 mm into the open gap,
        # letting the door move before the rendered claw touched its handle.
        model = make_model("door_open")
        expected_inner_faces = {
            "left_finger_a": (1, 0.0413),
            "left_finger_b": (-1, -0.0406),
            "right_finger_a": (-1, -0.0413),
            "right_finger_b": (1, 0.0406),
        }
        for body_name, (sign, expected) in expected_inner_faces.items():
            body = model.body(body_name)
            pads = [
                model.geom(i) for i in range(model.ngeom)
                if model.geom_bodyid[i] == body.id
                and "_contact_pad_" in model.geom(i).name
            ]
            self.assertEqual(len(pads), 16)
            self.assertTrue(all(g.type[0] == mujoco.mjtGeom.mjGEOM_BOX for g in pads))
            # Positive-Y jaws grip on pos-size; negative-Y jaws on pos+size.
            inner = min(g.pos[1] - g.size[1] for g in pads) if sign > 0 else max(
                g.pos[1] + g.size[1] for g in pads
            )
            with self.subTest(body=body_name):
                self.assertAlmostEqual(inner, expected, delta=0.0002)
                self.assertTrue(all(g.contype[0] == 2 and g.conaffinity[0] == 1 for g in pads))

    def test_8dof_gripper_physically_stops_on_door_handle(self):
        """The real jaws contact, attach, and cannot be driven through the handle."""
        from pioneer_humanoid.mujoco_bimanual_arm_8dof import set_home

        model = make_model("door_open")
        data = mujoco.MjData(model)
        set_home(model, data)
        mujoco.mj_forward(model, data)

        # Reorient the static cabinet for a deterministic straight-on grasp at
        # the home gripper pose.  The tested door, hinge and handle are otherwise
        # exactly the scene geometry used in VR.
        link = model.body("left_link7").id
        rotation = data.xmat[link].reshape(3, 3)
        handle_target = data.xpos[link] + rotation @ np.array([0, 0, -0.19])
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        cabinet = model.body("cabinet").id
        door = model.body("cabinet_door").id
        handle = model.geom("cabinet_handle")
        handle_in_cabinet = model.body_pos[door] + handle.pos
        model.body_quat[cabinet] = quaternion
        model.body_pos[cabinet] = handle_target - rotation @ handle_in_cabinet
        data.qpos[model.joint("cabinet_door_hinge").qposadr[0]] = 0.0
        data.ctrl[model.actuator("left_gripper").id] = 0.041
        mujoco.mj_forward(model, data)

        hook = scene_step("door_open")
        deepest = 0.0
        touched_a = touched_b = False
        for _ in range(100):
            # The teleop writes the trigger's full-close request before every
            # scene hook; the hook then applies its proximity/contact cap.
            data.ctrl[model.actuator("left_gripper").id] = 0.041
            hook(model, data)
            mujoco.mj_step(model, data, nstep=10)
            for contact in data.contact[:data.ncon]:
                names = {model.geom(contact.geom1).name, model.geom(contact.geom2).name}
                if "cabinet_handle" not in names:
                    continue
                deepest = min(deepest, float(contact.dist))
                other = next(name for name in names if name != "cabinet_handle")
                touched_a |= other.startswith("left_finger_a_contact_pad_")
                touched_b |= other.startswith("left_finger_b_contact_pad_")

        qa = float(data.qpos[model.joint("left_gripper").qposadr[0]])
        qb = float(data.qpos[model.joint("left_gripper_mimic").qposadr[0]])
        self.assertTrue(touched_a or touched_b)
        self.assertGreater(qa, 0.012)
        self.assertGreater(qb, 0.012)
        # The differential lets the first jaw stop while the other continues,
        # so an off-centre bar need not leave the two sliders symmetric. Their
        # mean must still stop well short of full closure, and neither jaw may
        # reach the actuator's 41 mm close request.
        self.assertLess((qa + qb) / 2, 0.032)
        self.assertLess(max(qa, qb), 0.040)
        self.assertGreater(data.userdata[1], 0.0)
        self.assertGreater(deepest, -0.002)

        equality_ids = [model.equality(name).id for name in ("cabinet_grasp_low", "cabinet_grasp_high")]
        self.assertTrue(all(data.eq_active[equality_ids]))

        # Push an arm actuator hard for two simulated seconds.  A friction-only
        # grasp used to let the wrist pass through and leave the door behind;
        # both attachment points must remain coincident instead.
        pull = model.actuator("left_shoulder_yaw").id
        data.ctrl[pull] -= 0.25
        max_anchor_error = 0.0
        for _ in range(200):
            data.ctrl[model.actuator("left_gripper").id] = 0.041
            hook(model, data)
            mujoco.mj_step(model, data, nstep=10)
            for equality_id in equality_ids:
                first = model.eq_obj1id[equality_id]
                second = model.eq_obj2id[equality_id]
                first_point = data.xpos[first] + data.xmat[first].reshape(3, 3) @ model.eq_data[equality_id, :3]
                second_point = data.xpos[second] + data.xmat[second].reshape(3, 3) @ model.eq_data[equality_id, 3:6]
                max_anchor_error = max(max_anchor_error, np.linalg.norm(first_point - second_point))
        self.assertLess(max_anchor_error, 0.001)

        # Opening the trigger is an intentional release, not a permanent weld.
        data.ctrl[model.actuator("left_gripper").id] = 0.0
        hook(model, data)
        self.assertFalse(any(data.eq_active[equality_ids]))
        self.assertEqual(data.userdata[1], 0.0)

    def test_door_grasp_handles_off_center_contact_but_never_grabs_from_air(self):
        """Every real pad contact captures; nearby non-contact never does."""
        from pioneer_humanoid.mujoco_bimanual_arm_8dof import set_home

        def close_at(offset_y, offset_z=-0.19):
            model = make_model("door_open")
            data = mujoco.MjData(model)
            set_home(model, data)
            mujoco.mj_forward(model, data)
            link = model.body("left_link7").id
            rotation = data.xmat[link].reshape(3, 3)
            target = data.xpos[link] + rotation @ np.array([0.0, offset_y, offset_z])
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, rotation.ravel())
            cabinet = model.body("cabinet").id
            door = model.body("cabinet_door").id
            handle = model.geom("cabinet_handle")
            model.body_quat[cabinet] = quaternion
            model.body_pos[cabinet] = target - rotation @ (model.body_pos[door] + handle.pos)
            data.qpos[model.joint("cabinet_door_hinge").qposadr[0]] = 0.0
            mujoco.mj_forward(model, data)
            hook = scene_step("door_open")
            actuator = model.actuator("left_gripper").id
            touched = False
            for tick in range(1000):
                if tick % 10 == 0:  # the VR control loop refreshes the trigger at 100 Hz
                    data.ctrl[actuator] = 0.041
                hook(model, data)   # scene contact logic runs at the 1 kHz physics rate
                mujoco.mj_step(model, data)
                for contact in data.contact[:data.ncon]:
                    names = {model.geom(contact.geom1).name, model.geom(contact.geom2).name}
                    if "cabinet_handle" in names:
                        other = next(name for name in names if name != "cabinet_handle")
                        touched |= other.startswith(("left_finger_a_contact_pad_", "left_finger_b_contact_pad_"))
                if data.userdata[2]:
                    break
            return model, data, touched

        for offset_y in (-0.015, -0.008, 0.0, 0.008, 0.015):
            with self.subTest(offset_y=offset_y):
                model, data, touched = close_at(offset_y)
                self.assertTrue(touched)
                self.assertEqual(data.userdata[2], 1.0)
                self.assertTrue(all(data.eq_active[
                    [model.equality("cabinet_grasp_low").id, model.equality("cabinet_grasp_high").id]
                ]))

        model, data, touched = close_at(0.0, offset_z=-0.15)
        self.assertFalse(touched)
        self.assertEqual(data.userdata[2], 0.0)
        self.assertFalse(any(data.eq_active[
            [model.equality("cabinet_grasp_low").id, model.equality("cabinet_grasp_high").id]
        ]))

    def test_zip_tie_strap_springs_rest_straight(self):
        # springref is given in radians: each bend spring relaxes at minus its threaded bend (a straight strap).
        from humanoid_mujoco_scenes.zip_tie.scene import _PATH, _wrap
        model = make_model("zip_tie")
        for i in range(1, len(_PATH)):
            joint = model.joint(f"strap{i}_bend")
            self.assertAlmostEqual(model.qpos_spring[joint.qposadr[0]], -_wrap(_PATH[i][2] - _PATH[i - 1][2]), places=9)

    def test_door_open_starts_clear_and_latches_success(self):
        from humanoid_mujoco_scenes.door_open.scene import (
            CABINET_CENTER_Y_RANGE,
            CABINET_FRONT_X_RANGE,
            CABINET_YAW_RANGE,
            INSTRUCTION,
            MIN_RESET_TRANSLATION,
            OPEN_ANGLE,
            door_angle,
        )
        from pioneer_humanoid.mujoco_bimanual_arm_8dof import set_home

        model = make_model("door_open")
        data = mujoco.MjData(model)
        set_home(model, data)
        mujoco.mj_forward(model, data)
        cabinet_contacts = [
            c for c in data.contact[:data.ncon]
            if model.geom(c.geom1).name.startswith("cabinet_")
            or model.geom(c.geom2).name.startswith("cabinet_")
        ]
        self.assertEqual(cabinet_contacts, [])

        reset = scene_reset("door_open")
        placements = []
        cabinet = model.body("cabinet").id
        previous = model.body_pos[cabinet, :2].copy()
        for seed in range(20):
            mujoco.mj_resetData(model, data)
            set_home(model, data)
            reset(model, data, np.random.default_rng(seed))
            mujoco.mj_forward(model, data)
            x, y = model.body_pos[cabinet, :2]
            yaw = 2 * np.arctan2(model.body_quat[cabinet, 3], model.body_quat[cabinet, 0])
            self.assertLessEqual(CABINET_FRONT_X_RANGE[0], x)
            self.assertLessEqual(x, CABINET_FRONT_X_RANGE[1])
            self.assertLessEqual(CABINET_CENTER_Y_RANGE[0], y)
            self.assertLessEqual(y, CABINET_CENTER_Y_RANGE[1])
            self.assertLessEqual(CABINET_YAW_RANGE[0], yaw)
            self.assertLessEqual(yaw, CABINET_YAW_RANGE[1])
            self.assertGreaterEqual(np.linalg.norm(model.body_pos[cabinet, :2] - previous), MIN_RESET_TRANSLATION)
            previous = model.body_pos[cabinet, :2].copy()
            placements.append((round(x, 4), round(y, 4), round(yaw, 4)))
            cabinet_contacts = [
                (model.geom(c.geom1).name, model.geom(c.geom2).name) for c in data.contact[:data.ncon]
                if model.geom(c.geom1).name.startswith("cabinet_")
                or model.geom(c.geom2).name.startswith("cabinet_")
            ]
            with self.subTest(seed=seed, placement=placements[-1]):
                self.assertEqual(cabinet_contacts, [])
        self.assertEqual(len(set(placements)), len(placements))

        reset(model, data, np.random.default_rng(7))
        self.assertLessEqual(-np.deg2rad(4), door_angle(model, data))
        self.assertLessEqual(door_angle(model, data), 0)
        self.assertEqual(scene_progress("door_open")(model, data), (0, 1, INSTRUCTION))
        self.assertFalse(scene_success("door_open")(model, data))

        data.qpos[model.joint("cabinet_door_hinge").qposadr[0]] = -OPEN_ANGLE
        mujoco.mj_forward(model, data)
        scene_step("door_open")(model, data)
        self.assertTrue(scene_success("door_open")(model, data))
        self.assertEqual(scene_progress("door_open")(model, data), (1, 1, INSTRUCTION))

    def test_two_consecutive_save_requests_immediately_randomize_door_pose(self):
        """Save requests—not delayed camera callbacks—each request one reset."""
        from pioneer_humanoid.mujoco_bimanual_arm_8dof import set_home

        teleop = HERE.parents[1] / "teleop" / "quest_controller_teleop"
        sys.path[:0] = [str(teleop), str(teleop / "mujoco"), str(HERE.parents[1] / "robot_learning")]
        from mujoco_sim import Recording

        class Flags:
            start = False
            success = False

        class Recorder:
            fps = 25
            num_buffered_frames = 10

            @staticmethod
            def start_flags():
                return Flags()

        class Window:
            saves = 0

            def save(self):
                self.saves += 1

            def discard(self):
                pass

        class Driver:
            status = ""

        model = make_model("door_open")
        data = mujoco.MjData(model)
        reset = scene_reset("door_open")
        recording = Recording(Recorder(), Window(), 4, Driver(), ("left",))
        rng = np.random.default_rng(901)
        placements = []
        for expected_save_marker in (1, 2):
            recording.start()
            recording.save()
            self.assertEqual(recording.window.saves, expected_save_marker)
            self.assertTrue(recording.consume_reset_after_save())
            self.assertFalse(recording.consume_reset_after_save())
            mujoco.mj_resetData(model, data)
            set_home(model, data)
            reset(model, data, rng)
            mujoco.mj_forward(model, data)
            placements.append(model.body_pos[model.body("cabinet").id, :2].copy())
            recording.saved()  # asynchronous encoder completion; must not request another reset

        self.assertGreaterEqual(np.linalg.norm(placements[1] - placements[0]), 0.10)
        self.assertFalse(recording.consume_reset_after_save())


if __name__ == "__main__":
    unittest.main()
