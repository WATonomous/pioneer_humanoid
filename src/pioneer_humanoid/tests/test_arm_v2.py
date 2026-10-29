"""Run: python -m unittest discover -s src/pioneer_humanoid/tests -v."""
import math
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from pioneer_humanoid.arm_v2_backend import mechanism, ASSET_ROOT
from pioneer_humanoid.arm_params import DEFAULT_JOINT_POS, LEFT_GRIPPER_CLOSED, RIGHT_GRIPPER_CLOSED
from pioneer_humanoid.mujoco_bimanual_arm import arm_spec, set_home
from pioneer_humanoid.urdf_joint_limits import JOINT_POS_LIMITS


class ArmV2Tests(unittest.TestCase):
    def test_names_meshes_and_limits(self):
        root = ET.parse(ASSET_ROOT/'urdf'/'pioneer_bimanual_arm.urdf').getroot()
        names = {j.get('name') for j in root.findall('joint')}
        self.assertTrue(set(DEFAULT_JOINT_POS) <= names)
        self.assertEqual(len(names), 22)
        self.assertEqual(len(root.findall('.//mesh')), 474)
        for name, value in {**DEFAULT_JOINT_POS, **LEFT_GRIPPER_CLOSED, **RIGHT_GRIPPER_CLOSED}.items():
            lo, hi = JOINT_POS_LIMITS[name]
            self.assertLessEqual(lo, value)
            self.assertLessEqual(value, hi)
        for node in root.findall('.//mesh'):
            path = ASSET_ROOT/'meshes'/node.get('filename').split('/')[-1]
            self.assertTrue(path.is_file())

    def test_all_independent_controls_and_closed_gripper(self):
        helper = mechanism()
        model, data = helper.build()
        for channel in range(14):
            values = np.zeros(14)
            values[channel] = math.radians(20)
            helper.set_controls(model, data, values)
            self.assertTrue(np.isfinite(data.xpos).all())
            self.assertLess(max(abs(data.efc_pos[:12])), 1e-6)
        for angle in np.linspace(-10, 90, 201):
            values = np.zeros(14)
            values[6] = values[13] = math.radians(angle)
            helper.set_controls(model, data, values)
            self.assertLess(max(abs(data.efc_pos[:12])), 1e-6)
            for name in ['joint7', 'joint8', 'joint7l', 'joint8l']:
                value = data.qpos[model.joint(name).qposadr[0]]
                lo, hi = JOINT_POS_LIMITS[name]
                self.assertLessEqual(lo, value)
                self.assertLessEqual(value, hi)

    def test_teleop_cameras_and_home(self):
        model = arm_spec(cameras={'ego': (480, 640), 'wrist_left': (480, 640), 'wrist_right': (480, 640)}).compile()
        data = mujoco.MjData(model)
        set_home(model, data)
        mujoco.mj_forward(model, data)
        self.assertEqual(model.nu, 16)
        self.assertEqual(model.neq, 4)
        self.assertEqual(model.ncam, 3)
        self.assertLess(max(abs(data.efc_pos[:12])), 1e-6)
        for name in DEFAULT_JOINT_POS:
            self.assertGreaterEqual(model.actuator(name).id, 0)

    def test_saved_mjcf_is_portable(self):
        model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT/'simulation'/'articulated_arm_scene.xml'))
        self.assertEqual((model.njnt, model.nmesh, model.neq), (22, 474, 4))


if __name__ == '__main__':
    unittest.main()
