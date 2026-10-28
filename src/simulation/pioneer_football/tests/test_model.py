import unittest

import mujoco
import numpy as np

from pioneer_football.model import JOINTS, REST, audit_source, ground_spec, rest_data


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = ground_spec().compile()

    def test_free_root_and_mass(self):
        m = self.m
        self.assertEqual(m.nv, 18)
        self.assertEqual(m.nu, 12)
        self.assertEqual(m.neq, 0)
        self.assertEqual(m.body(1).name, "base")
        self.assertFalse(np.any(m.body_gravcomp))
        self.assertAlmostEqual(m.body_mass.sum(), audit_source(), places=6)
        self.assertEqual([m.actuator(i).name for i in range(m.nu)], list(JOINTS))

    def test_stance_axes_and_contacts(self):
        m = self.m
        d = rest_data(m)
        self.assertGreater(d.xpos[m.body("l_foot").id, 1], 0.1)
        self.assertLess(d.xpos[m.body("r_foot").id, 1], -0.1)
        self.assertGreater(d.qpos[2], 0.8)
        self.assertFalse(any(c.dist < -1e-4 for c in d.contact))
        for j in JOINTS:
            lo, hi = m.joint(j).range
            self.assertLessEqual(lo, REST[j])
            self.assertGreaterEqual(hi, REST[j])

    def test_hip_and_knee_directions(self):
        m = self.m
        for side in ("L", "R"):
            d = rest_data(m)
            knee = m.joint("Knee_" + side).id
            ankle = m.joint("Ankle_P_" + side).id
            before = d.xanchor[knee].copy()
            d.qpos[m.joint("Hip_F_" + side).qposadr[0]] += 0.2
            mujoco.mj_forward(m, d)
            self.assertGreater(d.xanchor[knee, 0] - before[0], 0.03)
            d = rest_data(m)
            before = d.xanchor[ankle].copy()
            d.qpos[m.joint("Knee_" + side).qposadr[0]] -= 0.2
            mujoco.mj_forward(m, d)
            self.assertLess(d.xanchor[ankle, 0] - before[0], -0.03)

    def test_nonadjacent_leg_contacts_are_enabled(self):
        m = self.m
        left = m.geom("l_foot_collision_0").id
        right = m.geom("r_foot_collision_0").id
        self.assertTrue(m.geom_contype[left] & m.geom_conaffinity[right])
        self.assertEqual(m.nexclude, 2)  # only same-side ankle assembly exclusions

    def test_nominal_support(self):
        m = self.m
        d = rest_data(m)
        mujoco.mj_step(m, d, nstep=1500)
        self.assertGreater(d.qpos[2], 0.8)
        self.assertLess(np.linalg.norm(d.qpos[:2]), 0.03)
        self.assertFalse(np.any(d.warning.number))

    def test_sole_friction_controls_actual_ground_contacts(self):
        m = ground_spec().compile()
        floor = m.geom("floor").id
        m.geom_friction[floor, 0] = 1.0
        feet = [m.geom(f"{side}_foot_collision_0").id for side in ("l", "r")]
        for friction in (0.6, 1.0):
            m.geom_friction[feet, 0] = friction
            d = rest_data(m, ground_clearance=-0.001)
            contacts = [
                c
                for c in d.contact
                if floor in (c.geom1, c.geom2)
                and any(foot in (c.geom1, c.geom2) for foot in feet)
            ]
            self.assertGreater(len(contacts), 0)
            for contact in contacts:
                self.assertAlmostEqual(contact.friction[0], friction)


if __name__ == "__main__":
    unittest.main()
