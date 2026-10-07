"""KeyboardArm (mujoco_keyboard_teleop.py) driven headless by a scripted "operator" holding keys.

    pip install mujoco pytest && pytest src/teleop/keyboard_teleop/test_mujoco_keyboard_teleop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco_keyboard_teleop as kt  # noqa: E402  (puts the repo's packages on sys.path)
from humanoid_mujoco_scenes import make_model, scene_reset, scene_step  # noqa: E402
from humanoid_mujoco_scenes.tidy_table import scene as S  # noqa: E402
from pioneer_humanoid.arm_params import (  # noqa: E402
    LEFT_ARM_JOINTS, LEFT_GRIPPER_CLOSED, LEFT_GRIPPER_JOINTS, LEFT_GRIPPER_OPEN,
)
from pioneer_humanoid.mujoco_bimanual_arm import set_home  # noqa: E402


class Sim:
    def __init__(self, seed=3):
        self.m = make_model("tidy_table")
        self.m.actuator_biasprm[self.m.actuator("joint6l").id, 2] = -kt.WRIST_DAMPING
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetData(self.m, self.d)
        set_home(self.m, self.d)
        scene_reset("tidy_table")(self.m, self.d, np.random.default_rng(seed))
        mujoco.mj_forward(self.m, self.d)
        self.hook = scene_step("tidy_table")
        self.arm = kt.KeyboardArm(self.m, self.d)
        self.acts = [self.m.actuator(j).id for j in LEFT_ARM_JOINTS]
        self.gacts = [self.m.actuator(j).id for j in LEFT_GRIPPER_JOINTS]
        self.g_open = np.array([LEFT_GRIPPER_OPEN[j] for j in LEFT_GRIPPER_JOINTS])
        self.g_closed = np.array([LEFT_GRIPPER_CLOSED[j] for j in LEFT_GRIPPER_JOINTS])
        self.targets = []
        self.table = self.m.geom("table").id
        self.worst_table_pen = 0.0
        self.max_obj_speed = 0.0

    def run(self, held=(), seconds=0.01, fine=False, until=None):
        for _ in range(int(round(seconds / kt.CONTROL_DT))):
            q, g = self.arm.step(set(held), fine)
            self.targets.append(q)
            self.d.ctrl[self.acts] = q
            self.d.ctrl[self.gacts] = self.g_open + g * (self.g_closed - self.g_open)
            self.hook(self.m, self.d)
            mujoco.mj_step(self.m, self.d, nstep=10)
            for c in self.d.contact[:self.d.ncon]:
                if self.table in (c.geom1, c.geom2):
                    self.worst_table_pen = min(self.worst_table_pen, c.dist)
            for o in S.episode_objects(self.m, self.d):
                v = self.d.qvel[self.m.joint(o["name"]).dofadr[0]:][:3]
                self.max_obj_speed = max(self.max_obj_speed, float(np.linalg.norm(v)))
            if until is not None and until():
                return True
        return False

    def grasp_point(self):
        return self.arm.ik.fk(self.d.qpos, self.d.qpos[self.arm.qadr])[0]

    def drive_to(self, xyz, tol=0.004, seconds=8.0, fine_within=0.02):
        """An operator holding whichever keys point at `xyz`, fine near it, until within `tol` on every axis."""
        for _ in range(int(seconds / kt.CONTROL_DT)):
            err = np.asarray(xyz) - self.grasp_point()
            if np.all(np.abs(err) < tol):
                return True
            held = set()
            for axis, (pos_key, neg_key) in enumerate((("w", "s"), ("a", "d"), ("q", "e"))):
                if abs(err[axis]) >= tol:
                    held.add(pos_key if err[axis] > 0 else neg_key)
            self.run(held, kt.CONTROL_DT, fine=np.linalg.norm(err) < fine_within)
        return False


@pytest.fixture
def sim():
    return Sim()


def _seed_first_object_round():
    """A layout whose first object is a ball or a standing cylinder (graspable without turning)."""
    m = make_model("tidy_table")
    d = mujoco.MjData(m)
    for seed in range(100):
        mujoco.mj_resetData(m, d)
        set_home(m, d)
        scene_reset("tidy_table")(m, d, np.random.default_rng(seed))
        o = S.episode_objects(m, d)[0]
        if o["shape"] == "ball" or (o["shape"] == "cylinder" and not o["lying"]):
            return seed
    raise AssertionError("no such layout in 100 seeds")


def test_first_key_glides_to_ready_smoothly(sim):
    assert sim.arm.mode == "home"
    sim.run({"w"}, kt.GLIDE_S + 0.1)
    assert sim.arm.mode == "teleop"
    q = np.array(sim.targets)
    assert np.abs(np.diff(q, axis=0)).max() < 0.03          # no one-step jumps (the wrist snap)
    R = sim.d.xmat[sim.m.body("link6l").id].reshape(3, 3)
    assert R[2, 2] > 0.98                                   # gripper pointing down


def test_held_key_ramps_up_and_stops(sim):
    sim.run({"w"}, kt.GLIDE_S)
    sim.run((), 0.5)
    p0 = sim.grasp_point()
    sim.targets.clear()
    sim.run({"a"}, 1.0)
    p1 = sim.grasp_point()
    assert 0.08 < p1[1] - p0[1] < 0.14                      # ~SPEED for 1 s (the ramp costs a little)
    assert abs(p1[0] - p0[0]) < 0.01 and abs(p1[2] - p0[2]) < 0.01
    q = np.array(sim.targets)
    assert np.abs(np.diff(q, axis=0)).max() <= kt.JOINT_RATE * kt.CONTROL_DT + 1e-9
    sim.run((), 0.5)                                        # released: stops and stays
    p2 = sim.grasp_point()
    sim.run((), 0.5)
    assert np.linalg.norm(sim.grasp_point() - p2) < 0.002


def test_fingers_never_dig_into_the_table(sim):
    sim.run({"w"}, kt.GLIDE_S + 0.1)
    # over a clear patch of table between the zone and the robot, then hold "down" well past contact
    assert sim.drive_to((0.14, 0.40, 0.90))
    sim.run({"e"}, 4.0)
    tips = min(sim.d.geom_xpos[g][2] - abs(sim.d.geom_xmat[g].reshape(3, 3) @ sim.m.geom_size[g])[2]
               for g in range(sim.m.ngeom)
               if sim.m.body(sim.m.geom_bodyid[g]).name in ("link7l", "link8l") and sim.m.geom_contype[g])
    assert tips > kt.TABLE_Z - 0.0015                       # stops at the clearance, not inside the table
    assert sim.worst_table_pen > -0.0015


def test_pick_and_bin_an_object_by_keys():
    """Operator-by-keys tidies the first object: contacts stay calm and the scene counts the step."""
    sim = Sim(seed=_seed_first_object_round())
    sim.run({"w"}, kt.GLIDE_S + 0.1)
    obj = S.episode_objects(sim.m, sim.d)[0]
    p = sim.d.xpos[sim.m.body(obj["name"]).id].copy()
    grasp_z = kt.TABLE_Z + 0.004 + 0.087
    assert sim.drive_to((p[0], p[1], grasp_z + 0.10))
    assert sim.drive_to((p[0], p[1], grasp_z + 0.003), tol=0.003)
    sim.arm.toggle_grip()
    sim.run((), 0.8)
    assert sim.drive_to((p[0], p[1], grasp_z + 0.12))
    lifted = sim.d.xpos[sim.m.body(obj["name"]).id][2] - p[2]
    assert lifted > 0.08                                    # it came up with the gripper
    bx, by = obj["bin"]
    assert sim.drive_to((bx, by, kt.TABLE_Z + 0.20), tol=0.006)
    sim.arm.toggle_grip()
    assert sim.run((), 2.0, until=lambda: S.episode_status(sim.m, sim.d)["step"] >= 1)
    status = S.episode_status(sim.m, sim.d)
    assert not status["dropped"] and not status["toppled"] and not status["early"]
    assert sim.max_obj_speed < 1.5                          # nothing flung (a free fall from the release is ~0.9 m/s)


def test_turn_keys_turn_the_gripper_smoothly_and_stay_down(sim):
    sim.run({"w"}, kt.GLIDE_S)
    sim.run((), 0.3)
    sim.targets.clear()
    sim.run({"c"}, 0.5)
    yaw = sim.arm.ik.yaw_of(sim.d.xmat[sim.m.body("link6l").id].reshape(3, 3))
    assert 0.2 < yaw < 0.45                                  # ~YAW_RATE for 0.5 s, minus the ramp and lag
    assert np.abs(np.diff(np.array(sim.targets), axis=0)).max() <= kt.JOINT_RATE * kt.CONTROL_DT + 1e-9
    assert sim.d.xmat[sim.m.body("link6l").id][8] > 0.98     # still pointing down


def test_turning_far_from_the_robot_keeps_the_gripper_down_and_in_place(sim):
    """Far out a turn needs big joint swings, and past the yaw the arm reaches it just stops."""
    sim.run({"w"}, kt.GLIDE_S + 0.1)
    assert sim.drive_to((0.42, 0.45, 0.92))
    sim.run((), 0.4)
    p0 = sim.grasp_point()
    for held in ({"c"}, {"v"}):
        for _ in range(250):
            sim.run(held, kt.CONTROL_DT)
            assert sim.d.xmat[sim.m.body("link6l").id][8] > np.cos(np.radians(5))   # was ~20 deg off
            assert np.linalg.norm(sim.grasp_point() - p0) < 0.015                   # was ~9 cm away


def test_a_turned_gripper_still_moves_everywhere(sim):
    """Turned as far as it goes, a move the arm can't make at that yaw unwinds the turn instead of refusing."""
    sim.run({"w"}, kt.GLIDE_S + 0.1)
    assert sim.drive_to((0.27, 0.30, 0.90))
    sim.run({"c"}, 0.6)
    sim.run((), 0.4)
    p0 = sim.grasp_point()
    sim.run({"d"}, 0.7)                                      # toward the robot's side: no positive yaw there
    sim.run((), 0.3)
    d = sim.grasp_point() - p0
    assert d[1] < -0.06                                      # was 7 mm
    assert abs(d[0]) < 0.005 and abs(d[2]) < 0.005           # straight
    assert sim.d.xmat[sim.m.body("link6l").id][8] > np.cos(np.radians(3))


def test_home_glides_back(sim):
    sim.run({"w"}, kt.GLIDE_S + 0.5)
    sim.arm.go_home()
    sim.run((), kt.GLIDE_S + 0.1)
    assert sim.arm.mode == "home"
    assert np.allclose(sim.arm.q, sim.arm.home)


def test_pushing_into_something_presses_lightly_and_lets_go(sim):
    """Hold "down" with a finger over a bin wall: the arm holds where it stops instead of winding up force."""
    sim.run({"w"}, kt.GLIDE_S + 0.1)
    bx, by = S.BINS["box"]
    assert sim.drive_to((bx - 0.03, by - S.BIN_INNER / 2 - 0.045, 0.90), tol=0.005)   # a finger over the near wall
    f6, held = np.zeros(6), []
    for k in range(300):
        sim.run({"e"}, kt.CONTROL_DT)
        f = 0.0
        for i, c in enumerate(sim.d.contact[:sim.d.ncon]):
            if {sim.m.body(sim.m.geom_bodyid[c.geom1]).name, sim.m.body(sim.m.geom_bodyid[c.geom2]).name} \
                    & {"link7l", "link8l"}:
                mujoco.mj_contactForce(sim.m, sim.d, i, f6)
                f += abs(f6[0])
        held.append(f)
    assert max(held) > 0, "never touched the wall: the test's finger placement is off"
    assert np.mean(held[-100:]) < 15.0                       # leaning on it, not crushing it (was ~110 N)
    z0 = sim.grasp_point()[2]
    sim.run({"q"}, 0.5)
    assert sim.grasp_point()[2] - z0 > 0.03                  # and lifts straight off when asked
