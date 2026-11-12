"""URDF kinematics and a privileged-state Cartesian expert for the cargo task.

The expert outputs joint/gripper commands only. It never attaches, teleports,
or applies forces to the cargo. SciPy is used for bounded inverse kinematics.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, DEFAULT_JOINT_POS
from pioneer_humanoid.urdf_joint_limits import JOINT_POS_LIMITS


class ArmKinematics:
    def __init__(self):
        root = Path(__file__).resolve().parents[4]
        urdf = ET.parse(root / "assets/pioneer_bimanual_arm/urdf/pioneer_bimanual_arm.urdf")
        joints = {j.get("name"): j for j in urdf.findall("joint")}
        self.origins, self.axes = [], []
        for name in LEFT_ARM_JOINTS:
            joint = joints[name]
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
            transform[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
            self.origins.append(transform)
            self.axes.append(np.fromstring(joint.find("axis").get("xyz"), sep=" "))
        # Midpoint of canonical distal finger tips. Their opposite slide offsets cancel.
        self.tip_offset = np.mean([
            np.fromstring(joints["joint7l"].find("origin").get("xyz"), sep=" ") + [0, .02, 0],
            np.fromstring(joints["joint8l"].find("origin").get("xyz"), sep=" ") + [0, -.02, 0],
        ], axis=0)
        self.home = np.array([DEFAULT_JOINT_POS[j] for j in LEFT_ARM_JOINTS])
        self.lo, self.hi = np.array([JOINT_POS_LIMITS[j] for j in LEFT_ARM_JOINTS]).T

    def fk(self, q):
        transform = np.eye(4)
        for origin, axis, angle in zip(self.origins, self.axes, q):
            rotate = np.eye(4)
            rotate[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            transform = transform @ origin @ rotate
        return transform[:3, 3] + transform[:3, :3] @ self.tip_offset, Rotation.from_matrix(transform[:3, :3])

    def ik(self, pos, rot, seed):
        def residual(q):
            p, r = self.fk(q)
            return np.r_[p - pos, .15 * (rot * r.inv()).as_rotvec()]
        solution = least_squares(residual, np.clip(seed, self.lo + 1e-6, self.hi - 1e-6),
                                 bounds=(self.lo + 1e-6, self.hi - 1e-6), max_nfev=60,
                                 ftol=1e-7, xtol=1e-7, gtol=1e-7)
        p, r = self.fk(solution.x)
        error = np.linalg.norm(p - pos)
        angular = np.linalg.norm((rot * r.inv()).as_rotvec())
        return solution.x, error, angular


def smooth(t):
    t = np.clip(t, 0, 1)
    return t*t*t*(10 + t*(-15 + 6*t))


class CargoExpert:
    def __init__(self, kinematics, root_pos, measured_q, cargo_pos, cargo_quat_wxyz, target_xy, rng, retreat_height=.15, target_yaw=0.):
        self.kin = kinematics
        self.root = np.asarray(root_pos)
        self.q = np.asarray(measured_q).copy()
        self.goal = np.r_[target_xy, cargo_pos[2] + .025]
        self.aligned_place = False
        self.retreat_height = retreat_height
        p, rot = self.kin.fk(self.q)
        cargo_rotation = Rotation.from_quat(np.asarray(cargo_quat_wxyz)[[1, 2, 3, 0]])
        yaw = cargo_rotation.as_euler("xyz")[2]
        grasp_rot = Rotation.from_euler("z", yaw)
        # NASA handle arc top; grasp pad centres slightly below its upper surface.
        # The midpoint is the finger-body centre, whose collision hull extends
        # 68 mm down. Keep the hull off the bag body while pinching the loop top.
        grasp = np.asarray(cargo_pos) + cargo_rotation.apply([0, .08002, .363]) - self.root
        self.place_rot = Rotation.from_euler("z", target_yaw)
        place = np.r_[target_xy, cargo_pos[2]] + self.place_rot.apply([0, .08002, .363]) - self.root
        above = grasp + [0, 0, .105]
        carry = grasp + [0, 0, .12]
        target_above = place + [0, 0, .12]
        target_rot = self.place_rot
        speed = rng.uniform(.92, 1.08)
        self.points = [(p, rot, 0.0)]
        self.segments = []
        for name, endpoint, orientation, grip, duration in [
            ("approach", above, grasp_rot, 0., 3.0),
            ("descend", grasp, grasp_rot, 0., 2.0),
            ("close", grasp, grasp_rot, 1., 1.3),
            ("lift", carry, grasp_rot, 1., 2.0),
            ("transfer", target_above, target_rot, 1., 4.0),
            ("lower", place, target_rot, 1., 2.0),
            ("release", place, target_rot, 0., 1.0),
            ("retreat", target_above, target_rot, 0., 2.0),
            ("settle", target_above, target_rot, 0., 4.0),
        ]:
            self.segments.append((name, duration * speed))
            self.points.append((np.asarray(endpoint), orientation, grip))
        self.elapsed = 0.
        self.phase = "approach"
        self.max_error = 0.

    @property
    def duration(self):
        return sum(d for _, d in self.segments)

    def action(self, dt, cargo_pos=None, tip_pos=None):
        time = self.elapsed
        index = len(self.segments) - 1
        for i, (name, duration) in enumerate(self.segments):
            if time <= duration or i == len(self.segments) - 1:
                index = i
                fraction = smooth(time / duration)
                self.phase = name
                break
            time -= duration
        if self.phase == "lower" and not self.aligned_place and cargo_pos is not None:
            # Recompute placement from the actual carried object offset. A friction
            # grasp can settle differently from the nominal handle-to-tool frame.
            measured = np.asarray(tip_pos) - self.root
            endpoint = measured + self.goal - np.asarray(cargo_pos)
            self.points[index] = (measured, self.points[index][1], 1.)
            self.points[index + 1] = (endpoint, target_rot := self.place_rot, 1.)
            self.points[index + 2] = (endpoint, target_rot, 0.)
            for j in range(index + 3, len(self.points)):
                self.points[j] = (endpoint + [0, 0, self.retreat_height], target_rot, 0.)
            self.aligned_place = True
        start_p, start_r, start_g = self.points[index]
        end_p, end_r, end_g = self.points[index + 1]
        pos = start_p + fraction * (end_p - start_p)
        rot = Slerp([0, 1], Rotation.concatenate([start_r, end_r]))([fraction])[0]
        q, error, angular = self.kin.ik(pos, rot, self.q)
        self.max_error = max(error, self.max_error)
        if error > .018 or angular > .15:
            raise RuntimeError(f"Expert IK unreachable during {self.phase}: {error:.4f} m, {angular:.3f} rad")
        self.q = q
        self.elapsed += dt
        return np.r_[q, start_g + fraction * (end_g - start_g)].astype(np.float32)
