"""Quest controller mapping and Pioneer position-priority IK (no ROS/Isaac dependency)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


def rotation_vector(matrix):
    """Shortest rotation log, including the otherwise ill-conditioned pi case."""
    # Quaternion conversion via the largest diagonal of the symmetric K matrix.
    m = np.asarray(matrix)
    k = np.array([
        [m[0, 0]-m[1, 1]-m[2, 2], m[0, 1]+m[1, 0], m[0, 2]+m[2, 0], m[2, 1]-m[1, 2]],
        [m[0, 1]+m[1, 0], m[1, 1]-m[0, 0]-m[2, 2], m[1, 2]+m[2, 1], m[0, 2]-m[2, 0]],
        [m[0, 2]+m[2, 0], m[1, 2]+m[2, 1], m[2, 2]-m[0, 0]-m[1, 1], m[1, 0]-m[0, 1]],
        [m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1], np.trace(m)],
    ]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    if q[3] < 0:
        q = -q
    length = np.linalg.norm(q[:3])
    return q[:3] * (2 * math.atan2(length, q[3]) / length) if length > 1e-10 else 2*q[:3]


def rotation_exp(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew @ skew)


def quaternion_matrix(xyzw):
    q = np.asarray(xyzw, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
        raise ValueError("Invalid quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def cap_vector(vector, maximum):
    return vector * min(1., maximum / max(np.linalg.norm(vector), 1e-12))


def fit_pivot(samples):
    """Mean-centred least squares: p_i + R_i offset = stationary wrist.

    Reject single-axis/insufficient motion instead of saving an arbitrary pivot.
    Returns offset in controller grip coordinates and RMS residual in metres.
    """
    if len(samples) < 30:
        raise ValueError("Collect at least 30 poses while twisting about multiple axes")
    positions = np.array([s[0] for s in samples])
    rotations = np.array([s[1] for s in samples])
    a = (rotations - rotations.mean(axis=0)).reshape(-1, 3)
    b = -(positions - positions.mean(axis=0)).reshape(-1)
    offset, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    if rank < 3 or singular[-1] < 0.25 or singular[0]/singular[-1] > 30:
        raise ValueError("Insufficient rotation: twist about at least two different axes")
    points = positions + np.einsum("nij,j->ni", rotations, offset)
    residual = float(np.sqrt(np.mean(np.sum((points-points.mean(axis=0))**2, axis=1))))
    if np.linalg.norm(offset) > .3 or residual > .015:
        raise ValueError("Wrist moved during calibration; keep it still and retry")
    return offset, residual


@dataclass
class Input:
    position: np.ndarray
    rotation: np.ndarray
    clutch: bool
    trigger: float
    precision: bool
    calibrate: bool
    reset_scene: bool = False

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or value.get("tracked") is not True:
            return None
        p = np.asarray(value.get("position"), dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or np.linalg.norm(p) > 20:
            raise ValueError("Invalid controller position")
        trigger = float(value.get("trigger", 0))
        if not math.isfinite(trigger) or not 0 <= trigger <= 1:
            raise ValueError("Invalid trigger")
        return cls(p, quaternion_matrix(value.get("orientation")),
                   value.get("clutch") is True, trigger,
                   value.get("precision") is True, value.get("calibrate") is True,
                   value.get("reset_scene") is True)


@dataclass
class Target:
    gain: float = 1.5
    max_position_error: float = .08
    max_angle_error: float = .6
    rotation_gain: float = 1.
    # Fixed Quest-world -> robot-base calibration. Isaac supplies the original
    # stereo camera's calibrated basis, instead of silently assuming it is level.
    calibration: np.ndarray = field(default_factory=lambda: np.array(
        [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]]))
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    active: bool = False
    armed: bool = False
    position: np.ndarray | None = None
    rotation: np.ndarray | None = None
    requested_rotation: np.ndarray | None = None
    previous_position: np.ndarray | None = None
    previous_rotation: np.ndarray | None = None
    mapping: np.ndarray = field(default_factory=lambda: np.eye(3))
    saturation: float = 0.
    position_limited: bool = False
    orientation_limited: bool = False
    controller_rotation_delta_deg: float = 0.
    engagement_count: int = 0

    def __post_init__(self):
        self.set_calibration(self.calibration)

    def set_calibration(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or
                not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(matrix), 1., atol=1e-5)):
            raise ValueError("Controller calibration must be a proper rotation (no reflection/scaling)")
        self.calibration = matrix.copy()

    def lose_tracking(self):
        # A held grip after a disconnect MUST NOT reengage until released first.
        self.active = self.armed = False

    def update(self, controller, head_rotation, actual_position, actual_rotation):
        if controller is None or head_rotation is None:
            self.lose_tracking()
            return
        if not controller.clutch:
            self.active = False
            self.armed = True
            return
        if not self.armed:
            return
        position = controller.position + controller.rotation @ self.pivot
        if not self.active:
            # Horizontal headset forward; pitch/roll never tilt the control axes.
            forward = head_rotation @ np.array([0., 0., -1.])
            if np.linalg.norm(forward[[0, 2]]) < .1:
                self.lose_tracking()
                return
            yaw = math.atan2(-forward[0], -forward[2])
            self.mapping = self.calibration @ rotation_exp(np.array([0., -yaw, 0.]))
            self.position = actual_position.copy()
            self.rotation = actual_rotation.copy()
            self.requested_rotation = actual_rotation.copy()
            self.saturation = 0.
            self.position_limited = self.orientation_limited = False
            self.controller_rotation_delta_deg = 0.
            self.engagement_count += 1
            self.active = True
        else:
            scale = .25 if controller.precision else 1.
            delta = self.mapping @ (position-self.previous_position) * self.gain * scale
            dr = self.mapping @ (controller.rotation @ self.previous_rotation.T) @ self.mapping.T
            controller_angle = rotation_vector(dr)
            self.controller_rotation_delta_deg = float(np.degrees(np.linalg.norm(controller_angle)))
            proposed_position = self.position + delta
            # Preserve the requested orientation independently of the lead-
            # limited IK goal. Integrating onto a clipped goal loses angles:
            # fast outward and slow return gestures then end at different poses.
            reference = self.requested_rotation if self.requested_rotation is not None else self.rotation
            self.requested_rotation = rotation_exp(controller_angle*self.rotation_gain*scale) @ reference
            error = proposed_position-actual_position
            angle = rotation_vector(self.requested_rotation @ actual_rotation.T)
            self.position_limited = bool(np.linalg.norm(error) > self.max_position_error)
            self.orientation_limited = bool(np.linalg.norm(angle) > self.max_angle_error)
            self.saturation = float(max(np.linalg.norm(error)/self.max_position_error,
                                        np.linalg.norm(angle)/self.max_angle_error))
            self.position = actual_position + cap_vector(error, self.max_position_error)
            self.rotation = rotation_exp(cap_vector(angle, self.max_angle_error)) @ actual_rotation
        # Position overshoot is consumed. Orientation keeps its reference so
        # holding/reversing the controller preserves the requested pose.
        self.previous_position = position.copy()
        self.previous_rotation = controller.rotation.copy()


def dls(jacobian, error, baseline=.01, extra=.12, threshold=.015, bias=None, task_rank=3):
    # Product of singular values generalises |det J| to Pioneer's 3x4 position block.
    singular = np.linalg.svd(jacobian, compute_uv=False)
    manipulability = float(np.prod(singular)) if len(singular) >= task_rank else 0.
    rho = max(0., 1.-manipulability/threshold)
    damping_squared = baseline**2 + extra**2*rho**2
    spring = .002**2 if bias is not None else 0.
    j = jacobian
    rhs = j.T @ error
    if bias is not None:
        rhs += spring*bias
    return np.linalg.solve(j.T @ j + (damping_squared+spring)*np.eye(j.shape[1]), rhs), rho


def bounded_dls(jacobian, error, lower, upper, bias=None):
    """Minimize the same adaptive-DLS objective subject to joint-step bounds.

    A small active-set quadratic solve redistributes position motion when a
    joint reaches a stop. Clipping an unconstrained answer cannot do that.
    Bounds include both physical limits and this tick's speed allowance.
    """
    j = np.asarray(jacobian, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if np.any(lower > upper):
        raise ValueError("Invalid joint-step bounds")
    singular = np.linalg.svd(j, compute_uv=False)
    manipulability = float(np.prod(singular)) if len(singular) >= 3 else 0.
    rho = max(0., 1.-manipulability/.015)
    spring = .002**2 if bias is not None else 0.
    h = j.T @ j + (.01**2+.12**2*rho**2+spring)*np.eye(j.shape[1])
    rhs = j.T @ error
    if bias is not None:
        rhs += spring*bias
    x = np.clip(np.linalg.solve(h, rhs), lower, upper)
    fixed = upper-lower <= 1e-12
    active = np.where(x <= lower+1e-12, -1, np.where(x >= upper-1e-12, 1, 0))
    for _ in range(8*len(x)+8):
        free = np.flatnonzero((active == 0) & ~fixed)
        bound = np.flatnonzero((active != 0) | fixed)
        direction = np.zeros_like(x)
        if len(free):
            optimum = np.linalg.solve(h[np.ix_(free, free)], rhs[free]-h[np.ix_(free, bound)] @ x[bound])
            direction[free] = optimum-x[free]
        fraction, hit = 1., None
        for i in free:
            if direction[i] > 1e-12:
                limit = (upper[i]-x[i])/direction[i]
            elif direction[i] < -1e-12:
                limit = (lower[i]-x[i])/direction[i]
            else:
                continue
            if limit < fraction:
                fraction, hit = max(0., limit), i
        x = np.clip(x+fraction*direction, lower, upper)
        if hit is not None:
            active[hit] = 1 if direction[hit] > 0 else -1
            continue
        gradient = h @ x-rhs
        # Lower-bound optimum needs grad>=0; upper-bound optimum needs grad<=0.
        violation = np.where(active == -1, -gradient, np.where(active == 1, gradient, 0.))
        violation[fixed] = 0.
        release = int(np.argmax(violation))
        if violation[release] <= 1e-10:
            return x, rho
        active[release] = 0
    raise RuntimeError("Bounded position IK failed to converge")


class PioneerKinematics:
    """URDF geometry and analytic Jacobians, independent of control policy."""
    # The tracked "wrist" point: joint 6's origin, the wrist pivot, which joints 5 and 6 never move. It used to be
    # joint 5's origin (forearm roll), which the URDF puts 6.6 cm short of the pivot on the left arm but 12.8 cm
    # on the right (a roll joint's position along its own axis is arbitrary), so the right arm tracked worse.
    # Mirrored scripted reaches, final error left/right: joint 5 2.9/5.3 cm, joint 6 0.4/0.4 cm.
    anchor_joint = 5

    def __init__(self, names, velocity=.9, wrist_velocity=None):
        import xml.etree.ElementTree as ET
        from pioneer_humanoid.arm_params import DEFAULT_JOINT_POS
        from pioneer_humanoid.urdf_joint_limits import URDF_PATH, JOINT_POS_LIMITS
        joints = {j.get("name"): j for j in ET.parse(URDF_PATH).getroot().findall("joint")}
        self.origins, self.axes = [], []
        parent = "base_link"
        for name in names:
            joint = joints[name]
            if joint.find("parent").get("link") != parent:
                raise ValueError(f"Noncontiguous Pioneer chain at {name}")
            parent = joint.find("child").get("link")
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            transform = np.eye(4)
            transform[:3, :3] = rotation_exp([0, 0, yaw]) @ rotation_exp([0, pitch, 0]) @ rotation_exp([roll, 0, 0])
            transform[:3, 3] = xyz
            self.origins.append(transform)
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            self.axes.append(axis/np.linalg.norm(axis))
        self.qids = np.arange(6)
        self.physical_limits = np.array([JOINT_POS_LIMITS[name] for name in names])
        self.limits = self.physical_limits.copy()  # what the IK enforces; set_teleop_limits narrows it
        self.rest = np.array([DEFAULT_JOINT_POS[name] for name in names])
        self.command = self.rest.copy()
        self.velocity = np.array([velocity]*4 + [velocity if wrist_velocity is None else wrist_velocity]*2)
        self.warning = 0.
        self.diagnostics = {}


    def set_teleop_limits(self, limits):
        """Narrow the IK's joint ranges (rad, (6, 2)) inside the physical ones, e.g. to keep the elbow bending one way.

        Teleop-only: the robot model and its real limits are unchanged. A range that misses the physical one
        collapses to the nearest physical bound; an inverted one to its lower end.
        """
        low = np.clip(np.asarray(limits, dtype=float)[:, 0], *self.physical_limits.T)
        high = np.clip(np.asarray(limits, dtype=float)[:, 1], *self.physical_limits.T)
        self.limits = np.stack([low, np.maximum(low, high)], axis=1)

    def kinematics(self, qpos):
        transform = np.eye(4)
        positions, axes = [], []
        for i, (origin, axis) in enumerate(zip(self.origins, self.axes)):
            transform = transform @ origin
            positions.append(transform[:3, 3].copy())
            axes.append(transform[:3, :3] @ axis)
            transform[:3, :3] = transform[:3, :3] @ rotation_exp(axis*qpos[i])
        anchor = positions[self.anchor_joint]
        jp = np.zeros((3, 6))
        for i in range(4):
            jp[:, i] = np.cross(axes[i], anchor-positions[i])
        return anchor, transform[:3, :3].copy(), jp, np.array(axes).T


class UrdfPioneerIK(PioneerKinematics):
    """Legacy position-nullspace and two-axis orientation solvers."""
    def __init__(self, names, velocity=.9, wrist_velocity=None, orientation_mode="wrist"):
        if orientation_mode not in ("wrist", "full"):
            raise ValueError("Orientation mode must be 'wrist' or 'full'")
        super().__init__(names, velocity, wrist_velocity)
        self.orientation_mode = orientation_mode

    def solve(self, qpos, target, dt):
        if not target.active:
            return self.command.copy()
        actual_p, actual_r, _, _ = self.kinematics(qpos)
        # The reference solver warm-starts from the previous solution. Measured
        # FK still drives target anti-windup and the tracking diagnostics.
        seed = qpos.copy()
        seed[self.qids] = self.command
        q = self.command.copy()
        p, r, jp, _ = self.kinematics(seed)
        lower = np.maximum(self.limits[:, 0], q-self.velocity*dt)
        upper = np.minimum(self.limits[:, 1], q+self.velocity*dt)
        # Unlike the reference 3-DoF position block, ours has a redundant
        # shoulder DOF. Pull only the elbow toward its bent rest branch; pulling
        # the shoulder redundancy home would fight orientation every tick.
        posture = np.zeros(4)
        posture[3] = self.rest[3]-q[3]
        dp, rho_p = bounded_dls(jp[:, :4], target.position-p,
                                lower[:4]-q[:4], upper[:4]-q[:4], bias=posture)
        intermediate = seed.copy()
        intermediate[self.qids[:4]] = np.clip(q[:4]+dp, lower[:4], upper[:4])
        # Recompute at the position command we can actually send this tick,
        # including velocity/position bounds, not an uncapped hypothetical pose.
        _, r, jp, jr = self.kinematics(intermediate)
        rotation_error = rotation_vector(target.rotation @ r.T)
        parked = np.linalg.norm(rotation_error) > 2.2
        candidate = intermediate[self.qids].copy()
        step = np.zeros(6)
        rho_r = 0.
        locked = []
        if not parked:
            # Work in an orthonormal nullspace basis. When a joint is already
            # against a position/rate bound and the solve pushes outward, lock
            # that direction and resolve using the remaining DOFs. Independent
            # clipping after projection would destroy the position constraint.
            for _ in range(7):
                # A soft movement penalty alone still integrates into proximal
                # rotation over successive ticks. Wrist mode excludes proximal
                # orientation steps entirely and accepts angular residuals.
                fixed = list(range(4)) if self.orientation_mode == "wrist" else []
                fixed = sorted(set(fixed+locked))
                constraints = np.vstack([jp, np.eye(6)[fixed]]) if fixed else jp
                _, singular, vh = np.linalg.svd(constraints, full_matrices=True)
                rank = int(np.sum(singular > 1e-5))
                basis = vh[rank:].T
                if basis.shape[1] == 0:
                    step = np.zeros(6)
                    rho_r = 1.
                    break
                # A two-axis wrist is intentionally rank two. Treating its
                # missing third axis as a singularity would over-damp every
                # valid wrist motion.
                orient, rho_r = dls(jr @ basis, rotation_error, threshold=.2,
                                    task_rank=2 if self.orientation_mode == "wrist" else 3)
                step = basis @ orient
                outward = np.flatnonzero(
                    ((candidate <= lower+1e-10) & (step < -1e-10)) |
                    ((candidate >= upper-1e-10) & (step > 1e-10)))
                new_locks = [int(i) for i in outward if i not in locked]
                if not new_locks:
                    break
                locked.extend(new_locks)
        # Uniformly shorten the orientation step to the first bound, preserving
        # its direction in the position nullspace. Final clip is roundoff only.
        fraction = 1.
        for i, amount in enumerate(step):
            if amount > 1e-12:
                fraction = min(fraction, (upper[i]-candidate[i])/amount)
            elif amount < -1e-12:
                fraction = min(fraction, (lower[i]-candidate[i])/amount)
        fraction = max(0., fraction)
        self.command = np.clip(candidate+fraction*step, lower, upper)
        margin = np.minimum(self.command-self.limits[:, 0], self.limits[:, 1]-self.command)
        self.warning = float(np.clip(max(rho_p, rho_r, 1-margin.min()/.1,
                                         target.saturation-1, float(parked)), 0., 1.))
        sent_pose = seed.copy()
        sent_pose[self.qids] = self.command
        _, sent_rotation, _, _ = self.kinematics(sent_pose)
        requested_rotation = target.requested_rotation if target.requested_rotation is not None else target.rotation
        self.diagnostics = {
            "orientation_mode": self.orientation_mode,
            "position_error_m": float(np.linalg.norm(target.position-actual_p)),
            "orientation_error_deg": float(np.degrees(np.linalg.norm(rotation_vector(target.rotation @ actual_r.T)))),
            "absorbing_motion": target.position_limited,
            "angular_lead_limited": target.orientation_limited,
            "controller_rotation_delta_deg": target.controller_rotation_delta_deg,
            "controller_rotvec_deg": (np.degrees(rotation_vector(target.previous_rotation)).tolist()
                                       if target.previous_rotation is not None else None),
            "requested_rotvec_deg": np.degrees(rotation_vector(requested_rotation)).tolist(),
            "engagement_count": target.engagement_count,
            "requested_orientation_error_deg": float(np.degrees(np.linalg.norm(rotation_vector(requested_rotation @ actual_r.T)))),
            "requested_command_orientation_error_deg": float(np.degrees(np.linalg.norm(rotation_vector(requested_rotation @ sent_rotation.T)))),
            "orientation_parked": bool(parked),
            "orientation_damping": float(rho_r),
            "joint_limited": bool(margin.min() < np.radians(1)),
            "limited_joints": [int(i+1) for i in np.flatnonzero(margin < np.radians(1))],
            "command_orientation_error_deg": float(np.degrees(np.linalg.norm(rotation_vector(target.rotation @ sent_rotation.T)))),
            "actuator_orientation_error_deg": float(np.degrees(np.linalg.norm(rotation_vector(sent_rotation @ actual_r.T)))),
        }
        return self.command.copy()
