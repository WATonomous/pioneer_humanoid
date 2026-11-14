"""Quest teleop IK on mink (QP differential IK on MuJoCo) for pioneer_bimanual_arm_8dof.

Why (IK benchmark on the 8dof arm, simulation, mock Quest operator): against a sequential-DLS teleop IK, mink solved
82 % vs 51 % of random poses, returned to the start pose after wandering (0.4 vs 8-28 deg of joint drift) and held
hand orientation about 3x better. OpenArm's VR teleop also uses mink.

Each tick, one QP:
  minimise  |wrist-pivot position error|^2 + orientation_weight |gripper orientation error|^2
            + posture_weight |q - rest|^2 + damping
            (the 7th joint lets the home pull keep the elbow natural without costing tracking)
  subject to, per joint: the limit and this tick's speed cap, as ONE box on the step
             (a joint already outside its limit is driven back at its speed cap, never in one jump);
             optionally, upper arm / forearm capsules >= 2 cm from a torso keep-out box (base frame).
If the QP is infeasible with the keep-out, it is re-solved without it; if that also fails, the command holds.
The MuJoCo model is built from the URDF chain (no meshes): kinematics only.
"""
from __future__ import annotations

import math

import numpy as np

from control import PioneerKinematics, rotation_vector

# Approximate humanoid torso under the shoulder bar, base_link frame (x forward, y left, z up), metres.
# From the arm-shell session's draft-frame box (x +-0.15, y +-0.09, z 0.22..0.72 m in the draft frame) mapped with
# draft = (-y*1000, (x-0.02157)*1000+575, (z-0.06048)*1000+679) mm. The bimanual stand has no torso, so it is OFF by
# default (--body-keepout); on the stand it blocked legitimate moves toward the middle. Replace with the real torso.
TORSO_CENTRE = (0.0216, 0.0, -0.148)
TORSO_HALF = (0.09, 0.15, 0.25)


def _quat(R):
    """wxyz quaternion of a rotation matrix."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


class _JointBox:
    """mink limit: lo - q <= dq <= hi - q, and |dq| <= v dt, per arm joint, with recovery when outside."""

    def __init__(self, owner):
        self.owner = owner

    def compute_qp_inequalities(self, configuration, dt):
        import mink
        o = self.owner
        q = configuration.q
        step = o.velocity * dt
        upper = np.minimum(o.limits[:, 1] - q, step)
        lower = np.maximum(o.limits[:, 0] - q, -step)
        upper = np.maximum(upper, lower)   # outside the limit: the bound pushes back in at the speed cap
        n = len(q)
        return mink.Constraint(G=np.vstack([np.eye(n), -np.eye(n)]), h=np.concatenate([upper, -lower]))


class PioneerMinkIK(PioneerKinematics):
    def __init__(self, names, velocity=.9, wrist_velocity=None, orientation_weight=0.2, posture_weight=0.02,
                 body_keepout=False):
        super().__init__(names, velocity, wrist_velocity)
        import mink
        import mujoco
        self.mink, self.mujoco = mink, mujoco
        self.body_keepout = body_keepout
        self.model = mujoco.MjModel.from_xml_string(self._mjcf())
        self.cfg = mink.Configuration(self.model)
        # Position at the wrist pivot (the wrist-pitch joint's origin), orientation of the last link (the gripper): the
        # pivot sits 6 cm before the wrist-yaw joint, so one frame task on the last link would track the wrong point.
        self.task = mink.FrameTask("anchor", "site", position_cost=1.0, orientation_cost=0.0, lm_damping=1e-6)
        self.orient = mink.FrameTask("tool", "site", position_cost=0.0, orientation_cost=orientation_weight,
                                     lm_damping=1e-6)
        self.posture = mink.PostureTask(self.model, cost=posture_weight)
        self.posture.set_target(self.rest.copy())
        self.box = _JointBox(self)
        self.keepout = (mink.CollisionAvoidanceLimit(
            self.model, [(["upper_arm", "forearm", "wrist"], ["torso"])],
            minimum_distance_from_collisions=0.02, collision_detection_distance=0.12) if body_keepout else None)
        self.infeasible = 0

    def _mjcf(self):
        bodies, close = [], []
        chain = [o[:3, 3] for o in self.origins[1:]] + [np.zeros(3)]   # each body's child origin, in its frame
        capsules = {2: ("upper_arm", .05), 3: ("forearm", .045), 4: ("wrist", .04)}
        for i, (origin, axis) in enumerate(zip(self.origins, self.axes)):
            p, w = origin[:3, 3], _quat(origin[:3, :3])
            bodies.append(f'<body name="b{i}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
                          f'quat="{w[0]:.8f} {w[1]:.8f} {w[2]:.8f} {w[3]:.8f}">'
                          f'<joint name="j{i}" axis="{axis[0]:.8f} {axis[1]:.8f} {axis[2]:.8f}" limited="false"/>'
                          f'<inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>')
            if self.body_keepout and i in capsules:
                name, r = capsules[i]
                c = chain[i]
                bodies.append(f'<geom name="{name}" type="capsule" fromto="0 0 0 {c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" '
                              f'size="{r}" contype="1" conaffinity="1"/>')
            if i == self.anchor_joint:
                bodies.append('<site name="anchor" pos="0 0 0"/>')
            close.append("</body>")
        bodies.append('<site name="tool" pos="0 0 0"/>')
        c, h = TORSO_CENTRE, TORSO_HALF
        torso = (f'<geom name="torso" type="box" pos="{c[0]} {c[1]} {c[2]}" size="{h[0]} {h[1]} {h[2]}" '
                 f'contype="1" conaffinity="1"/>' if self.body_keepout else "")
        return (f'<mujoco><compiler angle="radian"/><worldbody>{torso}{"".join(bodies)}{"".join(close)}'
                f'</worldbody></mujoco>')

    def solve(self, qpos, target, dt):
        if not target.active:
            return self.command.copy()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("IK timestep must be positive and finite")
        mink = self.mink
        seed = self.command.copy()
        self.cfg.update(seed)
        goal = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(target.rotation), target.position)
        self.task.set_target(goal)
        self.orient.set_target(goal)
        tasks = [self.task, self.orient, self.posture]
        limits = [self.box] + ([self.keepout] if self.keepout is not None else [])
        try:
            v = mink.solve_ik(self.cfg, tasks, dt, "daqp", damping=1e-4, limits=limits)
        except mink.exceptions.NoSolutionFound:
            self.infeasible += 1
            try:   # e.g. already inside the keep-out after a reset: move on joint limits/speed alone
                v = mink.solve_ik(self.cfg, tasks, dt, "daqp", damping=1e-4, limits=[self.box])
            except mink.exceptions.NoSolutionFound:
                v = np.zeros(len(seed))
        step = self.velocity * dt
        lo = np.minimum(np.maximum(self.limits[:, 0], seed - step), seed + step)
        hi = np.maximum(np.minimum(self.limits[:, 1], seed + step), seed - step)
        self.command = np.clip(seed + v * dt, lo, hi)   # roundoff guard; the QP already respects the box
        p, r, _, _ = self.kinematics(self.command)
        margin = np.minimum(self.command - self.limits[:, 0], self.limits[:, 1] - self.command)
        pos_err = float(np.linalg.norm(target.position - p))
        ori_err = float(np.linalg.norm(rotation_vector(target.rotation @ r.T)))
        self.warning = float(np.clip(max(1 - margin.min() / .1, target.saturation - 1., pos_err / .08,
                                         ori_err / 1.), 0., 1.))
        self.diagnostics = {"position_error_m": pos_err, "orientation_error_deg": math.degrees(ori_err),
                            "limited_joints": [int(i + 1) for i in np.flatnonzero(margin < np.radians(1))],
                            "infeasible_ticks": self.infeasible}
        return self.command.copy()
