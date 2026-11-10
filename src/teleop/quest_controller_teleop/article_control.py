"""Fresh sequential-DLS controller based on Aurel Arnold's article/source.

Reference: Dream-Machines-Robotics/vr-teleop-kit, commit 8d37f3d6bd44c8ce646f7bc110ac98e802e0ab57.
Pioneer differs from DK1: its functional position group is joints 1/2/4,
and its three-axis orientation group is 3/5/6. Joint 3 also moves the wrist
anchor, so position is corrected on subsequent ticks, not exactly decoupled.
The mapper targets the wrist anchor directly instead of a tool-tip pose.
"""
from __future__ import annotations

import math
import numpy as np

from control import PioneerKinematics, Target, cap_vector, rotation_exp, rotation_vector


class ArticleTarget(Target):
    """Article's incremental clutch mapping, with absorbed lead on both channels.

Reuses only calibration/tracking state and rotation utilities from Target;
the previous persistent-orientation update is not used here.
"""
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
        pivot_position = controller.position + controller.rotation @ self.pivot
        if not self.active:
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
            precision = .25 if controller.precision else 1.
            dp = self.mapping @ (pivot_position-self.previous_position)*self.gain*precision
            dr = self.mapping @ (controller.rotation @ self.previous_rotation.T) @ self.mapping.T
            angle_increment = rotation_vector(dr)
            self.controller_rotation_delta_deg = float(np.degrees(np.linalg.norm(angle_increment)))
            increment = rotation_exp(angle_increment*self.rotation_gain*precision)
            # Accumulate onto the accepted target, not an unbounded request.
            proposed_rotation = increment @ self.rotation
            self.requested_rotation = proposed_rotation.copy()
            ep = self.position+dp-actual_position
            er = rotation_vector(proposed_rotation @ actual_rotation.T)
            self.position_limited = bool(np.linalg.norm(ep) > self.max_position_error)
            self.orientation_limited = bool(np.linalg.norm(er) > self.max_angle_error)
            self.saturation = float(max(np.linalg.norm(ep)/self.max_position_error,
                                        np.linalg.norm(er)/self.max_angle_error))
            self.position = actual_position+cap_vector(ep, self.max_position_error)
            self.rotation = rotation_exp(cap_vector(er, self.max_angle_error)) @ actual_rotation
        # Consume the input even when blocked: reversing acts immediately.
        self.previous_position = pivot_position.copy()
        self.previous_rotation = controller.rotation.copy()


def article_dls(jacobian, error, baseline, extra, threshold, rest_delta=None):
    """The article's 3x3 normal equations and independent damping ramp."""
    j = np.asarray(jacobian, dtype=float)
    w = abs(float(np.linalg.det(j)))
    ramp = max(0., 1.-w/threshold)
    damping = baseline**2+extra**2*ramp**2
    # Robot-specific weak position branch bias; no orientation/home spring.
    mu2 = .002**2 if rest_delta is not None else 0.
    rhs = j.T @ error
    if rest_delta is not None:
        rhs += mu2*rest_delta
    step = np.linalg.solve(j.T @ j+(damping+mu2)*np.eye(3), rhs)
    return step, ramp


class PioneerArticleIK(PioneerKinematics):
    """Position DLS, updated FK, orientation DLS, hard physical/rate bounds.

No nullspace solve, per-joint orientation preference, or discarded third axis.
Pioneer's orientation group is full rank at its bent-elbow home pose. A
literal DK1 joint-index split would be singular there on this robot.
"""
    position_ids = np.array([0, 1, 3])
    orientation_ids = np.array([2, 4, 5])
    orientation_mode = "article"

    def solve(self, qpos, target, dt):
        if not target.active:
            return self.command.copy()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("IK timestep must be positive and finite")
        seed = self.command.copy()
        lower = np.maximum(self.limits[:, 0], seed-self.velocity*dt)
        upper = np.minimum(self.limits[:, 1], seed+self.velocity*dt)
        p, _, jp, _ = self.kinematics(seed)
        ids = self.position_ids
        # Stabilize the elbow branch near straight-arm poses. With no task
        # error there is no reason to change the holding configuration.
        rest_delta = self.rest[ids]-seed[ids] if np.linalg.norm(target.position-p) > .001 else None
        dp, rho_p = article_dls(jp[:, ids], target.position-p,
                                 .05, .15, .015, rest_delta)
        candidate = seed.copy()
        candidate[ids] = np.clip(seed[ids]+dp, lower[ids], upper[ids])

        # Orientation sees exactly the bounded position update sent this tick.
        _, r, _, jr = self.kinematics(candidate)
        er = rotation_vector(target.rotation @ r.T)
        parked = bool(np.linalg.norm(er) > 2.2)
        ids = self.orientation_ids
        dr, rho_r = article_dls(jr[:, ids], er, .05, .4, .5)
        if parked:
            dr[:] = 0.
        unconstrained = candidate.copy()
        unconstrained[ids] += dr
        candidate[ids] = np.clip(unconstrained[ids], lower[ids], upper[ids])
        self.command = candidate

        margin = np.minimum(candidate-self.limits[:, 0], self.limits[:, 1]-candidate)
        self.warning = float(np.clip(max(rho_p, rho_r, 1-margin.min()/.1,
                                         target.saturation-1., float(parked)), 0., 1.))
        return self.command.copy()
