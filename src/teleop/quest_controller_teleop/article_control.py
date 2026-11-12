"""Controller-to-target mapping, from Aurel Arnold's VR teleoperation article: ArticleTarget (incremental clutch
mapping with lead limits) and AnchoredTarget (no drift: measured from the engagement pose). The IK is mink_control.py.
"""
from __future__ import annotations

import math
import numpy as np

from control import Target, cap_vector, rotation_exp, rotation_vector


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
            self._cap_lead(self.position+dp, proposed_rotation, actual_position, actual_rotation)
        # Consume the input even when blocked: reversing acts immediately.
        self.previous_position = pivot_position.copy()
        self.previous_rotation = controller.rotation.copy()

    def _cap_lead(self, position, rotation, actual_position, actual_rotation):
        """Set the IK goal to the requested pose, at most the lead limits away from the measured one."""
        ep = position-actual_position
        er = rotation_vector(rotation @ actual_rotation.T)
        self.position_limited = bool(np.linalg.norm(ep) > self.max_position_error)
        self.orientation_limited = bool(np.linalg.norm(er) > self.max_angle_error)
        self.saturation = float(max(np.linalg.norm(ep)/self.max_position_error,
                                    np.linalg.norm(er)/self.max_angle_error))
        self.position = actual_position+cap_vector(ep, self.max_position_error)
        self.rotation = rotation_exp(cap_vector(er, self.max_angle_error)) @ actual_rotation


class AnchoredTarget(ArticleTarget):
    """ArticleTarget without the drift: the request is measured from the engagement pose, not accumulated.

    ArticleTarget adds each frame's hand increment to the last *accepted* target, so whenever the lead limit
    caps it, the excess is lost for good and hand and arm drift apart; the article re-grabs to realign. With
    always-on control there is no re-grab, so offsets only accumulate. Here the request is
    `engagement target + gain x (hand now - hand at engagement)`, and the lead limit caps only what is sent
    to the IK this frame: once the arm catches up, it lines up with the hand again. Pressing or releasing
    precision (A/X, 25% motion) re-measures from the current request, so the arm never jumps.
    """
    def update(self, controller, head_rotation, actual_position, actual_rotation):
        if not self.active or controller is None or head_rotation is None or not controller.clutch:
            super().update(controller, head_rotation, actual_position, actual_rotation)  # engage, loss, release
            if self.active:  # just engaged at the measured pose: measure later hand motion from here
                self._anchor(self.position, self.rotation, 1.)
            return
        pivot_position = controller.position + controller.rotation @ self.pivot
        precision = .25 if controller.precision else 1.
        if precision != self._precision:
            self._anchor(self.requested_position, self.requested_rotation, precision)
        hand_turn = rotation_vector(self.mapping @ (controller.rotation @ self._hand_rotation.T) @ self.mapping.T)
        self.requested_position = self._target_position + self.mapping @ (pivot_position-self._hand_position)*self.gain*precision
        self.requested_rotation = rotation_exp(hand_turn*self.rotation_gain*precision) @ self._target_rotation
        self.controller_rotation_delta_deg = float(np.degrees(np.linalg.norm(
            rotation_vector(controller.rotation @ self.previous_rotation.T))))
        self._cap_lead(self.requested_position, self.requested_rotation, actual_position, actual_rotation)
        self.previous_position = pivot_position.copy()
        self.previous_rotation = controller.rotation.copy()

    def _anchor(self, target_position, target_rotation, precision):
        """Measure later hand motion from the last hand pose, starting at this target, at this precision scale."""
        self._hand_position, self._hand_rotation = self.previous_position.copy(), self.previous_rotation.copy()
        self._target_position, self._target_rotation = target_position.copy(), target_rotation.copy()
        self.requested_position, self.requested_rotation = target_position.copy(), target_rotation.copy()
        self._precision = precision
