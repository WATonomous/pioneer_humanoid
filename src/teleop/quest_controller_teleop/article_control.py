"""Controller-to-target mapping (AnchoredTarget), adapted from Aurel Arnold's VR teleoperation article. The IK is mink_control.py.

The article holds a grip "clutch" to engage incremental control and releases it to reposition the hand, because its
incremental mapping lets hand and arm drift apart. Measuring every frame from the engagement pose removes the drift,
so control here is always on while the controller and the headset are tracked: no clutch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from control import cap_vector, rotation_exp, rotation_vector


@dataclass
class AnchoredTarget:
    """Arm target = engagement target + gain x (hand now - hand at engagement), at most the lead limits ahead of the arm.

    It engages at the measured arm pose on the first frame with the controller and headset tracked (again after a
    loss, a reset or a calibration), so the arm never jumps. The lead limit caps only what is sent to the IK this
    frame, not the request: once the arm catches up, it lines up with the hand again. Pressing or releasing
    precision (A/X, 25% motion) re-measures from the current request, so the arm never jumps.
    """
    gain: float = 1.5
    max_position_error: float = .08   # m the IK goal may lead the measured wrist
    max_angle_error: float = .6       # rad
    rotation_gain: float = 1.
    # Fixed Quest-world -> robot-base calibration.
    calibration: np.ndarray = field(default_factory=lambda: np.array(
        [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]]))
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))   # wrist pivot in controller grip coordinates
    active: bool = False
    position: np.ndarray | None = None   # IK goal (lead-limited)
    rotation: np.ndarray | None = None
    mapping: np.ndarray = field(default_factory=lambda: np.eye(3))
    saturation: float = 0.

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
        self.active = False

    def update(self, controller, head_rotation, actual_position, actual_rotation):
        if controller is None or head_rotation is None:
            self.lose_tracking()
            return
        pivot_position = controller.position + controller.rotation @ self.pivot
        if not self.active:
            self._engage(pivot_position, controller.rotation, head_rotation, actual_position, actual_rotation)
            return
        precision = .25 if controller.precision else 1.
        if precision != self._precision:
            self._anchor(self.requested_position, self.requested_rotation, precision)
        hand_turn = rotation_vector(self.mapping @ (controller.rotation @ self._hand_rotation.T) @ self.mapping.T)
        self.requested_position = self._target_position + self.mapping @ (pivot_position-self._hand_position)*self.gain*precision
        self.requested_rotation = rotation_exp(hand_turn*self.rotation_gain*precision) @ self._target_rotation
        self._cap_lead(self.requested_position, self.requested_rotation, actual_position, actual_rotation)
        self.previous_position = pivot_position.copy()
        self.previous_rotation = controller.rotation.copy()

    def _engage(self, pivot_position, hand_rotation, head_rotation, actual_position, actual_rotation):
        """Start control at the measured arm pose, with the control axes turned to the headset's yaw."""
        # Horizontal headset forward; pitch/roll never tilt the control axes.
        forward = head_rotation @ np.array([0., 0., -1.])
        if np.linalg.norm(forward[[0, 2]]) < .1:
            return
        yaw = math.atan2(-forward[0], -forward[2])
        self.mapping = self.calibration @ rotation_exp(np.array([0., -yaw, 0.]))
        self.position = actual_position.copy()
        self.rotation = actual_rotation.copy()
        self.saturation = 0.
        self.active = True
        self.previous_position = pivot_position.copy()
        self.previous_rotation = hand_rotation.copy()
        self._anchor(self.position, self.rotation, 1.)

    def _anchor(self, target_position, target_rotation, precision):
        """Measure later hand motion from the last hand pose, starting at this target, at this precision scale."""
        self._hand_position, self._hand_rotation = self.previous_position.copy(), self.previous_rotation.copy()
        self._target_position, self._target_rotation = target_position.copy(), target_rotation.copy()
        self.requested_position, self.requested_rotation = target_position.copy(), target_rotation.copy()
        self._precision = precision

    def _cap_lead(self, position, rotation, actual_position, actual_rotation):
        """Set the IK goal to the requested pose, at most the lead limits away from the measured one."""
        ep = position-actual_position
        er = rotation_vector(rotation @ actual_rotation.T)
        self.saturation = float(max(np.linalg.norm(ep)/self.max_position_error,
                                    np.linalg.norm(er)/self.max_angle_error))
        self.position = actual_position+cap_vector(ep, self.max_position_error)
        self.rotation = rotation_exp(cap_vector(er, self.max_angle_error)) @ actual_rotation
