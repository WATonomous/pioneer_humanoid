"""One emulated CubeMars drive: what the firmware does on the far side of the CAN bus.

The MIT law itself is not in the repo (it runs in the drive), so it is written here from the manuals'
torque = kp*(p_des - p) + kd*(v_des - v) + t_ff. Packing uses the arm's real mit_protocol.cpp, so a
command reaches the law with the rounding it would have on the bus. Family differences (mit_protocol.hpp,
mit_profiles.yaml, can_node.cpp):

    ak   no ENTER/EXIT, status 0 = ok, feedback is the servo frame: 0.1 deg, 10 ERPM, 0.01 A; torque = current * kt
    gl2  ignores MIT until ENTER, limp on EXIT, status 1 = enabled, feedback is the MIT frame: 16/12-bit fields

Both answer only when spoken to: one feedback per command received. A GL II that hasn't entered motor
mode ignores MIT frames, so it stays silent too: ASSUMED from MotorCmd.msg ("ignores MIT commands until
MIT_ENTER"), to check on the bench. Otherwise joint_command would seed from a drive that isn't holding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from arm_emulator import mit_native
from arm_emulator.config import MotorSpec

# common_msgs/MotorCmd control types this arm uses.
MIT_CONTROL, MIT_ENTER, MIT_EXIT = 0, 9, 10
GL2_DISABLED, GL2_ENABLED = 0, 1
AK_MAX_CURRENT = 60.0  # A, humanoid.dbc FbkCurrent range


@dataclass
class Feedback:
    """One MotorFeedback, in can_node's units."""

    motor_id: int
    position: float     # deg, motor frame
    velocity: float     # AK: ERPM; GL II: "r/s"
    current: float      # A; 0 on GL II
    torque: float       # N.m
    temperature: int    # deg C
    error_code: int


class Drive:
    def __init__(self, spec: MotorSpec, ambient: float, can_timeout_s: float | None) -> None:
        self.spec = spec
        self.is_ak = spec.family == "ak"
        self.can_timeout_s = can_timeout_s if self.is_ak else None
        self.enabled = self.is_ak  # a GL II waits for MIT_ENTER
        self.p, self.v, self.kp, self.kd, self.t_ff = 0.0, 0.0, 0.0, 0.0, 0.0
        self.last_command_t: float | None = None
        self.ambient = ambient
        self.temperature = ambient
        self.tau = 0.0
        # Faults (faults.py).
        self.fault_code: int | None = None
        self.mute = False            # no feedback frames
        self.torque_scale = 1.0      # 0: the output has failed
        self.zero_shift_deg = 0.0    # the encoder's zero moved
        self.heat_offset = 0.0       # deg C

    def _encoder(self, q: float) -> float:
        return q + math.radians(self.zero_shift_deg)

    def command(self, control_type: int, position: float, velocity: float, kp: float, kd: float, torque: float,
                now: float) -> bool:
        """A MotorCmd as can_node would forward it; -> whether a frame reached the drive."""
        if control_type == MIT_ENTER:
            self.enabled = True
        elif control_type == MIT_EXIT:
            self.enabled = self.is_ak  # an AK has no motor mode to leave
            if not self.is_ak:
                self.kp, self.kd, self.t_ff = 0.0, 0.0, 0.0
        elif control_type == MIT_CONTROL:
            # can_node refuses these before they reach the bus.
            if not all(map(math.isfinite, (position, velocity, kp, kd, torque))):
                return False
            if not self.enabled:
                return False
            self._decode(mit_native.pack_command(self.is_ak, position, velocity, kp, kd, torque, self.spec.profile))
        else:
            return False
        self.last_command_t = now
        return True

    def _decode(self, frame: bytes) -> None:
        """The drive unpacking the payload (byte layouts: mit_protocol.hpp)."""
        b, pr = frame, self.spec.profile
        if self.is_ak:
            kp_i = (b[0] << 4) | (b[1] >> 4)
            kd_i = ((b[1] & 0xF) << 8) | b[2]
            p_i = (b[3] << 8) | b[4]
            v_i = (b[5] << 4) | (b[6] >> 4)
            t_i = ((b[6] & 0xF) << 8) | b[7]
        else:
            p_i = (b[0] << 8) | b[1]
            v_i = (b[2] << 4) | (b[3] >> 4)
            kp_i = ((b[3] & 0xF) << 8) | b[4]
            kd_i = (b[5] << 4) | (b[6] >> 4)
            t_i = ((b[6] & 0xF) << 8) | b[7]
        self.p = mit_native.unpack_value(p_i, pr["p_min"], pr["p_max"], 16)
        self.v = mit_native.unpack_value(v_i, pr["v_min"], pr["v_max"], 12)
        self.kp = mit_native.unpack_value(kp_i, 0.0, pr["kp_max"], 12)
        self.kd = mit_native.unpack_value(kd_i, 0.0, pr["kd_max"], 12)
        # A feed-forward of 0 lands on the middle code, a hair off zero: the same happens on the bus.
        self.t_ff = mit_native.unpack_value(t_i, pr["t_min"], pr["t_max"], 12)

    def step(self, q: float, qd: float, now: float, dt: float) -> float:
        """The motor-frame torque this tick (N.m), from its true angle (rad) and speed (rad/s)."""
        tau = 0.0
        timed_out = (self.can_timeout_s is not None and self.last_command_t is not None
                     and now - self.last_command_t > self.can_timeout_s)
        if self.enabled and self.fault_code is None and not timed_out and self.last_command_t is not None:
            tau = self.kp * (self.p - self._encoder(q)) + self.kd * (self.v - qd) + self.t_ff
            tau = self._motor_limits(tau, qd) * self.torque_scale
        self.tau = tau
        self._heat(tau, dt)
        return tau

    def _motor_limits(self, tau: float, qd: float) -> float:
        """Peak torque, falling to zero at no-load speed in the direction of travel."""
        peak, v0 = self.spec.model["peak_torque"], self.spec.model["no_load_speed"]
        high = min(max(peak * (1.0 - qd / v0), 0.0), peak)
        low = max(min(peak * (-1.0 - qd / v0), 0.0), -peak)
        return min(max(tau, low), high)

    def _heat(self, tau: float, dt: float) -> None:
        m = self.spec.model
        target = self.ambient + m["thermal_rise_at_rated"] * (tau / m["rated_torque"]) ** 2
        self.temperature += (target - self.temperature) * dt / m["thermal_tau"]

    def feedback(self, q: float, qd: float) -> Feedback | None:
        """The frame answering a command, rounded as it would be on the bus; None when muted."""
        if self.mute:
            return None
        encoder = self._encoder(q)
        temperature = int(self.temperature + self.heat_offset)
        pr = self.spec.profile
        if self.is_ak:
            # Servo frame factors: humanoid.dbc.
            position = round(math.degrees(encoder) / 0.1) * 0.1
            erpm = qd * 60.0 / (2.0 * math.pi) * self.spec.model["pole_pairs"] * self.spec.model["gear_ratio"]
            current = max(min(self.tau / pr["kt"], AK_MAX_CURRENT), -AK_MAX_CURRENT)
            current = round(current / 0.01) * 0.01
            # arm_actuators.yaml: "AK80-9: its MIT torque reading saturates at 18".
            torque = max(min(current * pr["kt"], pr["t_max"]), pr["t_min"])
            return Feedback(self.spec.motor_id, position, round(erpm / 10.0) * 10.0, current, torque, temperature,
                            self.fault_code or 0)
        position = mit_native.unpack_value(mit_native.pack_value(encoder, pr["p_min"], pr["p_max"], 16),
                                           pr["p_min"], pr["p_max"], 16)
        velocity = mit_native.unpack_value(mit_native.pack_value(qd, pr["v_min"], pr["v_max"], 12),
                                           pr["v_min"], pr["v_max"], 12)
        torque = mit_native.unpack_value(mit_native.pack_value(self.tau, pr["t_min"], pr["t_max"], 12),
                                         pr["t_min"], pr["t_max"], 12)
        status = self.fault_code if self.fault_code is not None else (GL2_ENABLED if self.enabled else GL2_DISABLED)
        return Feedback(self.spec.motor_id, math.degrees(position), velocity, 0.0, torque, temperature, status)
