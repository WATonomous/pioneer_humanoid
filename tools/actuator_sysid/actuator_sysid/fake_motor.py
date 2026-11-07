"""Fake CubeMars motor for tests and dry runs: answers like the real one on a virtual or vcan bus.

A 1-DOF rigid body (armature + optional load) with Coulomb + viscous friction and optional gravity,
driven by the MIT law tau = kp*(p_des - p) + kd*(v_des - v) + t, clipped to peak torque.
It is NOT a fitted model (Task 12 does that); it only has to behave plausibly so the bench code can
be exercised. Behaviour not verified on hardware is marked "assumed".

AK 3.0: streams 0x29 (and optionally 0x2A) at feedback_hz whether enabled or not; the first MIT
command enables; mode 15 disables and answers one 0x29 with error 0x77; mode 5 sets the origin.
AK 2.0: ignores MIT commands until FF..FC (enter); replies to every command; FD exits; FE zeros.
Both (assumed): no command for can_timeout while enabled -> limp, like the R-Link CAN timeout.
"""
import math
import threading
import time

from pioneer_humanoid.leg_params import MotorSpec

from . import protocol as pr
from .bus import CanBus


class FakeMotor:
    def __init__(self, spec: MotorSpec, motor_id: int, interface: str = "virtual", channel: str = "sysid",
                 position: float = 0.0, load_inertia: float = 0.0, coulomb: float = 0.2,
                 viscous: float = 0.05, gravity: float = 0.0, feedback_hz: float = 1000.0,
                 position_frame: bool = False, can_timeout: float = 0.2, temperature: int = 30,
                 reply_to_special: bool = False, substep: float = 2e-4):
        self.spec, self.motor_id = spec, motor_id
        self.inertia = spec.armature + load_inertia  # kg*m^2 at the output
        self.coulomb, self.viscous, self.gravity = coulomb, viscous, gravity  # Nm, Nm*s/rad, m*g*l (Nm)
        self.feedback_hz, self.position_frame, self.can_timeout = feedback_hz, position_frame, can_timeout
        self.reply_to_special, self.substep = reply_to_special, substep
        # State. theta = true angle; the motor reports theta - zero_offset.
        self.theta, self.omega, self.zero_offset = position, 0.0, 0.0
        self.enabled = False  # producing MIT torque
        self.mit_mode = spec.protocol == "ak3"  # AK 2.0 needs the enter frame
        self.command = pr.MitCommand(0.0, 0.0, 0.0, 0.0, 0.0)
        self.torque = 0.0  # last motor torque (Nm)
        # Fault injection (set from tests).
        self.temperature = temperature
        self.error = 0
        self.mute = False  # stop all feedback (watchdog tests)
        self.external_torque = 0.0  # Nm, e.g. a hand pushing
        # Counters for tests.
        self.received = []  # (kind, MitCommand or None)
        self._last_cmd_time = None
        self._ack_pending = False
        self._bus = CanBus(interface, channel, echo=False)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"fake-{spec.name}-{motor_id}", daemon=True)

    # --- lifecycle -------------------------------------------------------------------------------
    def start(self) -> "FakeMotor":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._bus.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    @property
    def position(self) -> float:
        """Reported position (rad)."""
        return self.theta - self.zero_offset

    # --- loop ------------------------------------------------------------------------------------
    def _run(self) -> None:
        last = time.monotonic()
        next_fb = last
        while not self._stop.is_set():
            rx = self._bus.recv(0.0002)
            while rx is not None:
                self._handle(rx.frame)
                rx = self._bus.recv(0.0)
            now = time.monotonic()
            self._integrate(min(now - last, 0.05))
            last = now
            if (self.enabled and self._last_cmd_time is not None
                    and now - self._last_cmd_time > self.can_timeout):
                self.enabled = False  # assumed CAN-timeout behaviour
            if self.spec.protocol == "ak3":
                if self._ack_pending:
                    self._ack_pending = False
                    self._send_ak3_status(error=0x77)
                if now >= next_fb:
                    self._send_ak3_status(self.error)
                    if self.position_frame and not self.mute:
                        self._bus.send(pr.encode_ak3_position(self.motor_id, self.position))
                    next_fb = max(next_fb + 1.0 / self.feedback_hz, now)

    def _integrate(self, dt: float) -> None:
        steps = max(1, int(dt / self.substep))
        h = dt / steps
        for _ in range(steps):
            c = self.command
            if self.enabled:
                tau_m = c.kp * (c.p - self.position) + c.kd * (c.v - self.omega) + c.t
                tau_m = max(-self.spec.peak_torque, min(self.spec.peak_torque, tau_m))
            else:
                tau_m = 0.0
            self.torque = tau_m
            tau = tau_m + self.external_torque - self.gravity * math.sin(self.theta) - self.viscous * self.omega
            if self.omega == 0.0 and abs(tau) <= self.coulomb:
                continue  # stuck (static friction)
            direction = math.copysign(1.0, self.omega if self.omega != 0.0 else tau)
            new_omega = self.omega + (tau - self.coulomb * direction) / self.inertia * h
            if self.omega != 0.0 and new_omega * self.omega < 0:
                new_omega = 0.0  # friction stopped it this step
            self.omega = new_omega
            self.theta += new_omega * h

    # --- frames ----------------------------------------------------------------------------------
    def _handle(self, frame: pr.CanFrame) -> None:
        if self.spec.protocol == "ak3":
            if not frame.is_extended_id:
                return
            mode, drive_id = pr.ak3_split_id(frame.arbitration_id)
            if drive_id != self.motor_id:
                return
            if mode == pr.AK3_MODE_MIT and len(frame.data) == 8:
                self._accept(pr.decode_mit(self.spec, pr.unpack_ak3_fields(frame.data)))
            elif mode == pr.AK3_MODE_DISABLE:
                self.enabled = False
                self._ack_pending = True
                self.received.append(("disable", None))
            elif mode == pr.AK3_MODE_SET_ORIGIN:
                self.zero_offset = self.theta
                self.received.append(("zero", None))
            return
        # AK 2.0
        if frame.is_extended_id or frame.arbitration_id != self.motor_id or len(frame.data) != 8:
            return
        special = pr.ak2_special_code(frame)
        if special is not None:
            kind = {pr.AK2_ENTER: "enter", pr.AK2_EXIT: "exit", pr.AK2_SET_ZERO: "zero"}[special]
            self.received.append((kind, None))
            if special == pr.AK2_ENTER:
                self.mit_mode = True
            elif special == pr.AK2_EXIT:
                self.mit_mode = self.enabled = False
            else:
                self.zero_offset = self.theta
            if self.reply_to_special:
                self._send_ak2_feedback()
            return
        if self.mit_mode:
            self._accept(pr.decode_mit(self.spec, pr.unpack_ak2_fields(frame.data)))
            self._send_ak2_feedback()

    def _accept(self, cmd: pr.MitCommand) -> None:
        self.command, self.enabled = cmd, True
        self._last_cmd_time = time.monotonic()
        self.received.append(("mit", cmd))

    def _send_ak3_status(self, error: int) -> None:
        if self.mute:
            return
        self._bus.send(pr.encode_ak3_status(self.spec, self.motor_id, self.position, self.omega,
                                            self.torque / self.spec.torque_constant, self.temperature, error))

    def _send_ak2_feedback(self) -> None:
        if self.mute:
            return
        self._bus.send(pr.encode_ak2_feedback(self.spec, self.motor_id, self.position, self.omega,
                                              self.torque, self.temperature, self.error))
