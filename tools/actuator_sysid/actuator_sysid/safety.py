"""Safety layer: the only code in the bench tools that sends frames to a motor.

What it does:
  - Limits: BenchLimits (lower bench caps) under a hard ceiling from leg_params.MotorSpec. Raising
    a cap above its default needs a reason, which goes into the run metadata.
  - Every command is checked before it is sent (check_command). The motor computes
    kp*(p_des - p) + kd*(v_des - v) + t itself, so software can't clip its output torque; instead the
    worst case kp*max_track_error + kd*(|v_des| + max_speed) + |t| must fit under max_torque, and
    feedback faults keep the tracking error and speed inside those bounds.
  - Every feedback frame is checked while armed (check_feedback): position window, speed, torque,
    temperature, tracking error, motor error byte.
  - Watchdog: no feedback for limits.watchdog (50 ms) while armed -> fault.
  - Seed from live position: arm() reads the motor before the first command; commands must stay
    within position_window of that start, and (kp > 0) within max_track_error of the measured position.
  - Fault or exit (normal, exception, Ctrl-C, SIGTERM/SIGHUP): damp (kp 0, kd fault_kd) for
    damp_time, then disable (user decision: option C). Nothing is sent if the motor was never armed.
Not covered by software: a hung PC or killed process. Then only the motor's own CAN timeout
(R-Link, Task 10) and the hardware E-stop act.
"""
import gc
import math
import signal
import threading
import time
from dataclasses import asdict, dataclass, fields, replace

from pioneer_humanoid.leg_params import MotorSpec

from .bus import CanBus, wall_time
from .motors import Motor
from .protocol import CanFrame, MitCommand, MotorFeedback

ZERO_TORQUE = MitCommand(p=0.0, v=0.0, kp=0.0, kd=0.0, t=0.0)  # p is ignored when kp = 0


class SafetyFault(RuntimeError):
    """A check failed. The motor has been (or is being) damped and disabled."""


@dataclass(frozen=True)
class BenchLimits:
    """Bench caps, output shaft. Defaults = first power-on values, to confirm with the D11 reviewer."""
    max_torque: float = 5.0  # Nm: worst-case commanded torque and measured |torque|
    max_speed: float = 1.0  # rad/s: |v_des| and measured |v|
    max_kp: float = 10.0  # Nm/rad
    max_kd: float = 1.0  # Nm*s/rad
    position_window: float = 0.5  # rad: commands within start +- this
    position_margin: float = 0.1  # rad: measured position may overshoot the window by this
    max_track_error: float = 0.3  # rad: |p_des - p| when kp > 0
    max_temperature: float = 70.0  # degC, driver temperature
    watchdog: float = 0.05  # s without feedback while armed
    fault_kd: float = 1.0  # Nm*s/rad: damping after a fault, before disable
    damp_time: float = 1.0  # s

    def check_against(self, spec: MotorSpec) -> None:
        """Raise ValueError listing every limit outside the motor's datasheet ceiling."""
        errors = []
        for f in fields(self):
            value = getattr(self, f.name)
            if not math.isfinite(value) or value < 0:
                errors.append(f"{f.name} = {value} must be finite and >= 0")
        speed_ceiling = min(spec.no_load_speed, spec.v_max)  # MIT feedback saturates at v_max
        ceilings = {
            "max_torque": spec.peak_torque,
            "max_speed": speed_ceiling,
            "max_kp": spec.kp_max,
            "max_kd": spec.kd_max,
            "fault_kd": min(self.max_kd, spec.kd_max),
            "position_window": spec.p_max - self.position_margin,
            "watchdog": 0.05,  # plan: 50 ms watchdog; shorter is allowed, longer is not
            "damp_time": 5.0,
        }
        for name, ceiling in ceilings.items():
            if getattr(self, name) > ceiling + 1e-12:
                errors.append(f"{name} = {getattr(self, name):g} > ceiling {ceiling:g} ({spec.name})")
        for name in ("max_torque", "max_speed", "watchdog"):
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be > 0")
        if errors:
            raise ValueError("; ".join(errors))

    def raised_above_defaults(self) -> dict[str, float]:
        """Fields set above their default (more permissive or stronger damping)."""
        default = BenchLimits()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) > getattr(default, f.name)}

    def with_overrides(self, overrides: dict[str, float], reason: str | None = None) -> "BenchLimits":
        """New limits; raising any value above its default requires a reason (logged in meta.json)."""
        unknown = set(overrides) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"unknown limit(s): {sorted(unknown)}")
        new = replace(self, **{k: float(v) for k, v in overrides.items()})
        if new.raised_above_defaults() and not (reason and reason.strip()):
            raise ValueError(f"raising {sorted(new.raised_above_defaults())} above the bench defaults needs a reason")
        return new

    def to_dict(self) -> dict:
        return asdict(self)


def worst_case_torque(cmd: MitCommand, limits: BenchLimits) -> float:
    """Largest torque the motor can produce for this command while the feedback checks hold."""
    return cmd.kp * limits.max_track_error + cmd.kd * (abs(cmd.v) + limits.max_speed) + abs(cmd.t)


def check_command(cmd: MitCommand, limits: BenchLimits, spec: MotorSpec, start: float,
                  measured: float | None = None) -> None:
    """Raise ValueError (all problems listed) if the command may not be sent."""
    if not all(math.isfinite(x) for x in cmd):
        raise ValueError(f"non-finite command {cmd}")
    errors = []
    if not 0 <= cmd.kp <= limits.max_kp:
        errors.append(f"kp {cmd.kp:g} outside 0..{limits.max_kp:g}")
    if not 0 <= cmd.kd <= limits.max_kd:
        errors.append(f"kd {cmd.kd:g} outside 0..{limits.max_kd:g}")
    if abs(cmd.v) > limits.max_speed:
        errors.append(f"|v_des| {abs(cmd.v):.4g} > {limits.max_speed:g} rad/s")
    if abs(cmd.p - start) > limits.position_window + 1e-12:
        errors.append(f"p_des {cmd.p:.4f} is {abs(cmd.p - start):.4f} rad from start {start:.4f} "
                      f"(window {limits.position_window:g})")
    if abs(cmd.p) > spec.p_max:
        errors.append(f"|p_des| {abs(cmd.p):.4f} > MIT range {spec.p_max:g} rad")
    worst = worst_case_torque(cmd, limits)
    if worst > limits.max_torque + 1e-12:
        errors.append(f"worst-case torque {worst:.4g} Nm > {limits.max_torque:g} Nm "
                      f"(kp*{limits.max_track_error:g} + kd*(|v|+{limits.max_speed:g}) + |t|)")
    if cmd.kp > 0 and measured is not None and abs(cmd.p - measured) > limits.max_track_error:
        errors.append(f"|p_des - p| {abs(cmd.p - measured):.4f} > max_track_error {limits.max_track_error:g}")
    if errors:
        raise ValueError("; ".join(errors))


def check_feedback(fb: MotorFeedback, limits: BenchLimits, start: float, last_cmd: MitCommand | None,
                   motor: Motor) -> str | None:
    """Reason text if this feedback must stop the run, else None."""
    errors = []
    reported = motor.fault(fb)
    if reported:
        errors.append(reported)
    if abs(fb.position - start) > limits.position_window + limits.position_margin:
        errors.append(f"position {fb.position:.4f} is {abs(fb.position - start):.4f} rad from start "
                      f"(window {limits.position_window:g} + margin {limits.position_margin:g})")
    if abs(fb.velocity) > limits.max_speed:
        errors.append(f"|velocity| {abs(fb.velocity):.4g} > {limits.max_speed:g} rad/s")
    if abs(fb.torque) > limits.max_torque:
        errors.append(f"|torque| {abs(fb.torque):.4g} > {limits.max_torque:g} Nm")
    if fb.temperature > limits.max_temperature:
        errors.append(f"temperature {fb.temperature} > {limits.max_temperature:g} degC")
    if last_cmd is not None and last_cmd.kp > 0 and abs(last_cmd.p - fb.position) > limits.max_track_error:
        errors.append(f"tracking error {abs(last_cmd.p - fb.position):.4f} > {limits.max_track_error:g} rad")
    return "; ".join(errors) or None


class Safety:
    """Owns all sending to one motor. Use as a context manager so exit always damps and disables.

    Listeners get on_rx(rx, feedback_or_None), on_command(wall_time, cmd, kind) with kind in
    "cmd" / "probe" / "damp", and on_event(wall_time, text).
    """

    def __init__(self, bus: CanBus, motor: Motor, limits: BenchLimits = BenchLimits(),
                 period: float = 0.002, clock=time.monotonic):
        limits.check_against(motor.spec)
        self.bus, self.motor, self.limits, self.period, self._clock = bus, motor, limits, period, clock
        self.listeners = []
        self.armed = False  # True from the first frame sent
        self.start_position: float | None = None
        self.feedback: MotorFeedback | None = None
        self.fault_reason: str | None = None
        self.disable_acknowledged: bool | None = None  # AK 3.0 only; None = not applicable / not sent
        self._last_fb_time: float | None = None
        self._last_cmd: MitCommand | None = None
        self._shut_down = False
        self._old_handlers = {}
        self._gc_was_enabled = True

    # --- context manager / signals -------------------------------------------------------------
    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGHUP):
                self._old_handlers[sig] = signal.signal(sig, _raise_system_exit)
        # Python's cycle collector can stop every thread for tens of ms (34 ms measured in the test
        # run, gen 2). Collect once now, then pause it until exit; the loop's objects have no cycles
        # and are freed by reference counting anyway.
        self._gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.freeze()
        gc.disable()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None and not issubclass(exc_type, SafetyFault):
                self._event(f"exit by {exc_type.__name__}: {exc}")
            self.shutdown()
        finally:
            for sig, handler in self._old_handlers.items():
                signal.signal(sig, handler)
            self._old_handlers = {}
            gc.unfreeze()
            if self._gc_was_enabled:
                gc.enable()
        return False

    # --- reading ---------------------------------------------------------------------------------
    def poll(self) -> list[MotorFeedback]:
        """Read all waiting frames. While armed, run the watchdog and feedback checks (may raise)."""
        new = self._drain()
        if self.armed and not self._shut_down:
            if self._clock() - self._last_fb_time > self.limits.watchdog:
                self._fault(f"watchdog: no feedback for {self._clock() - self._last_fb_time:.3f} s "
                            f"(limit {self.limits.watchdog:g} s)")
            for fb in new:
                reason = check_feedback(fb, self.limits, self.start_position, self._last_cmd, self.motor)
                if reason:
                    self._fault(reason)
        return new

    def wait_for_feedback(self, timeout: float) -> MotorFeedback | None:
        """Listen (send nothing) until a feedback frame arrives or `timeout` passes."""
        end = self._clock() + timeout
        while self._clock() < end:
            new = self._drain(wait=min(0.005, max(0.0, end - self._clock())))
            if new:
                return new[-1]
        return None

    def _drain(self, wait: float = 0.0) -> list[MotorFeedback]:
        new = []
        rx = self.bus.recv(wait)
        while rx is not None:
            fb = None if rx.is_echo else self.motor.parse(rx.frame)
            for listener in self.listeners:
                listener.on_rx(rx, fb)
            if fb is not None:
                self.feedback, self._last_fb_time = fb, self._clock()
                new.append(fb)
                if self.motor.is_disable_ack(fb):
                    self.disable_acknowledged = True
            rx = self.bus.recv(0.0)
        return new

    # --- sending ---------------------------------------------------------------------------------
    def arm(self) -> float:
        """Enable the motor and seed the start position from live feedback. Returns the start (rad).

        AK 3.0 streams feedback, so it is read before anything is sent. AK 2.0 replies only to
        commands: after the enter frame we send zero-torque probes (kp = kd = t = 0, so the motor
        can't produce torque whatever its position) until it answers.
        """
        if self.armed:
            raise SafetyFault("already armed")
        if self.motor.streams_feedback:
            if self.wait_for_feedback(0.2) is None:
                self._fault("no feedback within 0.2 s; not enabling the motor")  # never armed: sends nothing
            self._mark_armed()
            for frame in self.motor.enable_frames():
                self._send(frame, "enable")
        else:
            self._mark_armed()
            for frame in self.motor.enable_frames():
                self._send(frame, "enable")
            if not self._probe_until_feedback(0.2):
                self._fault("no reply to zero-torque probes within 0.2 s")
        self._reseed()
        self._event(f"armed, start position {self.start_position:.5f} rad")
        return self.start_position

    def command(self, cmd: MitCommand) -> None:
        """Check, then send one MIT command (output shaft units)."""
        if not self.armed or self.start_position is None:
            raise SafetyFault("command() before arm()")
        if self.fault_reason or self._shut_down:
            raise SafetyFault(f"motor is stopped ({self.fault_reason or 'shut down'})")
        measured = self.feedback.position if self.feedback else None
        try:
            check_command(cmd, self.limits, self.motor.spec, self.start_position, measured)
        except ValueError as e:
            self._fault(f"command rejected: {e}")
        self._send(self.motor.command(cmd), "cmd", cmd)
        self._last_cmd = cmd

    def set_zero(self) -> float:
        """Set the current position as zero (AK 2.0 FE frame; AK 3.0 temporary origin, no flash write).
        Only while the last command was zero torque, so no hold can jump to the new zero.
        Re-seeds the start position; returns it (should read ~0)."""
        if not self.armed:
            raise SafetyFault("set_zero() before arm()")
        last = self._last_cmd
        if last is not None and (last.kp != 0 or last.t != 0):
            self._fault("set_zero() while the motor holds or applies torque (kp or t nonzero)")
        self._send(self.motor.zero_frame(), "zero")
        time.sleep(0.02)
        self._drain()
        self.feedback = None
        if not self._probe_until_feedback(0.2):
            self._fault("no feedback after set_zero()")
        self._reseed()
        self._event(f"zero set, position now {self.start_position:.5f} rad")
        return self.start_position

    def begin_stage(self, limits: BenchLimits, label: str) -> float:
        """Switch limits and re-center the position window on the live position (bring-up stages)."""
        limits.check_against(self.motor.spec)
        if self.feedback is None:
            self._fault("begin_stage() without feedback")
        self.limits = limits
        self._reseed()
        self._event(f"stage {label}: limits {limits.to_dict()}, start {self.start_position:.5f} rad")
        return self.start_position

    def shutdown(self) -> None:
        """Damp for damp_time, then disable. Runs once; sends nothing if never armed."""
        if self._shut_down:
            return
        self._shut_down = True
        if not self.armed:
            self._event("never armed: nothing sent")
            return
        try:
            self._damp()
        except BaseException as e:  # Ctrl-C / SIGTERM / bus error during damping: still disable
            self._event(f"damping cut short by {type(e).__name__}: {e}")
        finally:
            self._disable()

    def note(self, text: str) -> None:
        """Add a timestamped event to the run log (listeners' on_event)."""
        self._event(text)

    # --- internals -------------------------------------------------------------------------------
    def _damp(self) -> None:
        end = self._clock() + self.limits.damp_time
        while self._clock() < end:
            try:
                self._drain()
            except Exception as e:  # keep damping even if a frame can't be read
                self._event(f"read error while damping: {e}")
            p = self.feedback.position if self.feedback else 0.0  # ignored by the motor (kp = 0)
            damp = MitCommand(p, 0.0, 0.0, self.limits.fault_kd, 0.0)
            self._send(self.motor.command(damp), "damp", damp)
            time.sleep(self.period)

    def _disable(self) -> None:
        sent = False
        for attempt in range(3):
            try:
                for frame in self.motor.disable_frames():
                    self._send(frame, "disable")
                sent = True
                time.sleep(0.01)
                self._drain()
                if self.disable_acknowledged:
                    break
            except KeyboardInterrupt:
                continue  # a second Ctrl-C must not skip the disable
            except Exception as e:
                self._event(f"disable attempt {attempt + 1} failed: {e}")
        if self.motor.streams_feedback and sent:
            deadline = self._clock() + 0.1
            while not self.disable_acknowledged and self._clock() < deadline:
                self._drain(wait=0.005)
            self.disable_acknowledged = bool(self.disable_acknowledged)
        self._event("disabled" + ("" if sent else ": DISABLE FRAME NOT SENT, USE THE E-STOP")
                    + (f", ack {self.disable_acknowledged}" if self.motor.streams_feedback else ""))

    def _probe_until_feedback(self, timeout: float) -> bool:
        end = self._clock() + timeout
        while self._clock() < end:
            self._send(self.motor.command(ZERO_TORQUE), "probe", ZERO_TORQUE)
            self._last_cmd = ZERO_TORQUE
            if self.wait_for_feedback(self.period) is not None:
                return True
        return False

    def _mark_armed(self) -> None:
        self.armed = True
        self._last_fb_time = self._clock()

    def _reseed(self) -> None:
        self.start_position = self.feedback.position
        self._last_fb_time = self._clock()

    def _send(self, frame: CanFrame, kind: str, cmd: MitCommand | None = None) -> None:
        now = wall_time()
        self.bus.send(frame)
        if cmd is not None:
            for listener in self.listeners:
                listener.on_command(now, cmd, kind)
        else:
            self._event(f"sent {kind} frame id {frame.arbitration_id:#x} {frame.data.hex()}")

    def _fault(self, reason: str):
        if self.fault_reason is None:
            self.fault_reason = reason
            self._event(f"FAULT: {reason}")
        raise SafetyFault(reason)

    def _event(self, text: str) -> None:
        now = wall_time()
        for listener in self.listeners:
            listener.on_event(now, text)


def _raise_system_exit(signum, frame):
    raise SystemExit(128 + signum)
