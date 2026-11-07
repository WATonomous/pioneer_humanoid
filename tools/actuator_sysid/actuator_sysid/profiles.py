"""Bench test profiles (Task 11 measurements) and the loop that runs them through safety.Safety.

A profile gives the MIT command at time t, with p RELATIVE to the start position (seeded from the
live position by Safety.arm()). Every profile starts at offset 0, ends at offset 0, and is checked
sample by sample against the bench limits before the first command (check_profile). run() is the
only way to execute one, and it sends through Safety.command().

Defaults fit the first-power-on bench caps (5 Nm, 1 rad/s, kp 10, kd 1, +-0.5 rad, 0.3 rad tracking).
Larger amplitudes need raised caps (with a reason) first.

  step          position steps +size / -size, PD hold (latency, armature, overshoot)
  chirp         exponential sine sweep f0 -> f1 (0.2 -> 10 Hz), constant amplitude (bandwidth)
  speed_sweep   constant-speed triangles at several speeds, PD + velocity feed-forward (friction)
  torque_ramp   slow feed-forward torque triangle, kp 0 (breakaway friction)
  torque_hold   feed-forward torque levels held still, kp 0 (torque constant, lever arm + scale)
  backlash      small slow torque triangle, kp 0, output locked (backlash)
Torque profiles (kp 0) stop early, without a fault, once the output moves abort_distance.
"""
import math
from dataclasses import asdict, dataclass

from pioneer_humanoid.leg_params import MotorSpec

from .logger import Ticker
from .protocol import MitCommand
from .safety import BenchLimits, Safety, check_command


@dataclass(frozen=True)
class _Segment:
    """Linear in p and t over [t0, t1]; v is the velocity feed-forward."""
    t0: float
    t1: float
    p0: float
    p1: float
    v: float = 0.0
    tau0: float = 0.0
    tau1: float = 0.0


class Profile:
    name = ""
    kp = 0.0
    kd = 0.0
    abort_distance: float | None = None

    @property
    def duration(self) -> float:
        raise NotImplementedError

    def setpoint(self, t: float) -> MitCommand:
        """Command at time t (s), p relative to the start position."""
        raise NotImplementedError

    def params(self) -> dict:
        return {"name": self.name, "duration_s": self.duration, "kp": self.kp, "kd": self.kd,
                "abort_distance": self.abort_distance, **asdict(self)}

    def sample_times(self, dt: float) -> list[float]:
        n = int(math.ceil(self.duration / dt))
        return [min(i * dt, self.duration) for i in range(n + 1)]


class _Piecewise(Profile):
    """Profiles built from linear segments."""

    def segments(self) -> list[_Segment]:
        raise NotImplementedError

    @property
    def duration(self) -> float:
        return self.segments()[-1].t1

    def setpoint(self, t: float) -> MitCommand:
        segs = self._cached()
        for s in segs:
            if t <= s.t1:
                break
        a = 0.0 if s.t1 == s.t0 else min(max((t - s.t0) / (s.t1 - s.t0), 0.0), 1.0)
        return MitCommand(s.p0 + a * (s.p1 - s.p0), s.v, self.kp, self.kd, s.tau0 + a * (s.tau1 - s.tau0))

    def sample_times(self, dt: float) -> list[float]:
        # Also sample just before and at every corner, so a jump between segments can't hide.
        corners = [x for s in self._cached() for x in (s.t0, s.t1)]
        return sorted(set(super().sample_times(dt)) | set(corners))

    def _cached(self):
        if not hasattr(self, "_segs"):
            object.__setattr__(self, "_segs", self.segments())
        return self._segs


def _chain(pieces) -> list[_Segment]:
    """(duration, p0, p1, v, tau0, tau1) tuples -> back-to-back segments."""
    out, t = [], 0.0
    for d, p0, p1, v, tau0, tau1 in pieces:
        out.append(_Segment(t, t + d, p0, p1, v, tau0, tau1))
        t += d
    return out


def _torque_triangle(peak: float, rate: float, cycles: int, lead: float):
    """0 -> +peak -> 0 -> -peak -> 0 at `rate` Nm/s, with `lead` s of zero torque before and after."""
    leg = peak / rate
    pieces = [(lead, 0, 0, 0, 0, 0)]
    for _ in range(cycles):
        pieces += [(leg, 0, 0, 0, 0, peak), (leg, 0, 0, 0, peak, 0),
                   (leg, 0, 0, 0, 0, -peak), (leg, 0, 0, 0, -peak, 0)]
    return _chain(pieces + [(lead, 0, 0, 0, 0, 0)])


@dataclass(frozen=True)
class Step(_Piecewise):
    size: float = 0.1  # rad
    hold: float = 1.0  # s per level
    repeats: int = 2  # each repeat: 0, +size, 0, -size
    kp: float = 5.0
    kd: float = 0.5
    name = "step"

    def segments(self):
        h, s = self.hold, self.size
        pieces = []
        for _ in range(self.repeats):
            pieces += [(h, 0, 0, 0, 0, 0), (h, s, s, 0, 0, 0), (h, 0, 0, 0, 0, 0), (h, -s, -s, 0, 0, 0)]
        return _chain(pieces + [(h, 0, 0, 0, 0, 0)])


@dataclass(frozen=True)
class Chirp(Profile):
    amplitude: float = 0.015  # rad. Peak speed 2*pi*f1*A = 0.94 rad/s at the defaults
    f0: float = 0.2  # Hz
    f1: float = 10.0  # Hz
    sweep: float = 30.0  # s
    lead: float = 1.0  # s of hold before and after
    kp: float = 10.0
    kd: float = 0.5
    name = "chirp"

    def __post_init__(self):
        if not 0 < self.f0 < self.f1:
            raise ValueError(f"need 0 < f0 < f1, got {self.f0}, {self.f1}")

    @property
    def duration(self) -> float:
        return self.sweep + 2 * self.lead

    def setpoint(self, t: float) -> MitCommand:
        """Exponential sweep: f(t) = f0 * k^(t/T), k = f1/f0; phase = 2*pi*f0*T/ln(k) * (k^(t/T) - 1)."""
        ts = t - self.lead
        if not 0 <= ts <= self.sweep:
            return MitCommand(0.0, 0.0, self.kp, self.kd, 0.0)
        k, T = self.f1 / self.f0, self.sweep
        growth = k ** (ts / T)
        phase = 2 * math.pi * self.f0 * T / math.log(k) * (growth - 1)
        omega = 2 * math.pi * self.f0 * growth
        return MitCommand(self.amplitude * math.sin(phase), self.amplitude * omega * math.cos(phase),
                          self.kp, self.kd, 0.0)

    def frequency(self, t: float) -> float:
        """Instantaneous sweep frequency (Hz), for plots and the fit."""
        ts = min(max(t - self.lead, 0.0), self.sweep)
        return self.f0 * (self.f1 / self.f0) ** (ts / self.sweep)


@dataclass(frozen=True)
class SpeedSweep(_Piecewise):
    speeds: tuple = (0.1, 0.2, 0.4, 0.6, 0.8)  # rad/s
    amplitude: float = 0.4  # rad; triangle 0 -> +A -> -A -> 0 at each speed
    cycles: int = 1
    hold: float = 0.5  # s at 0 between speeds
    kp: float = 5.0
    kd: float = 0.5
    name = "speed_sweep"

    def segments(self):
        a, pieces = self.amplitude, [(self.hold, 0, 0, 0, 0, 0)]
        for v in self.speeds:
            for _ in range(self.cycles):
                pieces += [(a / v, 0, a, v, 0, 0), (2 * a / v, a, -a, -v, 0, 0), (a / v, -a, 0, v, 0, 0)]
            pieces.append((self.hold, 0, 0, 0, 0, 0))
        return _chain(pieces)


@dataclass(frozen=True)
class TorqueRamp(_Piecewise):
    peak: float = 3.0  # Nm
    rate: float = 0.5  # Nm/s
    cycles: int = 1
    lead: float = 1.0
    kd: float = 0.5  # damping only; kp = 0
    abort_distance: float = 0.3  # rad: breakaway seen, stop
    name = "torque_ramp"

    def segments(self):
        return _torque_triangle(self.peak, self.rate, self.cycles, self.lead)


@dataclass(frozen=True)
class TorqueHold(_Piecewise):
    levels: tuple = (1.0, 2.0, 3.0, -1.0, -2.0, -3.0)  # Nm
    hold: float = 3.0  # s at each level
    rest: float = 2.0  # s at 0 between levels
    ramp: float = 0.2  # s up / down (no torque steps into the fixture)
    kd: float = 0.5
    abort_distance: float = 0.3
    name = "torque_hold"

    def segments(self):
        pieces = [(self.rest, 0, 0, 0, 0, 0)]
        for tau in self.levels:
            pieces += [(self.ramp, 0, 0, 0, 0, tau), (self.hold, 0, 0, 0, tau, tau),
                       (self.ramp, 0, 0, 0, tau, 0), (self.rest, 0, 0, 0, 0, 0)]
        return _chain(pieces)


@dataclass(frozen=True)
class BacklashProbe(_Piecewise):
    peak: float = 1.0  # Nm, small: take up the slack, don't load the fixture
    rate: float = 1.0  # Nm/s
    cycles: int = 3
    lead: float = 1.0
    kd: float = 0.2
    abort_distance: float = 0.2  # rad; backlash is ~0.003-0.004 rad, so more means the output moved
    name = "backlash"

    def segments(self):
        return _torque_triangle(self.peak, self.rate, self.cycles, self.lead)


PROFILES = {cls.name: cls for cls in (Step, Chirp, SpeedSweep, TorqueRamp, TorqueHold, BacklashProbe)}


def make_profile(name: str, **params) -> Profile:
    """Profile by CLI name. Tuple parameters may be given as comma-separated strings."""
    cls = PROFILES[name]
    fields = cls.__dataclass_fields__
    unknown = set(params) - set(fields)
    if unknown:
        raise ValueError(f"{name}: unknown parameter(s) {sorted(unknown)}; known: {sorted(fields)}")
    typed = {}
    for k, v in params.items():
        default = fields[k].default
        if isinstance(default, tuple):
            typed[k] = tuple(float(x) for x in v.split(",")) if isinstance(v, str) else tuple(v)
        else:
            typed[k] = type(default)(v)
    return cls(**typed)


def check_profile(profile: Profile, limits: BenchLimits, spec: MotorSpec, start: float = 0.0,
                  dt: float = 0.001) -> None:
    """Raise ValueError if any sample of the profile would be rejected by Safety, or the profile
    doesn't start and end at offset 0, or jumps further than the tracking limit between samples."""
    times = profile.sample_times(dt)
    first, last = profile.setpoint(0.0), profile.setpoint(profile.duration)
    if abs(first.p) > 1e-12 or abs(last.p) > 1e-12 or abs(first.t) > 1e-12 or abs(last.t) > 1e-12:
        raise ValueError(f"{profile.name}: must start and end at offset 0 with zero torque")
    if profile.abort_distance is not None and profile.abort_distance >= limits.position_window:
        raise ValueError(f"{profile.name}: abort_distance {profile.abort_distance:g} must be inside the "
                         f"position window {limits.position_window:g}")
    prev = None
    for t in times:
        cmd = profile.setpoint(t)
        try:
            check_command(cmd._replace(p=start + cmd.p), limits, spec, start)
        except ValueError as e:
            raise ValueError(f"{profile.name} at t = {t:.3f} s: {e}") from None
        if prev is not None and cmd.kp > 0 and abs(cmd.p - prev) > limits.max_track_error:
            raise ValueError(f"{profile.name} at t = {t:.3f} s: jumps {abs(cmd.p - prev):.4f} rad "
                             f"(> max_track_error {limits.max_track_error:g})")
        prev = cmd.p
    after = profile.setpoint(profile.duration + 1.0)
    if abs(after.p) > 1e-12 or abs(after.t) > 1e-12:
        raise ValueError(f"{profile.name}: setpoint after the end must be offset 0, zero torque")


@dataclass
class RunResult:
    ended: str  # "done" or "abort_distance"
    elapsed: float  # s
    ticker: dict


def run(safety: Safety, profile: Profile, period: float = 0.002, recorder=None) -> RunResult:
    """Run one profile at 1/period Hz through `safety` (already armed). Faults raise SafetyFault;
    the caller's `with Safety(...)` then damps and disables."""
    if not safety.armed or safety.start_position is None:
        raise RuntimeError("arm the motor (Safety.arm) before running a profile")
    start = safety.start_position
    check_profile(profile, safety.limits, safety.motor.spec, start)
    safety.note(f"profile {profile.name} start, params {profile.params()}")
    ticker = Ticker(period)
    ended = "done"
    while True:
        lateness = ticker.wait()
        safety.poll()
        t = ticker.elapsed
        if recorder is not None:
            recorder.on_tick(t, lateness)
        if t > profile.duration:
            break
        fb = safety.feedback
        if profile.abort_distance is not None and fb is not None and abs(fb.position - start) > profile.abort_distance:
            ended = "abort_distance"
            safety.note(f"profile {profile.name}: moved {abs(fb.position - start):.4f} rad "
                        f"> abort_distance {profile.abort_distance:g} at t = {t:.3f} s, stopping")
            break
        cmd = profile.setpoint(t)
        safety.command(cmd._replace(p=start + cmd.p))
    safety.note(f"profile {profile.name} end ({ended}), loop {ticker.stats()}")
    return RunResult(ended, ticker.elapsed, ticker.stats())
