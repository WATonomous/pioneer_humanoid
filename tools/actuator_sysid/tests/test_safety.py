"""safety.py on a virtual CAN bus with a fake motor: limits, seeding, watchdog, damp + disable on exit."""
import dataclasses
import os
import re
import signal
import time
from pathlib import Path

import pytest
from conftest import hold, rig

from actuator_sysid.motors import make_motor
from actuator_sysid.protocol import MitCommand, MotorFeedback
from actuator_sysid.safety import BenchLimits, Safety, SafetyFault, check_command, check_feedback, worst_case_torque
from pioneer_humanoid.leg_params import AK60_39, AK80_64, AKH70_48

LEG_MOTORS = [AKH70_48, AK80_64, AK60_39]
FAST = BenchLimits(damp_time=0.1)  # short damping so tests run quickly


class Events:
    """Listener that records everything Safety reports."""

    def __init__(self):
        self.rx, self.commands, self.events = [], [], []

    def on_rx(self, rx, fb):
        self.rx.append((rx, fb))

    def on_command(self, t, cmd, kind):
        self.commands.append((t, cmd, kind))

    def on_event(self, t, text):
        self.events.append(text)


def _safety(bus, motor, limits=FAST):
    s = Safety(bus, motor, limits)
    s.listeners.append(Events())
    return s


def _kinds(fake):
    return [k for k, _ in fake.received]


def _t_step(spec):
    """One 12-bit torque code. t = 0 can't be encoded exactly (4095 codes, no middle one): it arrives
    as -t_step/2 (AK60-39 -0.020, AK80-64 -0.035, AKH70-48 -0.068 Nm)."""
    return 2 * spec.t_max / 4095


def _assert_damped_then_disabled(fake, protocol, limits=FAST):
    """The fake saw damping commands (kp 0, kd fault_kd), then the disable frame, and nothing after it."""
    kinds = _kinds(fake)
    t_tol = _t_step(fake.spec)
    stop = "disable" if protocol == "ak3" else "exit"
    assert stop in kinds
    first_stop = kinds.index(stop)
    assert all(k == stop for k in kinds[first_stop:])
    damp = [c for k, c in fake.received[:first_stop] if k == "mit" and c.kp == 0 and c.kd > 0]
    assert damp, "no damping commands before disable"
    assert all(c.kd == pytest.approx(limits.fault_kd, abs=0.01) and abs(c.t) <= t_tol for c in damp)
    assert not fake.enabled


# --- Limits ------------------------------------------------------------------------------------
@pytest.mark.parametrize("spec", LEG_MOTORS, ids=lambda m: m.name)
def test_default_bench_limits_fit_every_motor(spec):
    BenchLimits().check_against(spec)


@pytest.mark.parametrize("field,value,spec", [
    ("max_torque", 223.0, AKH70_48),  # peak 222
    ("max_torque", 73.0, AK60_39),  # peak 72
    ("max_speed", 10.1, AK60_39),  # MIT v range 10 < no-load 10.26
    ("max_speed", 6.3, AK80_64),  # no-load 6.283 (D15)
    ("max_kd", 5.1, AK60_39),
    ("max_kp", 501.0, AK60_39),
    ("watchdog", 0.051, AK60_39),
    ("max_torque", float("nan"), AK60_39),
    ("max_speed", -1.0, AK60_39),
])
def test_limits_above_the_motor_ceiling_are_rejected(field, value, spec):
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(BenchLimits(), **{field: value}).check_against(spec)


def test_fault_kd_cannot_exceed_max_kd():
    with pytest.raises(ValueError, match="fault_kd"):
        BenchLimits(max_kd=0.5, fault_kd=1.0).check_against(AK60_39)


def test_raising_a_cap_needs_a_reason():
    with pytest.raises(ValueError, match="reason"):
        BenchLimits().with_overrides({"max_torque": 10})
    raised = BenchLimits().with_overrides({"max_torque": 10}, reason="Task 11 torque constant, lever arm")
    assert raised.max_torque == 10 and raised.raised_above_defaults() == {"max_torque": 10}
    assert BenchLimits().with_overrides({"max_torque": 2}).max_torque == 2  # lowering is always fine
    with pytest.raises(ValueError, match="unknown"):
        BenchLimits().with_overrides({"torque": 1})


def test_safety_refuses_limits_over_the_ceiling(channel):
    with rig(AK60_39, channel) as (fake, bus, motor):
        with pytest.raises(ValueError):
            Safety(bus, motor, BenchLimits(max_torque=100))


# --- Command check ---------------------------------------------------------------------------
L = BenchLimits()


def test_worst_case_torque_formula():
    # kp*0.3 + kd*(|v| + 1) + |t|
    assert worst_case_torque(MitCommand(0, 0.5, 10, 1, -0.5), L) == pytest.approx(3 + 1.5 + 0.5)


def test_command_inside_limits_passes():
    check_command(MitCommand(0.4, 0.2, 5, 0.5, 1.0), L, AK60_39, start=0.0, measured=0.2)


@pytest.mark.parametrize("cmd,measured,match", [
    (MitCommand(0, 0, 10.1, 0, 0), None, "kp"),
    (MitCommand(0, 0, 0, 1.1, 0), None, "kd"),
    (MitCommand(0, 0, -1, 0, 0), None, "kp"),
    (MitCommand(0, 1.01, 0, 0, 0), None, "v_des"),
    (MitCommand(0.51, 0, 0, 0, 0), None, "window"),
    (MitCommand(0, 0, 0, 0, 5.01), None, "worst-case"),
    (MitCommand(0, 0, 10, 1, 1.1), None, "worst-case"),  # 3 + 1 + 1.1
    (MitCommand(0.2, 0, 1, 0, 0), -0.11, "max_track_error"),
    (MitCommand(float("nan"), 0, 0, 0, 0), None, "non-finite"),
])
def test_command_outside_limits_is_rejected(cmd, measured, match):
    with pytest.raises(ValueError, match=match):
        check_command(cmd, L, AK60_39, start=0.0, measured=measured)


def test_tracking_check_skipped_with_kp_zero():
    check_command(MitCommand(0.4, 0, 0, 0.5, 1.0), L, AK60_39, start=0.0, measured=-0.4)


def test_command_beyond_mit_position_range_is_rejected():
    with pytest.raises(ValueError, match="MIT range"):
        check_command(MitCommand(12.6, 0, 0, 0, 0), L, AK60_39, start=12.4)


# --- Seeding and read-only ----------------------------------------------------------------------
@pytest.mark.parametrize("spec", LEG_MOTORS, ids=lambda m: m.name)
def test_arm_seeds_from_live_position(channel, spec):
    with rig(spec, channel, position=0.7, coulomb=0.0) as (fake, bus, motor):
        with _safety(bus, motor) as s:
            start = s.arm()
            assert start == pytest.approx(0.7, abs=2e-3)
            hold(s, 0.2)
            # Held where it was, no jump. Offset = t-zero bias / kp (see _t_step), up to 0.035 rad at kp 2.
            assert fake.position == pytest.approx(0.7, abs=_t_step(spec) / 2 / 1.95 + 3e-3)
            with pytest.raises(SafetyFault, match="window|track"):
                s.command(MitCommand(0.0, 0, 2, 0.2, 0))  # 0.7 rad away from where it is
        _assert_damped_then_disabled(fake, spec.protocol)


def test_ak2_enters_mit_mode_and_probes_with_zero_torque(channel):
    with rig(AK80_64, channel, position=0.3) as (fake, bus, motor):
        with _safety(bus, motor) as s:
            s.arm()
        kinds = _kinds(fake)
        assert kinds[0] == "enter"
        probes = [c for k, c in fake.received[1:] if k == "mit"][:1]
        assert probes and probes[0].kp == 0 and probes[0].kd == 0 and probes[0].t == pytest.approx(0, abs=0.05)


def test_ak2_works_if_the_motor_also_replies_to_enter(channel):
    with rig(AK80_64, channel, position=0.3, reply_to_special=True) as (fake, bus, motor):
        with _safety(bus, motor) as s:
            assert s.arm() == pytest.approx(0.3, abs=1e-3)


def test_read_only_sends_nothing(channel):
    with rig(AK60_39, channel, position=0.25) as (fake, bus, motor):
        with _safety(bus, motor) as s:
            fb = s.wait_for_feedback(0.2)
            assert fb.position == pytest.approx(0.25, abs=2e-3)
            for _ in range(20):
                s.poll()
                time.sleep(0.002)
        assert fake.received == [] and not s.armed
        assert "never armed: nothing sent" in s.listeners[0].events


def test_ak3_without_feedback_never_enables(channel):
    with rig(AK60_39, channel) as (fake, bus, motor):
        fake.mute = True
        time.sleep(0.02)
        while bus.recv(0.0):
            pass
        with pytest.raises(SafetyFault, match="no feedback"):
            with _safety(bus, motor) as s:
                s.arm()
        assert fake.received == []


def test_ak2_without_reply_disables(channel):
    with rig(AK80_64, channel) as (fake, bus, motor):
        fake.mute = True
        with pytest.raises(SafetyFault, match="no reply"):
            with _safety(bus, motor) as s:
                s.arm()
        assert _kinds(fake)[-1] == "exit"


def test_command_before_arm_is_refused(channel):
    with rig(AK60_39, channel) as (fake, bus, motor):
        with _safety(bus, motor) as s:
            with pytest.raises(SafetyFault, match="before arm"):
                s.command(MitCommand(0, 0, 0, 0, 0))
        assert fake.received == []


# --- Watchdog and feedback checks -----------------------------------------------------------------
@pytest.mark.parametrize("spec", [AK60_39, AK80_64], ids=lambda m: m.name)
def test_watchdog_trips_within_50_ms(channel, spec):
    with rig(spec, channel) as (fake, bus, motor):
        with pytest.raises(SafetyFault, match="watchdog"):
            with _safety(bus, motor) as s:
                s.arm()
                hold(s, 0.1)
                fake.mute = True
                t_mute = time.monotonic()
                try:
                    hold(s, 1.0)
                finally:
                    elapsed = time.monotonic() - t_mute
        assert 0.045 <= elapsed <= 0.08
        assert ("disable" if spec.protocol == "ak3" else "exit") in _kinds(fake)


@pytest.mark.parametrize("inject,match", [
    (lambda f: setattr(f, "external_torque", 3.0), "velocity|track|window"),
    (lambda f: setattr(f, "error", 2), "over-current"),
    (lambda f: setattr(f, "temperature", 71), "temperature"),
])
def test_feedback_faults_damp_and_disable(channel, inject, match):
    with rig(AK60_39, channel, coulomb=0.0) as (fake, bus, motor):
        with pytest.raises(SafetyFault, match=match):
            with _safety(bus, motor) as s:
                s.arm()
                hold(s, 0.05, kp=0.0, kd=0.0)
                inject(fake)
                hold(s, 2.0, kp=0.0, kd=0.0)
        assert s.fault_reason
        fake.error = 0
        _assert_damped_then_disabled(fake, "ak3")


def test_measured_torque_over_cap_faults():
    # The worst-case command check keeps commanded torque under the cap while the tracking check
    # holds (a push trips tracking first), so this is the backup for a motor that reports more.
    motor = make_motor(AK60_39, 1)
    fb = MotorFeedback(1, 0.0, 0.0, -5.2, -1.5, 30, 0)
    assert "torque" in check_feedback(fb, BenchLimits(), 0.0, None, motor)
    assert check_feedback(MotorFeedback(1, 0.0, 0.0, 4.9, 1.4, 30, 0), BenchLimits(), 0.0, None, motor) is None


def test_no_commands_after_a_fault(channel):
    with rig(AK60_39, channel) as (fake, bus, motor):
        with _safety(bus, motor) as s:
            s.arm()
            with pytest.raises(SafetyFault):
                s.command(MitCommand(0, 0, 50, 0, 0))
            with pytest.raises(SafetyFault, match="stopped"):
                s.command(MitCommand(s.start_position, 0, 0, 0, 0))


# --- Disable on exit ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec", LEG_MOTORS, ids=lambda m: m.name)
def test_normal_exit_damps_then_disables(channel, spec):
    with rig(spec, channel) as (fake, bus, motor):
        with _safety(bus, motor) as s:
            s.arm()
            hold(s, 0.1)
        _assert_damped_then_disabled(fake, spec.protocol)
        if spec.protocol == "ak3":
            assert s.disable_acknowledged is True  # 0x29 with error 0x77
        else:
            assert s.disable_acknowledged is None  # AK 2.0 has no documented ack


def test_damping_lasts_damp_time(channel):
    limits = BenchLimits(damp_time=0.3)
    with rig(AK60_39, channel) as (fake, bus, motor):
        with _safety(bus, motor, limits) as s:
            s.arm()
            hold(s, 0.05)
        damp = [t for t, c, k in s.listeners[0].commands if k == "damp"]
        assert 0.28 <= damp[-1] - damp[0] <= 0.4


@pytest.mark.parametrize("exc", [RuntimeError("bug in a profile"), KeyboardInterrupt()])
def test_exception_exit_damps_then_disables(channel, exc):
    with rig(AK60_39, channel) as (fake, bus, motor):
        with pytest.raises(type(exc)):
            with _safety(bus, motor) as s:
                s.arm()
                hold(s, 0.05)
                raise exc
        _assert_damped_then_disabled(fake, "ak3")
        assert any(e.startswith("exit by") for e in s.listeners[0].events)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_sigterm_and_sighup_damp_then_disable(channel, sig):
    before = signal.getsignal(sig)
    with rig(AK80_64, channel) as (fake, bus, motor):
        with pytest.raises(SystemExit):
            with _safety(bus, motor) as s:
                s.arm()
                os.kill(os.getpid(), sig)
                hold(s, 1.0)
        _assert_damped_then_disabled(fake, "ak2")
    assert signal.getsignal(sig) is before  # handler restored


def test_ctrl_c_during_damping_still_disables(channel):
    with rig(AK60_39, channel) as (fake, bus, motor):
        s = _safety(bus, motor, BenchLimits(damp_time=1.0))
        s.arm()
        calls = {"n": 0}
        real_drain = s._drain

        def drain_then_interrupt(wait=0.0):
            calls["n"] += 1
            if calls["n"] == 5:
                raise KeyboardInterrupt
            return real_drain(wait)
        s._drain = drain_then_interrupt
        s.shutdown()
        assert "disable" in _kinds(fake)
        assert any("cut short by KeyboardInterrupt" in e for e in s.listeners[0].events)


def test_cycle_collector_paused_only_while_active(channel):
    import gc
    assert gc.isenabled()
    with rig(AK60_39, channel) as (fake, bus, motor):
        with pytest.raises(RuntimeError):
            with _safety(bus, motor):
                assert not gc.isenabled()
                raise RuntimeError
    assert gc.isenabled()


# --- Zeroing and stages --------------------------------------------------------------------------
def test_ak2_set_zero_reseeds(channel):
    with rig(AK80_64, channel, position=0.7) as (fake, bus, motor):
        with _safety(bus, motor) as s:
            s.arm()
            assert s.set_zero() == pytest.approx(0.0, abs=1e-3)
            hold(s, 0.1)
            assert fake.position == pytest.approx(0.0, abs=5e-3)
        assert "zero" in _kinds(fake)


def test_set_zero_refused_while_holding(channel):
    with rig(AK80_64, channel, position=0.7) as (fake, bus, motor):
        with _safety(bus, motor) as s:
            s.arm()
            hold(s, 0.05)
            with pytest.raises(SafetyFault, match="set_zero"):
                s.set_zero()
        assert "zero" not in _kinds(fake)


def test_begin_stage_recenters_window_and_checks_limits(channel):
    with rig(AK60_39, channel, position=0.1) as (fake, bus, motor):
        with _safety(bus, motor) as s:
            s.arm()
            hold(s, 0.05, kp=0, kd=0)
            assert s.begin_stage(BenchLimits(damp_time=0.1, position_window=0.2), "test") == pytest.approx(0.1, abs=2e-3)
            with pytest.raises(ValueError):
                s.begin_stage(BenchLimits(max_torque=1000), "too much")


# --- Structure: only safety.py sends -----------------------------------------------------------
def test_only_safety_and_the_fake_motor_send_frames():
    pkg = Path(__file__).resolve().parents[1] / "actuator_sysid"
    allowed = {"bus.py", "safety.py", "fake_motor.py"}  # bus.py defines send(); fake only on virtual buses
    offenders = [p.name for p in pkg.glob("*.py") if p.name not in allowed and re.search(r"\.send\(", p.read_text())]
    assert offenders == []
