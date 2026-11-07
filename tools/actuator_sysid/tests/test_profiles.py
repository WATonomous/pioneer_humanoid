"""profiles.py: shapes, every default inside the bench limits, and runs through Safety on a fake motor."""
import math

import numpy as np
import pytest
from conftest import rig

from actuator_sysid.logger import Recorder
from actuator_sysid.profiles import (PROFILES, BacklashProbe, Chirp, SpeedSweep, Step, TorqueHold, TorqueRamp,
                                     check_profile, make_profile, run)
from actuator_sysid.protocol import MitCommand
from actuator_sysid.safety import BenchLimits, Safety, check_command
from pioneer_humanoid.leg_params import AK60_39, AK80_64, AKH70_48

LEG_MOTORS = [AKH70_48, AK80_64, AK60_39]
DEFAULTS = [cls() for cls in PROFILES.values()]


def _ids(x):
    return getattr(x, "name", str(x))


# --- Every default profile fits the default bench caps and the motor ceilings -------------------
@pytest.mark.parametrize("spec", LEG_MOTORS, ids=_ids)
@pytest.mark.parametrize("profile", DEFAULTS, ids=_ids)
def test_defaults_pass_the_bench_limits(profile, spec):
    check_profile(profile, BenchLimits(), spec, start=0.3)


@pytest.mark.parametrize("profile", DEFAULTS, ids=_ids)
def test_defaults_start_and_end_at_offset_zero(profile):
    for t in (0.0, profile.duration, profile.duration + 5):
        cmd = profile.setpoint(t)
        assert cmd.p == 0 and cmd.t == 0


@pytest.mark.parametrize("profile,match", [
    (Chirp(amplitude=0.02), "v_des"),  # peak speed 2*pi*10*0.02 = 1.26 rad/s
    (Step(size=0.35), "jumps"),  # step bigger than the 0.3 rad tracking limit
    (Step(kp=12), "kp"),
    (SpeedSweep(amplitude=0.6), "window"),
    (SpeedSweep(speeds=(1.2,)), "v_des"),
    (TorqueRamp(peak=5.0), "worst-case"),  # 5 + kd 0.5 * 1 rad/s
    (TorqueHold(levels=(4.6,)), "worst-case"),
    (TorqueRamp(abort_distance=0.5), "abort_distance"),
    (BacklashProbe(kd=1.5), "kd"),
])
def test_profiles_outside_the_limits_are_rejected(profile, match):
    with pytest.raises(ValueError, match=match):
        check_profile(profile, BenchLimits(), AK60_39)


def test_raised_caps_allow_bigger_profiles():
    raised = BenchLimits().with_overrides({"max_speed": 3.0, "max_torque": 10.0}, reason="test")
    check_profile(Chirp(amplitude=0.04), raised, AK60_39)


# --- Shapes ------------------------------------------------------------------------------------
def test_step_levels():
    p = Step(size=0.1, hold=1.0, repeats=1)
    assert p.duration == 5.0
    assert [p.setpoint(t).p for t in (0.5, 1.5, 2.5, 3.5, 4.5)] == [0, 0.1, 0, -0.1, 0]
    assert p.setpoint(1.5).kp == 5.0 and p.setpoint(1.5).v == 0


def test_chirp_frequency_and_velocity_feed_forward():
    c = Chirp()
    assert c.frequency(c.lead) == pytest.approx(0.2) and c.frequency(c.lead + c.sweep) == pytest.approx(10.0)
    # Halfway through an exponential sweep: geometric mean sqrt(0.2 * 10).
    assert c.frequency(c.lead + c.sweep / 2) == pytest.approx(math.sqrt(2.0))
    dt = 1e-6
    for t in np.linspace(c.lead + 0.1, c.lead + c.sweep - 0.1, 50):
        numeric = (c.setpoint(t + dt).p - c.setpoint(t - dt).p) / (2 * dt)
        assert c.setpoint(t).v == pytest.approx(numeric, abs=1e-4)
    peak_v = max(abs(c.setpoint(t).v) for t in np.arange(0, c.duration, 1e-4))
    assert peak_v == pytest.approx(2 * math.pi * 10 * 0.015, rel=0.01)


def test_chirp_rejects_bad_band():
    with pytest.raises(ValueError):
        Chirp(f0=10, f1=0.2)


def test_speed_sweep_feed_forward_matches_slope():
    s = SpeedSweep(speeds=(0.2, 0.8), amplitude=0.4)
    times = np.array(s.sample_times(0.001))
    p = np.array([s.setpoint(t).p for t in times])
    assert p.max() == pytest.approx(0.4) and p.min() == pytest.approx(-0.4)
    for t in np.linspace(0.6, s.duration - 0.6, 200):
        slope = (s.setpoint(t + 1e-5).p - s.setpoint(t - 1e-5).p) / 2e-5
        if abs(slope) > 1e-9:  # skip holds and corners
            assert s.setpoint(t).v == pytest.approx(slope, rel=1e-3)
    assert {abs(round(s.setpoint(t).v, 6)) for t in times} == {0.0, 0.2, 0.8}


def test_torque_ramp_rate_and_peak():
    r = TorqueRamp(peak=3.0, rate=0.5, lead=1.0)
    assert r.duration == pytest.approx(1 + 24 + 1)
    assert r.setpoint(1 + 6).t == pytest.approx(3.0) and r.setpoint(1 + 18).t == pytest.approx(-3.0)
    assert r.setpoint(1 + 3).t == pytest.approx(1.5)
    assert all(r.setpoint(t).kp == 0 for t in r.sample_times(0.1))


def test_torque_hold_levels():
    h = TorqueHold(levels=(1.0, -2.0), hold=3.0, rest=2.0, ramp=0.2)
    assert h.setpoint(2 + 0.2 + 1.5).t == pytest.approx(1.0)
    assert h.setpoint(2 + 3.4 + 2 + 0.2 + 1.5).t == pytest.approx(-2.0)
    assert h.setpoint(2 + 0.1).t == pytest.approx(0.5)  # ramped, no torque step


def test_backlash_probe_is_small_and_slow():
    b = BacklashProbe()
    taus = [b.setpoint(t).t for t in b.sample_times(0.01)]
    assert max(taus) == pytest.approx(1.0) and min(taus) == pytest.approx(-1.0)
    assert b.abort_distance < BenchLimits().position_window


def test_make_profile_parses_cli_strings():
    p = make_profile("speed_sweep", speeds="0.1,0.3", amplitude="0.2", cycles="2")
    assert p.speeds == (0.1, 0.3) and p.amplitude == 0.2 and p.cycles == 2
    assert make_profile("step", size=0.05).size == 0.05
    with pytest.raises(ValueError, match="unknown"):
        make_profile("step", amplitude=1)
    with pytest.raises(KeyError):
        make_profile("trapezoid")


def test_params_are_recorded():
    params = TorqueRamp().params()
    assert params["name"] == "torque_ramp" and params["kp"] == 0 and params["abort_distance"] == 0.3
    assert params["duration_s"] == pytest.approx(TorqueRamp().duration)


# --- Running through Safety on the fake motor -------------------------------------------------
SHORT = [
    Step(hold=0.2, repeats=1),
    Chirp(sweep=1.0, lead=0.2, f1=5.0, amplitude=0.02),
    SpeedSweep(speeds=(0.4, 0.8), amplitude=0.1, hold=0.1),
    TorqueRamp(peak=0.5, rate=5.0, lead=0.1),
    TorqueHold(levels=(0.5, -0.5), hold=0.2, rest=0.1, ramp=0.05),
    BacklashProbe(peak=0.5, rate=5.0, cycles=1, lead=0.1),
]


@pytest.mark.parametrize("spec", [AK60_39, AK80_64], ids=_ids)
@pytest.mark.parametrize("profile", SHORT, ids=_ids)
def test_profile_runs_through_safety_on_fake_motor(channel, spec, profile):
    # High friction: the torque profiles must not break away here (that's the abort test below).
    rec = Recorder()
    with rig(spec, channel, position=0.4, coulomb=1.0) as (fake, bus, motor):
        with Safety(bus, motor, BenchLimits(damp_time=0.1)) as s:
            s.listeners.append(rec)
            s.arm()
            result = run(s, profile, recorder=rec)
        assert result.ended == "done" and s.fault_reason is None
    sent = [c for c in rec.commands if c[6] == "cmd"]
    assert len(sent) >= 0.9 * profile.duration / 0.002
    start = s.start_position
    for _, p, v, kp, kd, t, _ in sent:  # every command the motor got was inside the limits
        check_command(MitCommand(p, v, kp, kd, t), BenchLimits(), spec, start)
    assert result.ticker["missed"] <= 5


def test_torque_profile_stops_cleanly_at_abort_distance(channel):
    with rig(AK60_39, channel, coulomb=0.2) as (fake, bus, motor):
        with Safety(bus, motor, BenchLimits(damp_time=0.1)) as s:
            s.arm()
            # 0.6 Nm vs 0.2 Nm friction: settles near (0.6 - 0.2) / (kd 0.5 + 0.05) = 0.7 rad/s, under 1 rad/s.
            profile = TorqueHold(levels=(0.6,), hold=2.0, rest=0.1, ramp=0.05)
            result = run(s, profile)
    assert result.ended == "abort_distance" and s.fault_reason is None
    assert result.elapsed < profile.duration


def test_run_refuses_unarmed_and_bad_profiles(channel):
    with rig(AK60_39, channel) as (fake, bus, motor):
        with Safety(bus, motor, BenchLimits(damp_time=0.1)) as s:
            with pytest.raises(RuntimeError, match="arm"):
                run(s, Step())
            s.arm()
            with pytest.raises(ValueError):
                run(s, Step(size=0.4))
        assert not any(k == "mit" and c.kp > 0 for k, c in fake.received)  # nothing of the step was sent
