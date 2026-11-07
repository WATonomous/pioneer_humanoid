"""The fake motor itself: sanity checks so the safety tests mean something."""
import time

import pytest

from actuator_sysid import protocol as pr
from actuator_sysid.bus import CanBus
from actuator_sysid.fake_motor import FakeMotor
from pioneer_humanoid.leg_params import AK60_39, AK80_64


def _send_for(bus, frame, seconds, period=0.002):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        bus.send(frame)
        time.sleep(period)


def test_ak3_pd_moves_to_target_and_streams_feedback(channel):
    with FakeMotor(AK60_39, 1, channel=channel, coulomb=0.0) as fake, CanBus("virtual", channel) as bus:
        _send_for(bus, pr.mit_command(AK60_39, 1, 0.2, 0, 5, 0.5, 0), 1.0)
        # t = 0 arrives as -0.0195 Nm (no middle code in 12 bits) -> offset 0.0195 / kp 4.88 = 0.004 rad.
        assert fake.position == pytest.approx(0.2 - 0.004, abs=1.5e-3)
        while bus.recv(0.0) is not None:  # drop what queued up during the moves
            pass
        n = 0
        end = time.monotonic() + 0.1
        while time.monotonic() < end:
            rx = bus.recv(0.01)
            n += rx is not None and not rx.is_echo and rx.frame.arbitration_id >> 8 == pr.AK3_STATUS
        assert 60 <= n <= 110  # ~1000 Hz


def test_ak2_needs_enter_and_replies_per_command(channel):
    with FakeMotor(AK80_64, 2, channel=channel) as fake, CanBus("virtual", channel, echo=False) as bus:
        bus.send(pr.mit_command(AK80_64, 2, 0, 0, 0, 0, 0))
        assert bus.recv(0.05) is None  # not in MIT mode yet
        bus.send(pr.ak2_enter(2))
        bus.send(pr.mit_command(AK80_64, 2, 0, 0, 0, 0, 0))
        rx = bus.recv(0.05)
        assert rx is not None and pr.decode_ak2_feedback(AK80_64, rx.frame).motor_id == 2
        assert [k for k, _ in fake.received] == ["enter", "mit"]


def test_can_timeout_makes_it_limp(channel):
    with FakeMotor(AK60_39, 1, channel=channel, can_timeout=0.1) as fake, CanBus("virtual", channel) as bus:
        bus.send(pr.mit_command(AK60_39, 1, 0, 0, 1, 0.1, 0))
        time.sleep(0.03)
        assert fake.enabled
        time.sleep(0.15)
        assert not fake.enabled


def test_gravity_pendulum_falls_when_limp(channel):
    # Not started: step the physics by hand.
    fake = FakeMotor(AK60_39, 1, channel=channel, position=0.5, gravity=5.0, load_inertia=0.05, coulomb=0.0)
    fake._integrate(0.05)
    assert fake.position < 0.5 and fake.omega < 0
    fake._bus.close()
