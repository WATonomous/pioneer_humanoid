"""Shared test helpers. Run in the sysid env: conda activate sysid && pytest tools/actuator_sysid/tests"""
import time
import uuid
from contextlib import contextmanager

import pytest

MOTOR_ID = 40
# Bus-related imports are inside the helpers, so test_protocol.py also runs without python-can.


@pytest.fixture
def channel():
    """A fresh python-can virtual channel name, so tests can't hear each other."""
    return f"test-{uuid.uuid4().hex[:10]}"


@contextmanager
def rig(spec, channel, interface="virtual", **fake_kw):
    """Fake motor + host bus + motor object on one bus."""
    from actuator_sysid.bus import CanBus
    from actuator_sysid.fake_motor import FakeMotor
    from actuator_sysid.motors import make_motor
    fake = FakeMotor(spec, MOTOR_ID, interface=interface, channel=channel, **fake_kw).start()
    bus = CanBus(interface, channel)
    try:
        yield fake, bus, make_motor(spec, MOTOR_ID)
    finally:
        bus.close()
        fake.stop()


def hold(safety, seconds, kp=2.0, kd=0.2, period=0.002):
    """Hold the start position with a PD command at 500 Hz (polls first, like the real loop)."""
    from actuator_sysid.protocol import MitCommand
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        safety.poll()
        safety.command(MitCommand(safety.start_position, 0.0, kp, kd, 0.0))
        time.sleep(period)
