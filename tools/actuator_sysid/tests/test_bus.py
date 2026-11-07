"""bus.py: hardware lock (D11), echo handling, timestamps. vcan tests skip unless vcan0 is up."""
import time

import pytest

from actuator_sysid import bus as busmod
from actuator_sysid.bus import CanBus, HardwareLocked, check_access, confirm_live, is_virtual_can
from actuator_sysid.protocol import CanFrame

VCAN = "vcan0"
needs_vcan = pytest.mark.skipif(not is_virtual_can(VCAN), reason="vcan0 not set up (needs sudo, see README)")


# --- Hardware lock ----------------------------------------------------------------------------
def test_lock_is_on_in_this_commit():
    # Flip only in a reviewed commit after the Task 10 reviewer checkpoint (D11).
    assert busmod.LIVE_HARDWARE_ENABLED is False


def test_virtual_bus_allowed():
    assert check_access("virtual", "anything") == "virtual"


@pytest.mark.parametrize("interface,channel", [("socketcan", "can0"), ("socketcan", "lo"), ("slcan", "/dev/ttyACM0"),
                                               ("pcan", "PCAN_USBBUS1"), ("socketcan", "../vcan0")])
def test_real_or_unknown_buses_refused_while_locked(interface, channel):
    with pytest.raises(HardwareLocked, match="D11"):
        check_access(interface, channel)
    with pytest.raises(HardwareLocked):
        check_access(interface, channel, live=True)  # --live alone doesn't unlock


def test_can_bus_refuses_before_opening_anything():
    with pytest.raises(HardwareLocked):
        CanBus("socketcan", "can0")


def test_unlocked_still_needs_live_and_socketcan(monkeypatch):
    monkeypatch.setattr(busmod, "LIVE_HARDWARE_ENABLED", True)
    with pytest.raises(HardwareLocked, match="live=True"):
        check_access("socketcan", "can0")
    with pytest.raises(HardwareLocked, match="not supported"):
        check_access("slcan", "/dev/ttyACM0", live=True)
    assert check_access("socketcan", "can0", live=True) == "hardware"


def test_is_virtual_can_checks_the_device_not_the_name():
    assert not is_virtual_can("can0")  # doesn't exist here (or is a real adapter)
    assert not is_virtual_can("lo")  # virtual, but not a CAN device
    assert not is_virtual_can("")
    assert not is_virtual_can("../../etc")


def test_confirm_live_needs_the_exact_motor_id():
    confirm_live(40, input_fn=lambda prompt: " 40 ")
    for answer in ("", "y", "41", "yes 40"):
        with pytest.raises(HardwareLocked):
            confirm_live(40, input_fn=lambda prompt, a=answer: a)


# --- Echo and timestamps ----------------------------------------------------------------------
def _recv_all(bus, timeout=0.2):
    out, end = [], time.monotonic() + timeout
    while time.monotonic() < end:
        rx = bus.recv(0.01)
        if rx is not None:
            out.append(rx)
    return out


def _echo_and_reply(interface, channel):
    host, other = CanBus(interface, channel), CanBus(interface, channel, echo=False)
    try:
        sent = CanFrame(0x28, bytes(range(8)), False)
        t0 = time.time()
        host.send(sent)
        got_other = _recv_all(other, 0.1)
        reply = CanFrame(0x28, bytes([0x28] + [0] * 7), False)  # same id as ours, like AK 2.0
        other.send(reply)
        got_host = _recv_all(host, 0.1)
        t1 = time.time()
    finally:
        host.close()
        other.close()
    assert [r.frame for r in got_other] == [sent] and not got_other[0].is_echo
    assert [(r.frame, r.is_echo) for r in got_host] == [(sent, True), (reply, False)]
    for r in got_host:
        assert t0 - 0.001 <= r.timestamp <= t1 + 0.001
    assert got_host[0].timestamp <= got_host[1].timestamp
    return host


def test_virtual_echo_and_reply(channel):
    assert _echo_and_reply("virtual", channel).timestamp_source == "python"


def test_lost_echo_does_not_shift_later_matches(channel):
    host = CanBus("virtual", channel)
    try:
        host._pending.append((0x99, False, b"lost"))  # pretend an echo never came back
        frame = CanFrame(0x10, b"\x01", False)
        host.send(frame)
        assert host.recv(0.1).is_echo
        assert len(host._pending) == 0
    finally:
        host.close()


@needs_vcan
def test_vcan_echo_reply_and_kernel_timestamps():
    # On vcan, python-can marks frames from any local socket as not received (MSG_DONTROUTE);
    # bus.py must still tell our echo from the other socket's reply.
    host = _echo_and_reply("socketcan", VCAN)
    assert host.kind == "vcan" and host.timestamp_source == "kernel"


@needs_vcan
def test_vcan_timestamps_are_set_by_the_kernel_before_we_read():
    host, other = CanBus("socketcan", VCAN), CanBus("socketcan", VCAN, echo=False)
    try:
        other.send(CanFrame(0x123, b"\x00", False))
        time.sleep(0.05)  # frame waits in the socket; a user-space timestamp would be taken at recv
        t_read = time.time()
        rx = host.recv(0.1)
    finally:
        host.close()
        other.close()
    assert rx is not None and t_read - rx.timestamp >= 0.04
