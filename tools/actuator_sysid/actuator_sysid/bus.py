"""CAN bus access for the bench tools, behind the real-hardware lock (D11).

Two kinds of bus are allowed today:
  - python-can "virtual": in-process, no root. Timestamps = time.time() at send ("python").
  - Linux vcan (SocketCAN virtual device, e.g. vcan0): same driver as the CANable, so the same
    code path. Timestamps come from the kernel (SO_TIMESTAMPNS, set by python-can) ("kernel").
Real adapters (e.g. CANable can0) are refused while LIVE_HARDWARE_ENABLED is False.

Every bus is opened with receive_own_messages, so each frame we send comes back as an "echo" with
its own timestamp: command time and feedback time then come from the same clock (latency, Task 11).
"""
import collections
import os
import time
from dataclasses import dataclass

import can

from .protocol import CanFrame

# D11: real hardware stays locked until the Task 10 reviewer checkpoint. Set this to True only in
# a reviewed commit after that review. Even then, opening a real adapter needs live=True (--live),
# and the scripts make the operator type the motor id (confirm_live).
LIVE_HARDWARE_ENABLED = False

_ARPHRD_CAN = 280  # /sys/class/net/<dev>/type for CAN devices (linux/if_arp.h)


class HardwareLocked(RuntimeError):
    """Raised when code tries to open a real CAN adapter."""


@dataclass(frozen=True)
class RxFrame:
    """A received frame with its timestamp (s, wall clock). is_echo = a frame we sent ourselves."""
    frame: CanFrame
    timestamp: float
    is_echo: bool


def is_virtual_can(channel: str) -> bool:
    """True if `channel` is a Linux virtual CAN device. Checked by where the kernel puts the device
    (/sys/devices/virtual/net), not by its name, so a real adapter renamed "vcan0" is still refused."""
    if not channel or "/" in channel:
        return False
    try:
        with open(f"/sys/class/net/{channel}/type") as f:
            is_can = int(f.read().strip()) == _ARPHRD_CAN
    except (OSError, ValueError):
        return False
    return is_can and os.path.isdir(f"/sys/devices/virtual/net/{channel}")


def check_access(interface: str, channel: str, live: bool = False) -> str:
    """Return "virtual", "vcan" or "hardware"; raise HardwareLocked if the bus isn't allowed."""
    if interface == "virtual":
        return "virtual"
    if interface == "socketcan" and is_virtual_can(channel):
        return "vcan"
    if not LIVE_HARDWARE_ENABLED:
        raise HardwareLocked(
            f"{interface}:{channel} is not a virtual bus. Real hardware is locked until the Task 10 "
            "reviewer checkpoint (D11): bus.LIVE_HARDWARE_ENABLED is False.")
    if not live:
        raise HardwareLocked(f"{interface}:{channel} is real hardware: pass live=True (--live).")
    if interface != "socketcan":
        # Only SocketCAN gives kernel timestamps; other interfaces were never tested here.
        raise HardwareLocked(f"interface {interface!r} is not supported for hardware; use socketcan")
    return "hardware"


def confirm_live(motor_id: int, input_fn=input) -> None:
    """Ask the operator to type the motor id before anything is sent to real hardware."""
    answer = input_fn(f"LIVE HARDWARE. E-stop in reach? Type the motor id ({motor_id}) to continue: ")
    if answer.strip() != str(motor_id):
        raise HardwareLocked("live run not confirmed")


class CanBus:
    """One CAN connection. Only safety.Safety (and the fake motor, on virtual buses) call send()."""

    def __init__(self, interface: str = "virtual", channel: str = "sysid", live: bool = False,
                 echo: bool = True):
        self.kind = check_access(interface, channel, live)
        self.interface, self.channel = interface, channel
        self.timestamp_source = "kernel" if interface == "socketcan" else "python"
        self._echo = echo
        self._bus = can.Bus(interface=interface, channel=channel, receive_own_messages=echo)
        # Frames sent but not yet echoed, oldest first. python-can marks a frame as not received
        # (is_rx False) when it came from ANY socket on this host (MSG_DONTROUTE), so on vcan a fake
        # motor's replies look like our own echoes. Matching against what we sent tells them apart.
        self._pending = collections.deque(maxlen=1024)

    def send(self, frame: CanFrame) -> None:
        msg = can.Message(arbitration_id=frame.arbitration_id, data=frame.data,
                          is_extended_id=frame.is_extended_id)
        if self._echo:
            self._pending.append(_key(frame))
        self._bus.send(msg)

    def recv(self, timeout: float = 0.0) -> RxFrame | None:
        """Next frame, or None after `timeout` seconds (0 = don't wait)."""
        msg = self._bus.recv(timeout)
        if msg is None:
            return None
        frame = CanFrame(msg.arbitration_id, bytes(msg.data), msg.is_extended_id)
        return RxFrame(frame, msg.timestamp, not msg.is_rx and self._match_echo(frame))

    def _match_echo(self, frame: CanFrame) -> bool:
        # Echoes come back in send order. Look a few entries deep in case one was lost, and drop
        # anything older than the match so a lost echo can't shift every later match.
        key = _key(frame)
        for i, pending in enumerate(list(self._pending)[:8]):
            if pending == key:
                for _ in range(i + 1):
                    self._pending.popleft()
                return True
        return False

    def close(self) -> None:
        self._bus.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _key(frame: CanFrame) -> tuple:
    return frame.arbitration_id, frame.is_extended_id, bytes(frame.data)


def wall_time() -> float:
    """Clock of the frame timestamps (kernel SO_TIMESTAMPNS and time.time() are both CLOCK_REALTIME)."""
    return time.time()
