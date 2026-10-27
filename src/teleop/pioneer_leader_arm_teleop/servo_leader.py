"""Shared STS3215 leader-arm reader.

Used by ``pioneer_leader_arm_teleop.py`` (leader -> sim, with optional recording) and
``encoder_test.py`` (bring-up readout).

The leader arm is an INPUT DEVICE. Torque is written OFF for every servo at
startup, on every re-zero and again on close, and is never enabled anywhere in
this module. Nothing here can drive a physical actuator -- the only thing that
moves is the simulated arm.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

# Servo label -> bus ID. A..F map onto the arm's six joints (joint1L..joint6l) in
# order; G is the gripper. Iteration order matters: read_radians() returns a tuple
# in this order and callers index it positionally.
SERVO_IDS: dict[str, int] = {
    "A": 2,  # joint1L shoulder flexion
    "B": 1,  # joint2l shoulder abduction
    "C": 3,  # joint3l shoulder rotation
    "D": 5,  # joint4l elbow flexion
    "E": 4,  # joint5l forearm rotation
    "F": 7,  # joint6l wrist
    "G": 6,  # gripper
}

# Labels that drive arm joints, in joint order, and the one that drives the gripper.
ARM_SERVOS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
GRIPPER_SERVO = "G"
assert tuple(SERVO_IDS) == ARM_SERVOS + (GRIPPER_SERVO,)

COUNTS_PER_REV = 4096
COUNTS_PER_RAD = COUNTS_PER_REV / (2.0 * math.pi)

# STS3215 control-table addresses.
TORQUE_ENABLE_ADDRESS = 40
PRESENT_POSITION_ADDRESS = 56

# Protocol endianness selector for the SCServo-flavoured SDK. 0 = STS/SMS
# (little-endian), which is what sms_sts hardcodes; 1 would be the older SCS
# series. Wrong value here reads plausible-looking but wrong counts.
_STS_PROTOCOL_END = 0


class _PortBoundHandler:
    """Adapts the SCServo SDK's packet handler to the STServo ``sms_sts`` API.

    Two different PyPI packages both install as ``scservo_sdk``:

    * Feetech's **STServo** SDK exposes ``sms_sts(port)``, whose methods take
      ``(servo_id, address)``.
    * The **SCServo** SDK exposes ``PacketHandler(protocol_end)``, whose methods
      take ``(port, servo_id, address)``.

    The wire protocol is the same; only the port binding differs. This wrapper
    binds the port once so the rest of this module can speak the ``sms_sts``
    API regardless of which package is installed.
    """

    def __init__(self, port, packet_handler):
        self._port = port
        self._handler = packet_handler

    def read2ByteTxRx(self, servo_id: int, address: int):
        return self._handler.read2ByteTxRx(self._port, servo_id, address)

    def write1ByteTxRx(self, servo_id: int, address: int, data: int):
        return self._handler.write1ByteTxRx(self._port, servo_id, address, data)


def open_servo_bus(port_name: str):
    """Return ``(port, servo)`` for ``port_name``, without opening the port."""
    try:
        from scservo_sdk import PortHandler
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "scservo_sdk is not importable: pip install feetech-servo-sdk "
            "(already in the simulation_isaac image)."
        ) from exc

    port = PortHandler(port_name)

    # Prefer the STServo class when present; fall back to the SCServo handler.
    try:
        from scservo_sdk import sms_sts

        return port, sms_sts(port)
    except ImportError:
        pass

    try:
        from scservo_sdk import PacketHandler
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "scservo_sdk exposes neither sms_sts nor PacketHandler; this is not "
            "a supported Feetech SDK build."
        ) from exc

    return port, _PortBoundHandler(port, PacketHandler(_STS_PROTOCOL_END))


def open_port(port, port_name: str, baud: int) -> None:
    """Open ``port`` at ``baud``, converting driver errors into actionable ones.

    pyserial raises rather than returning False when the device node exists but is
    not readable, which is by far the most common failure here: /dev/ttyACM0 is
    owned by root:dialout, and adding yourself to that group does not affect
    already-running login sessions.
    """
    try:
        opened = port.openPort()
    except Exception as exc:
        raise RuntimeError(_port_error_hint(port_name, exc)) from exc
    if not opened:
        raise RuntimeError(_port_error_hint(port_name, None))
    if not port.setBaudRate(baud):
        port.closePort()
        raise RuntimeError(f"could not set baud rate to {baud} on {port_name}")


def _port_error_hint(port_name: str, exc: Exception | None) -> str:
    detail = f": {exc}" if exc is not None else ""
    hint = (
        f"could not open {port_name}{detail}\n"
        f"  - is the leader arm plugged in?  ls -l {port_name}\n"
        "  - permission denied? you need the 'dialout' group:\n"
        "      sudo usermod -aG dialout $USER   # then LOG OUT and back in\n"
        "    'id -nG' must list dialout; /etc/group alone is not enough, because a\n"
        "    login session keeps the groups it started with.\n"
        "  - already open? only one process may hold the port at a time."
    )
    return hint


def wrapped_count_delta(position: int, previous: int) -> int:
    """Shortest signed displacement between two consecutive samples."""
    return (
        position - previous + COUNTS_PER_REV // 2
    ) % COUNTS_PER_REV - COUNTS_PER_REV // 2


class ServoLeader:
    """Read every leader servo in radians while keeping all torque off.

    With ``zeros`` (raw counts per servo ID, from a calibration file) angles are relative to that
    calibrated pose; without, relative to wherever the leader sits at startup.
    """

    def __init__(
        self,
        port_name: str,
        baud: int,
        servo_ids: dict[str, int] | None = None,
        zeros: dict[int, int] | None = None,
    ):
        # servo_ids lets a bring-up tool watch a subset (or a different bus layout)
        # without touching the module-level map the teleop scripts rely on.
        self.servo_ids = dict(SERVO_IDS if servo_ids is None else servo_ids)
        self._port, self._servo = open_servo_bus(port_name)
        self._zeros: dict[int, int] = {}
        self._last_positions: dict[int, int] = {}
        self._relative_counts: dict[int, int] = {}
        open_port(self._port, port_name, baud)
        try:
            self.rezero(announce=zeros is None)
            if zeros is not None:
                self._apply_zeros(zeros)
        except Exception:
            self.close()
            raise

    def _apply_zeros(self, zeros: dict[int, int]) -> None:
        """Measure from calibrated zeros instead of the startup pose.

        The startup offset takes the shortest way round (within +/-180 deg), so start the leader
        near its calibrated pose; reads then unwrap incrementally as usual.
        """
        missing = [sid for sid in self.servo_ids.values() if sid not in zeros]
        if missing:
            raise RuntimeError(f"leader calibration has no zero for servo IDs {missing}")
        for servo_id, start in self._last_positions.items():
            self._relative_counts[servo_id] = wrapped_count_delta(start, zeros[servo_id])
        self._zeros = {sid: zeros[sid] for sid in self.servo_ids.values()}

    def _read_position(self, label: str, servo_id: int) -> int:
        position, result, error = self._servo.read2ByteTxRx(
            servo_id, PRESENT_POSITION_ADDRESS
        )
        if result != 0 or error != 0:
            raise RuntimeError(
                f"leader {label} (ID {servo_id}) read failed: "
                f"result={result}, error={error}"
            )
        return int(position)

    def rezero(self, announce: bool = True) -> None:
        """Disable torque on every servo and latch the current pose as zero."""
        zeros: dict[int, int] = {}
        for label, servo_id in self.servo_ids.items():
            result, error = self._servo.write1ByteTxRx(
                servo_id, TORQUE_ENABLE_ADDRESS, 0
            )
            if result != 0 or error != 0:
                raise RuntimeError(
                    f"leader {label} (ID {servo_id}) torque-off failed: "
                    f"result={result}, error={error}"
                )
            time.sleep(0.02)
            zeros[servo_id] = self._read_position(label, servo_id)
        self._zeros = zeros
        self._last_positions = zeros.copy()
        self._relative_counts = {servo_id: 0 for servo_id in zeros}
        if not announce:
            return
        print(
            "[INFO] Leader zeroed: "
            + ", ".join(
                f"{label}=ID{servo_id}@{zeros[servo_id]}"
                for label, servo_id in self.servo_ids.items()
            ),
            flush=True,
        )

    def read_radians(self) -> tuple[float, ...]:
        """Startup-relative angle per servo, in SERVO_IDS order."""
        angles = []
        for label, servo_id in self.servo_ids.items():
            position = self._read_position(label, servo_id)
            # Unwrap incrementally instead of choosing the shortest path back
            # to the startup reading. This preserves direction through the
            # encoder's 4095 -> 0 boundary and across multiple revolutions.
            step = wrapped_count_delta(position, self._last_positions[servo_id])
            self._relative_counts[servo_id] += step
            self._last_positions[servo_id] = position
            angles.append(self._relative_counts[servo_id] / COUNTS_PER_RAD)
        return tuple(angles)

    def last_positions(self) -> dict[int, int]:
        """Raw encoder counts from the most recent read, keyed by servo ID."""
        return dict(self._last_positions)

    def close(self) -> None:
        """Re-assert torque-off on every servo and hand the serial port back."""
        for servo_id in self.servo_ids.values():
            try:
                self._servo.write1ByteTxRx(servo_id, TORQUE_ENABLE_ADDRESS, 0)
            except Exception:
                pass
        try:
            self._port.closePort()
        except Exception:
            pass


def calibration_entry(positions: dict[int, int], servo_ids: dict[str, int] | None = None) -> dict:
    """Raw zero counts keyed by servo label, with its bus ID (checked on load by calibration_zeros)."""
    servo_ids = SERVO_IDS if servo_ids is None else servo_ids
    return {label: {"id": sid, "zero": int(positions[sid])} for label, sid in servo_ids.items()}


def load_calibration(path: Path, servo_ids: dict[str, int] | None = None) -> dict[int, int]:
    """Raw zero counts keyed by servo ID; fails if the file's bus IDs disagree with ``servo_ids``."""
    return calibration_zeros(json.loads(Path(path).read_text()), str(path), servo_ids)


def calibration_zeros(data: dict, where: str, servo_ids: dict[str, int] | None = None) -> dict[int, int]:
    """Zero counts by servo ID from ``{label: {"id", "zero"}}`` (calibration_entry's format)."""
    servo_ids = SERVO_IDS if servo_ids is None else servo_ids
    zeros = {}
    for label, sid in servo_ids.items():
        entry = data.get(label)
        if entry is None or entry.get("id") != sid:
            raise ValueError(f"{where}: servo {label} must have id {sid}, got {entry}")
        zeros[sid] = int(entry["zero"])
    return zeros


def parse_signs(
    value: str, servo_ids: dict[str, int] | None = None
) -> tuple[float, ...]:
    """Parse a ``--signs``-style comma list into one direction per servo.

    ``servo_ids`` is the servo map the signs apply to (default: SERVO_IDS).
    """
    servo_ids = SERVO_IDS if servo_ids is None else servo_ids
    count = len(servo_ids)
    try:
        signs = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(f"signs must contain {count} numbers") from exc
    if len(signs) != count or any(
        not math.isfinite(sign) or sign == 0.0 for sign in signs
    ):
        raise ValueError(f"signs must contain {count} finite non-zero numbers")
    return signs
