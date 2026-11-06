"""CubeMars MIT ("force control") CAN codecs for the leg motors: AK 3.0 and AK 2.0.

Pure functions, bytes in / bytes out, no bus access (Task 9's bus.py sends the frames).
Encoding ranges come from leg_params.MotorSpec (p_max, v_max, t_max, kp_max, kd_max; symmetric
p/v/t, gains from 0), and so does the protocol choice (MotorSpec.protocol).

AK 3.0 (AKH70-48, AK60-39): "AK Series Module Product Manual V3.2.0".
  Command, §4.2: extended id (8 << 8) | drive_id, KP-first payload
      byte 0: kp[11:4]   byte 1: kp[3:0]<<4 | kd[11:8]   byte 2: kd[7:0]
      byte 3: p[15:8]    byte 4: p[7:0]
      byte 5: v[11:4]    byte 6: v[3:0]<<4 | t[11:8]     byte 7: t[7:0]
  No enter/exit frames. Disable = mode 15 (§4.1.8), set origin = mode 5 (§4.1.6).
  Feedback, §4.3.1 (same frame in servo and MIT mode): 0x29 status = int16 position * 0.1 deg,
  int16 speed * 10 ERPM, int16 current * 0.01 A, int8 driver temp degC, uint8 error.
  Optional 0x2A = int32 position * 0.01 deg. All big-endian.
AK 2.0 (AK80-64): "AK Series Driver Manual V1.0.18 for AK 2.0", §5.3.
  Command: standard id = motor id, position-first payload
      byte 0: p[15:8]    byte 1: p[7:0]
      byte 2: v[11:4]    byte 3: v[3:0]<<4 | kp[11:8]    byte 4: kp[7:0]
      byte 5: kd[11:4]   byte 6: kd[3:0]<<4 | t[11:8]    byte 7: t[7:0]
  Special frames FF FF FF FF FF FF FF + FC enter MIT, FD exit, FE set zero (needed every power-on).
  Feedback: DATA[0] motor id, p(16), v(12), t(12) in the command's ranges, DATA[6] temp (byte - 40),
  DATA[7] error.

Encoding is formula A (user decision, #335 log Task 7):
  code = trunc((x - min) * (2^bits - 1) / (max - min))
the exact inverse of the manuals' uint_to_float. Both manuals print 2^bits instead, which overflows
at x = max (can_node clamps that); the two differ by at most one code. Formula A fits more of the
V3.2.0 example bytes (9 of 11 vs 6 of 11).

Not verified on hardware yet (Task 10): that 0x29 position is output-shaft degrees, the AK 2.0
feedback id, and the AK 2.0 temperature offset.
"""
import math
from dataclasses import dataclass
from typing import NamedTuple

from pioneer_humanoid.leg_params import MotorSpec

_RPM_TO_RAD_S = 2.0 * math.pi / 60.0

# --- AK 3.0 extended-id modes (id = mode << 8 | drive_id) -------------------------------------
AK3_MODE_SET_ORIGIN = 5
AK3_MODE_MIT = 8
AK3_MODE_DISABLE = 15
AK3_STATUS = 0x29  # feedback: status frame
AK3_POSITION = 0x2A  # feedback: optional 32-bit position frame (enabled in R-Link / mode 16)

# AK 3.0 error byte (§4.3.1), plus the disable acknowledgement (§4.1.8).
AK3_ERRORS = {
    0: "no fault",
    1: "motor over-temperature",
    2: "over-current",
    3: "over-voltage",
    4: "under-voltage",
    5: "encoder fault",
    6: "MOSFET over-temperature",
    7: "motor lock-up",
    0x77: "disable acknowledged",
}

# --- AK 2.0 special frames: FF x7 + code --------------------------------------------------------
AK2_ENTER = 0xFC
AK2_EXIT = 0xFD
AK2_SET_ZERO = 0xFE

_AK2_TEMP_OFFSET = 40  # manual receive code: temperature = byte - 40 (range -40..215 degC)


@dataclass(frozen=True)
class CanFrame:
    """One CAN frame. Field names match python-can's Message, so bus code can pass them through."""
    arbitration_id: int
    data: bytes
    is_extended_id: bool


class MitCodes(NamedTuple):
    """Packed integer codes of one MIT command: p 16 bits, the others 12 bits."""
    p: int
    v: int
    kp: int
    kd: int
    t: int


@dataclass(frozen=True)
class MotorFeedback:
    """Decoded feedback, output shaft, SI units."""
    motor_id: int
    position: float  # rad
    velocity: float  # rad/s
    torque: float  # Nm
    current: float | None  # A (AK 3.0 reports current; AK 2.0 reports torque only -> None)
    temperature: int  # degC
    error: int  # raw error byte (AK 3.0 meanings in AK3_ERRORS)


# --- Fixed-point conversion -------------------------------------------------------------------
def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Physical value -> code in [0, 2^bits - 1]. Clamps to [x_min, x_max], then truncates."""
    if not math.isfinite(x):
        raise ValueError(f"cannot encode non-finite value {x}")
    top = (1 << bits) - 1
    if x <= x_min:
        return 0
    if x >= x_max:
        return top
    code = int((x - x_min) * top / (x_max - x_min))
    return min(max(code, 0), top)


def uint_to_float(code: int, x_min: float, x_max: float, bits: int) -> float:
    """Code -> physical value (manual uint_to_float)."""
    top = (1 << bits) - 1
    if not 0 <= code <= top:
        raise ValueError(f"code {code} does not fit in {bits} bits")
    return code * (x_max - x_min) / top + x_min


def erpm_to_rad_s(erpm: float, motor: MotorSpec) -> float:
    """Electrical rpm -> output shaft rad/s (V3.2.0 §4.3.1: ERPM / pole pairs / reduction ratio)."""
    return erpm / motor.pole_pairs / motor.gear_ratio * _RPM_TO_RAD_S


# --- MIT command ------------------------------------------------------------------------------
def encode_mit(motor: MotorSpec, p: float, v: float, kp: float, kd: float, t: float) -> MitCodes:
    """Physical command -> codes, using the motor's MIT ranges. Out-of-range values clamp (as on
    the motor); limits that matter for safety are enforced before this, in safety.py (Task 9)."""
    return MitCodes(
        p=float_to_uint(p, -motor.p_max, motor.p_max, 16),
        v=float_to_uint(v, -motor.v_max, motor.v_max, 12),
        kp=float_to_uint(kp, 0.0, motor.kp_max, 12),
        kd=float_to_uint(kd, 0.0, motor.kd_max, 12),
        t=float_to_uint(t, -motor.t_max, motor.t_max, 12),
    )


def _check_codes(c: MitCodes) -> None:
    if not 0 <= c.p <= 0xFFFF:
        raise ValueError(f"p code {c.p} does not fit in 16 bits")
    for name in ("v", "kp", "kd", "t"):
        value = getattr(c, name)
        if not 0 <= value <= 0xFFF:
            raise ValueError(f"{name} code {value} does not fit in 12 bits")


def pack_ak3_fields(c: MitCodes) -> bytes:
    """AK 3.0 payload, KP-first (see module docstring)."""
    _check_codes(c)
    return bytes([
        c.kp >> 4,
        ((c.kp & 0xF) << 4) | (c.kd >> 8),
        c.kd & 0xFF,
        c.p >> 8,
        c.p & 0xFF,
        c.v >> 4,
        ((c.v & 0xF) << 4) | (c.t >> 8),
        c.t & 0xFF,
    ])


def pack_ak2_fields(c: MitCodes) -> bytes:
    """AK 2.0 payload, position-first (see module docstring)."""
    _check_codes(c)
    return bytes([
        c.p >> 8,
        c.p & 0xFF,
        c.v >> 4,
        ((c.v & 0xF) << 4) | (c.kp >> 8),
        c.kp & 0xFF,
        c.kd >> 4,
        ((c.kd & 0xF) << 4) | (c.t >> 8),
        c.t & 0xFF,
    ])


def _check_motor_id(motor_id: int) -> None:
    # Both protocols carry the id in 8 bits (AK 3.0 id bits 7..0, AK 2.0 feedback DATA[0]).
    # Raise instead of masking: a masked id addresses a different motor.
    if not 0 <= motor_id <= 0xFF:
        raise ValueError(f"motor id {motor_id} is outside 0..255")


def ak3_can_id(mode: int, drive_id: int) -> int:
    """AK 3.0 extended id: mode in bits 28..8, drive id in bits 7..0."""
    _check_motor_id(drive_id)
    return (mode << 8) | drive_id


def mit_command(motor: MotorSpec, motor_id: int, p: float, v: float, kp: float, kd: float,
                t: float) -> CanFrame:
    """MIT command frame for the motor's protocol. Units: rad, rad/s, Nm/rad, Nm*s/rad, Nm."""
    codes = encode_mit(motor, p, v, kp, kd, t)
    if motor.protocol == "ak3":
        return CanFrame(ak3_can_id(AK3_MODE_MIT, motor_id), pack_ak3_fields(codes), True)
    if motor.protocol == "ak2":
        _check_motor_id(motor_id)
        return CanFrame(motor_id, pack_ak2_fields(codes), False)
    raise ValueError(f"{motor.name}: unknown protocol {motor.protocol!r}")


# --- AK 3.0 control frames --------------------------------------------------------------------
def ak3_disable(drive_id: int) -> CanFrame:
    """Disable (§4.1.8). No data; the motor answers one 0x29 frame with error byte 0x77."""
    return CanFrame(ak3_can_id(AK3_MODE_DISABLE, drive_id), b"", True)


def ak3_set_origin(drive_id: int, permanent: bool = False) -> CanFrame:
    """Set the current position as zero (§4.1.6). permanent=True writes flash."""
    return CanFrame(ak3_can_id(AK3_MODE_SET_ORIGIN, drive_id), bytes([1 if permanent else 0]), True)


# --- AK 2.0 special frames --------------------------------------------------------------------
def _ak2_special(motor_id: int, code: int) -> CanFrame:
    _check_motor_id(motor_id)
    return CanFrame(motor_id, bytes([0xFF] * 7 + [code]), False)


def ak2_enter(motor_id: int) -> CanFrame:
    """Enter MIT mode (must come before any command)."""
    return _ak2_special(motor_id, AK2_ENTER)


def ak2_exit(motor_id: int) -> CanFrame:
    """Exit MIT mode (motor goes limp)."""
    return _ak2_special(motor_id, AK2_EXIT)


def ak2_set_zero(motor_id: int) -> CanFrame:
    """Set the current position as zero. AK80-64 has no absolute position: do this every session."""
    return _ak2_special(motor_id, AK2_SET_ZERO)


# --- Feedback ---------------------------------------------------------------------------------
def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _int16(data: bytes, i: int) -> int:
    return int.from_bytes(data[i:i + 2], "big", signed=True)


def decode_ak3_status(motor: MotorSpec, frame: CanFrame) -> MotorFeedback:
    """AK 3.0 status frame 0x29 -> output-shaft SI units. Torque = Kt * current."""
    _require(motor.protocol == "ak3", f"{motor.name} is not an AK 3.0 motor")
    _require(frame.is_extended_id, "AK 3.0 feedback is an extended frame")
    _require(frame.arbitration_id >> 8 == AK3_STATUS, f"id {frame.arbitration_id:#x} is not a 0x29 frame")
    _require(len(frame.data) == 8, f"status frame needs 8 bytes, got {len(frame.data)}")
    d = frame.data
    current = _int16(d, 4) * 0.01
    return MotorFeedback(
        motor_id=frame.arbitration_id & 0xFF,
        position=math.radians(_int16(d, 0) * 0.1),
        velocity=erpm_to_rad_s(_int16(d, 2) * 10.0, motor),
        torque=current * motor.torque_constant,
        current=current,
        temperature=int.from_bytes(d[6:7], "big", signed=True),
        error=d[7],
    )


def decode_ak3_position(frame: CanFrame) -> float:
    """AK 3.0 optional frame 0x2A -> output position in rad (int32 * 0.01 deg)."""
    _require(frame.is_extended_id, "AK 3.0 feedback is an extended frame")
    _require(frame.arbitration_id >> 8 == AK3_POSITION, f"id {frame.arbitration_id:#x} is not a 0x2A frame")
    _require(len(frame.data) >= 4, f"position frame needs 4 bytes, got {len(frame.data)}")
    return math.radians(int.from_bytes(frame.data[0:4], "big", signed=True) * 0.01)


def decode_ak2_feedback(motor: MotorSpec, frame: CanFrame) -> MotorFeedback:
    """AK 2.0 MIT reply -> SI units, using the motor's MIT ranges."""
    _require(motor.protocol == "ak2", f"{motor.name} is not an AK 2.0 motor")
    _require(not frame.is_extended_id, "AK 2.0 feedback is a standard frame")
    _require(len(frame.data) == 8, f"feedback needs 8 bytes, got {len(frame.data)}")
    d = frame.data
    p_code = (d[1] << 8) | d[2]
    v_code = (d[3] << 4) | (d[4] >> 4)
    t_code = ((d[4] & 0xF) << 8) | d[5]
    return MotorFeedback(
        motor_id=d[0],
        position=uint_to_float(p_code, -motor.p_max, motor.p_max, 16),
        velocity=uint_to_float(v_code, -motor.v_max, motor.v_max, 12),
        torque=uint_to_float(t_code, -motor.t_max, motor.t_max, 12),
        current=None,
        temperature=d[6] - _AK2_TEMP_OFFSET,
        error=d[7],
    )
