"""Tests for actuator_sysid.protocol (CubeMars MIT codecs, AK 3.0 + AK 2.0).

Run on the host: conda activate watonomous_env && pytest tools/actuator_sysid/tests -p no:cacheprovider

Sources: AK 3.0 = "AK Series Module Product Manual V3.2.0" (2026-01-17) §4.1, §4.2, §4.3.1, §4.4.1.
AK 2.0 = "AK Series Driver Manual V1.0.18 for AK 2.0" §5.3 (it has no MIT example bytes, so its tests
check bit positions field by field instead).
Encoding = formula A: code = trunc((x - min) * (2^bits - 1) / (max - min)), the exact inverse of the
manuals' uint_to_float (user decision, see the #335 log, Task 7).
"""
import dataclasses
import math

import pytest

from actuator_sysid import protocol as pr
from pioneer_humanoid.leg_params import AK60_39, AK80_64, AKH70_48

LEG_MOTORS = [AKH70_48, AK80_64, AK60_39]

# The V3.2.0 §4.4.1 example frames decode with p +-12.5, v +-28, t +-54 (AK10-9 speed/torque, the
# older p range), not with any leg motor's table values. Same MotorSpec type, other ranges.
MANUAL_EXAMPLE_MOTOR = dataclasses.replace(AK60_39, p_max=12.5, v_max=28.0, t_max=54.0)


def _step(lo, hi, bits):
    """Physical size of one code step."""
    return (hi - lo) / ((1 << bits) - 1)


# --- float_to_uint / uint_to_float ------------------------------------------------------------
def test_ends_of_range_map_to_first_and_last_code():
    assert pr.float_to_uint(-12.56, -12.56, 12.56, 16) == 0
    assert pr.float_to_uint(12.56, -12.56, 12.56, 16) == 65535
    assert pr.float_to_uint(80.0, -80.0, 80.0, 12) == 4095


def test_out_of_range_is_clamped_not_wrapped():
    # The manual's 2^bits formula gives 4096 at max, which wraps to 0 in a 12-bit field
    # ("max torque" becomes "min torque"). Formula A can't, and inputs beyond the range clamp.
    assert pr.float_to_uint(1e6, -80.0, 80.0, 12) == 4095
    assert pr.float_to_uint(-1e6, -80.0, 80.0, 12) == 0


def test_truncates():
    # Mid-range of +-1 in 12 bits is code 2047.5 -> 2047 (drop decimals, as the manual does).
    assert pr.float_to_uint(0.0, -1.0, 1.0, 12) == 2047
    # 40 Nm below max on +-80: 40 * 4095 / 160 = 1023.75 -> 1023.
    assert pr.float_to_uint(-40.0, -80.0, 80.0, 12) == 1023


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_input_raises(bad):
    with pytest.raises(ValueError):
        pr.float_to_uint(bad, -1.0, 1.0, 12)


def test_decode_ends_and_bad_codes():
    assert pr.uint_to_float(0, -144.0, 144.0, 12) == pytest.approx(-144.0)
    assert pr.uint_to_float(4095, -144.0, 144.0, 12) == pytest.approx(144.0)
    for bad in (-1, 4096):
        with pytest.raises(ValueError):
            pr.uint_to_float(bad, -144.0, 144.0, 12)


@pytest.mark.parametrize("motor", LEG_MOTORS, ids=lambda m: m.name)
def test_round_trip_within_one_step(motor):
    fields = [  # (lo, hi, bits)
        (-motor.p_max, motor.p_max, 16),
        (-motor.v_max, motor.v_max, 12),
        (0.0, motor.kp_max, 12),
        (0.0, motor.kd_max, 12),
        (-motor.t_max, motor.t_max, 12),
    ]
    for lo, hi, bits in fields:
        step = _step(lo, hi, bits)
        for frac in (0.0, 0.013, 0.25, 0.5, 0.731, 0.999, 1.0):
            x = lo + frac * (hi - lo)
            back = pr.uint_to_float(pr.float_to_uint(x, lo, hi, bits), lo, hi, bits)
            # Truncation only ever rounds down, by less than one step.
            assert -1e-9 <= x - back < step + 1e-9, (motor.name, lo, hi, x, back)


# --- Ranges come from MotorSpec -------------------------------------------------------------
@pytest.mark.parametrize("motor", LEG_MOTORS, ids=lambda m: m.name)
def test_spec_maxima_give_all_ones_codes(motor):
    codes = pr.encode_mit(motor, p=motor.p_max, v=motor.v_max, kp=motor.kp_max, kd=motor.kd_max,
                          t=motor.t_max)
    assert codes == pr.MitCodes(p=0xFFFF, v=0xFFF, kp=0xFFF, kd=0xFFF, t=0xFFF)
    codes = pr.encode_mit(motor, p=-motor.p_max, v=-motor.v_max, kp=0.0, kd=0.0, t=-motor.t_max)
    assert codes == pr.MitCodes(p=0, v=0, kp=0, kd=0, t=0)


@pytest.mark.parametrize("motor", LEG_MOTORS, ids=lambda m: m.name)
def test_kd_5_is_4095(motor):
    assert pr.encode_mit(motor, p=0.0, v=0.0, kp=0.0, kd=5.0, t=0.0).kd == 4095


def test_codec_follows_the_spec_ranges():
    # Same torque, different t_max -> different code: the range is read from the MotorSpec.
    wide = dataclasses.replace(AK60_39, t_max=160.0)
    assert pr.encode_mit(AK60_39, p=0, v=0, kp=0, kd=0, t=40.0).t == 3071  # 120 * 4095 / 160 = 3071.25
    assert pr.encode_mit(wide, p=0, v=0, kp=0, kd=0, t=40.0).t == 2559  # 200 * 4095 / 320 = 2559.4


def test_gains_below_zero_clamp_to_zero():
    codes = pr.encode_mit(AKH70_48, p=0, v=0, kp=-3.0, kd=-0.1, t=0)
    assert (codes.kp, codes.kd) == (0, 0)


# --- AK 3.0 command frames (V3.2.0 §4.2, examples §4.4.1, drive id 0x68) ----------------------
def _hex(s: str) -> bytes:
    return bytes.fromhex(s)


@pytest.mark.parametrize(
    "label, p, v, kp, kd, t, manual",
    [
        ("velocity -6 rad/s, kd 2", 0.0, -6.0, 0.0, 2.0, 0.0, "00 06 66 7F FF 64 87 FF"),
        ("position -6 rad, kp 2, kd 2", -6.0, 0.0, 2.0, 2.0, 0.0, "01 06 66 42 8F 7F F7 FF"),
        ("torque 2 Nm", 0.0, 0.0, 0.0, 0.0, 2.0, "00 00 00 7F FF 7F F8 4B"),
        ("torque 4 Nm", 0.0, 0.0, 0.0, 0.0, 4.0, "00 00 00 7F FF 7F F8 97"),
    ],
)
def test_ak3_matches_manual_examples_exactly(label, p, v, kp, kd, t, manual):
    frame = pr.mit_command(MANUAL_EXAMPLE_MOTOR, 0x68, p=p, v=v, kp=kp, kd=kd, t=t)
    assert frame.data == _hex(manual), label
    assert frame.arbitration_id == 0x868  # manual "00 00 08 68"
    assert frame.is_extended_id


def test_ak3_manual_position_plus_6_within_one_count():
    # Manual BD 70 (48496); formula A gives 48495.9 -> 48495 (BD 6F). Every other field is exact.
    # (The manual's +6 rad/s example, 98 67 -> 0x986, fits no formula; -6 rad/s does, so it's
    # left out as a likely typo for 0x9B6.)
    manual = _hex("01 06 66 BD 70 7F F7 FF")
    got = pr.mit_command(MANUAL_EXAMPLE_MOTOR, 0x68, p=6.0, v=0.0, kp=2.0, kd=2.0, t=0.0).data
    assert got[:3] == manual[:3] and got[5:] == manual[5:]
    assert abs(int.from_bytes(got[3:5], "big") - int.from_bytes(manual[3:5], "big")) <= 1


def test_ak3_bit_positions():
    # KP-first: kp(12) kd(12) p(16) v(12) t(12). Distinct nibbles show any swapped field.
    data = pr.pack_ak3_fields(pr.MitCodes(p=0xABCD, v=0x123, kp=0x456, kd=0x789, t=0xABC))
    assert data == _hex("45 67 89 AB CD 12 3A BC")


def test_ak3_ids():
    assert pr.ak3_can_id(pr.AK3_MODE_MIT, 0x28) == 0x828
    assert pr.ak3_can_id(pr.AK3_STATUS, 0x28) == 0x2928
    for bad in (-1, 256):
        with pytest.raises(ValueError):
            pr.ak3_can_id(pr.AK3_MODE_MIT, bad)


def test_ak3_disable_and_set_origin_frames():
    off = pr.ak3_disable(0x28)  # §4.1.8: mode 15, no data
    assert (off.arbitration_id, off.data, off.is_extended_id) == (0xF28, b"", True)
    zero = pr.ak3_set_origin(0x28)  # §4.1.6: mode 5, 0 = temporary (lost at power-off)
    assert (zero.arbitration_id, zero.data, zero.is_extended_id) == (0x528, b"\x00", True)
    assert pr.ak3_set_origin(0x28, permanent=True).data == b"\x01"  # written to flash


# --- AK 3.0 feedback (V3.2.0 §4.3.1: frame 0x29 status, optional 0x2A position) ---------------
def _ak3_status_frame(drive_id, pos, spd, cur, temp, err):
    data = (pos.to_bytes(2, "big", signed=True) + spd.to_bytes(2, "big", signed=True)
            + cur.to_bytes(2, "big", signed=True) + temp.to_bytes(1, "big", signed=True) + bytes([err]))
    return pr.CanFrame(arbitration_id=(0x29 << 8) | drive_id, data=data, is_extended_id=True)


def test_ak3_status_decodes_units():
    # pos 900 * 0.1 = 90 deg; speed 1008 * 10 = 10080 ERPM / 21 pole pairs / 48 = 10 rpm;
    # current -150 * 0.01 = -1.5 A * Kt 8.8123 = -13.21845 Nm.
    fb = pr.decode_ak3_status(AKH70_48, _ak3_status_frame(0x28, 900, 1008, -150, 35, 0))
    assert fb.motor_id == 0x28
    assert fb.position == pytest.approx(math.pi / 2)
    assert fb.velocity == pytest.approx(10 * 2 * math.pi / 60)
    assert fb.current == pytest.approx(-1.5)
    assert fb.torque == pytest.approx(-13.21845)
    assert fb.temperature == 35
    assert fb.error == 0


def test_ak3_status_negatives():
    fb = pr.decode_ak3_status(AK60_39, _ak3_status_frame(0x2B, -900, -546, 200, -5, 2))
    assert fb.position == pytest.approx(-math.pi / 2)
    assert fb.velocity == pytest.approx(-5460 / 14 / 39 * 2 * math.pi / 60)  # = -10 rpm
    assert fb.torque == pytest.approx(2.0 * 3.4616)
    assert fb.temperature == -5
    assert fb.error == 2 and pr.AK3_ERRORS[2] == "over-current"


def test_ak3_disable_ack_is_error_0x77():
    fb = pr.decode_ak3_status(AKH70_48, _ak3_status_frame(0x28, 0, 0, 0, 30, 0x77))
    assert fb.error == 0x77 and pr.AK3_ERRORS[0x77] == "disable acknowledged"


def test_ak3_status_rejects_wrong_frames():
    good = _ak3_status_frame(0x28, 0, 0, 0, 30, 0)
    with pytest.raises(ValueError):  # not a 0x29 frame
        pr.decode_ak3_status(AKH70_48, dataclasses.replace(good, arbitration_id=0x2A28))
    with pytest.raises(ValueError):  # standard frame
        pr.decode_ak3_status(AKH70_48, dataclasses.replace(good, is_extended_id=False))
    with pytest.raises(ValueError):  # short payload
        pr.decode_ak3_status(AKH70_48, dataclasses.replace(good, data=good.data[:7]))
    with pytest.raises(ValueError):  # AK 2.0 motor
        pr.decode_ak3_status(AK80_64, good)


def test_ak3_position_frame_0x2a():
    # int32 * 0.01 deg, big-endian (DBC ExtPosition 7|32@0-).
    def frame(raw):
        return pr.CanFrame((0x2A << 8) | 0x28, raw.to_bytes(4, "big", signed=True), True)
    assert pr.decode_ak3_position(frame(9000)) == pytest.approx(math.pi / 2)
    assert pr.decode_ak3_position(frame(-36000)) == pytest.approx(-2 * math.pi)


def test_erpm_to_output_rad_s():
    assert pr.erpm_to_rad_s(21 * 48 * 60, AKH70_48) == pytest.approx(60 * 2 * math.pi / 60)


# --- AK 2.0 (V1.0.18 §5.3): standard frames, position-first -----------------------------------
def test_ak2_bit_positions():
    # Position-first: p(16) v(12) kp(12) kd(12) t(12).
    data = pr.pack_ak2_fields(pr.MitCodes(p=0xABCD, v=0x123, kp=0x456, kd=0x789, t=0xABC))
    assert data == _hex("AB CD 12 34 56 78 9A BC")


def test_ak2_command_frame():
    frame = pr.mit_command(AK80_64, 0x41, p=12.5, v=-8.0, kp=0.0, kd=5.0, t=144.0)
    assert frame.arbitration_id == 0x41 and not frame.is_extended_id
    # p max -> FFFF, v min -> 000, kp 0 -> 000, kd 5 -> FFF, t max -> FFF.
    assert frame.data == _hex("FF FF 00 00 00 FF FF FF")


def test_ak2_negative_values():
    # p -6.25 on +-12.5: 6.25 * 65535 / 25 = 16383.75 -> 16383 (3F FF); v -4 on +-8: 4 * 4095 / 16
    # = 1023.75 -> 1023 (3FF); t -72 on +-144: 72 * 4095 / 288 = 1023.75 -> 1023 (3FF).
    frame = pr.mit_command(AK80_64, 0x41, p=-6.25, v=-4.0, kp=0.0, kd=0.0, t=-72.0)
    assert frame.data == _hex("3F FF 3F F0 00 00 03 FF")


def test_ak2_special_frames():
    for fn, last in ((pr.ak2_enter, 0xFC), (pr.ak2_exit, 0xFD), (pr.ak2_set_zero, 0xFE)):
        frame = fn(0x41)
        assert frame.data == bytes([0xFF] * 7 + [last])
        assert frame.arbitration_id == 0x41 and not frame.is_extended_id


def test_ak2_feedback_bit_positions_and_temperature_offset():
    # DATA[0] id, p(16), v(12), t(12), DATA[6] temp, DATA[7] error. Manual receive code: temp = byte - 40.
    data = bytes([0x41, 0xAB, 0xCD, 0x12, 0x34, 0x56, 65, 0])
    frame = pr.CanFrame(arbitration_id=0x041, data=data, is_extended_id=False)
    fb = pr.decode_ak2_feedback(AK80_64, frame)
    assert fb.motor_id == 0x41
    assert fb.position == pytest.approx(pr.uint_to_float(0xABCD, -12.5, 12.5, 16))
    assert fb.velocity == pytest.approx(pr.uint_to_float(0x123, -8.0, 8.0, 12))
    assert fb.torque == pytest.approx(pr.uint_to_float(0x456, -144.0, 144.0, 12))
    assert fb.current is None
    assert fb.temperature == 25
    assert fb.error == 0


def test_ak2_feedback_extremes():
    data = bytes([0x41, 0xFF, 0xFF, 0x00, 0x0F, 0xFF, 0, 0])
    fb = pr.decode_ak2_feedback(AK80_64, pr.CanFrame(0x041, data, False))
    assert fb.position == pytest.approx(12.5)
    assert fb.velocity == pytest.approx(-8.0)
    assert fb.torque == pytest.approx(144.0)
    assert fb.temperature == -40


def test_ak2_feedback_rejects_wrong_frames():
    good = pr.CanFrame(0x041, bytes(8), False)
    with pytest.raises(ValueError):
        pr.decode_ak2_feedback(AK80_64, dataclasses.replace(good, data=bytes(7)))
    with pytest.raises(ValueError):
        pr.decode_ak2_feedback(AK80_64, dataclasses.replace(good, is_extended_id=True))
    with pytest.raises(ValueError):
        pr.decode_ak2_feedback(AKH70_48, good)


# --- Dispatch and ids -------------------------------------------------------------------------
def test_mit_command_picks_frame_type_from_spec():
    assert pr.mit_command(AKH70_48, 0x28, 0, 0, 0, 0, 0).is_extended_id
    assert pr.mit_command(AK60_39, 0x2B, 0, 0, 0, 0, 0).is_extended_id
    assert not pr.mit_command(AK80_64, 0x29, 0, 0, 0, 0, 0).is_extended_id


@pytest.mark.parametrize("bad_id", [-1, 256])
def test_motor_id_out_of_range_raises(bad_id):
    # The C++ akMitCanId masks to 8 bits (0x1FF -> 0xFF, a different motor); raise instead.
    for motor in LEG_MOTORS:
        with pytest.raises(ValueError):
            pr.mit_command(motor, bad_id, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        pr.ak2_enter(bad_id)


def test_unknown_protocol_raises():
    with pytest.raises(ValueError):
        pr.mit_command(dataclasses.replace(AK60_39, protocol="gl2"), 1, 0, 0, 0, 0, 0)


# --- Motor side (Task 9 fake motor): round trips against the host-side functions ---------------
_COMMANDS = [(0.0, 0.0, 0.0, 0.0, 0.0), (1.234, -2.5, 10.0, 0.7, -3.3), (-11.0, 4.0, 499.0, 4.9, 60.0)]


@pytest.mark.parametrize("motor", LEG_MOTORS, ids=lambda m: m.name)
@pytest.mark.parametrize("cmd", _COMMANDS)
def test_command_round_trip_within_one_step(motor, cmd):
    p, v, kp, kd, t = cmd
    frame = pr.mit_command(motor, 0x28, p, v, kp, kd, t)
    unpack = pr.unpack_ak3_fields if motor.protocol == "ak3" else pr.unpack_ak2_fields
    out = pr.decode_mit(motor, unpack(frame.data))
    v_c, t_c = max(-motor.v_max, min(motor.v_max, v)), max(-motor.t_max, min(motor.t_max, t))
    for got, want, lo, hi, bits in [(out.p, p, -motor.p_max, motor.p_max, 16), (out.v, v_c, -motor.v_max, motor.v_max, 12),
                                    (out.kp, kp, 0, motor.kp_max, 12), (out.kd, kd, 0, motor.kd_max, 12),
                                    (out.t, t_c, -motor.t_max, motor.t_max, 12)]:
        assert want - _step(lo, hi, bits) - 1e-9 <= got <= want + 1e-9  # truncation rounds down


def test_unpack_is_exact_inverse_of_pack():
    codes = pr.MitCodes(p=0xBEEF, v=0xABC, kp=0x123, kd=0xFED, t=0x456)
    assert pr.unpack_ak3_fields(pr.pack_ak3_fields(codes)) == codes
    assert pr.unpack_ak2_fields(pr.pack_ak2_fields(codes)) == codes


@pytest.mark.parametrize("motor", [AKH70_48, AK60_39], ids=lambda m: m.name)
def test_ak3_status_encode_round_trip(motor):
    frame = pr.encode_ak3_status(motor, 0x28, position=-1.2345, velocity=0.8, current=-3.21, temperature=-5, error=2)
    fb = pr.decode_ak3_status(motor, frame)
    assert fb.motor_id == 0x28 and fb.temperature == -5 and fb.error == 2
    assert fb.position == pytest.approx(-1.2345, abs=math.radians(0.05))  # 0.1 deg counts, rounded
    erpm_step = pr.erpm_to_rad_s(10.0, motor)
    assert fb.velocity == pytest.approx(0.8, abs=erpm_step / 2 + 1e-12)
    assert fb.current == pytest.approx(-3.21, abs=0.005 + 1e-12)
    assert fb.torque == pytest.approx(fb.current * motor.torque_constant)


def test_ak3_status_encode_saturates():
    frame = pr.encode_ak3_status(AK60_39, 1, position=1e6, velocity=-1e6, current=1e6, temperature=500)
    assert frame.data[:6] == bytes.fromhex("7fff80007fff") and frame.data[6] == 127


def test_ak3_position_frame_round_trip():
    assert pr.decode_ak3_position(pr.encode_ak3_position(5, -2.71828)) == pytest.approx(-2.71828, abs=math.radians(0.005))


def test_ak2_feedback_encode_round_trip():
    m = AK80_64
    frame = pr.encode_ak2_feedback(m, 0x32, position=0.5, velocity=-1.5, torque=12.0, temperature=-3, error=1)
    assert not frame.is_extended_id and frame.arbitration_id == 0x32
    fb = pr.decode_ak2_feedback(m, frame)
    assert fb.motor_id == 0x32 and fb.temperature == -3 and fb.error == 1
    assert 0.5 - _step(-m.p_max, m.p_max, 16) <= fb.position <= 0.5 + 1e-9
    assert -1.5 - _step(-m.v_max, m.v_max, 12) <= fb.velocity <= -1.5 + 1e-9
    assert 12.0 - _step(-m.t_max, m.t_max, 12) <= fb.torque <= 12.0 + 1e-9
    assert pr.encode_ak2_feedback(m, 0x32, 0, 0, 0, 25, arbitration_id=0).arbitration_id == 0


def test_ak2_special_code():
    assert pr.ak2_special_code(pr.ak2_enter(7)) == pr.AK2_ENTER
    assert pr.ak2_special_code(pr.ak2_exit(7)) == pr.AK2_EXIT
    assert pr.ak2_special_code(pr.ak2_set_zero(7)) == pr.AK2_SET_ZERO
    assert pr.ak2_special_code(pr.mit_command(AK80_64, 7, 0, 0, 0, 0, 0)) is None
    assert pr.ak2_special_code(pr.CanFrame(7, bytes([0xFF] * 7 + [0x00]), False)) is None
    assert pr.ak2_special_code(pr.CanFrame(7, bytes([0xFF] * 7 + [0xFC]), True)) is None


def test_ak2_feedback_id_accepts_zero_or_motor_id():
    assert pr.is_ak2_feedback_id(0x00, 0x32) and pr.is_ak2_feedback_id(0x32, 0x32)
    assert not pr.is_ak2_feedback_id(0x33, 0x32)


def test_ak3_split_id_inverts_can_id():
    assert pr.ak3_split_id(pr.ak3_can_id(pr.AK3_MODE_DISABLE, 0x41)) == (15, 0x41)
