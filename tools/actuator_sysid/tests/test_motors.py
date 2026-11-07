"""motors.py: which frames each protocol needs, and which replies it accepts."""
import pytest

from actuator_sysid import protocol as pr
from actuator_sysid.motors import MOTORS, Ak2Motor, Ak3Motor, make_motor, short_name
from actuator_sysid.protocol import MitCommand
from pioneer_humanoid.leg_params import AK60_39, AK80_64, AKH70_48


def test_cli_names_and_factory():
    assert set(MOTORS) == {"AKH70-48", "AK80-64", "AK60-39"}
    assert short_name(AK80_64) == "AK80-64"
    assert isinstance(make_motor(AKH70_48, 1), Ak3Motor) and isinstance(make_motor(AK60_39, 1), Ak3Motor)
    assert isinstance(make_motor(AK80_64, 1), Ak2Motor)


def test_protocol_mismatch_and_bad_id_raise():
    with pytest.raises(ValueError):
        Ak3Motor(AK80_64, 1)
    with pytest.raises(ValueError):
        Ak2Motor(AK60_39, 1)
    with pytest.raises(ValueError):
        make_motor(AK60_39, 256)


def test_ak3_frames():
    m = make_motor(AK60_39, 0x41)
    assert m.streams_feedback and m.enable_frames() == []
    assert m.disable_frames() == [pr.ak3_disable(0x41)]
    assert m.zero_frame() == pr.ak3_set_origin(0x41, permanent=False)
    assert m.command(MitCommand(0.1, 0, 1, 0.1, 0)) == pr.mit_command(AK60_39, 0x41, 0.1, 0, 1, 0.1, 0)


def test_ak2_frames():
    m = make_motor(AK80_64, 0x32)
    assert not m.streams_feedback
    assert m.enable_frames() == [pr.ak2_enter(0x32)] and m.disable_frames() == [pr.ak2_exit(0x32)]
    assert m.zero_frame() == pr.ak2_set_zero(0x32)


def test_ak3_parse_only_own_status_frames():
    m = make_motor(AK60_39, 0x41)
    own = pr.encode_ak3_status(AK60_39, 0x41, 0.5, 0.0, 1.0, 30)
    assert m.parse(own).position == pytest.approx(0.5, abs=1e-3)
    assert m.parse(pr.encode_ak3_status(AK60_39, 0x42, 0.5, 0.0, 1.0, 30)) is None  # other motor
    assert m.parse(pr.encode_ak3_position(0x41, 0.5)) is None  # 0x2A: raw log only
    assert m.parse(pr.mit_command(AK60_39, 0x41, 0, 0, 0, 0, 0)) is None  # a command, not feedback
    assert m.parse(pr.CanFrame(own.arbitration_id, own.data, False)) is None


def test_ak2_parse_only_own_replies():
    m = make_motor(AK80_64, 0x32)
    own = pr.encode_ak2_feedback(AK80_64, 0x32, 0.5, 0.0, 1.0, 30)
    assert m.parse(own).position == pytest.approx(0.5, abs=1e-3)
    assert m.parse(pr.encode_ak2_feedback(AK80_64, 0x32, 0.5, 0, 1, 30, arbitration_id=0)) is not None
    assert m.parse(pr.encode_ak2_feedback(AK80_64, 0x33, 0.5, 0.0, 1.0, 30)) is None
    assert m.parse(pr.ak2_enter(0x32)) is None
    assert m.parse(pr.CanFrame(own.arbitration_id, own.data, True)) is None


def test_fault_texts():
    ak3, ak2 = make_motor(AK60_39, 1), make_motor(AK80_64, 1)
    fb = pr.MotorFeedback(1, 0, 0, 0, 0, 30, 0)
    assert ak3.fault(fb) is None and ak2.fault(fb) is None
    assert ak3.fault(pr.MotorFeedback(1, 0, 0, 0, 0, 30, 0x77)) is None  # disable ack is not a fault
    assert "over-current" in ak3.fault(pr.MotorFeedback(1, 0, 0, 0, 0, 30, 2))
    assert ak3.is_disable_ack(pr.MotorFeedback(1, 0, 0, 0, 0, 30, 0x77))
    assert "0x05" in ak2.fault(pr.MotorFeedback(1, 0, 0, 0, 0, 30, 5))
