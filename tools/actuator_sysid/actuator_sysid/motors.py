"""Ak3Motor / Ak2Motor: which frames a motor needs and how to read its replies.

These classes only build and parse frames (all bytes come from protocol.py). They never send:
safety.Safety does, after its checks.
"""
from pioneer_humanoid.leg_params import AK60_39, AK80_64, AKH70_48, MotorSpec

from . import protocol as pr
from .protocol import CanFrame, MitCommand, MotorFeedback

# CLI name -> spec. Short name = first word of MotorSpec.name (it contains spaces).
MOTORS = {spec.name.split()[0]: spec for spec in (AKH70_48, AK80_64, AK60_39)}


def short_name(spec: MotorSpec) -> str:
    return spec.name.split()[0]


class Motor:
    """Common interface. `streams_feedback`: the motor sends feedback without being commanded."""
    protocol = ""
    streams_feedback = False

    def __init__(self, spec: MotorSpec, motor_id: int):
        if spec.protocol != self.protocol:
            raise ValueError(f"{spec.name} uses {spec.protocol}, not {self.protocol}")
        if not 0 <= motor_id <= 0xFF:
            raise ValueError(f"motor id {motor_id} is outside 0..255")
        self.spec, self.motor_id = spec, motor_id

    def command(self, cmd: MitCommand) -> CanFrame:
        return pr.mit_command(self.spec, self.motor_id, cmd.p, cmd.v, cmd.kp, cmd.kd, cmd.t)

    def enable_frames(self) -> list[CanFrame]:
        raise NotImplementedError

    def disable_frames(self) -> list[CanFrame]:
        raise NotImplementedError

    def zero_frame(self) -> CanFrame:
        raise NotImplementedError

    def parse(self, frame: CanFrame) -> MotorFeedback | None:
        """Feedback from this motor, or None if the frame is something else."""
        raise NotImplementedError

    def is_disable_ack(self, fb: MotorFeedback) -> bool:
        return False

    def fault(self, fb: MotorFeedback) -> str | None:
        """Text if the feedback reports a motor fault, else None."""
        raise NotImplementedError


class Ak3Motor(Motor):
    """AK 3.0 (AKH70-48, AK60-39). Streams 0x29 status at the rate set in R-Link (Task 10)."""
    protocol = "ak3"
    streams_feedback = True

    def enable_frames(self) -> list[CanFrame]:
        return []  # AK 3.0 has no enter frame: the first MIT command enables it

    def disable_frames(self) -> list[CanFrame]:
        return [pr.ak3_disable(self.motor_id)]

    def zero_frame(self) -> CanFrame:
        return pr.ak3_set_origin(self.motor_id, permanent=False)  # never writes flash

    def parse(self, frame: CanFrame) -> MotorFeedback | None:
        if not frame.is_extended_id or len(frame.data) != 8:
            return None
        mode, drive_id = pr.ak3_split_id(frame.arbitration_id)
        if drive_id != self.motor_id or mode != pr.AK3_STATUS:
            return None  # 0x2A and other frames are kept in the raw log only
        return pr.decode_ak3_status(self.spec, frame)

    def is_disable_ack(self, fb: MotorFeedback) -> bool:
        return fb.error == 0x77

    def fault(self, fb: MotorFeedback) -> str | None:
        if fb.error in (0, 0x77):
            return None
        return f"motor error {fb.error}: {pr.AK3_ERRORS.get(fb.error, 'unknown code')}"


class Ak2Motor(Motor):
    """AK 2.0 (AK80-64). Replies only to commands; needs the enter frame first."""
    protocol = "ak2"
    streams_feedback = False

    def enable_frames(self) -> list[CanFrame]:
        return [pr.ak2_enter(self.motor_id)]

    def disable_frames(self) -> list[CanFrame]:
        return [pr.ak2_exit(self.motor_id)]

    def zero_frame(self) -> CanFrame:
        return pr.ak2_set_zero(self.motor_id)

    def parse(self, frame: CanFrame) -> MotorFeedback | None:
        if (frame.is_extended_id or len(frame.data) != 8 or frame.data[0] != self.motor_id
                or not pr.is_ak2_feedback_id(frame.arbitration_id, self.motor_id)
                or pr.ak2_special_code(frame) is not None):
            return None
        return pr.decode_ak2_feedback(self.spec, frame)

    def fault(self, fb: MotorFeedback) -> str | None:
        if fb.error == 0:
            return None
        return f"motor error byte {fb.error:#04x} (AK 2.0 codes are not listed in manual §5.3)"


def make_motor(spec: MotorSpec, motor_id: int) -> Motor:
    if spec.protocol == "ak3":
        return Ak3Motor(spec, motor_id)
    if spec.protocol == "ak2":
        return Ak2Motor(spec, motor_id)
    raise ValueError(f"{spec.name}: unknown protocol {spec.protocol!r}")
