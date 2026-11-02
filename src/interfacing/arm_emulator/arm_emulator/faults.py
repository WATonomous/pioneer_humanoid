"""Faults to inject, to prove the real safety layers react. Each is a small change to one drive or joint.

    "kind:motor_id[:value]", e.g. "mute:11", "code:14:2", "zero_shift:11:30", "hot:14:60", "dead:11", "stall:11"
"""
from __future__ import annotations

from dataclasses import dataclass

KINDS = {
    "mute": "the drive stops answering (feedback timeout)",
    "code": "the drive reports this fault code and stops producing torque",
    "dead": "the output produces no torque (tracking error as the joint sinks)",
    "stall": "the joint is mechanically locked (torque climbs, tracking error grows)",
    "zero_shift": "the encoder zero moves by this many degrees (stale calibration)",
    "hot": "the temperature reads this many degrees higher",
}


@dataclass(frozen=True)
class Fault:
    kind: str
    motor_id: int
    value: float = 0.0


def parse(spec: str) -> Fault:
    parts = spec.split(":")
    if len(parts) not in (2, 3) or parts[0] not in KINDS:
        raise ValueError(f"bad fault {spec!r}: use kind:motor_id[:value], kinds {sorted(KINDS)}")
    return Fault(parts[0], int(parts[1]), float(parts[2]) if len(parts) == 3 else 0.0)
