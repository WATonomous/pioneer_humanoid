"""Jaw opening as a closure fraction (0 open .. 1 closed, the datasets' gripper convention).

``encoder`` (default): an STS3215 servo used as an encoder on the jaw linkage, read exactly as the leader arm reads
its servos (pioneer_leader_arm_teleop/servo_leader.py: torque written OFF at start and on close, never enabled, so
it can't move). ``trigger``: the mounted controller's trigger, linked to the jaws; no extra hardware, coarser.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

from umi_gripper._paths import LEADER_TELEOP, add

add(LEADER_TELEOP)
from servo_leader import COUNTS_PER_REV, wrapped_count_delta  # noqa: E402  (pioneer_leader_arm_teleop)

WIDTH_SERVO_LABEL = "W"


def closure_from_counts(count: int, open_count: int, closed_count: int) -> float:
    """Raw encoder counts -> closure in [0, 1]. Wrap-safe: the jaw travel is well under half a servo turn."""
    span = wrapped_count_delta(int(closed_count), int(open_count))
    if span == 0:
        raise ValueError("width calibration: open and closed counts are equal; recalibrate (calibrate_width.py)")
    closure = wrapped_count_delta(int(count), int(open_count)) / span
    if not math.isfinite(closure):
        return 0.0
    return min(1.0, max(0.0, closure))


def check_width_calibration(width_cfg: dict) -> tuple[int, int]:
    open_count, closed_count = width_cfg.get("open_count"), width_cfg.get("closed_count")
    if open_count is None or closed_count is None:
        raise ValueError("the width encoder is not calibrated: run python -m umi_gripper.calibration.calibrate_width "
                         "(or collect with --width trigger)")
    for value in (open_count, closed_count):
        if not 0 <= int(value) < COUNTS_PER_REV:
            raise ValueError(f"width calibration counts must be in [0, {COUNTS_PER_REV}), got {value}")
    if abs(wrapped_count_delta(int(closed_count), int(open_count))) < 50:
        raise ValueError("width calibration: open and closed are < 50 counts apart; recalibrate")
    return int(open_count), int(closed_count)


def encoder_port(width_cfg: dict) -> str:
    """The encoder board's port: found by USB serial (as the leader arm), else width.port."""
    if width_cfg.get("usb_serial"):
        add(LEADER_TELEOP)
        from leader_mapping import port_of
        return port_of(str(width_cfg["usb_serial"]))
    if width_cfg.get("port"):
        return str(width_cfg["port"])
    raise ValueError("width encoder: set width.usb_serial (calibrate_width.py stores it) or width.port")


class RawEncoder:
    """One STS3215, torque off, read in raw counts."""

    def __init__(self, port: str, servo_id: int, baud: int):
        from servo_leader import ServoLeader
        self.servo_id = int(servo_id)
        self._leader = ServoLeader(port, baud, servo_ids={WIDTH_SERVO_LABEL: self.servo_id})

    def read_count(self) -> int:
        self._leader.read_radians()
        return int(self._leader.last_positions()[self.servo_id])

    def close(self) -> None:
        self._leader.close()


class EncoderWidth:
    """Polls the encoder on a thread; drain() returns (host_s, closure, raw_count) since the last drain."""

    source = "encoder"

    def __init__(self, width_cfg: dict, encoder: RawEncoder | None = None):
        self.open_count, self.closed_count = check_width_calibration(width_cfg)
        self.period = 1.0 / float(width_cfg.get("rate_hz", 100))
        self.encoder = encoder or RawEncoder(encoder_port(width_cfg), width_cfg["servo_id"], int(width_cfg["baud"]))
        self.samples: deque[tuple[float, float, float]] = deque()
        self.latest: float | None = None
        self.error: BaseException | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="width-encoder", daemon=True)

    def start(self) -> "EncoderWidth":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                before = time.monotonic()
                count = self.encoder.read_count()
                stamp = (before + time.monotonic()) / 2   # the read's midpoint; the rest is latency.width_s
                closure = closure_from_counts(count, self.open_count, self.closed_count)
                with self._lock:
                    self.samples.append((stamp, closure, float(count)))
                    self.latest = closure
                self._stop.wait(max(0.0, self.period - (time.monotonic() - before)))
        except BaseException as exc:   # surfaced to collect.py, which stops the take
            self.error = exc

    def drain(self) -> list[tuple[float, float, float]]:
        with self._lock:
            out = list(self.samples)
            self.samples.clear()
        return out

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.encoder.close()


class TriggerWidth:
    """Width from the mounted controller's trigger: collect.py feeds it the pose samples."""

    source = "trigger"

    def __init__(self):
        self.samples: deque[tuple[float, float, float]] = deque()
        self.latest: float | None = None
        self.error = None

    def start(self) -> "TriggerWidth":
        return self

    def feed(self, pose_samples) -> None:
        for sample in pose_samples:
            if sample.tracked:
                self.samples.append((sample.host_s, float(sample.trigger), float(sample.trigger)))
                self.latest = float(sample.trigger)

    def drain(self) -> list[tuple[float, float, float]]:
        out = list(self.samples)
        self.samples.clear()
        return out

    def stop(self) -> None:
        pass


def from_config(cfg: dict):
    return EncoderWidth(cfg["width"]) if cfg["width"]["source"] == "encoder" else TriggerWidth()
