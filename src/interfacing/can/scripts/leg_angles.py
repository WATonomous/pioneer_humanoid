#!/usr/bin/env python3
"""Leg joint angles straight off the CAN bus, plus the MIT commands sent to each drive.

This script reads CubeMars AK V3 (akv3) drives only; it does not show akv2 motors. V3 drives never send MIT replies: everything they report comes in
the servo status frame (ext id 0x2900|id, humanoid.dbc ServoStatusFeedback), about 50 Hz,
whether or not they are being commanded:
  position  int16 x 0.1 deg       speed  int16 x 10 ERPM (rotor, electrical rpm)
  current   int16 x 0.01 A        temp   int8 deg C          error  uint8
MIT commands (ext id 0x800|id, KP-first) are decoded with mit_profiles.yaml's ranges, i.e. as the
drive should read them.

The summary checks the drive's speed sign against the way its position moves. Opposite signs are
a warning, not proof: the speed is the rotor's, and a gearbox can legitimately reverse it. A
powered test with the joint held between physical stops is what decides.

Read-only: a receive-only raw CAN socket. It never sends a frame, so it cannot move a motor.

Run (interfacing container, which has can0)::

    ros2 run can leg_angles.py                      # left leg, live from can0
    ros2 run can leg_angles.py --ids 44             # just the knee
    ros2 run can leg_angles.py --log /outputs/can_step1.log   # a candump -L capture

Each row: command-frame deg [motor-frame deg], speed (ERPM), current (A), drive temperature, and
the last MIT command seen ("tx": target in motor deg, kp, kd). "---" = nothing in the last 0.5 s.
Ctrl-C (or the end of a log) prints a per-joint summary.
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import sys
import time
from collections import deque
from pathlib import Path

import yaml

import joint_config as jc

SERVO_BASE = 0x2900   # ServoStatusFeedback (humanoid.dbc), ext id = base | drive id
MIT_CMD_BASE = 0x800  # AK V3 MIT command (mit_protocol.cpp akMitCanId), ext id = base | id
STALE_AFTER_S = 0.5
SLOPE_WINDOW_S = 0.1  # position change measured over this long (0.1 deg steps are coarse)
MIN_SLOPE_DPS = 10.0  # only compare signs while the joint is clearly moving
MIN_ERPM = 50.0

CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_FRAME = struct.Struct("=IB3x8s")


def load_ak_profiles(ids) -> dict[int, dict]:
    """mit_profiles.yaml entries for these ids, akv3 only (the frames this script decodes)."""
    path = jc._first_existing(jc.MIT_PROFILES)
    if path is None:
        raise SystemExit(f"could not find mit_profiles.yaml (looked in: {', '.join(jc.MIT_PROFILES)})")
    motors = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("motors", {}) or {}
    out = {}
    for motor_id in ids:
        prof = motors.get(str(motor_id))
        if prof and prof.get("family") == "akv3":
            out[motor_id] = prof
    print(f"MIT profiles: {path}")
    return out


class Joint:
    def __init__(self, motor_id: int, info: dict, profile: dict | None):
        self.id = motor_id
        self.info = info
        self.profile = profile
        self.fb: tuple[float, float, float, int, int, float] | None = None  # deg, erpm, A, C, err, t
        self.tx: tuple[float, float, float, float] | None = None  # (p motor deg, kp, kd, t)
        self.n_servo = self.n_tx = 0
        self.pos_min = math.inf
        self.pos_max = -math.inf
        self.recent: deque[tuple[float, float]] = deque()  # (t, deg) over SLOPE_WINDOW_S
        self.same_sign = self.opposite_sign = 0
        self.errors: set[int] = set()

    def on_servo(self, data: bytes, t: float) -> None:
        pos_raw, erpm_raw, cur_raw, temp, err = struct.unpack(">hhhbB", data[:8])
        deg, erpm, amps = pos_raw * 0.1, erpm_raw * 10.0, cur_raw * 0.01
        self.recent.append((t, deg))
        while t - self.recent[0][0] > SLOPE_WINDOW_S:
            self.recent.popleft()
        t0, deg0 = self.recent[0]
        if t - t0 > 0.5 * SLOPE_WINDOW_S:
            slope = (deg - deg0) / (t - t0)
            if abs(slope) > MIN_SLOPE_DPS and abs(erpm) > MIN_ERPM:
                if (slope > 0) == (erpm > 0):
                    self.same_sign += 1
                else:
                    self.opposite_sign += 1
        self.fb = (deg, erpm, amps, temp, err, t)
        self.n_servo += 1
        self.pos_min, self.pos_max = min(self.pos_min, deg), max(self.pos_max, deg)
        if err:
            self.errors.add(err)

    def on_mit_cmd(self, data: bytes, t: float) -> None:
        """packAkMitCommand, KP-first: kp12 kd12 p16 v12 t12. Gains: code * max / 4096."""
        p = self.profile
        kp = ((data[0] << 4) | (data[1] >> 4)) * p["kp_max"] / 4096
        kd = (((data[1] & 0xF) << 8) | data[2]) * p["kd_max"] / 4096
        pos = ((data[3] << 8) | data[4]) * (p["p_max"] - p["p_min"]) / 65536 + p["p_min"]
        self.tx = (math.degrees(pos), kp, kd, t)
        self.n_tx += 1

    def _angle(self, motor_deg: float) -> str:
        if not self.info["calibrated"]:
            return f"  uncal [{motor_deg:+7.1f}]"
        cmd = jc.motor_to_cmd_deg(self.info, motor_deg)
        out = self.info["limit_range"] and not self.info["lower"] <= cmd <= self.info["upper"]
        return f"{cmd:+7.1f}{'!' if out else ' '}[{motor_deg:+7.1f}]"

    def cell(self, now: float) -> str:
        fb = self.fb if self.fb and now - self.fb[5] <= STALE_AFTER_S else None
        tx = self.tx if self.tx and now - self.tx[3] <= STALE_AFTER_S else None
        if fb:
            deg, erpm, amps, temp, err, _ = fb
            f = (f"{self._angle(deg)}  spd {erpm:+7.0f}erpm  cur {amps:+5.2f}A  {temp:3d}C"
                 + (f"  ERR {err}" if err else ""))
        else:
            f = f"{'---':>17}"
        x = (f"p[{tx[0]:+7.1f}] kp{tx[1]:5.1f} kd{tx[2]:4.2f}" if tx else "---")
        return f"{self.info['name']}  {f}  tx {x}"

    def summary(self) -> list[str]:
        lines = [f"{self.info['name']} (id {self.id}): {self.n_servo} servo frames, "
                 f"{self.n_tx} MIT commands"]
        if self.n_servo:
            lines.append(f"  motor deg {self.pos_min:+.1f} .. {self.pos_max:+.1f}"
                         f"  (moved {self.pos_max - self.pos_min:.1f} deg)")
        else:
            lines.append("  no servo frames: drive off, or wrong id/interface")
        if self.errors:
            lines.append(f"  drive error codes seen: {sorted(self.errors)}")
        lines.append("  speed sign: " + self.speed_verdict())
        return lines

    def speed_verdict(self) -> str:
        n = self.same_sign + self.opposite_sign
        counts = (f"{self.same_sign} same / {self.opposite_sign} opposite to the position change "
                  "(samples while moving)")
        if n < 10:
            return counts + ". Too little motion to tell: turn the joint by hand, both ways, ~20 deg/s."
        if self.same_sign >= 0.9 * n:
            return counts + ". SAME: speed and position agree."
        if self.opposite_sign >= 0.9 * n:
            return counts + (". OPPOSITE: the drive's speed reading runs against its position. "
                             "Suspicious (its damping may push instead of brake), but a reversing "
                             "gearbox also does this: Step 2 between stops decides.")
        return counts + ". Mixed: look at the rows."


def frames_from_socket(ifname: str):
    """(t, ext, can_id, data) from a receive-only raw socket. Nothing is ever sent on it."""
    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((ifname,))
    sock.settimeout(0.1)
    while True:
        try:
            raw = sock.recv(CAN_FRAME.size)
        except socket.timeout:
            yield time.monotonic(), None, None, None
            continue
        can_id, dlc, data = CAN_FRAME.unpack(raw)
        if can_id & (CAN_ERR_FLAG | CAN_RTR_FLAG):
            continue
        ext = bool(can_id & CAN_EFF_FLAG)
        yield time.monotonic(), ext, can_id & (0x1FFFFFFF if ext else 0x7FF), data[:dlc]


def frames_from_log(path: Path):
    """(t, ext, can_id, data) from a `candump -L` log: "(ts) can0 ID#DATA"."""
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 3 or "#" not in parts[2]:
            continue
        ident, data = parts[2].split("#", 1)
        try:  # skips RTR ("123#R") and CAN FD ("123##1...") frames
            yield float(parts[0].strip("()")), len(ident) == 8, int(ident, 16), bytes.fromhex(data)
        except ValueError:
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", default="left", help="leg side in the calibration (default left)")
    parser.add_argument("--calibration", default=None,
                        help="leg_calibration.yaml (default: auto, as joint_command)")
    parser.add_argument("--ids", default=None, help="comma-separated CAN ids (default: the whole leg)")
    parser.add_argument("--interface", default="can0", help="CAN interface for live mode (default can0)")
    parser.add_argument("--log", default=None, help="decode this candump -L file instead of the bus")
    parser.add_argument("--period", type=float, default=0.5, help="seconds between rows (default 0.5)")
    args = parser.parse_args()

    calibration = jc.find_calibration(args.calibration, "leg")
    joint_map = jc.load_joint_map(calibration, args.side, "leg")
    if args.ids:
        try:
            wanted = [int(x) for x in args.ids.split(",") if x.strip()]
        except ValueError:
            raise SystemExit(f"--ids: expected comma-separated integers, got {args.ids!r}")
        unknown = [i for i in wanted if i not in joint_map]
        if unknown:
            raise SystemExit(f"--ids: {unknown} not in the {args.side} leg. Known: {sorted(joint_map)}")
        joint_map = {i: joint_map[i] for i in wanted}
    print(f"Calibration: {calibration}")
    profiles = load_ak_profiles(joint_map)
    joints = {mid: Joint(mid, info, profiles.get(mid)) for mid, info in joint_map.items()}
    source = f"log {args.log}" if args.log else f"live {args.interface}"
    print(f"Source: {source}. Read-only: no frame is sent.")
    print("Angle: command-frame deg [motor-frame deg]; '!' = outside the calibrated limits.\n")

    frames = frames_from_log(Path(args.log)) if args.log else frames_from_socket(args.interface)
    t0 = next_row = None
    try:
        for t, ext, can_id, data in frames:
            if t0 is None:
                t0 = next_row = t
            if data is not None and ext and len(data) >= 8 and (can_id & 0xFF) in joints:
                j = joints[can_id & 0xFF]
                if (can_id & ~0xFF) == SERVO_BASE:
                    j.on_servo(data, t)
                elif (can_id & ~0xFF) == MIT_CMD_BASE and j.profile:
                    j.on_mit_cmd(data, t)
            if t >= next_row:
                for j in joints.values():
                    print(f"{t - t0:7.2f}s  {j.cell(t)}")
                if len(joints) > 1:
                    print()
                next_row = t + args.period
    except KeyboardInterrupt:
        pass

    print("\n=== Summary ===")
    for j in joints.values():
        print("\n".join(j.summary()))
    print("Nothing was sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
