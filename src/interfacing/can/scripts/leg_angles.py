#!/usr/bin/env python3
"""Leg joint angles straight off the CAN bus, one column per feedback source.

can_node merges two reports of the same shaft into MotorFeedback.position:
  servo  servo status frame, ext id 0x2900|id, int16 x 0.1 deg (fixed scale)
  mit    MIT reply, std id 0x000 (master), byte 0 = id, uint16 over mit_profiles.yaml's p range
This script keeps them apart, so you can check that mit_profiles.yaml's p range matches the
drive (KNEE_RUNAWAY_DEBUG.md, T3): at the same pose the two must agree. It also shows the MIT
command frames (ext id 0x800|id) on the bus, decoded the way the drive should read them.

Read-only: a receive-only raw CAN socket. It never sends a frame, so it cannot move a motor.

Run (interfacing container, which has can0)::

    ros2 run can leg_angles.py                      # left leg, live from can0
    ros2 run can leg_angles.py --ids 44             # just the knee
    ros2 run can leg_angles.py --log /outputs/can_t3_063150.log   # a candump -L capture

Each row: command-frame deg (motor-frame deg in brackets) per source, d = servo - mit (motor
deg), and the last MIT command seen ("tx"). "---" = nothing from that source in the last
0.5 s. Ctrl-C (or the end of a log) prints a per-joint summary with a verdict.
"""

from __future__ import annotations

import argparse
import math
import socket
import statistics
import struct
import sys
import time
from pathlib import Path

import yaml

import joint_config as jc

SERVO_BASE = 0x2900   # ServoStatusFeedback (humanoid.dbc), ext id = base | drive id
MIT_CMD_BASE = 0x800  # AK V3 MIT command (mit_protocol.cpp akMitCanId), ext id = base | id
MIT_MASTER_ID = 0x000  # can params.yaml mit_master_id
STALE_AFTER_S = 0.5
PAIR_WINDOW_S = 0.1   # an MIT reply and a servo frame this close count as the same pose
H2C_RATIO = 50.0 / 12.56  # servo/mit span ratio if the drive decodes p over +-50 (debug log H2c)

CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_FRAME = struct.Struct("=IB3x8s")


def load_ak_profiles(ids) -> dict[int, dict]:
    """mit_profiles.yaml entries for these ids, AK family only (the leg drives)."""
    path = jc._first_existing(jc.MIT_PROFILES)
    if path is None:
        raise SystemExit(f"could not find mit_profiles.yaml (looked in: {', '.join(jc.MIT_PROFILES)})")
    motors = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("motors", {}) or {}
    out = {}
    for motor_id in ids:
        prof = motors.get(str(motor_id))
        if prof and prof.get("family") == "ak":
            out[motor_id] = prof
    print(f"MIT profiles: {path}")
    return out


def unpack(code: int, lo: float, hi: float, bits: int) -> float:
    """mit_protocol.cpp unpackMitValue (feedback)."""
    return code * (hi - lo) / ((1 << bits) - 1) + lo


class Joint:
    def __init__(self, motor_id: int, info: dict, profile: dict | None):
        self.id = motor_id
        self.info = info
        self.profile = profile
        self.servo: tuple[float, float] | None = None   # (motor deg, t)
        self.mit: tuple[float, float] | None = None
        self.tx: tuple[float, float, float, float] | None = None  # (p motor deg, kp, kd, t)
        self.n_servo = self.n_mit = self.n_tx = 0
        self.servo_all: list[float] = []
        self.mit_all: list[float] = []
        self.pairs: list[tuple[float, float]] = []  # (servo, mit) motor deg, same moment

    def on_servo(self, data: bytes, t: float) -> None:
        deg = struct.unpack(">h", data[0:2])[0] * 0.1
        self.servo = (deg, t)
        self.n_servo += 1
        self.servo_all.append(deg)

    def on_mit_reply(self, data: bytes, t: float) -> None:
        p = self.profile
        rad = unpack((data[1] << 8) | data[2], p["p_min"], p["p_max"], 16)
        deg = math.degrees(rad)
        self.mit = (deg, t)
        self.n_mit += 1
        self.mit_all.append(deg)
        if self.servo and t - self.servo[1] <= PAIR_WINDOW_S:
            self.pairs.append((self.servo[0], deg))

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
        servo = self.servo if self.servo and now - self.servo[1] <= STALE_AFTER_S else None
        mit = self.mit if self.mit and now - self.mit[1] <= STALE_AFTER_S else None
        tx = self.tx if self.tx and now - self.tx[3] <= STALE_AFTER_S else None
        s = self._angle(servo[0]) if servo else f"{'---':>17}"
        m = self._angle(mit[0]) if mit else f"{'---':>17}"
        d = f"{servo[0] - mit[0]:+6.1f}" if servo and mit else f"{'---':>6}"
        x = (f"p[{tx[0]:+7.1f}] kp{tx[1]:5.1f} kd{tx[2]:4.2f}" if tx else "---")
        return f"{self.info['name']}  servo {s}  mit {m}  d {d}  tx {x}"

    def summary(self) -> list[str]:
        lines = [f"{self.info['name']} (id {self.id}): {self.n_servo} servo frames, "
                 f"{self.n_mit} MIT replies, {self.n_tx} MIT commands"]
        if self.servo_all:
            lines.append(f"  servo motor deg {min(self.servo_all):+.1f} .. {max(self.servo_all):+.1f}")
        if self.mit_all:
            lines.append(f"  mit   motor deg {min(self.mit_all):+.1f} .. {max(self.mit_all):+.1f}"
                         f"  (p range +-{self.profile['p_max']} rad from mit_profiles.yaml)")
        lines.append("  verdict: " + self.verdict())
        return lines

    def verdict(self) -> str:
        if self.profile is None:
            return "no AK profile in mit_profiles.yaml; MIT frames not decoded"
        if self.n_mit == 0:
            if self.n_tx == 0:
                return ("no MIT commands or replies on the bus: nothing commands this drive in MIT. "
                        "Is joint_command's motor_cmd_topic still /dry_run/motorCMD?")
            return "MIT commands sent but no replies: drive off, wrong id, or not in MIT mode"
        if not self.pairs:
            return "no servo frame within 0.1 s of an MIT reply: can't compare the two sources"
        servo = [s for s, _ in self.pairs]
        mit = [m for _, m in self.pairs]
        diff = statistics.median(abs(s - m) for s, m in self.pairs)
        servo_span = max(servo) - min(servo)
        mit_span = max(mit) - min(mit)
        base = f"median |servo - mit| {diff:.2f} deg; spans servo {servo_span:.1f} / mit {mit_span:.1f} deg"
        if servo_span < 5.0:
            return base + ". Moved < 5 deg: hold the joint at several poses to check the scale."
        ratio = servo_span / mit_span if mit_span > 0 else float("inf")
        if diff < 0.5 and abs(ratio - 1.0) < 0.05:
            return base + ". AGREE: mit_profiles.yaml's p range matches the drive."
        if abs(ratio - H2C_RATIO) < 0.3:
            return base + f". MIT moves {ratio:.2f}x less than servo: the drive decodes p over about +-50 (H2c)."
        if abs(ratio - 1 / H2C_RATIO) < 0.05:
            return base + f". MIT moves {1 / ratio:.2f}x more than servo: the drive's p range is narrower than the file's."
        return base + f". DISAGREE (span ratio {ratio:.2f}): look at the rows."


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
    print("Columns: command-frame deg [motor-frame deg]; '!' = outside the calibrated limits.\n")

    frames = frames_from_log(Path(args.log)) if args.log else frames_from_socket(args.interface)
    t0 = next_row = None
    try:
        for t, ext, can_id, data in frames:
            if t0 is None:
                t0 = next_row = t
            if data is not None and len(data) >= 8:
                if ext and (can_id & ~0xFF) == SERVO_BASE and (can_id & 0xFF) in joints:
                    joints[can_id & 0xFF].on_servo(data, t)
                elif ext and (can_id & ~0xFF) == MIT_CMD_BASE and (can_id & 0xFF) in joints:
                    j = joints[can_id & 0xFF]
                    if j.profile:
                        j.on_mit_cmd(data, t)
                elif not ext and can_id == MIT_MASTER_ID and data[0] in joints:
                    j = joints[data[0]]
                    if j.profile:
                        j.on_mit_reply(data, t)
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
