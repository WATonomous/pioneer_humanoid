"""Jaw encoder range: raw counts fully open and fully closed, and the board's USB serial (width.* in
umi_calibration.json). The STS3215 is read with torque OFF, as the leader arm; nothing here can drive it.

    python -m umi_gripper.calibration.calibrate_width [--port /dev/ttyACM0] [--servo-id 1]

Bring-up first if the servo doesn't answer: pioneer_leader_arm_teleop/encoder_test.py --ids <id> --raw.
"""
from __future__ import annotations

import argparse

import numpy as np

from umi_gripper import config as umi_config
from umi_gripper._paths import LEADER_TELEOP, add
from umi_gripper.width import RawEncoder, check_width_calibration

add(LEADER_TELEOP)


def average_count(encoder: RawEncoder, samples: int = 50) -> int:
    from servo_leader import COUNTS_PER_REV, wrapped_count_delta
    counts = [encoder.read_count() for _ in range(samples)]
    deltas = np.array([wrapped_count_delta(c, counts[0]) for c in counts])   # wrap-safe mean
    if np.ptp(deltas) > 20:
        raise ValueError("the jaws moved while sampling; hold them still")
    return int(round(counts[0] + deltas.mean())) % COUNTS_PER_REV


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default=None, help="default: the one USB serial board plugged in")
    parser.add_argument("--servo-id", type=int, default=None, help="default: width.servo_id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = umi_config.load_config()
    from leader_mapping import connected_port, usb_serial
    port = args.port or connected_port()
    servo_id = args.servo_id if args.servo_id is not None else int(cfg["width"]["servo_id"])
    encoder = RawEncoder(port, servo_id, int(cfg["width"]["baud"]))
    try:
        input("Open the jaws FULLY and hold them, then Enter ")
        open_count = average_count(encoder)
        input("Close the jaws FULLY (fingertips touching) and hold, then Enter ")
        closed_count = average_count(encoder)
    finally:
        encoder.close()
    width = {"servo_id": servo_id, "open_count": open_count, "closed_count": closed_count,
             "usb_serial": usb_serial(port)}
    try:
        check_width_calibration(width)
    except ValueError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"open {open_count}, closed {closed_count} counts; board USB serial {width['usb_serial']}")
    if width["usb_serial"] is None:
        print("this board reports no USB serial: set width.port instead")
    if not args.dry_run:
        umi_config.update_calibration({"width": width})
        print(f"saved to {umi_config.DEFAULT_CALIBRATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
