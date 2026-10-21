"""Calibrate one board of the leader unit: its hanging-pose zero, servo directions and USB identity.

The pose is hanging straight down under gravity, gripper fully open. That is joint zero for the
leader, the real arm (calibrated the same way) and the sim (URDF zero), so leader angles map
one-to-one. Run once per board, or after a servo is re-mounted; it updates that side's entry in
leader_arm.json (here, kept out of git) and enables it:

    python calibrate_leader.py --side left --port /dev/ttyACM0
    python calibrate_leader.py --side right --port /dev/ttyACM1 --signs=<A..G>   # = because signs may start with -
    python calibrate_leader.py --side left --port /dev/ttyACM0 --from leader_calibration.json  # convert

The board's USB serial number is stored too: the teleop finds each board by it, so plugging the two
boards in the other order can't swap the arms. Rotation joints (shoulder rotation, forearm
rotation, wrist) do not settle under gravity: line them up with their alignment marks. Leader
torque stays off throughout.
"""
import argparse
import json
import sys
from pathlib import Path

from leader_mapping import DEFAULT_LEADER_ARM, DEFAULT_SIGNS, SIDES, usb_serial
from servo_leader import SERVO_IDS, ServoLeader, calibration_entry, load_calibration, parse_signs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", choices=SIDES, required=True, help="which arm this board drives")
    parser.add_argument("--port", required=True, help="the board's serial port now (it is found by USB serial later)")
    parser.add_argument("--signs", help=f"direction per servo A..G (left default: {DEFAULT_SIGNS}; required for right)")
    parser.add_argument("--from", dest="old", type=Path,
                        help="take the zeros from an old leader_calibration.json instead of reading the leader")
    parser.add_argument("--baud", type=int, default=1_000_000, help="serial baud rate")
    parser.add_argument("--out", type=Path, default=DEFAULT_LEADER_ARM)
    args = parser.parse_args()

    signs = args.signs or (DEFAULT_SIGNS if args.side == "left" else None)
    if signs is None:
        parser.error("--side right needs --signs (its servos are mirrored; no default)")
    try:
        parse_signs(signs)
        serial_number = usb_serial(args.port)
    except ValueError as exc:
        parser.error(str(exc))
    data = json.loads(args.out.read_text()) if args.out.is_file() else {}
    other = next(side for side in SIDES if side != args.side)
    if (data.get(other) or {}).get("enabled") and (serial_number is None or data[other]["usb_serial"] == serial_number):
        parser.error(f"this board's USB serial is {serial_number!r}; the {other} board in {args.out} has "
                     f"{data[other]['usb_serial']!r}. Two boards need different USB serials to be told apart.")

    if args.old is not None:
        positions = load_calibration(args.old)
    else:
        try:
            leader = ServoLeader(args.port, args.baud)
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"ERROR: {exc}")
            return 1
        try:
            input(
                "\nLet the leader hang straight down, gripper fully OPEN, rotation joints on their marks.\n"
                "Press Enter to store this pose as zero (Ctrl+C to abort)... "
            )
            leader.read_radians()
            positions = leader.last_positions()
        except KeyboardInterrupt:
            print("\nAborted; nothing written.")
            return 1
        finally:
            leader.close()

    data[args.side] = {"enabled": True, "port": args.port, "usb_serial": serial_number, "signs": signs,
                       "servos": calibration_entry(positions)}
    data.setdefault(other, {"enabled": False})
    args.out.write_text(json.dumps({side: data[side] for side in SIDES}, indent=2) + "\n")
    print(f"Saved {args.side} board (port {args.port}, USB serial {serial_number}, signs {signs}) to {args.out}: "
          + ", ".join(f"{label}={positions[sid]}" for label, sid in SERVO_IDS.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
