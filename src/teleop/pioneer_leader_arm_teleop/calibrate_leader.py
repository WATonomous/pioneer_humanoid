"""Calibrate one board of the leader unit: its hanging-pose zero, servo directions and USB identity.

The pose is hanging straight down under gravity, gripper fully open. That is joint zero for the
leader, the real arm (calibrated the same way) and the sim (URDF zero), so leader angles map
one-to-one. Run once per board, or after a servo is re-mounted; it updates that side's entry in
leader_arm.json (here; commit it: it describes the team's leader unit) and enables it:

    python calibrate_leader.py --side left                            # the one USB serial board plugged in
    python calibrate_leader.py --side right --port <your-port>        # several plugged in: say which
    python calibrate_leader.py --side right --signs=<A..G>            # = because signs may start with -
    python calibrate_leader.py --side left --from leader_calibration.json  # convert
    python calibrate_leader.py --side right --robot 8dof              # the 8dof leader (+ servo H): leader_arm_8dof.json

Without --signs the servo directions (A..G; A..F, H, G for --robot 8dof) are the side's ROBOT_SIDE_SIGNS in
leader_mapping.py.

The board's USB serial number is stored too: the teleop finds each board by it, so plugging the two
boards in the other order can't swap the arms. Rotation joints (shoulder rotation, forearm
rotation, wrist) do not settle under gravity: line them up with their alignment marks. Leader
torque stays off throughout.
"""
import argparse
import json
import sys
from pathlib import Path

from leader_mapping import DEFAULT_LEADER_ARMS, ROBOT_SIDE_SIGNS, SIDES, connected_port, usb_serial
from servo_leader import LAYOUTS, ServoLeader, calibration_entry, load_calibration, parse_signs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", choices=SIDES, required=True, help="which arm this board drives")
    parser.add_argument("--port", help="the board's serial port now (default: the one USB serial board plugged in); "
                                       "not stored: the teleop finds the board by its USB serial")
    parser.add_argument("--robot", choices=tuple(LAYOUTS), default="armv2",
                        help="which leader: armv2 (servos A..G, the old arms) or 8dof (A..F, H = wrist yaw, G)")
    parser.add_argument("--signs", help="direction per servo, in reading order (A..G, or A..F,H,G for --robot 8dof; "
                                        "default: the side's in leader_mapping.ROBOT_SIDE_SIGNS)")
    parser.add_argument("--from", dest="old", type=Path,
                        help="take the zeros from an old leader_calibration.json instead of reading the leader")
    parser.add_argument("--baud", type=int, default=1_000_000, help="serial baud rate")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: leader_arm.json here, leader_arm_8dof.json for --robot 8dof")
    args = parser.parse_args()
    servo_ids = LAYOUTS[args.robot].servo_ids
    if args.out is None:
        args.out = DEFAULT_LEADER_ARMS[args.robot]

    signs = args.signs or ROBOT_SIDE_SIGNS[args.robot][args.side]
    try:
        parse_signs(signs, servo_ids)
        port = args.port or connected_port()
        serial_number = usb_serial(port)
    except ValueError as exc:
        parser.error(str(exc))
    if serial_number is None:
        parser.error(f"the board on {port} reports no USB serial number, so the teleop could not find it again")
    data = json.loads(args.out.read_text()) if args.out.is_file() else {}
    other = next(side for side in SIDES if side != args.side)
    if (data.get(other) or {}).get("enabled") and data[other]["usb_serial"] == serial_number:
        parser.error(f"this board's USB serial is {serial_number!r}; the {other} board in {args.out} has "
                     f"{data[other]['usb_serial']!r}. Two boards need different USB serials to be told apart.")

    if args.old is not None:
        positions = load_calibration(args.old, servo_ids)
    else:
        try:
            leader = ServoLeader(port, args.baud, servo_ids=servo_ids)
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

    lock_wrist = (data.get(args.side) or {}).get("lock_wrist", False)  # a setting, not calibration: keep it
    data[args.side] = {"enabled": True, "usb_serial": serial_number, "signs": signs, "lock_wrist": lock_wrist,
                       "servos": calibration_entry(positions, servo_ids)}
    data.setdefault(other, {"enabled": False})
    args.out.write_text(json.dumps({side: data[side] for side in SIDES}, indent=2) + "\n")
    print(f"Saved {args.side} board (port {port}, USB serial {serial_number}, signs {signs}) to {args.out}: "
          + ", ".join(f"{label}={positions[sid]}" for label, sid in servo_ids.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
