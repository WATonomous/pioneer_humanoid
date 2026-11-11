"""Live STS3215 encoder readout for the whole leader arm -- no simulator.

Bring-up tool. Reads every leader servo at once and prints each one's angle
relative to wherever it sat when the program started, so you can move a joint by
hand and confirm it is wired, addressable, and turning the direction you expect
before Isaac Sim is involved.

It drives ``ServoLeader``, the same class ``pioneer_leader_arm_teleop.py`` uses, so a
clean run here means the read path the teleop depends on is working -- not merely
that something answered on the bus.

The leader arm is an INPUT DEVICE. Torque is written OFF for every servo at
startup and again on exit, and is never enabled. Nothing here can drive a
physical actuator.

    python encoder_test.py                  # all seven servos
    python encoder_test.py --ids 2          # just servo A
    python encoder_test.py --ids 2,3 --raw  # two servos, with encoder counts
    python encoder_test.py --ids 7,6        # wrist (F) + gripper (G) servos
"""

import argparse
import math
import sys
import time

from servo_leader import LAYOUTS, ServoLeader

POLL_PERIOD = 0.02  # 50 Hz


def _select_servos(spec: str | None, servo_ids: dict[str, int]) -> dict[str, int]:
    """servo_ids, or the subset named by a comma list of bus IDs."""
    if not spec:
        return dict(servo_ids)
    try:
        wanted = [int(item.strip()) for item in spec.split(",") if item.strip()]
    except ValueError:
        raise SystemExit(f"--ids: expected comma-separated integers, got {spec!r}")
    by_id = {servo_id: label for label, servo_id in servo_ids.items()}
    unknown = [i for i in wanted if i not in by_id]
    if unknown:
        raise SystemExit(
            f"--ids: {unknown} not in the leader map. Known IDs: {sorted(by_id)}"
        )
    return {by_id[i]: i for i in wanted}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="/dev/ttyACM0", help="leader serial port")
    parser.add_argument("--baud", type=int, default=1_000_000, help="serial baud rate")
    parser.add_argument(
        "--ids",
        default=None,
        help="comma-separated bus IDs to watch (default: all of --robot's servos)",
    )
    parser.add_argument("--robot", choices=tuple(LAYOUTS), default="armv2",
                        help="leader layout: armv2 (A..G, 7 servos) or 8dof (A..H: 8 servos)")
    parser.add_argument(
        "--raw", action="store_true", help="also show raw encoder counts"
    )
    args = parser.parse_args()

    servos = _select_servos(args.ids, LAYOUTS[args.robot].servo_ids)

    try:
        leader = ServoLeader(args.port, args.baud, servo_ids=servos)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print()
    print("=" * 62)
    print(" STS3215 LEADER ENCODER TEST")
    print("=" * 62)
    print("Servos: " + ", ".join(f"{label}=ID{sid}" for label, sid in servos.items()))
    print()
    print("Move the arm by hand. Angles are relative to the startup pose.")
    print("Press CTRL+C to stop.")
    print()

    try:
        while True:
            try:
                angles = leader.read_radians()
            except RuntimeError as exc:
                # One servo dropping out should not kill the readout -- this is the
                # tool you reach for precisely when a servo has stopped answering.
                print(f"\nREAD ERROR: {exc}")
                time.sleep(0.5)
                continue

            raw = leader.last_positions()
            cells = []
            for (label, servo_id), angle in zip(servos.items(), angles):
                cell = f"{label}:{math.degrees(angle):+7.1f}d"
                if args.raw:
                    cell += f"[{raw[servo_id]:4d}]"
                cells.append(cell)
            print("\r" + "  ".join(cells), end="", flush=True)
            time.sleep(POLL_PERIOD)
    except KeyboardInterrupt:
        pass
    finally:
        # Re-assert torque-off and hand the port back even on an unexpected
        # exception: only one process may hold the serial port at a time.
        leader.close()
        print("\nStopped. Leader torque is OFF.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
