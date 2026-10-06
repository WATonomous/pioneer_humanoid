#!/usr/bin/env python3
"""Interactive per-joint arm calibration.

Requires can_node running (feedback + motorCMD topics).

Typical flow per joint:
  1. Confirm / set motor id (Enter = yes, or type the correct id e.g. 14)
  2. Move joint to the pose you want as 0° → Enter  (records zero)
  3. Move to one range end → Enter, other end → Enter  (records min/max)
  4. Results written to arm_calibration.yaml and/or a sidecar file

MIT drives only answer when spoken to, so they are polled with zero-gain MIT frames: those
joints are LIMP while you calibrate -- support the arm. Stop joint_command_node first.

Examples (inside interfacing container, ROS sourced)::

  python3 /root/ament_ws/src/interfacing/can/scripts/calibrate_arm.py \\
    --arm-side left --write-mapping --calibration /calibration/arm_calibration.yaml
  ros2 run can calibrate_arm.py --arm-side left --write-mapping
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from common_msgs.msg import MotorCmd, MotorFeedback

import joint_config as jc


SET_ORIGIN = 5
DISABLE = 8


def _qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=50,
    )


def flatten_joints(
    mapping: Dict[str, Any],
    arm_side: str,
    include_hand: bool,
    include_gripper: bool,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return [(dotted.name, joint_dict), ...] for one arm side."""
    if arm_side not in mapping:
        raise KeyError(f"arm side '{arm_side}' not in mapping (keys={list(mapping)})")

    out: List[Tuple[str, Dict[str, Any]]] = []

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        if "can_id" in node:
            out.append((prefix, node))
            return
        for key, child in node.items():
            name = f"{prefix}.{key}" if prefix else key
            if key == "hand" and not include_hand:
                continue
            if key == "gripper" and not include_gripper:
                continue
            walk(child, name)

    walk(mapping[arm_side], arm_side)
    return out


def parse_can_id(value: Any) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def parse_id_list(text: str) -> set:
    """Parse '1,2,0x0A' or '10 14' into a set of ints."""
    if not text or not text.strip():
        return set()
    out = set()
    for part in text.replace(",", " ").split():
        out.add(int(part, 0))
    return out


def parse_name_list(text: str) -> set:
    if not text or not text.strip():
        return set()
    return {p.strip() for p in text.replace(",", " ").split() if p.strip()}


def joint_matches_skip(name: str, motor_id: int, skip_motors: set, skip_joints: set) -> bool:
    if motor_id in skip_motors:
        return True
    # Allow full name or suffix (e.g. shoulder.pitch or left.shoulder.pitch)
    if name in skip_joints:
        return True
    for skip in skip_joints:
        if name.endswith("." + skip) or name == skip:
            return True
    return False


class CalibrateNode(Node):
    def __init__(self) -> None:
        super().__init__("calibrate_arm")
        self._lock = threading.Lock()
        self._latest: Dict[int, MotorFeedback] = {}

        self._pub = self.create_publisher(MotorCmd, "/interfacing/motorCMD", 10)
        self.create_subscription(
            MotorFeedback,
            "/interfacing/motorFeedback",
            self._on_feedback,
            _qos(),
        )

    def _on_feedback(self, msg: MotorFeedback) -> None:
        with self._lock:
            self._latest[int(msg.motor_id)] = msg

    def latest(self, motor_id: int) -> Optional[MotorFeedback]:
        with self._lock:
            return self._latest.get(motor_id)

    def seen_motor_ids(self) -> List[int]:
        with self._lock:
            return sorted(self._latest.keys())

    def wait_for_feedback(self, motor_id: int, timeout_s: float = 3.0) -> MotorFeedback:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            fb = self.latest(motor_id)
            if fb is not None:
                return fb
            time.sleep(0.05)
        raise TimeoutError(
            f"No feedback for motor_id={motor_id} within {timeout_s}s. "
            "Is can_node running? Try: ros2 topic echo /interfacing/motorFeedback"
        )

    def set_origin(self, motor_id: int, temporary: bool) -> None:
        msg = MotorCmd()
        msg.motor_id = motor_id
        msg.control_type = SET_ORIGIN
        msg.temporary = temporary
        self._pub.publish(msg)
        # Give the bus a moment
        time.sleep(0.05)
        self._pub.publish(msg)

    # --- MIT drives only answer when spoken to --------------------------------------------
    def start_mit_poke(self, motor_ids: List[int], rate_hz: float = 20.0) -> None:
        """Zero-gain MIT frames (no torque) so the drives keep reporting while moved by hand."""
        self._mit_ids = set(motor_ids)
        self._mit_ticks = 0
        self._mit_timer = self.create_timer(1.0 / rate_hz, self._mit_poke)

    def add_mit_motor(self, motor_id: int) -> None:
        if getattr(self, "_mit_timer", None) is not None:
            self._mit_ids.add(motor_id)

    def _mit_poke(self) -> None:
        # Re-enter once a second: the first MIT_ENTER can go out before DDS matches can_node.
        enter = self._mit_ticks % 20 == 0
        self._mit_ticks += 1
        for motor_id in sorted(self._mit_ids):
            if enter:
                self._mit_special(motor_id, MotorCmd.MIT_ENTER)
            msg = MotorCmd()
            msg.motor_id = motor_id
            msg.control_type = MotorCmd.MIT_CONTROL
            msg.position = msg.velocity = msg.torque = 0.0
            msg.kp = msg.kd = 0.0
            self._pub.publish(msg)

    def stop_mit_poke(self) -> None:
        timer = getattr(self, "_mit_timer", None)
        if timer is None:
            return
        timer.cancel()
        self._mit_timer = None
        for _ in range(3):  # a dropped frame would leave the drive in motor mode
            for motor_id in sorted(self._mit_ids):
                self._mit_special(motor_id, MotorCmd.MIT_EXIT)
            time.sleep(0.01)

    def _mit_special(self, motor_id: int, control_type: int) -> None:
        msg = MotorCmd()
        msg.motor_id = motor_id
        msg.control_type = control_type
        self._pub.publish(msg)

    def other_motor_cmd_publishers(self) -> List[str]:
        return sorted({i.node_name for i in self.get_publishers_info_by_topic("/interfacing/motorCMD")
                       if i.node_name != self.get_name()})

    def disable(self, motor_id: int) -> None:
        msg = MotorCmd()
        msg.motor_id = motor_id
        msg.control_type = DISABLE
        self._pub.publish(msg)


def prompt(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        print()
        return "q"


def read_position(node: CalibrateNode, motor_id: int, timeout_s: float = 5.0) -> float:
    fb = node.wait_for_feedback(motor_id, timeout_s=timeout_s)
    return float(fb.position)


def capture_limits(
    node: CalibrateNode,
    name: str,
    motor_id: int,
) -> Tuple[str, Optional[Tuple[float, float, float]]]:
    """Move to one end → Enter, other end → Enter.

    Returns (status, data) where status is 'ok' | 'skip' | 'quit'.
    """
    ans = prompt(
        f"[{name}] Move joint to ONE end of its safe range, then press Enter "
        f"(s=skip limits, q=quit): "
    ).strip().lower()
    if ans == "q":
        return "quit", None
    if ans == "s":
        print("  Skipped limits.")
        return "skip", None

    a = read_position(node, motor_id)
    print(f"  End A: {a:.3f}°")

    ans = prompt(
        f"[{name}] Move joint to the OTHER end, then press Enter "
        f"(s=skip limits, q=quit): "
    ).strip().lower()
    if ans == "q":
        return "quit", None
    if ans == "s":
        print("  Skipped limits.")
        return "skip", None

    b = read_position(node, motor_id)
    print(f"  End B: {b:.3f}°")
    lo, hi = (a, b) if a <= b else (b, a)
    return "ok", (lo, hi, b)


def load_mapping(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML required. Install: apt-get install -y python3-yaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected mapping dict in {path}")
    return data


def save_mapping(path: Path, data: Dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML required")
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists() and not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  Backup: {backup}")
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  Wrote: {path}")


def save_sidecar(path: Path, results: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(results, f, default_flow_style=False, sort_keys=False)
    else:
        path = path.with_suffix(".json")
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    print(f"  Sidecar: {path}")


def apply_result_to_joint(joint: Dict[str, Any], result: Dict[str, Any]) -> None:
    if "can_id" in result:
        # Store as int; yaml.safe_dump will emit decimal by default.
        joint["can_id"] = int(result["can_id"])
    if "zero_offset" in result:
        joint["zero_offset"] = float(result["zero_offset"])
    if "lower_limit" in result:
        joint["lower_limit"] = float(result["lower_limit"])
    if "upper_limit" in result:
        joint["upper_limit"] = float(result["upper_limit"])
    if "direction" in result:
        joint["direction"] = int(result["direction"])


def mit_motor_ids(joints: List[Tuple[str, Dict[str, Any]]]) -> List[int]:
    """Drives that only answer MIT frames: every GL II, plus joints arm_actuators.yaml puts on
    control_type 0 (an AK brought up in MIT)."""
    safety = jc.find_actuators(None)
    ids = set()
    for name, joint in joints:
        motor_id = parse_can_id(joint["can_id"])
        short = name.split(".", 1)[1] if "." in name else name  # drop the arm side
        control_type = jc.joint_safety(safety, short).get("control_type")
        if jc.drive_family(motor_id) == "gl2" or (control_type is not None and
                                                  int(control_type) == MotorCmd.MIT_CONTROL):
            ids.add(motor_id)
    return sorted(ids)


def resolve_motor_id(
    node: "CalibrateNode",
    name: str,
    joint: Dict[str, Any],
    seen: set,
) -> Optional[int]:
    """Ask: is mapped id correct? Enter=yes, else type the real id. s=skip, q=quit."""
    motor_id = parse_can_id(joint["can_id"])
    while True:
        bus = sorted(node.seen_motor_ids())
        on_bus = motor_id in seen or motor_id in bus or node.latest(motor_id) is not None
        status = "on bus" if on_bus else "NOT on bus"
        print(f"  YAML can_id={motor_id} (0x{motor_id:02X})  [{status}]")
        print(f"  motors seen: {bus or '(none)'}")
        ans = prompt(
            f"[{name}] Is motor id {motor_id}? "
            f"[Enter]=yes  |  type correct id (e.g. 14)  |  s=skip  |  q=quit: "
        ).strip()

        if ans == "":
            if not on_bus:
                force = prompt(
                    f"  id {motor_id} not seen — use it anyway? [y/N]: "
                ).strip().lower()
                if force not in ("y", "yes"):
                    continue
            return motor_id

        low = ans.lower()
        if low in ("s", "skip"):
            print("  Skipped.")
            return None
        if low in ("q", "quit"):
            return -1
        if low in ("y", "yes"):
            return motor_id

        # Anything else: treat as a new motor id (decimal or 0xHH)
        try:
            motor_id = int(ans, 0)
        except ValueError:
            print(f"  Invalid id '{ans}' — Enter to confirm, or type a number like 14 / 0x0E")
            continue
        joint["can_id"] = motor_id
        print(f"  Using motor_id={motor_id} (0x{motor_id:02X})")
        time.sleep(0.2)


def run(args: argparse.Namespace) -> int:
    mapping_path = Path(args.mapping).expanduser().resolve()
    mapping = load_mapping(mapping_path)
    joints = flatten_joints(
        mapping,
        args.arm_side,
        include_hand=args.include_hand,
        include_gripper=not args.arm_only,
    )
    if not joints:
        print("No joints found in mapping.", file=sys.stderr)
        return 1

    skip_motors = parse_id_list(args.skip_motors)
    only_motors = parse_id_list(args.only_motors)
    skip_joints = parse_name_list(args.skip_joints)

    rclpy.init()
    node = CalibrateNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    time.sleep(1.0)  # DDS discovery, so the publisher check below sees everyone
    others = node.other_motor_cmd_publishers()
    if others:
        print(f"Refusing: {', '.join(others)} also publishes /interfacing/motorCMD. Stop it "
              f"(e.g. joint_command_node) first -- two commanders would fight over the drives.",
              file=sys.stderr)
        rclpy.shutdown()  # stops the spin thread; destroying a node mid-spin aborts the process
        spin_thread.join(timeout=2.0)
        return 1
    mit_ids = mit_motor_ids(joints)
    if mit_ids:
        print(f"MIT drives {mit_ids}: polling with zero-gain frames so they report. They are "
              f"LIMP (zero torque) until this exits -- support the arm.")
        node.start_mit_poke(mit_ids)

    print("Waiting for /interfacing/motorFeedback ...")
    time.sleep(max(1.0, args.discover_seconds))
    seen = set(node.seen_motor_ids())
    print(f"Motors seen on bus: {sorted(seen) if seen else '(none yet — move/power arm?)'}")

    if args.skip_missing:
        missing = [
            (name, parse_can_id(joint["can_id"]))
            for name, joint in joints
            if parse_can_id(joint["can_id"]) not in seen
        ]
        if missing:
            print("YAML can_id not on bus (still offered — you can type a new id at the per-joint prompt, or s to skip):")
            for name, mid in missing:
                print(f"  {name}  mapped={mid} (0x{mid:02X})")

    # CLI filters only; per-joint skip / motor-id remap is interactive
    selected: List[Tuple[str, Dict[str, Any]]] = []
    for name, joint in joints:
        mid = parse_can_id(joint["can_id"])
        if only_motors and mid not in only_motors:
            print(f"Skip {name} (not in --only-motors)")
            continue
        if joint_matches_skip(name, mid, skip_motors, skip_joints):
            print(f"Skip {name} motor_id={mid} (0x{mid:02X})")
            continue
        selected.append((name, joint))

    print()
    print("Arm joint calibration")
    print("  zero-mode:", args.zero_mode)
    print("  mapping:  ", mapping_path)
    print("  joints:   ", len(selected), f"(of {len(joints)} in file)")
    print("  Per joint: Enter=yes that id is correct  |  type new id  |  s=skip  |  q=quit")
    print()

    if not selected:
        print("Nothing to calibrate after filters.", file=sys.stderr)
        node.stop_mit_poke()
        node.destroy_node()
        rclpy.shutdown()
        return 1

    results: Dict[str, Any] = {
        "arm_side": args.arm_side,
        "zero_mode": args.zero_mode,
        "joints": {},
    }

    try:
        for name, joint in selected:
            print("=" * 60)
            print(f"Joint: {name}")
            print(
                f"  current YAML: can_id={joint.get('can_id')}  "
                f"zero_offset={joint.get('zero_offset')}  "
                f"limits=[{joint.get('lower_limit')}, {joint.get('upper_limit')}]"
            )

            motor_id = resolve_motor_id(node, name, joint, seen)
            if motor_id is None:
                continue
            if motor_id == -1:
                break
            if jc.drive_family(motor_id) == "gl2":
                node.add_mit_motor(motor_id)  # remapped to a GL40 id: poke it too

            ans = prompt(
                f"[{name}] Move joint to HOME / zero pose, then press Enter "
                "(s=skip this joint, q=quit): "
            ).strip().lower()
            if ans == "q":
                break
            if ans == "s":
                print("  Skipped.")
                continue

            fb = node.wait_for_feedback(motor_id, timeout_s=args.feedback_timeout)
            home_pos = float(fb.position)
            print(f"  Feedback at home: {home_pos:.3f}°")

            joint_result: Dict[str, Any] = {
                "can_id": motor_id,
                "home_position_raw": home_pos,
            }
            # Persist remapped id into YAML joint dict
            joint["can_id"] = motor_id

            if args.zero_mode == "software":
                # joint_command: motor = direction * (cmd - zero_offset)
                # Want cmd=0 → motor holds current home_pos:
                #   home_pos = direction * (0 - zero_offset)  ⇒  zero_offset = -home_pos / direction
                direction = int(joint.get("direction", 1))
                if direction == 0:
                    direction = 1
                joint_result["zero_offset"] = -home_pos / float(direction)
                joint_result["direction"] = direction
                print(f"  software zero_offset := {joint_result['zero_offset']:.3f}")
            elif args.zero_mode == "temporary":
                node.set_origin(motor_id, temporary=True)
                time.sleep(0.2)
                fb2 = node.wait_for_feedback(motor_id)
                print(f"  SET_ORIGIN temporary sent; feedback now {fb2.position:.3f}°")
                joint_result["zero_offset"] = 0.0
                joint_result["set_origin"] = "temporary"
            elif args.zero_mode == "permanent":
                confirm = prompt(
                    "  PERMANENT EEPROM origin — type 'yes' to confirm: "
                ).strip().lower()
                if confirm != "yes":
                    print("  Aborted permanent origin for this joint.")
                    continue
                node.set_origin(motor_id, temporary=False)
                time.sleep(0.2)
                fb2 = node.wait_for_feedback(motor_id)
                print(f"  SET_ORIGIN permanent sent; feedback now {fb2.position:.3f}°")
                joint_result["zero_offset"] = 0.0
                joint_result["set_origin"] = "permanent"
            else:
                raise ValueError(args.zero_mode)

            status, ends = capture_limits(node, name, motor_id)
            if status == "quit":
                results["joints"][name] = joint_result
                apply_result_to_joint(joint, joint_result)
                break
            if status == "ok" and ends is not None:
                lo, hi, last = ends
                direction = int(joint_result.get("direction", joint.get("direction", 1))) or 1
                zero_offset = float(joint_result.get("zero_offset", joint.get("zero_offset", 0.0)))
                # Limits in arm_calibration.yaml are command-frame (pre-calibration) degrees.
                # cmd = zero_offset + motor / direction
                cmd_a = zero_offset + lo / float(direction)
                cmd_b = zero_offset + hi / float(direction)
                cmd_lo, cmd_hi = (cmd_a, cmd_b) if cmd_a <= cmd_b else (cmd_b, cmd_a)
                pad = float(args.limit_pad)
                joint_result["lower_limit"] = cmd_lo + pad
                joint_result["upper_limit"] = cmd_hi - pad
                if joint_result["lower_limit"] > joint_result["upper_limit"]:
                    mid = 0.5 * (joint_result["lower_limit"] + joint_result["upper_limit"])
                    joint_result["lower_limit"] = mid - 1.0
                    joint_result["upper_limit"] = mid + 1.0
                print(
                    f"  motor range [{lo:.2f}, {hi:.2f}]° → cmd limits "
                    f"[{joint_result['lower_limit']:.2f}, {joint_result['upper_limit']:.2f}] "
                    f"(pad={pad})"
                )
                joint_result["last_position"] = last
                joint_result["motor_range"] = [lo, hi]

            apply_result_to_joint(joint, joint_result)
            results["joints"][name] = joint_result
            print(f"  Saved {name}")

    except (TimeoutError, RuntimeError, KeyboardInterrupt) as exc:
        print(f"\nStopped: {exc}")
    finally:
        node.stop_mit_poke()
        node.destroy_node()
        rclpy.shutdown()

    sidecar = Path(args.sidecar).expanduser()
    save_sidecar(sidecar, results)

    if args.write_mapping and results["joints"]:
        save_mapping(mapping_path, mapping)
    else:
        print("Mapping file not modified (pass --write-mapping to update it).")

    print("Done.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    default_mapping = (
        Path(__file__).resolve().parents[3]
        / "interfacing"
        / "joint_command"
        / "config"
        / "arm_calibration.yaml"
    )
    # In-container mount path used by docker-compose.interfacing.yaml
    container_mapping = Path("/calibration/arm_calibration.yaml")
    if container_mapping.exists():
        default_mapping = container_mapping
    elif not default_mapping.exists():
        default_mapping = Path("arm_calibration.yaml")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm-side", default="left", choices=["left", "right"], help="Side in arm_calibration.yaml")
    p.add_argument("--calibration", "--mapping", dest="mapping", default=str(default_mapping),
                   help="Path to arm_calibration.yaml")
    p.add_argument(
        "--sidecar",
        default=str(Path.home() / ".cache" / "humanoid" / "calibration" / "last_calibration.yaml"),
        help="Always write results here too",
    )
    p.add_argument(
        "--zero-mode",
        default="software",
        choices=["software", "temporary", "permanent"],
        help="software=YAML zero_offset only; temporary/permanent=MotorCmd SET_ORIGIN",
    )
    p.add_argument("--include-hand", action="store_true", help="Also calibrate hand joints (if present in YAML)")
    p.add_argument("--arm-only", action="store_true", help="Skip gripper; only 6 arm DOFs")
    p.add_argument(
        "--skip-motors",
        default="",
        help="Comma/space list of motor IDs to skip, e.g. '1,2' or '0x01 0x02'",
    )
    p.add_argument(
        "--only-motors",
        default="",
        help="If set, only calibrate these motor IDs (whitelist)",
    )
    p.add_argument(
        "--skip-joints",
        default="",
        help="Joint name suffixes to skip, e.g. 'shoulder.pitch gripper.open_close'",
    )
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="Warn about YAML can_ids not on the bus (still prompt; type a new id at the per-joint prompt, or s to skip)",
    )
    p.add_argument(
        "--discover-seconds",
        type=float,
        default=2.0,
        help="Seconds to listen for motor IDs before filtering",
    )
    p.add_argument("--limit-pad", type=float, default=2.0, help="Degrees to shrink recorded limits")
    p.add_argument("--feedback-timeout", type=float, default=5.0)
    p.add_argument(
        "--write-mapping",
        action="store_true",
        help="Update arm_calibration.yaml in place (creates .bak once)",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    if yaml is None:
        print("Missing PyYAML. In the container: apt-get update && apt-get install -y python3-yaml", file=sys.stderr)
        return 1
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
