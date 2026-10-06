"""Real-arm backend of pioneer_leader_arm_teleop.py: DRY RUN, leader vs real left arm, side by side.

READ-ONLY: subscribes to /interfacing/motorFeedback and never creates a publisher, so it cannot
move the arm. (--live, which will publish /arm/joint_targets, is a later step of #327.)

Run in the `simulation_mj` container (ROS, common_msgs, the leader's servo SDK) while
`interfacing` is up and the arm is powered:

    python3 pioneer_leader_arm_teleop.py --target real --port /dev/ttyACM1

Every angle is shown in the COMMAND frame, the frame ArmPose / joint_command use:

    leader:  q_urdf = sign * leader angle, clamped to the URDF limits      (as in sim)
             q_cmd  = urdf_direction * (q_urdf - urdf_offset_deg)          (safety_limits.yaml)
    real:    q_cmd  = zero_offset + motor / direction                       (hardware_mapping.yaml)

Pose both arms the same (motors off) and check every joint agrees within 1-2 deg across its range;
fix --signs, the calibrations or urdf_direction / urdf_offset_deg until it does.

Check the math without ROS or the leader:  python3 pioneer_leader_arm_teleop.py --target real --self-test
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
# joint_config (hardware_mapping / safety_limits readers) and pioneer_humanoid (URDF limits, home).
sys.path.insert(0, str(_SRC / "interfacing" / "can" / "scripts"))
sys.path.insert(0, str(_SRC / "pioneer_humanoid"))

from joint_config import find_mapping, find_safety_limits, joint_safety, load_joint_map, motor_to_cmd_deg  # noqa: E402

from arm_limits import gripper_fraction, limited_target  # noqa: E402
from leader_mapping import add_leader_args, check_leader_args  # noqa: E402
from servo_leader import ARM_SERVOS, GRIPPER_SERVO, SERVO_IDS, parse_signs  # noqa: E402

# The --live gate (#327), previewed here: leader within HOME_TOL_DEG of home and within
# AGREE_TOL_DEG of the real arm on every joint.
HOME_TOL_DEG = 3.0
AGREE_TOL_DEG = 5.0
# Calibration target for the dry-run check.
GOOD_TOL_DEG = 2.0
STALE_AFTER_S = 0.5
PRINT_PERIOD_S = 0.2
READ_PERIOD_S = 0.02
# hardware_mapping.yaml left.gripper.open_close (the GL40): shown, not compared, until the
# gripper command path exists (#327).
GRIPPER_CAN_ID = 21


def urdf_to_cmd_deg(joint: dict, urdf_deg: float) -> float:
    """Inverse of q_urdf = urdf_direction * q_cmd + urdf_offset_deg (urdf_direction is +-1)."""
    return joint["urdf_direction"] * (urdf_deg - joint["urdf_offset_deg"])


def build_joints(mapping: Path, safety: Path | None, urdf_joints: list[str],
                 urdf_limits: dict[str, tuple[float, float]]) -> list[dict]:
    """The six left-arm joints in ArmPose slot order (= leader A..F = LEFT_ARM_JOINTS)."""
    joints: list[dict | None] = [None] * len(urdf_joints)
    for motor_id, info in load_joint_map(mapping, "left").items():
        if info["slot"] is None:  # gripper
            continue
        block = joint_safety(safety, info["name"])
        direction = int(block.get("urdf_direction") or 1)
        if direction not in (1, -1):
            raise SystemExit(f"{safety}: {info['name']} urdf_direction must be +1 or -1, got {direction}")
        urdf_joint = urdf_joints[info["slot"]]
        lo, hi = urdf_limits[urdf_joint]
        joints[info["slot"]] = dict(
            info,
            motor_id=motor_id,
            urdf_joint=urdf_joint,
            urdf_direction=direction,
            urdf_offset_deg=float(block.get("urdf_offset_deg") or 0.0),
            urdf_limits_deg=(math.degrees(lo), math.degrees(hi)),
        )
    missing = [urdf_joints[i] for i, j in enumerate(joints) if j is None]
    if missing:
        raise SystemExit(f"{mapping}: no left-arm motor for {missing}")
    return joints


def leader_cmd_deg(joint: dict, angle_rad: float, sign: float) -> tuple[float, bool]:
    """Leader angle -> command-frame target (deg), and whether the URDF clamp cut it."""
    raw = math.degrees(angle_rad * sign)
    urdf = math.degrees(limited_target(angle_rad, sign, 1.0, joint["urdf_limits_deg"]))
    return urdf_to_cmd_deg(joint, urdf), math.isfinite(raw) and abs(raw - urdf) > 1e-6


def compare(joints: list[dict], leader_rad: tuple[float, ...], signs: list[float],
            real_cmd: dict[int, float], home_urdf_rad: list[float]) -> list[dict]:
    """One row per joint: leader / real / home in the command frame, and the flags."""
    rows = []
    for i, joint in enumerate(joints):
        leader, clamped = leader_cmd_deg(joint, leader_rad[i], signs[i])
        real = real_cmd.get(joint["motor_id"])
        home = urdf_to_cmd_deg(joint, math.degrees(home_urdf_rad[i]))
        out_of_hw = joint["limit_range"] and not joint["lower"] <= leader <= joint["upper"]
        rows.append(dict(
            joint=joint, leader=leader, real=real, diff=None if real is None else leader - real,
            home_err=leader - home, clamped=clamped, out_of_hw=out_of_hw,
        ))
    return rows


def format_rows(rows: list[dict], grip: float, real_grip: float | None) -> list[str]:
    lines = [f"{'joint':<16}{'urdf':<9}{'id':>4}{'leader':>9}{'real':>9}{'diff':>8}{'home':>8}  flags",
             "-" * 72]
    for row in rows:
        j = row["joint"]
        real = "  ---" if row["real"] is None else f"{row['real']:+8.1f}"
        diff = "  ---" if row["diff"] is None else f"{row['diff']:+7.1f}"
        flags = []
        if row["real"] is None:
            flags.append("NO FEEDBACK")
        elif abs(row["diff"]) > AGREE_TOL_DEG:
            flags.append(f">{AGREE_TOL_DEG:g}")
        elif abs(row["diff"]) > GOOD_TOL_DEG:
            flags.append(f">{GOOD_TOL_DEG:g}")
        if row["clamped"]:
            flags.append("urdf-clamp")
        if row["out_of_hw"]:
            flags.append(f"outside hw [{j['lower']:.0f},{j['upper']:.0f}]")
        lines.append(f"{j['name']:<16}{j['urdf_joint']:<9}{j['motor_id']:>4}{row['leader']:+9.1f}"
                     f"{real:>9}{diff:>8}{row['home_err']:+8.1f}  {' '.join(flags)}")
    real_g = "---" if real_grip is None else f"{real_grip:.1f} (cmd units, not compared)"
    lines.append(f"{'gripper':<16}leader closure {grip:.2f} (0 open .. 1 closed)   real id 21: {real_g}")

    diffs = [abs(r["diff"]) for r in rows if r["diff"] is not None]
    worst_home = max(abs(r["home_err"]) for r in rows)
    if len(diffs) < len(rows):
        agree = "no (missing feedback)"
    else:
        agree = f"{'yes' if max(diffs) <= AGREE_TOL_DEG else 'no'} (worst {max(diffs):.1f} deg)"
    lines.append(f"leader at home (<= {HOME_TOL_DEG:g} deg): "
                 f"{'yes' if worst_home <= HOME_TOL_DEG else 'no'} (worst {worst_home:.1f})   "
                 f"leader = real (<= {AGREE_TOL_DEG:g} deg): {agree}")
    lines.append("DRY RUN: nothing is published.  Ctrl-C to stop.")
    return lines


def self_test() -> None:
    joint = {"urdf_direction": 1, "urdf_offset_deg": 0.0, "urdf_limits_deg": (-90.0, 90.0),
             "limit_range": True, "lower": -45.0, "upper": 45.0, "motor_id": 1,
             "name": "shoulder.pitch", "urdf_joint": "joint1L"}
    assert urdf_to_cmd_deg(joint, 30.0) == 30.0
    assert urdf_to_cmd_deg(dict(joint, urdf_direction=-1, urdf_offset_deg=10.0), 30.0) == -20.0
    # Round trip with live_arm_mjviser's q_urdf = urdf_direction * q_cmd + urdf_offset_deg.
    for d, off in ((1, 0.0), (-1, 0.0), (1, 25.0), (-1, -40.0)):
        j = dict(joint, urdf_direction=d, urdf_offset_deg=off)
        assert abs(d * urdf_to_cmd_deg(j, 12.5) + off - 12.5) < 1e-9

    cmd, clamped = leader_cmd_deg(joint, math.radians(20.0), -1.0)
    assert abs(cmd + 20.0) < 1e-9 and not clamped
    cmd, clamped = leader_cmd_deg(joint, math.radians(120.0), 1.0)
    assert abs(cmd - 90.0) < 1e-9 and clamped

    rows = compare([joint], (math.radians(50.0),), [1.0], {1: 47.0}, [0.0])
    assert abs(rows[0]["diff"] - 3.0) < 1e-9 and rows[0]["out_of_hw"]
    assert compare([joint], (0.0,), [1.0], {}, [0.0])[0]["diff"] is None

    # The real config: six joints, slot order, URDF limits loaded.
    from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS
    from pioneer_humanoid.urdf_joint_limits import JOINT_POS_LIMITS
    mapping = find_mapping(None)
    joints = build_joints(mapping, find_safety_limits(None), LEFT_ARM_JOINTS, JOINT_POS_LIMITS)
    assert [j["urdf_joint"] for j in joints] == LEFT_ARM_JOINTS
    assert [j["name"] for j in joints] == ["shoulder.pitch", "shoulder.roll", "shoulder.yaw",
                                           "elbow.pitch", "elbow.roll", "wrist.pitch"], joints
    print(f"self-test ok ({len(joints)} joints from {mapping})")


def run() -> None:
    parser = argparse.ArgumentParser(
        description="DRY RUN: leader arm vs the real left arm (reads /interfacing/motorFeedback, publishes nothing).")
    add_leader_args(parser, scene_help="ignored for --target real")
    parser.add_argument("--mapping", default=None, help="hardware_mapping.yaml (default: auto, as joint_command)")
    parser.add_argument("--safety", default=None, help="safety_limits.yaml (default: auto, as joint_command)")
    parser.add_argument("--self-test", action="store_true", help="check the angle math and exit (no ROS, no leader)")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    check_leader_args(parser, args)

    from pioneer_humanoid.arm_params import DEFAULT_JOINT_POS, LEFT_ARM_JOINTS
    from pioneer_humanoid.urdf_joint_limits import JOINT_POS_LIMITS

    mapping, safety = find_mapping(args.mapping), find_safety_limits(args.safety)
    joints = build_joints(mapping, safety, LEFT_ARM_JOINTS, JOINT_POS_LIMITS)
    by_id = {j["motor_id"]: j for j in joints}
    grip_info = load_joint_map(mapping, "left").get(GRIPPER_CAN_ID)
    signs = parse_signs(args.signs)
    grip_axis = tuple(SERVO_IDS).index(GRIPPER_SERVO)
    home = [DEFAULT_JOINT_POS[j] for j in LEFT_ARM_JOINTS]

    print(f"left arm, DRY RUN (read-only). mapping: {mapping}  safety: {safety}")
    for label, j, sign in zip(ARM_SERVOS, joints, signs):
        print(f"  leader {label} {sign:+.0f} -> {j['urdf_joint']:<8} -> {j['name']:<15} id {j['motor_id']:>3}  "
              f"urdf_direction {j['urdf_direction']:+d}  urdf_offset_deg {j['urdf_offset_deg']:+g}")
    print("urdf_direction / urdf_offset_deg are NOT YET VERIFIED on hardware (see live_arm_mjviser.py).")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    from common_msgs.msg import MotorFeedback
    from leader_mapping import LeaderInput

    lock = threading.Lock()
    real_cmd: dict[int, float] = {}
    last_seen: dict[int, float] = {}

    def on_feedback(msg: MotorFeedback) -> None:
        motor_id = int(msg.motor_id)
        info = by_id.get(motor_id) or (grip_info if motor_id == GRIPPER_CAN_ID else None)
        if info is None:
            return
        with lock:
            real_cmd[motor_id] = motor_to_cmd_deg(info, float(msg.position))
            last_seen[motor_id] = time.monotonic()

    rclpy.init()
    # Subscriptions only: this node has no publisher, so the dry run cannot command a motor.
    node = Node("leader_arm_real_dry_run")
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50)
    node.create_subscription(MotorFeedback, "/interfacing/motorFeedback", on_feedback, qos)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    leader = LeaderInput(args)
    redraw = sys.stdout.isatty()
    next_print = 0.0
    try:
        while rclpy.ok():
            angles = leader.read()
            now = time.monotonic()
            if now >= next_print:
                next_print = now + PRINT_PERIOD_S
                with lock:
                    fresh = {i: q for i, q in real_cmd.items() if now - last_seen[i] <= STALE_AFTER_S}
                rows = compare(joints, angles, signs, fresh, home)
                grip = gripper_fraction(angles[grip_axis], signs[grip_axis])
                text = "\n".join(format_rows(rows, grip, fresh.get(GRIPPER_CAN_ID)))
                print(("\033[H\033[J" if redraw else "\n") + text, flush=True)
            time.sleep(READ_PERIOD_S)
    except KeyboardInterrupt:
        pass
    finally:
        leader.close()
        node.destroy_node()
        rclpy.shutdown()
        print("\n[INFO] Stopped. Nothing was published; leader torque is OFF.", flush=True)
