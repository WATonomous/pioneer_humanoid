"""Real-arm backend of pioneer_leader_arm_teleop.py: DRY RUN, leader vs real left arm, side by side.

READ-ONLY: subscribes to /interfacing/motorFeedback and never creates a publisher, so it cannot
move the arm. (--live, which will publish /arm/joint_targets, is a later step of #327.)

Run in the `simulation_mj` container (ROS, common_msgs, the leader's servo SDK) while
`interfacing` is up and the arm is powered:

    python3 pioneer_leader_arm_teleop.py --target real --port /dev/ttyACM1

Every angle is in the URDF frame, which is also the command frame ArmPose / joint_command use:

    leader:  sign * leader angle, clamped to the URDF limits      (as in sim)
    real:    zero_offset + motor / direction                       (arm_calibration.yaml)

The gripper is compared as its position, 0 open .. 1 closed, as joint_command maps it.

Pose both arms the same (motors off) and check every joint agrees within 5 deg across its range;
fix --signs (leader) or the real arm's direction / calibration (calibrate_arm.py) until it does.

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
# joint_config (arm_calibration.yaml reader) and pioneer_humanoid (URDF limits, home).
sys.path.insert(0, str(_SRC / "interfacing" / "can" / "scripts"))
sys.path.insert(0, str(_SRC / "pioneer_humanoid"))

from joint_config import find_calibration, gripper_position, load_joint_map, motor_to_cmd_deg  # noqa: E402

from arm_limits import gripper_fraction, limited_target  # noqa: E402
from leader_mapping import add_leader_args, check_leader_args  # noqa: E402
from servo_leader import ARM_SERVOS, GRIPPER_SERVO, SERVO_IDS, parse_signs  # noqa: E402

# The --live gate (#327), previewed here: leader within HOME_TOL_DEG of home and within
# AGREE_TOL_DEG of the real arm on every joint.
HOME_TOL_DEG = 3.0
AGREE_TOL_DEG = 5.0
STALE_AFTER_S = 0.5
PRINT_PERIOD_S = 0.2
READ_PERIOD_S = 0.02
# arm_calibration.yaml left.gripper.open_close (the GL40), compared as position 0 open .. 1 closed.
GRIPPER_CAN_ID = 21
GRIP_TOL = 0.1


def build_joints(mapping: Path, urdf_joints: list[str],
                 urdf_limits: dict[str, tuple[float, float]]) -> list[dict]:
    """The six left-arm joints in ArmPose slot order (= leader A..F = LEFT_ARM_JOINTS)."""
    joints: list[dict | None] = [None] * len(urdf_joints)
    for motor_id, info in load_joint_map(mapping, "left").items():
        if info["slot"] is None:  # gripper
            continue
        urdf_joint = urdf_joints[info["slot"]]
        lo, hi = urdf_limits[urdf_joint]
        joints[info["slot"]] = dict(
            info,
            motor_id=motor_id,
            urdf_joint=urdf_joint,
            urdf_limits_deg=(math.degrees(lo), math.degrees(hi)),
        )
    missing = [urdf_joints[i] for i, j in enumerate(joints) if j is None]
    if missing:
        raise SystemExit(f"{mapping}: no left-arm motor for {missing}")
    return joints


def leader_deg(joint: dict, angle_rad: float, sign: float) -> tuple[float, bool]:
    """Leader angle -> URDF (= command-frame) target in deg, and whether the URDF clamp cut it."""
    raw = math.degrees(angle_rad * sign)
    urdf = math.degrees(limited_target(angle_rad, sign, 1.0, joint["urdf_limits_deg"]))
    return urdf, math.isfinite(raw) and abs(raw - urdf) > 1e-6


def compare(joints: list[dict], leader_rad: tuple[float, ...], signs: list[float],
            real_cmd: dict[int, float], home_urdf_rad: list[float]) -> list[dict]:
    """One row per joint: leader / real / home in the URDF (= command) frame, and the flags."""
    rows = []
    for i, joint in enumerate(joints):
        leader, clamped = leader_deg(joint, leader_rad[i], signs[i])
        real = real_cmd.get(joint["motor_id"])
        home = math.degrees(home_urdf_rad[i])
        out_of_hw = joint["limit_range"] and not joint["lower"] <= leader <= joint["upper"]
        rows.append(dict(
            joint=joint, leader=leader, real=real, diff=None if real is None else leader - real,
            home_err=leader - home, clamped=clamped, out_of_hw=out_of_hw,
        ))
    return rows


def format_rows(rows: list[dict], grip: float, real_grip: float | None) -> list[str]:
    """real_grip: the real gripper's position (0 open .. 1 closed), None without feedback."""
    lines = [f"{'joint':<16}{'urdf':<9}{'id':>4}{'leader':>9}{'real':>9}{'diff':>8}{'home':>8}  warnings",
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
        if row["clamped"]:
            flags.append("urdf-clamp")
        if row["out_of_hw"]:
            flags.append(f"outside hw [{j['lower']:.0f},{j['upper']:.0f}]")
        lines.append(f"{j['name']:<16}{j['urdf_joint']:<9}{j['motor_id']:>4}{row['leader']:+9.1f}"
                     f"{real:>9}{diff:>8}{row['home_err']:+8.1f}  {' '.join(flags)}")
    if real_grip is None:
        real, diff, warn = "  ---", "  ---", "NO FEEDBACK"
    else:
        real, diff = f"{real_grip:+9.2f}", f"{grip - real_grip:+8.2f}"
        warn = f">{GRIP_TOL:g}" if abs(grip - real_grip) > GRIP_TOL else ""
    lines.append(f"{'gripper':<16}{'position':<9}{GRIPPER_CAN_ID:>4}{grip:+9.2f}{real:>9}{diff:>8}{'':>8}  {warn}")

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
    joint = {"urdf_limits_deg": (-90.0, 90.0), "limit_range": True, "lower": -45.0, "upper": 45.0,
             "motor_id": 1, "name": "shoulder.pitch", "urdf_joint": "joint1L"}
    deg, clamped = leader_deg(joint, math.radians(20.0), -1.0)
    assert abs(deg + 20.0) < 1e-9 and not clamped
    deg, clamped = leader_deg(joint, math.radians(120.0), 1.0)
    assert abs(deg - 90.0) < 1e-9 and clamped

    assert "NO FEEDBACK" in format_rows(compare([joint], (0.0,), [1.0], {}, [0.0]), 0.5, None)[-3]
    assert format_rows(compare([joint], (0.0,), [1.0], {1: 0.0}, [0.0]), 0.5, 0.3)[-3].endswith(">0.1")
    assert format_rows(compare([joint], (0.0,), [1.0], {1: 0.0}, [0.0]), 0.5, 0.45)[-3].split()[-1] == "+0.05"

    rows = compare([joint], (math.radians(50.0),), [1.0], {1: 47.0}, [0.0])
    assert abs(rows[0]["diff"] - 3.0) < 1e-9 and rows[0]["out_of_hw"]
    assert compare([joint], (0.0,), [1.0], {}, [0.0])[0]["diff"] is None

    # The real config: six joints, slot order, URDF limits loaded.
    from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS
    from pioneer_humanoid.urdf_joint_limits import JOINT_POS_LIMITS
    mapping = find_calibration(None)
    joints = build_joints(mapping, LEFT_ARM_JOINTS, JOINT_POS_LIMITS)
    assert [j["urdf_joint"] for j in joints] == LEFT_ARM_JOINTS
    assert [j["name"] for j in joints] == ["shoulder.pitch", "shoulder.roll", "shoulder.yaw",
                                           "elbow.pitch", "elbow.roll", "wrist.pitch"], joints
    print(f"self-test ok ({len(joints)} joints from {mapping})")


def run() -> None:
    parser = argparse.ArgumentParser(
        description="DRY RUN: leader arm vs the real left arm (reads /interfacing/motorFeedback, publishes nothing).")
    add_leader_args(parser, scene_help="ignored for --target real")
    # --calibration is the leader's own file (add_leader_args).
    parser.add_argument("--arm-calibration", "--mapping", dest="mapping", default=None,
                        help="the real arm's arm_calibration.yaml (default: auto, as joint_command)")
    parser.add_argument("--self-test", action="store_true", help="check the angle math and exit (no ROS, no leader)")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    check_leader_args(parser, args)

    from pioneer_humanoid.arm_params import DEFAULT_JOINT_POS, LEFT_ARM_JOINTS
    from pioneer_humanoid.urdf_joint_limits import JOINT_POS_LIMITS

    mapping = find_calibration(args.mapping)
    joints = build_joints(mapping, LEFT_ARM_JOINTS, JOINT_POS_LIMITS)
    by_id = {j["motor_id"]: j for j in joints}
    grip_info = load_joint_map(mapping, "left").get(GRIPPER_CAN_ID)
    signs = parse_signs(args.signs)
    grip_axis = tuple(SERVO_IDS).index(GRIPPER_SERVO)
    home = [DEFAULT_JOINT_POS[j] for j in LEFT_ARM_JOINTS]

    print(f"left arm, DRY RUN (read-only). mapping: {mapping}")
    for label, j, sign in zip(ARM_SERVOS, joints, signs):
        print(f"  leader {label} {sign:+.0f} -> {j['urdf_joint']:<8} -> {j['name']:<15} id {j['motor_id']:>3}  "
              f"direction {j['direction']:+g}")

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
        cmd = motor_to_cmd_deg(info, float(msg.position))
        with lock:
            # The gripper is kept as position 0..1, the arm joints as degrees.
            real_cmd[motor_id] = gripper_position(info, cmd) if info is grip_info else cmd
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
