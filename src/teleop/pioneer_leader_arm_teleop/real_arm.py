"""Real-arm backend of pioneer_leader_arm_teleop.py: leader vs real left arm, side by side.

Dry run by default: subscribes to /interfacing/motorFeedback and never creates a publisher, so it
cannot move the arm. --live publishes the leader as ArmPose on /arm/joint_targets (50 Hz) through
joint_command, but only after the gate passes (every active joint: leader within 3 deg of home and
within 10 deg of the real arm; gripper within 0.1) and you type `live`. Before the gate, --live
offers to ramp the real elbow.pitch from its measured angle to home at --home-rate deg/s (default
5) and hold it there; it refuses if home is outside the joint's calibrated limits or it is inactive.

Run in the `simulation_mj` container (ROS, common_msgs, the leader's servo SDK) while
`interfacing` is up and the arm is powered:

    python3 pioneer_leader_arm_teleop.py --target real [--live]   # left board from leader_arm.json

Every angle is in the URDF frame, which is also the command frame ArmPose / joint_command use:

    leader:  sign * leader angle, clamped to the URDF limits      (as in sim)
    real:    zero_offset + motor / direction                       (arm_calibration.yaml)

The gripper is compared as its position, 0 open .. 1 closed, as joint_command maps it.

Pose both arms the same (motors off) and check every joint agrees within 10 deg across its range;
fix the leader's signs (calibrate_leader.py --signs) or the real arm's direction / calibration
(calibrate_arm.py) until it does.

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

from joint_config import (find_actuators, find_calibration, gripper_position, joint_safety,  # noqa: E402
                          load_joint_map, load_joint_safety, motor_to_cmd_deg)

from arm_limits import gripper_fraction, limited_target  # noqa: E402
from leader_mapping import add_leader_args, check_leader_args  # noqa: E402
from servo_leader import ARM_SERVOS, GRIPPER_SERVO, SERVO_IDS  # noqa: E402

# The --live gate: leader within HOME_TOL_DEG of home and within AGREE_TOL_DEG of the real arm on
# every active joint, gripper within GRIP_TOL.
HOME_TOL_DEG = 3.0
AGREE_TOL_DEG = 10.0
STALE_AFTER_S = 0.5
PRINT_PERIOD_S = 0.2
READ_PERIOD_S = 0.02
# arm_calibration.yaml left.gripper.open_close (the GL40), compared as position 0 open .. 1 closed.
GRIPPER_CAN_ID = 21
GRIP_TOL = 0.1
# --live, before the gate: optionally ramp the real HOME_JOINT from its measured angle to home at
# --home-rate deg/s (the others held where they are), then hold it there until the leader takes over.
HOME_JOINT = "elbow.pitch"
HOME_RATE_DEG_S = 5.0
HOME_SETTLE_S = 3.0
FEEDBACK_WAIT_S = 2.0


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


def gate_failures(rows: list[dict], grip: float, real_grip: float | None,
                  active: set[str], grip_active: bool) -> list[str]:
    """Why --live may not start yet; empty when it may. Inactive joints are not driven, so skipped."""
    out = []
    for row in rows:
        name = row["joint"]["name"]
        if name not in active:
            continue
        if row["real"] is None:
            out.append(f"{name}: no feedback")
        elif abs(row["diff"]) > AGREE_TOL_DEG:
            out.append(f"{name}: leader {row['diff']:+.1f} deg from real")
        if abs(row["home_err"]) > HOME_TOL_DEG:
            out.append(f"{name}: leader {row['home_err']:+.1f} deg from home")
    if grip_active:
        if real_grip is None:
            out.append("gripper: no feedback")
        elif abs(grip - real_grip) > GRIP_TOL:
            out.append(f"gripper: leader {grip - real_grip:+.2f} from real")
    return out


def ramp(start: float, goal: float, rate_deg_s: float, elapsed: float) -> float:
    """Linear from start toward goal at rate_deg_s, stopping at goal."""
    step = rate_deg_s * elapsed
    return goal if abs(goal - start) <= step else start + math.copysign(step, goal - start)


def home_refusal(joint: dict, home_deg: float, active: set[str]) -> str | None:
    """Why the real joint may not be homed; None when it may."""
    if joint["name"] not in active:
        return f"{joint['name']} is not active in arm_actuators.yaml, so joint_command would not drive it"
    if joint["limit_range"] and not joint["lower"] <= home_deg <= joint["upper"]:
        return (f"home {home_deg:+.1f} deg is outside {joint['name']}'s calibrated limits "
                f"[{joint['lower']:.1f}, {joint['upper']:.1f}] and joint_command would clamp it; "
                "re-run calibrate_arm.py sweeping past home first")
    return None


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
    return lines


def self_test() -> None:
    joint = {"urdf_limits_deg": (-90.0, 90.0), "limit_range": True, "lower": -45.0, "upper": 45.0,
             "motor_id": 1, "name": "shoulder.pitch", "urdf_joint": "joint1L"}
    deg, clamped = leader_deg(joint, math.radians(20.0), -1.0)
    assert abs(deg + 20.0) < 1e-9 and not clamped
    deg, clamped = leader_deg(joint, math.radians(120.0), 1.0)
    assert abs(deg - 90.0) < 1e-9 and clamped

    assert "NO FEEDBACK" in format_rows(compare([joint], (0.0,), [1.0], {}, [0.0]), 0.5, None)[-2]
    assert format_rows(compare([joint], (0.0,), [1.0], {1: 0.0}, [0.0]), 0.5, 0.3)[-2].endswith(">0.1")
    assert format_rows(compare([joint], (0.0,), [1.0], {1: 0.0}, [0.0]), 0.5, 0.45)[-2].split()[-1] == "+0.05"

    active = {"shoulder.pitch"}
    assert gate_failures(compare([joint], (0.0,), [1.0], {1: 1.0}, [0.0]), 0.0, 0.05, active, True) == []
    assert gate_failures(compare([joint], (0.0,), [1.0], {}, [0.0]), 0.0, 0.0, active, True) == [
        "shoulder.pitch: no feedback"]
    assert len(gate_failures(compare([joint], (math.radians(15.0),), [1.0], {1: 0.0}, [0.0]),
                             0.0, None, active, True)) == 3  # off real, off home, gripper silent
    assert gate_failures(compare([joint], (math.radians(10.0),), [1.0], {}, [0.0]), 0.5, None, set(), False) == []

    rows = compare([joint], (math.radians(50.0),), [1.0], {1: 47.0}, [0.0])
    assert abs(rows[0]["diff"] - 3.0) < 1e-9 and rows[0]["out_of_hw"]
    assert compare([joint], (0.0,), [1.0], {}, [0.0])[0]["diff"] is None

    assert ramp(0.0, -90.0, 5.0, 2.0) == -10.0 and ramp(0.0, -90.0, 5.0, 100.0) == -90.0
    assert ramp(10.0, 12.0, 5.0, 1.0) == 12.0 and ramp(3.0, 3.0, 5.0, 0.0) == 3.0
    assert home_refusal(joint, 0.0, active) is None
    assert "outside" in home_refusal(joint, -90.0, active)
    assert "not active" in home_refusal(joint, 0.0, set())

    # The real config: six joints, slot order, URDF limits loaded.
    from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS
    from pioneer_humanoid.urdf_joint_limits import JOINT_POS_LIMITS
    mapping = find_calibration(None)
    joints = build_joints(mapping, LEFT_ARM_JOINTS, JOINT_POS_LIMITS)
    assert [j["urdf_joint"] for j in joints] == LEFT_ARM_JOINTS
    assert [j["name"] for j in joints] == ["shoulder.pitch", "shoulder.roll", "shoulder.yaw",
                                           "elbow.pitch", "elbow.roll", "wrist.pitch"], joints
    print(f"self-test ok ({len(joints)} joints from {mapping})")
    from pioneer_humanoid.arm_params import DEFAULT_JOINT_POS
    slot = next(i for i, j in enumerate(joints) if j["name"] == HOME_JOINT)
    home_deg = math.degrees(DEFAULT_JOINT_POS[LEFT_ARM_JOINTS[slot]])
    print(f"--live homing of {HOME_JOINT} to {home_deg:+.1f} deg (limits only): "
          f"{home_refusal(joints[slot], home_deg, {HOME_JOINT}) or 'allowed'}")


def arm_pose(msg_type, q: list[float], grip: float, include_gripper: bool):
    """ArmPose of six targets (deg, URDF = command frame); joint_command clamps and rate-limits."""
    msg = msg_type()
    msg.name = "leader_arm_teleop"
    msg.is_left = True
    q = [float(a) for a in q]
    msg.shoulder.position, msg.elbow.position, msg.wrist.position = q[0:3], q[3:5], q[5:6]
    msg.include_gripper = include_gripper
    msg.gripper_position = float(grip)
    return msg


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Leader arm vs the real left arm: a dry run (publishes nothing) unless --live.")
    add_leader_args(parser, scene_help="ignored for --target real")
    # --leader-arm is the leader's own file (add_leader_args); this drives the left arm from its left board.
    parser.add_argument("--arm-calibration", "--mapping", dest="mapping", default=None,
                        help="the real arm's arm_calibration.yaml (default: auto, as joint_command)")
    parser.add_argument("--live", action="store_true",
                        help="publish /arm/joint_targets once the gate passes and you type `live`")
    parser.add_argument("--home-rate", type=float, default=HOME_RATE_DEG_S,
                        help=f"--live: deg/s of the optional ramp of the real {HOME_JOINT} to home")
    parser.add_argument("--self-test", action="store_true", help="check the angle math and exit (no ROS, no leader)")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    board = check_leader_args(parser, args, ["left"])["left"]

    from pioneer_humanoid.arm_params import DEFAULT_JOINT_POS, LEFT_ARM_JOINTS
    from pioneer_humanoid.urdf_joint_limits import JOINT_POS_LIMITS

    mapping = find_calibration(args.mapping)
    joints = build_joints(mapping, LEFT_ARM_JOINTS, JOINT_POS_LIMITS)
    by_id = {j["motor_id"]: j for j in joints}
    grip_info = load_joint_map(mapping, "left").get(GRIPPER_CAN_ID)
    signs = board.signs
    grip_axis = tuple(SERVO_IDS).index(GRIPPER_SERVO)
    home = [DEFAULT_JOINT_POS[j] for j in LEFT_ARM_JOINTS]
    # arm_actuators.yaml `active`: joint_command leaves inactive joints alone, so the gate skips them.
    # It drives the gripper only if the gripper has a block there.
    actuators = find_actuators(None)
    _, blocks = load_joint_safety(actuators)
    active = {j["name"] for j in joints if joint_safety(actuators, j["name"]).get("active") is not False}
    grip_active = (grip_info is not None and grip_info["name"] in blocks and
                   blocks[grip_info["name"]].get("active") is not False)
    home_vmax = float(joint_safety(actuators, HOME_JOINT).get("velocity_max") or 0.0)
    if not 0.0 < args.home_rate <= home_vmax:
        parser.error(f"--home-rate must be in (0, {home_vmax:g}], {HOME_JOINT}'s velocity_max")

    print(f"left arm, {'LIVE once the gate passes' if args.live else 'DRY RUN (read-only)'}. "
          f"mapping: {mapping}, actuators: {actuators}")
    for label, j, sign in zip(ARM_SERVOS, joints, signs):
        print(f"  leader {label} {sign:+.0f} -> {j['urdf_joint']:<8} -> {j['name']:<15} id {j['motor_id']:>3}  "
              f"direction {j['direction']:+g}{'' if j['name'] in active else '  (inactive)'}")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    from common_msgs.msg import ArmPose, MotorFeedback
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
    # No publisher until --live passes its gate, so the dry run cannot command a motor.
    node = Node("leader_arm_real")
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50)
    node.create_subscription(MotorFeedback, "/interfacing/motorFeedback", on_feedback, qos)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    def fresh_real() -> dict[int, float]:
        now = time.monotonic()
        with lock:
            return {i: q for i, q in real_cmd.items() if now - last_seen[i] <= STALE_AFTER_S}

    def snapshot():
        angles = leader.read()
        fresh = fresh_real()
        grip = gripper_fraction(angles[grip_axis], signs[grip_axis])
        return compare(joints, angles, signs, fresh, home), grip, fresh.get(GRIPPER_CAN_ID)

    def publish(publisher, q: list[float]) -> None:
        # include_gripper False: joint_command holds the gripper where it is.
        msg = arm_pose(ArmPose, q, 0.0, False)
        msg.header.stamp = node.get_clock().now().to_msg()
        publisher.publish(msg)

    def measured_active() -> dict[int, float] | None:
        """Fresh feedback of every active joint, or None after FEEDBACK_WAIT_S without it."""
        need = [j["motor_id"] for j in joints if j["name"] in active]
        deadline = time.monotonic() + FEEDBACK_WAIT_S
        while True:
            real = fresh_real()
            if all(i in real for i in need):
                return real
            if time.monotonic() > deadline:
                return None
            time.sleep(READ_PERIOD_S)

    def home_real_joint():
        """Offer to ramp the real HOME_JOINT to home; -> (publisher, hold timer), or None if not homed."""
        nonlocal published
        slot = next(i for i, j in enumerate(joints) if j["name"] == HOME_JOINT)
        joint, goal = joints[slot], math.degrees(home[slot])
        refusal = home_refusal(joint, goal, active)
        if refusal:
            print(f"[INFO] Not homing the real {HOME_JOINT}: {refusal}.", flush=True)
            return None
        real = measured_active()
        if real is None:
            print(f"[INFO] Not homing the real {HOME_JOINT}: no feedback from every active joint.", flush=True)
            return None
        answer = input(f"\nRamp the real {HOME_JOINT} from {real[joint['motor_id']]:+.1f} to {goal:+.1f} deg "
                       f"at {args.home_rate:g} deg/s, holding the other joints where they are? "
                       "Hand on the E-stop. [y/N]: ")
        if answer.strip().lower() != "y":
            return None
        # Start from where the arm is now, not where it was before the prompt.
        real = measured_active()
        if real is None:
            raise SystemExit(f"[ERROR] lost feedback while at the prompt; {HOME_JOINT} not homed.")
        # Inactive joints are ignored by joint_command; any finite angle will do for their slots.
        q = [real.get(j["motor_id"], math.degrees(home[i])) for i, j in enumerate(joints)]
        start = q[slot]
        publisher = node.create_publisher(ArmPose, "/arm/joint_targets", 10)
        published = True
        t0 = time.monotonic()
        settle_by = None
        while True:
            now = time.monotonic()
            q[slot] = ramp(start, goal, args.home_rate, now - t0)
            publish(publisher, q)
            measured = fresh_real().get(joint["motor_id"])
            if measured is None:
                raise SystemExit(f"[ERROR] {HOME_JOINT} feedback went stale while homing; stopped publishing.")
            print(f"\r[HOME] {HOME_JOINT} target {q[slot]:+6.1f}  real {measured:+6.1f}  home {goal:+6.1f}",
                  end="", flush=True)
            if q[slot] == goal:
                settle_by = settle_by or now + HOME_SETTLE_S
                if abs(measured - goal) <= HOME_TOL_DEG or now > settle_by:
                    break
            time.sleep(READ_PERIOD_S)
        if abs(measured - goal) > HOME_TOL_DEG:
            print(f"\n[WARN] {HOME_JOINT} settled {measured - goal:+.1f} deg from home.", flush=True)
        # Keep publishing the home pose from the spin thread: input() below blocks this loop, and
        # joint_command drops to kp = 0 after command_timeout_sec without an ArmPose.
        hold_q = list(q)
        return publisher, node.create_timer(READ_PERIOD_S, lambda: publish(publisher, hold_q))

    leader = LeaderInput(board, args.baud)
    redraw = sys.stdout.isatty()
    next_print = 0.0
    pub = None
    homed = None
    published = False
    try:
        if args.live:
            homed = home_real_joint()
        while rclpy.ok():
            rows, grip, real_grip = snapshot()
            failures = [] if pub else gate_failures(rows, grip, real_grip, active, grip_active)
            if args.live and pub is None and not failures:
                answer = input("\nGate passed: leader at home and matching the real arm. Hand on the "
                               "E-stop, then type `live` to start publishing (anything else quits): ")
                if answer.strip() != "live":
                    break
                # Re-check: either arm may have moved while waiting at the prompt.
                rows, grip, real_grip = snapshot()
                failures = gate_failures(rows, grip, real_grip, active, grip_active)
                if failures:
                    print("[WARN] moved while at the prompt: " + "; ".join(failures), flush=True)
                    time.sleep(2.0)
                elif homed is not None:
                    # The leader takes over from the held home pose: the gate just put it within
                    # AGREE_TOL_DEG of the real arm.
                    pub, hold = homed
                    hold.cancel()
                else:
                    pub = node.create_publisher(ArmPose, "/arm/joint_targets", 10)
                    published = True
            if pub is not None:
                msg = arm_pose(ArmPose, [row["leader"] for row in rows], grip, grip_active)
                msg.header.stamp = node.get_clock().now().to_msg()
                pub.publish(msg)
            now = time.monotonic()
            if now >= next_print:
                next_print = now + PRINT_PERIOD_S
                lines = format_rows(rows, grip, real_grip)
                if pub is not None:
                    lines.append("LIVE: publishing /arm/joint_targets.  Ctrl-C to stop.")
                elif homed is not None:
                    lines.append("waiting for the gate: " + ("; ".join(failures) or "ok"))
                    lines.append(f"HOLDING the real {HOME_JOINT} at home.  Ctrl-C to stop (the arm then sinks).")
                else:
                    if args.live:
                        lines.append("waiting for the gate: " + ("; ".join(failures) or "ok"))
                    lines.append("DRY RUN: nothing is published.  Ctrl-C to stop.")
                print(("\033[H\033[J" if redraw else "\n") + "\n".join(lines), flush=True)
            time.sleep(READ_PERIOD_S)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        leader.close()
        node.destroy_node()
        rclpy.shutdown()
        if not published:
            print("\n[INFO] Stopped. Nothing was published; leader torque is OFF.", flush=True)
        else:
            print("\n[INFO] Stopped publishing. joint_command holds the last pose for its "
                  "command_timeout_sec, then sets kp = 0 and the arm sinks: support it.", flush=True)
