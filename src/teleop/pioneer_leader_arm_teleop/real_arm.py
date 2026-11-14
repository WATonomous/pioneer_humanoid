# Old 6-DoF arm only (today's real arm). 8dof is the default; the real arm moves to 8dof soon.
"""Real-arm backend of pioneer_leader_arm_teleop.py: leader vs real arm(s), side by side.

--arms left (default), right or bimanual: each arm is driven by its own leader board (leader_arm.json).

Dry run by default: subscribes to /interfacing/motorFeedback and never creates a publisher, so it
cannot move an arm. --live first asks "start at home? [y/N]":

  y  every active joint of every arm ramps from its measured angle to home at --home-rate deg/s
     (default 5) and holds there; you then put the leader at home by hand. Refused (nothing moves)
     if an active joint's home is outside its calibrated limits.
  n  no homing: start from wherever the arm is.

Then it publishes each leader as ArmPose on /arm/<side>/joint_targets (50 Hz) through joint_command,
but only after the gate passes and you press Enter; one gate and one Enter for both arms. The gate, on
every active joint of every arm: real arm inside its calibrated limits, leader within 15 deg of the
real arm (and within 15 deg of home after y); gripper inside its calibrated limits and within 0.1.

Run in the `simulation_mj` container (ROS, common_msgs, the leader's servo SDK) while
`interfacing` is up and the arm is powered:

    python3 pioneer_leader_arm_teleop.py --target real [--arms right|bimanual] [--live]

Every angle is in the URDF frame, which is also the command frame ArmPose / joint_command use:

    leader:  sign * leader angle, clamped to the calibrated limits (arm_calibration.yaml;
             the URDF limits for a joint without limit_range)
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
import termios
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[2]
# joint_config (arm_calibration.yaml reader), pioneer_humanoid (URDF limits, home) and
# humanoid_robot_learning (the --arms choices).
sys.path.insert(0, str(_SRC / "interfacing" / "can" / "scripts"))
sys.path.insert(0, str(_SRC / "pioneer_humanoid"))
sys.path.insert(0, str(_SRC / "robot_learning"))

from humanoid_robot_learning.recording.schema import ARM_SETS  # noqa: E402
from joint_config import (arm_pose_topic, find_actuators, find_calibration, gripper_position,  # noqa: E402
                          joint_safety, load_joint_map, load_joint_safety, motor_to_cmd_deg)

from arm_limits import gripper_fraction, limited_target  # noqa: E402
from leader_mapping import LeaderBoard, add_leader_args, check_leader_args  # noqa: E402
from servo_leader import ARM_SERVOS, GRIPPER_SERVO, SERVO_IDS  # noqa: E402

# The --live gate: leader within AGREE_TOL_DEG of the real arm on every active joint (and within
# HOME_TOL_DEG of home after homing), gripper within GRIP_TOL.
HOME_TOL_DEG = 15.0
AGREE_TOL_DEG = 15.0
STALE_AFTER_S = 0.5
PRINT_PERIOD_S = 0.2
READ_PERIOD_S = 0.02
# Each arm's gripper (a GL40) in arm_calibration.yaml, compared as position 0 open .. 1 closed.
GRIPPER_JOINT = "gripper.open_close"
GRIP_TOL = 0.1
# --live, before the gate, on y: ramp every active real joint from its measured angle to home at
# --home-rate deg/s, then hold it there until the leader takes over.
HOME_RATE_DEG_S = 5.0
HOME_SETTLE_S = 3.0
FEEDBACK_WAIT_S = 2.0


def build_joints(mapping: Path, side: str, urdf_joints: list[str],
                 urdf_limits: dict[str, tuple[float, float]]) -> list[dict]:
    """The six joints of the `side` arm in ArmPose slot order (= leader A..F = urdf_joints).

    limits_deg, what the leader is clamped to: the calibrated limits joint_command enforces, or
    the URDF limits for a joint without limit_range."""
    joints: list[dict | None] = [None] * len(urdf_joints)
    for motor_id, info in load_joint_map(mapping, side).items():
        if info["slot"] is None:  # gripper
            continue
        urdf_joint = urdf_joints[info["slot"]]
        lo, hi = urdf_limits[urdf_joint]
        joints[info["slot"]] = dict(
            info,
            motor_id=motor_id,
            urdf_joint=urdf_joint,
            limits_deg=((info["lower"], info["upper"]) if info["limit_range"]
                        else (math.degrees(lo), math.degrees(hi))),
        )
    missing = [urdf_joints[i] for i, j in enumerate(joints) if j is None]
    if missing:
        raise SystemExit(f"{mapping}: no {side}-arm motor for {missing}")
    return joints


@dataclass
class RealArm:
    """One leader board driving one real arm: its calibration, what the gate checks, its publisher."""

    side: str
    board: LeaderBoard
    joints: list[dict]  # build_joints, ArmPose slot order
    grip_id: int | None  # the gripper's motor id; None without one in arm_calibration.yaml
    grip_info: dict | None
    grip_range: tuple[float, float] | None  # its calibrated limits as positions
    home: list[float]  # rad, URDF (= command) frame
    active: set[str]  # joints joint_command drives (arm_actuators.yaml `active`)
    grip_active: bool
    leader: Any = None  # LeaderInput, opened inside run()'s try so a failed open still closes the others
    pub: Any = None  # ArmPose publisher, once live
    homed: Any = None  # (publisher, hold timer) after home_real_arms


def load_arm(side: str, board: LeaderBoard, mapping: Path, actuators: Path | None) -> RealArm:
    """The `side` arm as joint_command loads it: joints, gripper, home and what is active."""
    from pioneer_humanoid.arm_params import DEFAULT_JOINT_POS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
    from pioneer_humanoid.urdf_joint_limits import JOINT_POS_LIMITS

    urdf_joints = {"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS}[side]
    joints = build_joints(mapping, side, urdf_joints, JOINT_POS_LIMITS)
    joint_map = load_joint_map(mapping, side)
    grip_id = next((i for i, info in joint_map.items() if info["name"] == GRIPPER_JOINT), None)
    grip_info = joint_map.get(grip_id)
    # The gripper's calibrated limits as positions, to compare with its feedback.
    grip_range = None
    if grip_info is not None and grip_info["limit_range"]:
        grip_range = (gripper_position(grip_info, grip_info["lower"]),
                      gripper_position(grip_info, grip_info["upper"]))
    # arm_actuators.yaml `active`: joint_command leaves inactive joints alone, so the gate skips them.
    # It drives the gripper only if the gripper has a block there.
    _, blocks = load_joint_safety(actuators, side)
    active = {j["name"] for j in joints if joint_safety(actuators, side, j["name"]).get("active") is not False}
    grip_active = (grip_info is not None and grip_info["name"] in blocks and
                   blocks[grip_info["name"]].get("active") is not False)
    return RealArm(side=side, board=board, joints=joints, grip_id=grip_id, grip_info=grip_info,
                   grip_range=grip_range, home=[DEFAULT_JOINT_POS[j] for j in urdf_joints],
                   active=active, grip_active=grip_active)


def leader_deg(joint: dict, angle_rad: float, sign: float) -> tuple[float, bool]:
    """Leader angle -> URDF (= command-frame) target in deg, and whether the limits_deg clamp cut it."""
    raw = math.degrees(angle_rad * sign)
    urdf = math.degrees(limited_target(angle_rad, sign, 1.0, joint["limits_deg"]))
    return urdf, math.isfinite(raw) and abs(raw - urdf) > 1e-6


def compare(joints: list[dict], leader_rad: tuple[float, ...], signs: list[float],
            real_cmd: dict[int, float], home_urdf_rad: list[float]) -> list[dict]:
    """One row per joint: leader / real / home in the URDF (= command) frame, and the flags."""
    rows = []
    for i, joint in enumerate(joints):
        leader, clamped = leader_deg(joint, leader_rad[i], signs[i])
        real = real_cmd.get(joint["motor_id"])
        home = math.degrees(home_urdf_rad[i])
        # The real arm outside its calibrated limits (the leader is clamped inside them).
        out_of_hw = (real is not None and joint["limit_range"]
                     and not joint["lower"] <= real <= joint["upper"])
        rows.append(dict(
            joint=joint, leader=leader, real=real, diff=None if real is None else leader - real,
            home_err=leader - home, clamped=clamped, out_of_hw=out_of_hw,
        ))
    return rows


def gate_failures(rows: list[dict], grip: float, real_grip: float | None,
                  active: set[str], grip_active: bool,
                  grip_range: tuple[float, float] | None = None, check_home: bool = True) -> list[str]:
    """Why --live may not start yet; empty when it may. Inactive joints are not driven, so skipped.

    grip_range: the gripper's calibrated limits as positions (low, high); None to skip that check.
    check_home: also require the leader within HOME_TOL_DEG of home (after the real arm was homed)."""
    out = []
    for row in rows:
        j = row["joint"]
        name = j["name"]
        if name not in active:
            continue
        if row["real"] is None:
            out.append(f"{name}: no feedback")
        else:
            if row["out_of_hw"]:
                out.append(f"{name}: real {row['real']:+.1f} deg outside calibrated "
                           f"[{j['lower']:.1f}, {j['upper']:.1f}]")
            if abs(row["diff"]) > AGREE_TOL_DEG:
                out.append(f"{name}: leader {row['diff']:+.1f} deg from real")
        if check_home and abs(row["home_err"]) > HOME_TOL_DEG:
            out.append(f"{name}: leader {row['home_err']:+.1f} deg from home")
    if grip_active:
        if real_grip is None:
            out.append("gripper: no feedback")
            return out
        if grip_range is not None and not grip_range[0] <= real_grip <= grip_range[1]:
            out.append(f"gripper: real {real_grip:+.2f} outside calibrated "
                       f"[{grip_range[0]:.2f}, {grip_range[1]:.2f}]")
        if abs(grip - real_grip) > GRIP_TOL:
            out.append(f"gripper: leader {grip - real_grip:+.2f} from real")
    return out


def ramp(start: float, goal: float, rate_deg_s: float, elapsed: float) -> float:
    """Linear from start toward goal at rate_deg_s, stopping at goal."""
    step = rate_deg_s * elapsed
    return goal if abs(goal - start) <= step else start + math.copysign(step, goal - start)


def home_refusals(joints: list[dict], home_rad: list[float], active: set[str]) -> list[str]:
    """Why the real arm may not be ramped to home; empty when it may. An active joint whose home is
    outside its calibrated limits would be clamped by joint_command; inactive joints are not driven."""
    out = []
    for joint, home in zip(joints, home_rad):
        home_deg = math.degrees(home)
        if joint["name"] in active and joint["limit_range"] and not joint["lower"] <= home_deg <= joint["upper"]:
            out.append(f"{joint['name']}: home {home_deg:+.1f} deg is outside its calibrated limits "
                       f"[{joint['lower']:.1f}, {joint['upper']:.1f}]")
    return out


def home_rate_error(rate: float, arms: list[RealArm], actuators: Path | None) -> str | None:
    """Why --home-rate is not allowed: it must be > 0 and at most every active joint's velocity_max."""
    for arm in arms:
        for j in arm.joints:
            if j["name"] in arm.active:
                vmax = float(joint_safety(actuators, arm.side, j["name"]).get("velocity_max") or 0.0)
                if not 0.0 < rate <= vmax:
                    return f"--home-rate must be in (0, {vmax:g}], the {arm.side} {j['name']}'s velocity_max"
    return None


def format_rows(rows: list[dict], grip: float, real_grip: float | None,
                grip_range: tuple[float, float] | None = None, grip_id: int | None = None) -> list[str]:
    """real_grip: the real gripper's position (0 open .. 1 closed), None without feedback;
    grip_range: its calibrated limits as positions, as gate_failures; grip_id: its motor id."""
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
            flags.append("limit-clamp")
        if row["out_of_hw"]:
            flags.append(f"real outside hw [{j['lower']:.0f},{j['upper']:.0f}]")
        lines.append(f"{j['name']:<16}{j['urdf_joint']:<9}{j['motor_id']:>4}{row['leader']:+9.1f}"
                     f"{real:>9}{diff:>8}{row['home_err']:+8.1f}  {' '.join(flags)}")
    if real_grip is None:
        real, diff, warn = "  ---", "  ---", "NO FEEDBACK"
    else:
        real, diff = f"{real_grip:+9.2f}", f"{grip - real_grip:+8.2f}"
        warn = f">{GRIP_TOL:g}" if abs(grip - real_grip) > GRIP_TOL else ""
        if grip_range is not None and not grip_range[0] <= real_grip <= grip_range[1]:
            warn = f"{warn} real outside hw [{grip_range[0]:.2f},{grip_range[1]:.2f}]".strip()
    grip_label = "---" if grip_id is None else str(grip_id)
    lines.append(f"{'gripper':<16}{'position':<9}{grip_label:>4}{grip:+9.2f}{real:>9}{diff:>8}{'':>8}  {warn}")

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
    joint = {"limits_deg": (-90.0, 90.0), "limit_range": True, "lower": -45.0, "upper": 45.0,
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
    off = math.radians(max(HOME_TOL_DEG, AGREE_TOL_DEG) + 1.0)
    assert len(gate_failures(compare([joint], (off,), [1.0], {1: 0.0}, [0.0]),
                             0.0, None, active, True)) == 3  # off real, off home, gripper silent
    assert gate_failures(compare([joint], (math.radians(10.0),), [1.0], {}, [0.0]), 0.5, None, set(), False) == []

    rows = compare([joint], (math.radians(50.0),), [1.0], {1: 47.0}, [0.0])
    assert abs(rows[0]["diff"] - 3.0) < 1e-9 and rows[0]["out_of_hw"]
    assert compare([joint], (0.0,), [1.0], {}, [0.0])[0]["diff"] is None
    assert not compare([joint], (math.radians(50.0),), [1.0], {1: 44.0}, [0.0])[0]["out_of_hw"]

    # The real arm outside its calibrated limits blocks the gate, the gripper too.
    rows = compare([joint], (math.radians(50.0),), [1.0], {1: 50.0}, [math.radians(50.0)])
    assert gate_failures(rows, 0.0, None, active, False) == [
        "shoulder.pitch: real +50.0 deg outside calibrated [-45.0, 45.0]"]
    rows = compare([joint], (0.0,), [1.0], {1: 0.0}, [0.0])
    assert gate_failures(rows, 1.2, 1.2, active, True, (-0.3, 1.0)) == [
        "gripper: real +1.20 outside calibrated [-0.30, 1.00]"]
    assert gate_failures(rows, 1.0, 1.0, active, True, (-0.3, 1.0)) == []
    assert "real outside hw" in format_rows(rows, 1.2, 1.2, (-0.3, 1.0))[-2]

    # Not homed ("n"): the leader only has to match the real arm, wherever both are.
    far = math.radians(30.0)  # leader 30 deg from home
    rows = compare([joint], (far,), [1.0], {1: 25.0}, [0.0])
    assert gate_failures(rows, 0.0, None, active, False) == ["shoulder.pitch: leader +30.0 deg from home"]
    assert gate_failures(rows, 0.0, None, active, False, check_home=False) == []
    rows = compare([joint], (far,), [1.0], {1: 10.0}, [0.0])
    assert gate_failures(rows, 0.0, None, active, False, check_home=False) == ["shoulder.pitch: leader +20.0 deg from real"]

    assert ramp(0.0, -90.0, 5.0, 2.0) == -10.0 and ramp(0.0, -90.0, 5.0, 100.0) == -90.0
    assert ramp(10.0, 12.0, 5.0, 1.0) == 12.0 and ramp(3.0, 3.0, 5.0, 0.0) == 3.0
    other = dict(joint, motor_id=2, name="elbow.pitch")
    assert home_refusals([joint, other], [0.0, math.radians(90.0)], {"shoulder.pitch"}) == [], "inactive elbow skipped"
    assert home_refusals([joint, other], [0.0, math.radians(90.0)], {"shoulder.pitch", "elbow.pitch"}) == [
        "elbow.pitch: home +90.0 deg is outside its calibrated limits [-45.0, 45.0]"]

    # is_left follows the side: joint_command ignores a pose whose is_left doesn't match its topic.
    from types import SimpleNamespace

    class FakePose:  # the fields arm_pose sets on an ArmPose
        def __init__(self):
            self.shoulder, self.elbow, self.wrist = SimpleNamespace(), SimpleNamespace(), SimpleNamespace()

    q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for side in ("left", "right"):
        msg = arm_pose(FakePose, side, q, 0.5, True)
        assert msg.is_left == (side == "left"), side
        assert msg.shoulder.position + msg.elbow.position + msg.wrist.position == q

    # The real config, both arms: six joints in slot order, URDF limits loaded, a gripper, no shared id.
    from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
    mapping = find_calibration(None)
    actuators = find_actuators(None)
    ids = {}
    for side, urdf_joints in (("left", LEFT_ARM_JOINTS), ("right", RIGHT_ARM_JOINTS)):
        board = LeaderBoard(side=side, port="", usb_serial="", signs=(1.0,) * len(SERVO_IDS), zeros={})
        arm = load_arm(side, board, mapping, actuators)
        assert [j["urdf_joint"] for j in arm.joints] == urdf_joints
        assert [j["name"] for j in arm.joints] == ["shoulder.pitch", "shoulder.roll", "shoulder.yaw",
                                                   "elbow.pitch", "elbow.yaw", "wrist.pitch"], arm.joints
        for j in arm.joints:
            if j["limit_range"]:
                assert j["limits_deg"] == (j["lower"], j["upper"]), j
        assert arm.grip_id is not None, f"{side}: no {GRIPPER_JOINT} in {mapping}"
        ids[side] = {j["motor_id"] for j in arm.joints} | {arm.grip_id}
        refusals = home_refusals(arm.joints, arm.home, arm.active)
        print(f"{side}: {len(arm.joints)} joints + gripper (id {arm.grip_id}); active {sorted(arm.active)}; "
              f"--live homing: {'; '.join(refusals) or 'allowed'}")
    assert not ids["left"] & ids["right"], ids
    print(f"self-test ok (both arms from {mapping})")


def arm_pose(msg_type, side: str, q: list[float], grip: float, include_gripper: bool):
    """ArmPose of six targets for the `side` arm (deg, URDF = command frame); joint_command clamps
    and rate-limits."""
    msg = msg_type()
    msg.name = "leader_arm_teleop"
    msg.is_left = side == "left"
    q = [float(a) for a in q]
    msg.shoulder.position, msg.elbow.position, msg.wrist.position = q[0:3], q[3:5], q[5:6]
    msg.include_gripper = include_gripper
    msg.gripper_position = float(grip)
    return msg


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Leader arm vs the real arm(s): a dry run (publishes nothing) unless --live.")
    add_leader_args(parser, scene_help="ignored for --target real")
    # --leader-arm is the leader's own file (add_leader_args); each arm is driven by its own board.
    parser.add_argument("--arms", choices=tuple(ARM_SETS), default="left",
                        help="real arms to drive, one leader board each (leader_arm.json)")
    parser.add_argument("--arm-calibration", "--mapping", dest="mapping", default=None,
                        help="the real arm's arm_calibration.yaml (default: auto, as joint_command)")
    parser.add_argument("--live", action="store_true",
                        help="publish /arm/<side>/joint_targets once the gate passes and you press Enter")
    parser.add_argument("--home-rate", type=float, default=HOME_RATE_DEG_S,
                        help="--live: deg/s of each real joint's ramp to home, if you answer y")
    parser.add_argument("--self-test", action="store_true", help="check the angle math and exit (no ROS, no leader)")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    sides = list(ARM_SETS[args.arms])
    boards = check_leader_args(parser, args, sides)

    mapping = find_calibration(args.mapping)
    actuators = find_actuators(None)
    arms = [load_arm(side, boards[side], mapping, actuators) for side in sides]
    bimanual = len(arms) > 1
    # motor id -> its joint (or gripper) info, over every arm: feedback is matched by id alone.
    by_id: dict[int, dict] = {}
    for arm in arms:
        infos = [(j["motor_id"], j) for j in arm.joints]
        if arm.grip_info is not None:
            infos.append((arm.grip_id, arm.grip_info))
        for motor_id, info in infos:
            if motor_id in by_id:
                raise SystemExit(f"{mapping}: motor {motor_id} is in both arms")
            by_id[motor_id] = info
    grip_axis = tuple(SERVO_IDS).index(GRIPPER_SERVO)
    rate_error = home_rate_error(args.home_rate, arms, actuators)
    if rate_error:
        parser.error(rate_error)

    print(f"{' + '.join(sides)} arm{'s' * bimanual}, "
          f"{'LIVE once the gate passes' if args.live else 'DRY RUN (read-only)'}. "
          f"mapping: {mapping}, actuators: {actuators}")
    for arm in arms:
        for label, j, sign in zip(ARM_SERVOS, arm.joints, arm.board.signs):
            print(f"  {arm.side:<5} leader {label} {sign:+.0f} -> {j['urdf_joint']:<8} -> {j['name']:<15} "
                  f"id {j['motor_id']:>3}  direction {j['direction']:+g}"
                  f"{'' if j['name'] in arm.active else '  (inactive)'}")

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
        info = by_id.get(motor_id)
        if info is None:
            return
        cmd = motor_to_cmd_deg(info, float(msg.position))
        with lock:
            # The gripper is kept as position 0..1, the arm joints as degrees.
            real_cmd[motor_id] = gripper_position(info, cmd) if info["name"] == GRIPPER_JOINT else cmd
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

    def snapshot(arm: RealArm):
        angles = arm.leader.read()
        fresh = fresh_real()
        grip = gripper_fraction(angles[grip_axis], arm.board.signs[grip_axis])
        return compare(arm.joints, angles, arm.board.signs, fresh, arm.home), grip, fresh.get(arm.grip_id)

    def gate(snaps) -> list[str]:
        """Every arm's gate_failures, named by side when there are two."""
        out = []
        for arm, (rows, grip, real_grip) in zip(arms, snaps):
            prefix = f"{arm.side} " if bimanual else ""
            out += [prefix + failure for failure in
                    gate_failures(rows, grip, real_grip, arm.active, arm.grip_active, arm.grip_range,
                                  check_home=arm.homed is not None)]
        return out

    def publish(arm: RealArm, publisher, q: list[float]) -> None:
        # include_gripper False: joint_command holds the gripper where it is.
        msg = arm_pose(ArmPose, arm.side, q, 0.0, False)
        msg.header.stamp = node.get_clock().now().to_msg()
        publisher.publish(msg)

    def measured_active(arm: RealArm) -> dict[int, float] | None:
        """Fresh feedback of every active joint of `arm`, or None after FEEDBACK_WAIT_S without it."""
        need = [j["motor_id"] for j in arm.joints if j["name"] in arm.active]
        deadline = time.monotonic() + FEEDBACK_WAIT_S
        while True:
            real = fresh_real()
            if all(i in real for i in need):
                return real
            if time.monotonic() > deadline:
                return None
            time.sleep(READ_PERIOD_S)

    def home_real_arms() -> None:
        """Ask once: start at home? On y, ramp every active joint of every arm from its measured angle to
        home at --home-rate, then keep holding that pose (arm.homed) until the leader takes over."""
        nonlocal published
        answer = input("\nStart at home? y: the real arm ramps to home and holds there, then you put the leader "
                       f"at home. n: start from where the arm is (leader within {AGREE_TOL_DEG:g} deg of it). [y/N]: ")
        if answer.strip().lower() != "y":
            print("[INFO] Not homing: starting from where the arm is.", flush=True)
            return
        refusals = [f"{arm.side} {r}" if bimanual else r
                    for arm in arms for r in home_refusals(arm.joints, arm.home, arm.active)]
        if refusals:
            print("[INFO] Not homing, nothing moved: " + "; ".join(refusals) + " (joint_command would clamp it; "
                  "re-run calibrate_arm.py sweeping past home). Starting from where the arm is.", flush=True)
            return
        # Start from where each arm is NOW (measured), never from an assumed pose.
        plans = []  # (arm, publisher, q, {slot: start deg}, {slot: home deg})
        for arm in arms:
            real = measured_active(arm)
            if real is None:
                raise SystemExit(f"[ERROR] no feedback from every active {arm.side} joint; nothing moved.")
            # Inactive joints are ignored by joint_command; any finite angle will do for their slots.
            q = [real.get(j["motor_id"], math.degrees(arm.home[i])) for i, j in enumerate(arm.joints)]
            slots = [i for i, j in enumerate(arm.joints) if j["name"] in arm.active]
            plans.append((arm, None, q, {i: q[i] for i in slots}, {i: math.degrees(arm.home[i]) for i in slots}))
        far = max((abs(goals[i] - starts[i]) for _, _, _, starts, goals in plans for i in starts), default=0.0)
        print(f"[HOME] Ramping every active joint to home at {args.home_rate:g} deg/s "
              f"(furthest {far:.1f} deg, ~{far / args.home_rate:.0f} s). Hand on the E-stop.", flush=True)
        plans = [(arm, node.create_publisher(ArmPose, arm_pose_topic(arm.side), 10), q, starts, goals)
                 for arm, _, q, starts, goals in plans]
        published = True
        t0 = time.monotonic()
        settle_by = None
        while True:
            now = time.monotonic()
            done, worst = True, (0.0, "")
            fresh = fresh_real()
            for arm, publisher, q, starts, goals in plans:
                for i in starts:
                    q[i] = ramp(starts[i], goals[i], args.home_rate, now - t0)
                    done = done and q[i] == goals[i]
                publish(arm, publisher, q)
                for i in starts:
                    name = f"{arm.side} {arm.joints[i]['name']}" if bimanual else arm.joints[i]["name"]
                    measured = fresh.get(arm.joints[i]["motor_id"])
                    if measured is None:
                        raise SystemExit(f"[ERROR] {name} feedback went stale while homing; stopped publishing.")
                    worst = max(worst, (abs(measured - goals[i]), name))
            print(f"\r[HOME] furthest from home: {worst[1]} {worst[0]:5.1f} deg   ", end="", flush=True)
            if done:
                settle_by = settle_by or now + HOME_SETTLE_S
                if worst[0] <= HOME_TOL_DEG or now > settle_by:
                    break
            time.sleep(READ_PERIOD_S)
        if worst[0] > HOME_TOL_DEG:
            print(f"\n[WARN] {worst[1]} settled {worst[0]:.1f} deg from home.", flush=True)
        # Keep publishing the home pose from the spin thread: input() below blocks this loop, and
        # joint_command drops to kp = 0 after command_timeout_sec without an ArmPose.
        for arm, publisher, q, _, _ in plans:
            hold_q = list(q)
            arm.homed = (publisher, node.create_timer(
                READ_PERIOD_S, lambda arm=arm, publisher=publisher, hold_q=hold_q: publish(arm, publisher, hold_q)))
        print("\n[HOME] Holding at home. Now put the leader at home by hand (upper arm down, forearm "
              "forward, gripper matching the real one).", flush=True)

    redraw = sys.stdout.isatty()
    next_print = 0.0
    live = False
    published = False
    try:
        for arm in arms:
            arm.leader = LeaderInput(arm.board, args.baud, label=arm.side.upper() if bimanual else "")
        if args.live:
            home_real_arms()
        while rclpy.ok():
            snaps = [snapshot(arm) for arm in arms]
            failures = [] if live else gate(snaps)
            if args.live and not live and not failures:
                # Drop keys typed before the prompt, so only an Enter pressed after it starts live.
                if sys.stdin.isatty():
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                at_home = " at home and" if any(arm.homed is not None for arm in arms) else ""
                answer = input(f"\nGate passed: every leader{at_home} matching its real arm. Hand on the "
                               "E-stop, then press Enter to start publishing (anything else quits): ")
                if answer.strip():
                    break
                # Re-check: any arm may have moved while waiting at the prompt.
                snaps = [snapshot(arm) for arm in arms]
                failures = gate(snaps)
                if failures:
                    print("[WARN] moved while at the prompt: " + "; ".join(failures), flush=True)
                    time.sleep(2.0)
                else:
                    for arm in arms:
                        if arm.homed is not None:
                            # The leader takes over from the held home pose: the gate just put it within
                            # AGREE_TOL_DEG of the real arm.
                            arm.pub, hold = arm.homed
                            hold.cancel()
                        else:
                            arm.pub = node.create_publisher(ArmPose, arm_pose_topic(arm.side), 10)
                    live = published = True
            if live:
                for arm, (rows, grip, _) in zip(arms, snaps):
                    msg = arm_pose(ArmPose, arm.side, [row["leader"] for row in rows], grip, arm.grip_active)
                    msg.header.stamp = node.get_clock().now().to_msg()
                    arm.pub.publish(msg)
            now = time.monotonic()
            if now >= next_print:
                next_print = now + PRINT_PERIOD_S
                lines = []
                for arm, (rows, grip, real_grip) in zip(arms, snaps):
                    if bimanual:
                        lines.append(f"== {arm.side.upper()} ==")
                    lines += format_rows(rows, grip, real_grip, arm.grip_range, arm.grip_id)
                holding = [arm.side for arm in arms if arm.homed is not None]
                if live:
                    lines.append(f"LIVE: publishing {', '.join(arm_pose_topic(side) for side in sides)}.  "
                                 "Ctrl-C to stop.")
                elif holding:
                    lines.append("waiting for the gate: " + ("; ".join(failures) or "ok"))
                    lines.append(f"HOLDING the real {' and '.join(holding)} arm{'s' * (len(holding) > 1)} at home: "
                                 "put the leader at home.  Ctrl-C to stop (the arm then sinks).")
                else:
                    if args.live:
                        lines.append("waiting for the gate: " + ("; ".join(failures) or "ok"))
                    lines.append("DRY RUN: nothing is published.  Ctrl-C to stop.")
                print(("\033[H\033[J" if redraw else "\n") + "\n".join(lines), flush=True)
            time.sleep(READ_PERIOD_S)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        for arm in arms:
            if arm.leader is not None:
                arm.leader.close()
        node.destroy_node()
        rclpy.shutdown()
        if not published:
            print("\n[INFO] Stopped. Nothing was published; leader torque is OFF.", flush=True)
        else:
            print("\n[INFO] Stopped publishing. joint_command holds the last pose for its "
                  "command_timeout_sec, then sets kp = 0 and each arm sinks: support it.", flush=True)
