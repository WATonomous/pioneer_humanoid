# Old 6-DoF arm only (today's real arm). 8dof is the default; the real arm moves to 8dof soon.
"""Leader arm -> real arm(s) teleop with a phased start: no need to put the leader at home.

    python3 real_arm_teleop.py [--arms right|bimanual] [--slow-rate 15] [--home-rate 5] [--skip-zero]

At start it asks "start at home? [y/N]". y adds the HOME phase between CHECK and SLOW; n skips it.

  ZERO   read-only. Hang the leader AND the real arm (each --arms arm) straight down, leader gripper
         open, then Enter: the leader's current reading becomes its zero (this run only, not saved).
         Refuses if any active real joint reads more than ZERO_TOL_DEG from 0 there: its calibration
         is stale, re-run calibrate_arm.py. The real arm's zero is never changed here (joint_command
         loads it from arm_calibration.yaml). --skip-zero keeps leader_arm.json's zeros.
  CHECK  read-only. Leader vs real arm, side by side. Nothing is published.
         Enter starts HOME (y) or SLOW (n) once every active joint reports feedback and is inside its
         calibrated limits.
  HOME   (y only) every active real joint ramps from its MEASURED angle to home at --home-rate deg/s and
         holds there; put the leader at home by hand. Enter starts SLOW once the ramp is done and every
         active leader joint is within HOME_TOL_DEG of home. Refused at the question (nothing moves) if
         an active joint's home is outside its calibrated limits.
  SLOW   the real arm moves to wherever the leader is, starting from its MEASURED pose, each joint at
         most --slow-rate deg/s, then keeps following at that speed: test the joints here.
         Enter starts LIVE once the target has caught up with the leader.
  LIVE   normal teleop (joint_command's own velocity_max still applies). Enter drops back to SLOW.

Ctrl-C stops publishing: joint_command holds the last pose for its command_timeout_sec, then sets
kp = 0 and the arm sinks, so support it. Everything goes through joint_command (/arm/<side>/joint_targets),
which still clamps, rate-limits and runs the MIT watchdog. Inactive joints (arm_actuators.yaml) are skipped.

Run in the `simulation_mj` container while `interfacing` and `joint_command` are up.
Check the logic without ROS or the leader:  python3 real_arm_teleop.py --self-test
"""
from __future__ import annotations

import argparse
import math
import queue
import sys
import termios
import threading
import time
from pathlib import Path

# real_arm puts joint_config, pioneer_humanoid and humanoid_robot_learning on sys.path.
from real_arm import (AGREE_TOL_DEG, GRIPPER_JOINT, HOME_RATE_DEG_S, HOME_TOL_DEG,  # noqa: E402
                      PRINT_PERIOD_S, READ_PERIOD_S, STALE_AFTER_S, RealArm, arm_pose, compare,
                      home_rate_error, home_refusals, load_arm)
from humanoid_robot_learning.recording.schema import ARM_SETS  # noqa: E402
from joint_config import (arm_pose_topic, find_actuators, find_calibration, gripper_position,  # noqa: E402
                          joint_safety, motor_to_cmd_deg)

from arm_limits import gripper_fraction  # noqa: E402
from leader_mapping import DEFAULT_LEADER_ARM, load_leader_boards  # noqa: E402
from servo_leader import GRIPPER_SERVO, SERVO_IDS  # noqa: E402

SLOW_RATE_DEG_S = 15.0
# SLOW's gripper speed, as position (0 open .. 1 closed) per second: full travel in 4 s.
SLOW_GRIP_RATE = 0.25
# LIVE may start once every active joint's target is this close to the leader (and the gripper's
# within GRIP_CAUGHT_UP), so the switch is no step in the target.
CAUGHT_UP_DEG = 2.0
GRIP_CAUGHT_UP = 0.05
# A stall longer than this (slow terminal, GC) must not turn into one big SLOW step.
MAX_DT_S = 0.1

# ZERO: every active real joint must read within this of 0 (hanging) for the leader re-zero.
ZERO_TOL_DEG = 5.0

ZERO, CHECK, HOME, SLOW, LIVE = "ZERO", "CHECK", "HOME", "SLOW", "LIVE"


def zero_failures(rows: list[dict], arm: RealArm) -> list[str]:
    """Why ZERO may not re-zero the leader: an active real joint without feedback, or not hanging
    (|real| > ZERO_TOL_DEG, which also means a stale calibration). Empty when it may."""
    out = []
    for row in rows:
        name = row["joint"]["name"]
        if name not in arm.active:
            continue
        if row["real"] is None:
            out.append(f"{name}: no feedback")
        elif abs(row["real"]) > ZERO_TOL_DEG:
            out.append(f"{name}: real reads {row['real']:+.1f} deg, not ~0 -- hang it straight down, "
                       "or recalibrate (calibrate_arm.py)")
    return out


def step_toward(current: float, goal: float, max_step: float) -> float:
    """current moved toward goal by at most max_step."""
    return goal if abs(goal - current) <= max_step else current + math.copysign(max_step, goal - current)


def start_failures(rows: list[dict], real_grip: float | None, arm: RealArm) -> list[str]:
    """Why SLOW may not start for `arm`: an active joint without feedback or outside its calibrated
    limits (a stale calibration). Empty when it may. Leader position does not matter."""
    out = []
    for row in rows:
        name = row["joint"]["name"]
        if name not in arm.active:
            continue
        if row["real"] is None:
            out.append(f"{name}: no feedback")
        elif row["out_of_hw"]:
            j = row["joint"]
            out.append(f"{name}: real {row['real']:+.1f} deg outside calibrated [{j['lower']:.1f}, {j['upper']:.1f}]"
                       " -- recalibrate")
    if arm.grip_active:
        if real_grip is None:
            out.append("gripper: no feedback")
        elif arm.grip_range is not None and not arm.grip_range[0] <= real_grip <= arm.grip_range[1]:
            out.append(f"gripper: real {real_grip:+.2f} outside calibrated "
                       f"[{arm.grip_range[0]:.2f}, {arm.grip_range[1]:.2f}]")
    return out


def lagging(rows: list[dict], target: list[float], grip: float, grip_target: float, arm: RealArm) -> list[str]:
    """Active joints whose SLOW target has not caught up with the leader yet. Empty: LIVE may start."""
    out = [f"{row['joint']['name']} {row['leader'] - t:+.1f}"
           for row, t in zip(rows, target)
           if row["joint"]["name"] in arm.active and abs(row["leader"] - t) > CAUGHT_UP_DEG]
    if arm.grip_active and abs(grip - grip_target) > GRIP_CAUGHT_UP:
        out.append(f"gripper {grip - grip_target:+.2f}")
    return out


def not_home(rows: list[dict], target: list[float], arm: RealArm) -> list[str]:
    """HOME: active joints whose ramp target has not reached home yet, then active leader joints more
    than HOME_TOL_DEG from home. Empty: SLOW may start."""
    out = [f"{row['joint']['name']} ramping" for row, t, h in zip(rows, target, arm.home)
           if row["joint"]["name"] in arm.active and t != math.degrees(h)]
    out += [f"{row['joint']['name']} leader {row['home_err']:+.1f} from home" for row in rows
            if row["joint"]["name"] in arm.active and abs(row["home_err"]) > HOME_TOL_DEG]
    return out


def format_arm(rows: list[dict], target: list[float] | None, grip: float, grip_target: float | None,
               real_grip: float | None, arm: RealArm) -> list[str]:
    """One line per joint: leader, the target being sent (--- in CHECK), real, real - target."""
    lines = [f"{'joint':<16}{'leader':>9}{'target':>9}{'real':>9}{'lag':>8}  notes", "-" * 64]
    for i, row in enumerate(rows):
        name = row["joint"]["name"]
        t = None if target is None else target[i]
        real = row["real"]
        notes = []
        if name not in arm.active:
            notes.append("inactive")
        else:
            if real is None:
                notes.append("NO FEEDBACK")
            if row["out_of_hw"]:
                notes.append("real outside limits")
            if row["clamped"]:
                notes.append("leader at limit")
        lines.append(f"{name:<16}{row['leader']:+9.1f}"
                     f"{'---' if t is None else f'{t:+.1f}':>9}"
                     f"{'---' if real is None else f'{real:+.1f}':>9}"
                     f"{'---' if t is None or real is None else f'{real - t:+.1f}':>8}  {' '.join(notes)}")
    if arm.grip_info is not None:
        lines.append(f"{'gripper (0..1)':<16}{grip:+9.2f}"
                     f"{'---' if grip_target is None else f'{grip_target:+.2f}':>9}"
                     f"{'---' if real_grip is None else f'{real_grip:+.2f}':>9}"
                     f"{'':>8}  {'' if arm.grip_active else 'inactive'}")
    return lines


def self_test() -> None:
    assert step_toward(0.0, 10.0, 0.3) == 0.3 and step_toward(0.0, -10.0, 0.3) == -0.3
    assert step_toward(5.0, 5.2, 0.3) == 5.2

    joint = {"limits_deg": (-90.0, 90.0), "limit_range": True, "lower": -45.0, "upper": 45.0,
             "motor_id": 1, "name": "shoulder.pitch", "urdf_joint": "joint1"}
    other = dict(joint, motor_id=2, name="elbow.pitch")
    arm = RealArm(side="right", board=None, joints=[joint, other], grip_id=None, grip_info=None, grip_range=None,
                  home=[0.0, 0.0], active={"shoulder.pitch"}, grip_active=False)

    # The leader may be anywhere: only feedback and the real arm's limits gate SLOW.
    rows = compare(arm.joints, (math.radians(80.0), 0.0), [1.0, 1.0], {1: 10.0}, arm.home)
    assert start_failures(rows, None, arm) == [], "inactive elbow without feedback is skipped"
    rows = compare(arm.joints, (0.0, 0.0), [1.0, 1.0], {}, arm.home)
    assert start_failures(rows, None, arm) == ["shoulder.pitch: no feedback"]
    rows = compare(arm.joints, (0.0, 0.0), [1.0, 1.0], {1: 50.0}, arm.home)
    assert "outside calibrated" in start_failures(rows, None, arm)[0]

    # SLOW from the measured 10 deg to a leader at 40 deg: 15 deg/s -> 2 s, then caught up.
    rows = compare(arm.joints, (math.radians(40.0), 0.0), [1.0, 1.0], {1: 10.0}, arm.home)
    target, t = [10.0, 0.0], 0.0
    while t < 1.0 - 1e-9:
        target[0] = step_toward(target[0], rows[0]["leader"], SLOW_RATE_DEG_S * READ_PERIOD_S)
        t += READ_PERIOD_S
    assert abs(target[0] - 25.0) < 1e-6, target
    assert lagging(rows, target, 0.0, 0.0, arm) == ["shoulder.pitch +15.0"]
    target[0] = 39.0
    assert lagging(rows, target, 0.0, 0.0, arm) == [], "within CAUGHT_UP_DEG; the inactive elbow is ignored"

    lines = format_arm(rows, None, 0.0, None, None, arm)
    assert "inactive" in lines[3] and "---" in lines[2]

    # ZERO: only active joints, each within ZERO_TOL_DEG of 0.
    rows = compare(arm.joints, (0.0, 0.0), [1.0, 1.0], {1: 3.0}, arm.home)
    assert zero_failures(rows, arm) == []
    rows = compare(arm.joints, (0.0, 0.0), [1.0, 1.0], {1: 8.0}, arm.home)
    assert "not ~0" in zero_failures(rows, arm)[0]
    rows = compare(arm.joints, (0.0, 0.0), [1.0, 1.0], {}, arm.home)
    assert zero_failures(rows, arm) == ["shoulder.pitch: no feedback"]

    # HOME: SLOW waits for the ramp to reach home and for the leader near home (active joints only).
    rows = compare(arm.joints, (math.radians(30.0), math.radians(60.0)), [1.0, 1.0], {1: 2.0}, arm.home)
    assert not_home(rows, [2.0, 0.0], arm) == ["shoulder.pitch ramping", "shoulder.pitch leader +30.0 from home"]
    rows = compare(arm.joints, (math.radians(10.0), math.radians(60.0)), [1.0, 1.0], {1: 0.0}, arm.home)
    assert not_home(rows, [0.0, 0.0], arm) == [], "home reached, leader within HOME_TOL_DEG; inactive elbow ignored"
    print("self-test ok")


def run() -> None:
    parser = argparse.ArgumentParser(description="Leader arm -> real arm teleop: CHECK -> SLOW -> LIVE, Enter to advance.")
    parser.add_argument("--arms", choices=tuple(ARM_SETS), default="left",
                        help="real arms to drive, one leader board each (leader_arm.json)")
    parser.add_argument("--leader-arm", type=Path, default=DEFAULT_LEADER_ARM,
                        help="leader boards file from calibrate_leader.py (default: leader_arm.json here)")
    parser.add_argument("--baud", type=int, default=1_000_000, help="leader serial baud rate")
    parser.add_argument("--arm-calibration", dest="mapping", default=None,
                        help="the real arm's arm_calibration.yaml (default: auto, as joint_command)")
    parser.add_argument("--slow-rate", type=float, default=SLOW_RATE_DEG_S,
                        help=f"SLOW phase: max deg/s per joint (default {SLOW_RATE_DEG_S:g})")
    parser.add_argument("--home-rate", type=float, default=HOME_RATE_DEG_S,
                        help=f"HOME phase (if you answer y): deg/s per joint (default {HOME_RATE_DEG_S:g})")
    parser.add_argument("--skip-zero", action="store_true",
                        help="skip ZERO: keep the leader zeros from leader_arm.json")
    parser.add_argument("--self-test", action="store_true", help="check the logic and exit (no ROS, no leader)")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    sides = list(ARM_SETS[args.arms])
    try:
        boards = load_leader_boards(args.leader_arm, sides)
    except ValueError as exc:
        parser.error(str(exc))
    mapping = find_calibration(args.mapping)
    actuators = find_actuators(None)
    arms = [load_arm(side, boards[side], mapping, actuators) for side in sides]
    bimanual = len(arms) > 1

    # SLOW must be slower than every active joint's own velocity_max, or it is not the limit.
    for arm in arms:
        for j in arm.joints:
            if j["name"] in arm.active:
                vmax = float(joint_safety(actuators, arm.side, j["name"]).get("velocity_max") or 0.0)
                if not 0.0 < args.slow_rate <= vmax:
                    parser.error(f"--slow-rate must be in (0, {vmax:g}], the {arm.side} {j['name']}'s velocity_max")
    rate_error = home_rate_error(args.home_rate, arms, actuators)
    if rate_error:
        parser.error(rate_error)

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

    print(f"{' + '.join(sides)} arm{'s' * bimanual}. mapping: {mapping}, actuators: {actuators}")
    for arm in arms:
        print(f"  {arm.side}: active {sorted(arm.active)}{' + gripper' if arm.grip_active else ''}")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    from common_msgs.msg import ArmPose, MotorFeedback
    from leader_mapping import LeaderInput

    lock = threading.Lock()
    real_cmd: dict[int, float] = {}
    last_seen: dict[int, float] = {}

    def on_feedback(msg: MotorFeedback) -> None:
        info = by_id.get(int(msg.motor_id))
        if info is None:
            return
        cmd = motor_to_cmd_deg(info, float(msg.position))
        with lock:
            real_cmd[int(msg.motor_id)] = gripper_position(info, cmd) if info["name"] == GRIPPER_JOINT else cmd
            last_seen[int(msg.motor_id)] = time.monotonic()

    def fresh_real() -> dict[int, float]:
        now = time.monotonic()
        with lock:
            return {i: q for i, q in real_cmd.items() if now - last_seen[i] <= STALE_AFTER_S}

    # Enter arrives as an empty line from a reader thread, so the control loop never blocks on input.
    keys: queue.Queue[str] = queue.Queue()

    def read_keys() -> None:
        for line in sys.stdin:
            keys.put(line.strip().lower())
        keys.put("eof")

    rclpy.init()
    # No publisher until SLOW starts, so CHECK cannot command a motor.
    node = Node("leader_arm_real_teleop")
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=50)
    node.create_subscription(MotorFeedback, "/interfacing/motorFeedback", on_feedback, qos)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    phase = CHECK if args.skip_zero else ZERO
    # Per arm, the leader reading (rad, per servo) taken as zero in ZERO; subtracted from every read.
    leader_zero: dict[str, tuple[float, ...]] = {arm.side: (0.0,) * len(SERVO_IDS) for arm in arms}
    targets: dict[str, list[float]] = {}
    grip_targets: dict[str, float] = {}
    pubs: dict[str, object] = {}
    message = ""
    redraw = sys.stdout.isatty()
    next_print = 0.0
    last_tick = time.monotonic()
    try:
        for arm in arms:
            arm.leader = LeaderInput(arm.board, args.baud, label=arm.side.upper() if bimanual else "")
        # Asked before the key reader starts; nothing moves until HOME (after CHECK + Enter).
        answer = input("\nStart at home? y: after CHECK the real arm ramps to home and holds there, then you put "
                       "the leader at home. n: the real arm moves to wherever the leader is. [y/N]: ")
        start_home = answer.strip().lower() == "y"
        if start_home:
            refusals = [f"{arm.side} {r}" if bimanual else r
                        for arm in arms for r in home_refusals(arm.joints, arm.home, arm.active)]
            if refusals:
                start_home = False
                print("[INFO] Not homing: " + "; ".join(refusals) + " (joint_command would clamp it; re-run "
                      "calibrate_arm.py sweeping past home). Continuing without HOME.", flush=True)
                time.sleep(3.0)
        home_deg = {arm.side: [math.degrees(h) for h in arm.home] for arm in arms}
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)  # nothing typed before the start counts
        threading.Thread(target=read_keys, daemon=True).start()

        while rclpy.ok():
            now = time.monotonic()
            dt = min(max(now - last_tick, 0.0), MAX_DT_S)
            last_tick = now
            fresh = fresh_real()
            snaps = []
            raw: dict[str, tuple[float, ...]] = {}
            for arm in arms:
                raw[arm.side] = arm.leader.read()
                angles = tuple(a - z for a, z in zip(raw[arm.side], leader_zero[arm.side]))
                rows = compare(arm.joints, angles, arm.board.signs, fresh, arm.home)
                grip = gripper_fraction(angles[grip_axis], arm.board.signs[grip_axis])
                snaps.append((rows, grip, fresh.get(arm.grip_id)))

            # One Enter per tick; extra ones typed meanwhile are dropped.
            key = None
            try:
                key = keys.get_nowait()
                while not keys.empty():
                    keys.get_nowait()
            except queue.Empty:
                pass
            if key in ("q", "eof"):
                break

            failures = [f"{arm.side} {f}" if bimanual else f
                        for arm, (rows, _, real_grip) in zip(arms, snaps)
                        for f in start_failures(rows, real_grip, arm)]

            if phase == ZERO and key == "":
                not_hanging = [f"{arm.side} {f}" if bimanual else f
                               for arm, (rows, _, _) in zip(arms, snaps) for f in zero_failures(rows, arm)]
                if not_hanging:
                    message = "Not zeroing: " + "; ".join(not_hanging)
                else:
                    for arm in arms:
                        leader_zero[arm.side] = raw[arm.side]
                    phase, message = CHECK, "Leader zeroed at the hanging pose (this run only)."
            elif phase == CHECK and key == "":
                if failures:
                    message = "Not starting SLOW: " + "; ".join(failures)
                else:
                    # Start from where each arm IS, not from the leader.
                    for arm, (rows, grip, real_grip) in zip(arms, snaps):
                        targets[arm.side] = [row["real"] if row["real"] is not None else row["leader"]
                                             for row in rows]
                        grip_targets[arm.side] = real_grip if real_grip is not None else grip
                        pubs[arm.side] = node.create_publisher(ArmPose, arm_pose_topic(arm.side), 10)
                    if start_home:
                        phase, message = HOME, f"HOME: ramping to home at {args.home_rate:g} deg/s."
                    else:
                        phase, message = SLOW, f"SLOW: moving to the leader at {args.slow_rate:g} deg/s."
            elif phase == HOME and key == "":
                waiting = [f"{arm.side} {w}" if bimanual else w
                           for arm, (rows, _, _) in zip(arms, snaps) for w in not_home(rows, targets[arm.side], arm)]
                if waiting:
                    message = "Not starting SLOW: " + "; ".join(waiting)
                else:
                    phase, message = SLOW, f"SLOW: moving to the leader at {args.slow_rate:g} deg/s."
            elif phase == SLOW and key == "":
                behind = [f"{arm.side} {b}" if bimanual else b
                          for arm, (rows, grip, _) in zip(arms, snaps)
                          for b in lagging(rows, targets[arm.side], grip, grip_targets[arm.side], arm)]
                if behind:
                    message = "Not LIVE yet, still catching up: " + ", ".join(behind)
                else:
                    phase, message = LIVE, "LIVE."
            elif phase == LIVE and key == "":
                phase, message = SLOW, f"Back to SLOW ({args.slow_rate:g} deg/s)."
            elif key:
                message = f"Ignored '{key}': press Enter (or q + Enter to quit)."

            if phase in (HOME, SLOW, LIVE):
                # Lost feedback on an active joint: hold the last target (still published, so
                # joint_command does not time out and drop the arm) and fall back to SLOW (HOME stays HOME).
                stale = [f for f in failures if "no feedback" in f]
                for arm, (rows, grip, _) in zip(arms, snaps):
                    target = targets[arm.side]
                    for i, row in enumerate(rows):
                        if row["joint"]["name"] not in arm.active:
                            target[i] = row["leader"]  # joint_command ignores it; any finite value
                        elif stale:
                            pass
                        elif phase == HOME:
                            target[i] = step_toward(target[i], home_deg[arm.side][i], args.home_rate * dt)
                        elif phase == LIVE:
                            target[i] = row["leader"]
                        else:
                            target[i] = step_toward(target[i], row["leader"], args.slow_rate * dt)
                    # HOME holds the gripper where it was measured.
                    if arm.grip_active and not stale and phase != HOME:
                        grip_targets[arm.side] = (grip if phase == LIVE else
                                                  step_toward(grip_targets[arm.side], grip, SLOW_GRIP_RATE * dt))
                    msg = arm_pose(ArmPose, arm.side, target, grip_targets[arm.side], arm.grip_active)
                    msg.header.stamp = node.get_clock().now().to_msg()
                    pubs[arm.side].publish(msg)
                if stale and phase == LIVE:
                    phase = SLOW
                if stale:
                    message = "HOLDING (feedback lost): " + "; ".join(stale)

            if now >= next_print:
                next_print = now + PRINT_PERIOD_S
                lines = []
                for arm, (rows, grip, real_grip) in zip(arms, snaps):
                    if bimanual:
                        lines.append(f"== {arm.side.upper()} ==")
                    lines += format_arm(rows, targets.get(arm.side), grip, grip_targets.get(arm.side), real_grip, arm)
                lines.append("")
                if phase == ZERO:
                    not_hanging = [f for arm, (rows, _, _) in zip(arms, snaps) for f in zero_failures(rows, arm)]
                    lines.append("ZERO (read-only): hang the leader and the real arm straight down, leader gripper "
                                 "open, then Enter. "
                                 + (f"Real arm hanging (within {ZERO_TOL_DEG:g} deg)." if not not_hanging
                                    else "Waiting: " + "; ".join(not_hanging)))
                elif phase == CHECK:
                    lines.append("CHECK (read-only, nothing published). "
                                 + ("Ready: " if not failures else "Waiting: " + "; ".join(failures) + ". ")
                                 + "Hand on the E-stop, then Enter: the arm moves to "
                                 + (f"home at {args.home_rate:g} deg/s." if start_home
                                    else f"the leader at {args.slow_rate:g} deg/s."))
                elif phase == HOME:
                    waiting = [w for arm, (rows, _, _) in zip(arms, snaps) for w in not_home(rows, targets[arm.side], arm)]
                    ramping = any(w.endswith("ramping") for w in waiting)
                    lines.append(f"HOME ({args.home_rate:g} deg/s): "
                                 + ("ramping the real arm to home." if ramping else
                                    "real arm at home. Put the leader at home by hand (upper arm down, forearm "
                                    "forward), then Enter for SLOW. ")
                                 + ("" if ramping or not waiting else "Waiting: " + "; ".join(waiting)))
                elif phase == SLOW:
                    behind = [b for arm, (rows, grip, _) in zip(arms, snaps)
                              for b in lagging(rows, targets[arm.side], grip, grip_targets[arm.side], arm)]
                    lines.append(f"SLOW ({args.slow_rate:g} deg/s): "
                                 + ("caught up. Test the joints, then Enter for LIVE." if not behind
                                    else "catching up: " + ", ".join(behind)))
                else:
                    lines.append("LIVE: full-speed teleop. Enter = back to SLOW.")
                worst = max((abs(row["real"] - t) for arm, (rows, _, _) in zip(arms, snaps) if arm.side in targets
                             for row, t in zip(rows, targets[arm.side])
                             if row["joint"]["name"] in arm.active and row["real"] is not None), default=0.0)
                if phase != CHECK and worst > AGREE_TOL_DEG:
                    lines.append(f"[WARN] real arm {worst:.1f} deg behind its target")
                lines.append("Ctrl-C stops (joint_command then times out and the arm sinks: support it).")
                if message:
                    lines.append(message)
                print(("\033[H\033[J" if redraw else "\n") + "\n".join(lines), flush=True)
            time.sleep(READ_PERIOD_S)
    except KeyboardInterrupt:
        pass
    finally:
        for arm in arms:
            if arm.leader is not None:
                arm.leader.close()
        node.destroy_node()
        rclpy.shutdown()
        if not pubs:
            print("\n[INFO] Stopped. Nothing was published.", flush=True)
        else:
            print("\n[INFO] Stopped publishing. joint_command holds the last pose for its "
                  "command_timeout_sec, then sets kp = 0 and each arm sinks: support it.", flush=True)


if __name__ == "__main__":
    run()
