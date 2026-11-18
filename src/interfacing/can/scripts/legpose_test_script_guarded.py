#!/usr/bin/env python3
"""Move ONE leg joint (or a few) through a short list of poses, then back, via joint_command.

Publishes LegPose on /leg/<side>/joint_targets exactly like a policy would, so joint_command's clamp,
velocity limit, MIT watchdog and seeding all apply. Nothing here talks to the motors directly.

Poses are OFFSETS in command-frame degrees from where the joint is when the script starts (read
from motor feedback), six numbers per pose in LegPose order::

    hip.pitch  hip.roll  hip.yaw  knee.pitch  ankle.pitch  ankle.roll

and poses are separated by ';'. The script goes origin -> pose 1 -> pose 2 -> ... -> origin.

By default it only PRINTS THE PLAN (and reads feedback) and publishes nothing. Add --live to move.

Examples (joint_command_node and can_node running, `legs` enabled, nothing else publishing)::

  # dry run: show the plan for the knee going +5 then +10 and back
  legpose_test_script.py --joints knee.pitch --poses "0,0,0,5,0,0; 0,0,0,10,0,0"
  # the same, for real
  legpose_test_script.py --joints knee.pitch --poses "0,0,0,5,0,0; 0,0,0,10,0,0" --live

Only the joints named in --joints may move: a pose with a nonzero offset for any other joint is
refused. Every other slot is held at its measured angle (or 0 if that motor reports nothing; joint_command
ignores inactive joints).

Safety, in order:
  * refuses unless joint_command is listening and nothing else publishes the topic
  * refuses if a --joints motor is silent, or its feedback is older than --feedback-timeout
  * refuses a target within --limit-margin of a limit, farther than --max-delta, or a speed over
    --vel (hard caps below)
  * holds the origin for --hold seconds first and aborts if the joint drifts more than --seed-tolerance
  * ramps on a cosine profile (peak speed <= --vel); joint_command rate-limits it again
  * a tracking error over --max-track-err, or Ctrl-C, ramps back to the origin; a second Ctrl-C stops
    streaming; lost feedback PAUSES the move: the last pose keeps being sent, and the move resumes from
    the same point if feedback returns (joint_command's own watchdog faults on stale feedback anyway)
  * when the script ends, joint_command stops getting poses: after its command_timeout_sec (10 s) it
    drops MIT joints to zero stiffness. Use --hold-at-end to keep holding until Ctrl-C.

HARDWARE E-STOP within reach for every run. Start with the joint supported or free on a bench, low
--vel, small offsets. Read src/interfacing/ARM_BRINGUP.md and the real-hardware-safety rules first.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, NoReturn, Optional

import yaml
import rclpy
from rclpy.node import Node

from common_msgs.msg import LegPose, MotorFeedback

# LegPose slot order, identical to common_msgs/msg/LegPose.msg and joint_command_core.cpp legJointPaths().
LEG_JOINTS = [("hip", "pitch"), ("hip", "roll"), ("hip", "yaw"), ("knee", "pitch"), ("ankle", "pitch"), ("ankle", "roll")]
LEG_NAMES = [f"{group}.{joint}" for group, joint in LEG_JOINTS]

# Hard caps: the flags below cannot be raised past these. Change them only with bench data.
HARD_MAX_VEL_DEG_S = 15.0
HARD_MAX_DELTA_DEG = 30.0

REPO_CALIBRATION = Path(__file__).resolve().parents[2] / "joint_command" / "config" / "leg_calibration.yaml"


class Stop(Exception):
    """Stop streaming now (second Ctrl-C)."""


class Abort(Exception):
    """Ramp back to the origin, then exit non-zero."""


def fail(message: str) -> NoReturn:
    print(f"REFUSED: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_poses(text: str) -> List[List[float]]:
    poses = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values = [float(v) for v in chunk.split(",")]
        except ValueError:
            fail(f"pose {chunk!r} is not six comma-separated numbers")
        if len(values) != len(LEG_NAMES):
            fail(f"pose {chunk!r} has {len(values)} numbers; need {len(LEG_NAMES)}: {', '.join(LEG_NAMES)}")
        if not all(math.isfinite(v) for v in values):
            fail(f"pose {chunk!r} has a non-finite number")
        poses.append(values)
    if not poses:
        fail("no poses given")
    return poses


def segment_time(max_delta_deg: float, vel_deg_s: float, floor_s: float) -> float:
    """Seconds for a cosine ramp whose PEAK speed (pi/2 x the average) stays at or below vel."""
    return max(floor_s, (math.pi / 2.0) * max_delta_deg / vel_deg_s)


def load_calibration(path: Path, side: str, selected: List[int]) -> List[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = data.get(side)
    if not block:
        fail(f"{path} has no '{side}' block")
    joints = []
    for slot, (group, joint) in enumerate(LEG_JOINTS):
        node = (block.get(group) or {}).get(joint)
        if not node:
            fail(f"{path}: no {side}.{group}.{joint}")
        info = {"name": LEG_NAMES[slot], "id": node.get("can_id"), "dir": node.get("direction"),
                "zero": node.get("zero_offset"), "lo": node.get("lower_limit"), "hi": node.get("upper_limit")}
        if slot in selected:
            for key in ("id", "dir", "zero", "lo", "hi"):
                if info[key] is None:
                    fail(f"{path}: {side}.{info['name']} has a null {key}. Calibrate it (calibrate_actuators.py --limb leg) "
                         f"or set it by hand before moving it.")
            if info["dir"] not in (1, -1):
                fail(f"{path}: {side}.{info['name']} direction must be 1 or -1, got {info['dir']}")
        joints.append(info)
    return joints


def installed_calibration() -> Optional[Path]:
    """The leg_calibration.yaml joint_command_node actually loaded (its installed copy), if we can find it."""
    try:
        from ament_index_python.packages import get_package_share_directory
        path = Path(get_package_share_directory("joint_command")) / "config" / "leg_calibration.yaml"
        return path if path.exists() else None
    except Exception:
        return None


class LegPoseNode(Node):
    def __init__(self, side: str, joints: List[dict], pose_topic: str, feedback_topic: str):
        super().__init__("legpose_test_script")
        self.side = side
        self.joints = joints
        self.pub = self.create_publisher(LegPose, pose_topic, 10)
        self.fb_deg: Dict[int, float] = {}   # motor id -> motor-frame degrees
        self.fb_time: Dict[int, float] = {}
        self.create_subscription(MotorFeedback, feedback_topic, self._on_feedback, 100)

    def _on_feedback(self, msg: MotorFeedback) -> None:
        self.fb_deg[int(msg.motor_id)] = float(msg.position)
        self.fb_time[int(msg.motor_id)] = time.monotonic()

    def age(self, slot: int) -> float:
        t = self.fb_time.get(self.joints[slot]["id"])
        return math.inf if t is None else time.monotonic() - t

    def measured(self, slot: int, timeout: float) -> Optional[float]:
        """Command-frame degrees from the latest feedback, or None if silent / stale."""
        info = self.joints[slot]
        if info["id"] is None or info["dir"] not in (1, -1) or info["zero"] is None or self.age(slot) > timeout:
            return None
        return info["zero"] + self.fb_deg[info["id"]] / info["dir"]

    def publish_pose(self, angles: List[float]) -> None:
        msg = LegPose()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = "legpose_test_script"
        msg.is_left = self.side == "left"
        msg.hip.position = [float(a) for a in angles[0:3]]
        msg.knee.position = [float(angles[3])]
        msg.ankle.position = [float(a) for a in angles[4:6]]
        self.pub.publish(msg)


def spin_for(node: Node, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.01)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--poses", "--offset", dest="poses", required=True,
                    help="';'-separated poses, each six OFFSET degrees in LegPose order: " + " ".join(LEG_NAMES))
    ap.add_argument("--joints", required=True, help="comma-separated joints allowed to move, e.g. knee.pitch or hip.pitch,knee.pitch")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--live", action="store_true", help="actually publish. Without it: print the plan and exit.")
    ap.add_argument("--vel", type=float, default=3.0, help=f"peak deg/s (default 3, hard cap {HARD_MAX_VEL_DEG_S})")
    ap.add_argument("--max-delta", type=float, default=10.0, help=f"largest offset per joint, deg (default 10, hard cap {HARD_MAX_DELTA_DEG})")
    ap.add_argument("--limit-margin", type=float, default=2.0, help="refuse a target this close to a limit, deg")
    ap.add_argument("--max-track-err", type=float, default=8.0, help="ramp back if measured differs from commanded by more, deg")
    ap.add_argument("--seed-tolerance", type=float, default=2.0, help="abort if the joint drifts this far from the origin while holding, deg")
    ap.add_argument("--feedback-timeout", type=float, default=0.5, help="feedback older than this counts as lost, s")
    ap.add_argument("--hold", type=float, default=2.0, help="s holding the origin before moving")
    ap.add_argument("--dwell", type=float, default=2.0, help="s at each pose")
    ap.add_argument("--rest", type=float, default=2.0, help="s at the origin after returning")
    ap.add_argument("--min-segment", type=float, default=1.0, help="shortest ramp, s")
    ap.add_argument("--rate", type=float, default=50.0, help="LegPose rate, Hz (joint_command's tick is 50)")
    ap.add_argument("--hold-at-end", action="store_true", help="keep publishing the origin after finishing, until Ctrl-C")
    ap.add_argument("--calibration", help="leg_calibration.yaml (default: the copy joint_command installed)")
    ap.add_argument("--allow-config-mismatch", action="store_true", help="use --calibration even if it differs from the installed copy")
    ap.add_argument("--pose-topic", help="default /leg/<side>/joint_targets (override only for a dry-run domain)")
    ap.add_argument("--feedback-topic", default="/interfacing/motorFeedback", help="override only for a dry-run domain")
    args = ap.parse_args(argv)

    # ---- argument checks (nothing is published yet)
    names = [n.strip() for n in args.joints.split(",") if n.strip()]
    unknown = [n for n in names if n not in LEG_NAMES]
    if unknown:
        fail(f"unknown joint(s) {unknown}; choose from {LEG_NAMES}")
    selected = sorted({LEG_NAMES.index(n) for n in names})
    if not selected:
        fail("--joints is empty")
    if not (0.0 < args.vel <= HARD_MAX_VEL_DEG_S):
        fail(f"--vel must be in (0, {HARD_MAX_VEL_DEG_S}] deg/s")
    if not (0.0 < args.max_delta <= HARD_MAX_DELTA_DEG):
        fail(f"--max-delta must be in (0, {HARD_MAX_DELTA_DEG}] deg")
    if args.rate <= 0 or args.hold < 0 or args.dwell < 0 or args.rest < 0 or args.min_segment <= 0:
        fail("--rate and --min-segment must be > 0; --hold, --dwell and --rest must be >= 0")
    offsets = parse_poses(args.poses)
    for pose in offsets:
        for slot, value in enumerate(pose):
            if slot not in selected and value != 0.0:
                fail(f"pose moves {LEG_NAMES[slot]} by {value} but it is not in --joints")
            if abs(value) > args.max_delta:
                fail(f"offset {value} deg on {LEG_NAMES[slot]} exceeds --max-delta {args.max_delta}")

    # ---- which calibration: the one joint_command enforces
    installed = installed_calibration()
    cal_path = Path(args.calibration) if args.calibration else (installed or REPO_CALIBRATION)
    if not cal_path.exists():
        fail(f"{cal_path} does not exist; pass --calibration")
    if installed and cal_path.resolve() != installed.resolve():
        same = yaml.safe_load(cal_path.read_text(encoding="utf-8")) == yaml.safe_load(installed.read_text(encoding="utf-8"))
        if not same and not args.allow_config_mismatch:
            fail(f"{cal_path} differs from the leg_calibration.yaml joint_command installed ({installed}). joint_command enforces the "
                 f"installed one: rebuild after editing, or pass --allow-config-mismatch.")
    joints = load_calibration(cal_path, args.side, selected)
    print(f"calibration: {cal_path}")

    rclpy.init()
    node = LegPoseNode(args.side, joints, args.pose_topic or f"/leg/{args.side}/joint_targets", args.feedback_topic)
    pose_topic = args.pose_topic or f"/leg/{args.side}/joint_targets"
    try:
        # ---- who is on the topic, and is joint_command listening
        spin_for(node, 1.5)
        others = node.count_publishers(pose_topic) - 1  # minus ourselves
        if others > 0:
            fail(f"{others} other publisher(s) on {pose_topic}; stop them first (two sources would fight)")
        if node.count_subscribers(pose_topic) < 1:
            fail(f"nothing is listening on {pose_topic}: is joint_command_node running with `legs` including '{args.side}'?")

        # ---- the origin, from live feedback
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and any(node.measured(s, args.feedback_timeout) is None for s in selected):
            rclpy.spin_once(node, timeout_sec=0.05)
        origin = [0.0] * len(LEG_NAMES)
        for slot, info in enumerate(joints):
            measured = node.measured(slot, args.feedback_timeout)
            if slot in selected:
                if measured is None:
                    fail(f"no fresh feedback from {info['name']} (motor {info['id']}) on {args.feedback_topic}: is it powered, is can_node "
                         f"running, does the can_id match the drive?")
                if not (info["lo"] <= measured <= info["hi"]):
                    fail(f"{info['name']} is at {measured:.1f} deg, outside its limits [{info['lo']}, {info['hi']}]: stale calibration?")
                origin[slot] = measured
            elif measured is not None and info["lo"] is not None and info["lo"] <= measured <= info["hi"]:
                origin[slot] = measured  # a joint we do not move holds where it is
            elif info["lo"] is not None and info["hi"] is not None:
                origin[slot] = min(max(0.0, info["lo"]), info["hi"])  # filler: joint_command ignores a silent / inactive joint

        waypoints = [[origin[s] + pose[s] for s in range(len(LEG_NAMES))] for pose in offsets]
        for pose in waypoints:
            for slot in selected:
                info = joints[slot]
                if not (info["lo"] + args.limit_margin <= pose[slot] <= info["hi"] - args.limit_margin):
                    fail(f"target {pose[slot]:.1f} deg on {info['name']} is within {args.limit_margin} deg of its limits "
                         f"[{info['lo']}, {info['hi']}]")

        # ---- the plan
        path = [origin] + waypoints + [origin]
        plan = []
        for a, b in zip(path, path[1:]):
            plan.append(segment_time(max(abs(b[s] - a[s]) for s in selected), args.vel, args.min_segment))
        print(f"\nside: {args.side}   moving: {', '.join(LEG_NAMES[s] for s in selected)}   vel <= {args.vel} deg/s   rate {args.rate} Hz")
        for s in selected:
            info = joints[s]
            print(f"  {info['name']:12s} motor {info['id']:>3}  origin {origin[s]:8.2f}  limits [{info['lo']}, {info['hi']}]")
        for k, (a, b, t) in enumerate(zip(path, path[1:], plan), 1):
            label = f"to pose {k}" if k <= len(waypoints) else "back to origin"
            print(f"  step {k}: {label:15s} " + "  ".join(f"{LEG_NAMES[s]} {a[s]:7.2f} -> {b[s]:7.2f}" for s in selected) + f"   ({t:.1f} s)")
        total = args.hold + sum(plan) + args.dwell * len(waypoints) + args.rest
        print(f"  total about {total:.0f} s")
        if not args.live:
            print("\nDRY RUN: nothing was published. Re-run with --live to move.")
            return 0

        return run(node, args, joints, selected, origin, path, plan)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run(node: LegPoseNode, args, joints: List[dict], selected: List[int], origin: List[float], path: List[List[float]], plan: List[float]) -> int:
    interrupts = {"n": 0}
    state = {"cmd": list(origin), "max_err": 0.0, "frozen": False, "returning": False, "hold_end": False, "t0": time.monotonic(), "last_print": 0.0}

    def on_sigint(_sig, _frame):
        interrupts["n"] += 1
        if state["hold_end"] or interrupts["n"] >= 2:
            print("\nCtrl-C: stopping", flush=True)
        else:
            print("\nCtrl-C: ramping back to the origin (Ctrl-C again to stop streaming)", flush=True)

    signal.signal(signal.SIGINT, on_sigint)
    dt = 1.0 / args.rate
    next_tick = {"t": time.monotonic()}

    def pace() -> None:
        next_tick["t"] += dt
        while True:
            remaining = next_tick["t"] - time.monotonic()
            if remaining <= 0:
                break
            rclpy.spin_once(node, timeout_sec=min(remaining, 0.005))
        if next_tick["t"] < time.monotonic() - 1.0:
            next_tick["t"] = time.monotonic()

    def show(angles: List[float], phase: str) -> None:
        now = time.monotonic()
        if now - state["last_print"] < 0.5:
            return
        state["last_print"] = now
        cols = "  ".join(
            f"{joints[s]['name']} cmd {angles[s]:7.2f} meas "
            + ("   lost" if node.measured(s, args.feedback_timeout) is None else f"{node.measured(s, args.feedback_timeout):7.2f}")
            for s in selected)
        print(f"  t={now - state['t0']:6.1f}s  {phase:14s} {cols}", flush=True)

    def send(angles: List[float], phase: str) -> bool:
        """Publish one tick. False = paused: feedback is lost, so the LAST pose was re-sent and the profile must not advance."""
        if interrupts["n"] >= 2:
            raise Stop()
        if interrupts["n"] == 1 and not state["returning"]:
            raise Abort("Ctrl-C")
        worst = 0.0
        lost = []
        for slot in selected:
            measured = node.measured(slot, args.feedback_timeout)
            if measured is None:
                lost.append(joints[slot]["name"])
            else:
                worst = max(worst, abs(measured - angles[slot]))
        if lost:
            if not state["frozen"]:
                print(f"\nFEEDBACK LOST from {', '.join(lost)}: holding the last pose and pausing (Ctrl-C twice to stop streaming)", flush=True)
            state["frozen"] = True
            node.publish_pose(state["cmd"])
            show(state["cmd"], phase)
            return False
        if state["frozen"]:
            print("feedback is back: resuming", flush=True)
        state["frozen"] = False
        state["cmd"] = list(angles)
        node.publish_pose(angles)
        state["max_err"] = max(state["max_err"], worst)
        if worst > args.max_track_err and not state["returning"]:
            raise Abort(f"tracking error {worst:.1f} deg > --max-track-err {args.max_track_err}")
        show(angles, phase)
        return True

    # The profile runs on its own clock that only advances on ticks that were really sent, so a feedback outage
    # pauses the move instead of letting it run on unseen.
    def hold(angles: List[float], seconds: float, phase: str) -> None:
        elapsed = 0.0
        while elapsed < seconds:
            if send(angles, phase):
                elapsed += dt
            pace()

    def ramp(a: List[float], b: List[float], seconds: float, phase: str) -> None:
        elapsed = 0.0
        while True:
            alpha = min(1.0, elapsed / seconds)
            s = 0.5 * (1.0 - math.cos(math.pi * alpha))
            if send([x + (y - x) * s for x, y in zip(a, b)], phase):
                if alpha >= 1.0:
                    return
                elapsed += dt
            pace()

    reason = None
    try:
        print("\nLIVE. Hold at the origin first.", flush=True)
        hold(origin, args.hold, "hold origin")
        for slot in selected:
            measured = node.measured(slot, args.feedback_timeout)
            if measured is not None and abs(measured - origin[slot]) > args.seed_tolerance:
                raise Abort(f"{joints[slot]['name']} drifted to {measured:.1f} deg while holding the origin {origin[slot]:.1f} "
                            f"(more than --seed-tolerance {args.seed_tolerance})")
        for k in range(len(path) - 1):
            last = k == len(path) - 2
            ramp(path[k], path[k + 1], plan[k], "back to origin" if last else f"to pose {k + 1}")
            if not last:
                hold(path[k + 1], args.dwell, f"dwell pose {k + 1}")
        hold(origin, args.rest, "rest")
    except Abort as exc:
        reason = str(exc)
        print(f"\nABORT: {reason}. Ramping back to the origin.", flush=True)
        state["returning"] = True
        try:
            span = max(abs(state["cmd"][s] - origin[s]) for s in selected)
            ramp(list(state["cmd"]), origin, segment_time(span, args.vel, args.min_segment), "return")
            hold(origin, args.rest, "rest")
        except Stop:
            print("stopped.", flush=True)
    except Stop:
        print("stopped streaming.", flush=True)
        reason = "stopped"

    print("\n=== summary ===")
    print(f"largest tracking error while streaming: {state['max_err']:.2f} deg")
    for slot in selected:
        measured = node.measured(slot, args.feedback_timeout)
        end = "no feedback" if measured is None else f"{measured:.2f} deg ({measured - origin[slot]:+.2f} from the origin)"
        print(f"  {joints[slot]['name']:12s} origin {origin[slot]:.2f}  now {end}")
    if args.hold_at_end and reason != "stopped":
        print("\n--hold-at-end: still publishing the origin. Ctrl-C to stop.", flush=True)
        interrupts["n"] = 0
        state["hold_end"] = True
        state["returning"] = False  # so the first Ctrl-C ends the hold
        try:
            while True:
                send(origin, "hold at end")
                pace()
        except (Stop, Abort):
            pass  # the first Ctrl-C ends the hold: we are already at the origin
    else:
        print("\nPublishing has stopped: joint_command goes to zero stiffness after its command_timeout_sec.")
    return 1 if reason else 0


if __name__ == "__main__":
    sys.exit(main())
