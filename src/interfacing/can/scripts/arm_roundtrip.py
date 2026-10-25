#!/usr/bin/env python3
"""Angle benchmark for the real arm: ramp out to a pose, dwell, ramp back, settle.

Publishes ArmPose on /arm/joint_targets exactly like teleop, so joint_command's clamp, velocity
limits and MIT watchdog all apply. Nothing here bypasses them.

Run (joint_command container; joint_command_node and can_node running; nothing else
publishing /arm/joint_targets)::

  python3 /opt/humanoid_scripts/arm_roundtrip.py --joints elbow.roll --offset "0,0,0,0,5,0"
  python3 /opt/humanoid_scripts/arm_roundtrip.py --pose "0,0,10,20,0,0" --max-delta 25
  tools/arm_roundtrip.sh ...        # the same from the host, plus plots

Sequence (command-frame degrees):
  1. read the origin from feedback; refuse if a --joints joint is silent or any joint is outside
     its limits (silent joints not in --joints are allowed: joint_command excludes them)
  2. target = origin + --offset (or --pose); refuse within --limit-margin of a limit or beyond
     --max-delta
  3. hold    publish the origin; abort unless joint_command's setpoint matches the measured pose
  4. ramp    cosine profile, peak <= --vel and each joint's velocity_max
  5. dwell   at the target
  6. return  to the origin on the same profile
  7. rest    and report the per-joint error

--exercise-limits sends the request as given (past limits, faster than velocity_max) over
--duration; the summary checks joint_command clamped both::

  arm_roundtrip.py --exercise-limits --joints elbow.roll --pose "0,0,0,0,200,0" --duration 2

Stopping: Ctrl-C or tracking error > --max-track-err ramps back to the origin; a second Ctrl-C
stops streaming; lost feedback freezes in place. Then joint_command's stale-stream handling
applies. The node's installed configs must match the repo's. Recalibrate the AK80-9s after every
power-on (src/interfacing/README.md).

HARDWARE E-STOP within reach for every run.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from typing import Dict, List, Optional

import rclpy

from common_msgs.msg import ArmPose, MotorFeedback

import joint_config as jc
from telemetry import RunFolder
from telemetry_record import TelemetryRecorder

N = len(jc.ARM_POSE_JOINTS)


class RoundTripNode(TelemetryRecorder):
    """The telemetry recorder, plus the ArmPose publisher and feedback timestamps."""

    def __init__(self, joint_map, log, rate_hz):
        super().__init__(joint_map, log, None, rate_hz)
        self.fb_time: Dict[int, float] = {}
        self.pub = self.create_publisher(ArmPose, "/arm/joint_targets", 10)

    def _on_feedback(self, msg: MotorFeedback) -> None:
        super()._on_feedback(msg)
        self.fb_time[int(msg.motor_id)] = time.monotonic()

    def publish_pose(self, pose: List[float]) -> None:
        msg = ArmPose()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = "arm_roundtrip"
        msg.is_left = True
        msg.shoulder.position = [float(v) for v in pose[0:3]]
        msg.elbow.position = [float(v) for v in pose[3:5]]
        msg.wrist.position = [float(v) for v in pose[5:6]]
        msg.include_hand_pose = False
        self.pub.publish(msg)

    def measured(self, motor_id: int) -> Optional[float]:
        fb = self.feedback.get(motor_id)
        if fb is None:
            return None
        return jc.motor_to_cmd_deg(self.joint_map[motor_id], float(fb.position))


class Stop(Exception):
    pass


def parse_six(text: str, flag: str) -> List[float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != N:
        raise SystemExit(f"{flag} wants {N} comma-separated degrees "
                         f"({','.join(jc.ARM_POSE_NAMES)}), got {text!r}")
    return [float(p) for p in parts]


def cosine(alpha: float) -> float:
    """0 -> 1 with zero slope at both ends; peak slope pi/2 (vs 1 for a linear ramp)."""
    return 0.5 - 0.5 * math.cos(math.pi * min(max(alpha, 0.0), 1.0))


def profile_time(a: List[float], b: List[float], vel: List[float], floor: float) -> float:
    """Shortest cosine-profile duration keeping every joint's PEAK speed <= its vel."""
    need = [math.pi / 2 * abs(y - x) / v for x, y, v in zip(a, b, vel) if v > 0]
    return max([floor] + need)


def resolve_config(args):
    """The configs joint_command_node enforces (its INSTALLED copy); refuse if the repo differs.

    Explicit --calibration / --actuators skip the check.
    """
    installed = jc.installed_joint_command_config()
    live_mapping = jc.find_calibration(args.mapping)
    live_safety = jc.find_actuators(args.actuators)
    if live_safety is None:
        sys.exit("could not find arm_actuators.yaml; pass --actuators PATH")
    if installed is None or (args.mapping and args.actuators):
        return live_mapping, live_safety
    stale = []
    mapping, safety = live_mapping, live_safety
    if not args.mapping:
        mapping = installed / "arm_calibration.yaml"
        if not jc.same_yaml(mapping, live_mapping):
            stale.append(f"{mapping} != {live_mapping}")
    if not args.actuators:
        safety = installed / "arm_actuators.yaml"
        if not jc.same_yaml(safety, live_safety):
            stale.append(f"{safety} != {live_safety}")
    if stale:
        sys.exit("joint_command_node is enforcing an older config than the repo:\n  "
                 + "\n  ".join(stale) + "\nRebuild it so it picks up the current calibration "
                 "and limits (host: ./watod build joint_command && ./watod up -d), relaunch "
                 "joint_command, then rerun.")
    return mapping, safety


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--offset", help="six degrees RELATIVE to the current pose "
                                        "(shoulder p,r,y, elbow p,r, wrist p)")
    where.add_argument("--pose", help="six ABSOLUTE command-frame degrees")
    ap.add_argument("--joints", help="comma-separated joints to move, e.g. elbow.roll "
                                     "(default: every joint with a non-zero move)")
    ap.add_argument("--vel", type=float, default=5.0,
                    help="peak deg/s of the profile (default 5; capped per joint at its "
                         "velocity_max)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds for each ramp: a minimum by default (stretched to respect "
                         "--vel); exact with --exercise-limits (required there)")
    ap.add_argument("--hold", type=float, default=2.0, help="s at the origin before moving")
    ap.add_argument("--dwell", type=float, default=3.0, help="s at the target")
    ap.add_argument("--rest", type=float, default=3.0, help="s at the origin after returning")
    ap.add_argument("--max-delta", type=float, default=30.0,
                    help="refuse any joint moving further than this, deg (default 30)")
    ap.add_argument("--limit-margin", type=float, default=2.0,
                    help="refuse targets closer than this to a joint limit, deg (default 2)")
    ap.add_argument("--max-track-err", type=float, default=8.0,
                    help="measured vs joint_command setpoint, deg: above this the arm turns "
                         "around and returns (default 8)")
    ap.add_argument("--seed-tolerance", type=float, default=2.0,
                    help="max |setpoint - measured| before moving, deg (default 2)")
    ap.add_argument("--feedback-timeout", type=float, default=0.5,
                    help="s without feedback from a motor before freezing (default 0.5)")
    ap.add_argument("--exercise-limits", action="store_true",
                    help="send the request exactly as given -- past a joint's limits and faster "
                         "than velocity_max if so -- over exactly --duration, and let "
                         "joint_command clamp it. Each phase then waits until joint_command's "
                         "setpoint has settled at the CLAMPED target. --max-delta applies to the "
                         "clamped travel; --vel and --limit-margin are ignored")
    ap.add_argument("--hold-at-end", action="store_true",
                    help="keep streaming the origin after the run until Ctrl-C")
    ap.add_argument("--rate", type=float, default=50.0, help="ArmPose / sampling rate, Hz")
    ap.add_argument("--label", default="", help="run folder name (default: derived)")
    ap.add_argument("--calibration", "--mapping", dest="mapping",
                    help="arm_calibration.yaml (default: joint_command's)")
    ap.add_argument("--actuators", "--safety-limits", dest="actuators",
                    help="arm_actuators.yaml (default: joint_command's)")
    ap.add_argument("--arm-side", default="left")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)

    mapping, safety_path = resolve_config(args)
    joint_map = jc.load_joint_map(mapping, args.arm_side)
    slot_motor: List[int] = []
    for slot, name in enumerate(jc.ARM_POSE_NAMES):
        mid = next((m for m, info in joint_map.items() if info["slot"] == slot), None)
        if mid is None:
            sys.exit(f"{name} has no can_id in {mapping}")
        slot_motor.append(mid)
    infos = [joint_map[m] for m in slot_motor]

    blocks = [jc.joint_safety(safety_path, name) for name in jc.ARM_POSE_NAMES]
    vel = []
    for name, blk in zip(jc.ARM_POSE_NAMES, blocks):
        vmax = float(blk.get("velocity_max") or 0.0)
        if vmax <= 0.0:
            sys.exit(f"{name} has no velocity_max in {safety_path}; refusing to guess a speed")
        vel.append(min(args.vel, vmax))
    if args.vel <= 0:
        sys.exit("--vel must be > 0")
    if args.exercise_limits and args.duration <= 0:
        sys.exit("--exercise-limits needs --duration S: how long the request takes to go out")
    vmax_joint = [float(blk.get("velocity_max")) for blk in blocks]

    rel = parse_six(args.offset, "--offset") if args.offset else None
    absolute = parse_six(args.pose, "--pose") if args.pose else None
    moving = set(jc.ARM_POSE_NAMES)
    if args.joints:
        moving = {j.strip() for j in args.joints.split(",") if j.strip()}
        unknown = moving - set(jc.ARM_POSE_NAMES)
        if unknown:
            sys.exit(f"unknown joint(s) {sorted(unknown)}; choose from {jc.ARM_POSE_NAMES}")

    label = args.label or "rt-" + "-".join(
        f"{n.replace('.', '')[:6]}{v:+g}".replace("+", "p").replace("-", "m")
        for n, v in zip(jc.ARM_POSE_NAMES, rel or absolute) if n in moving and v)

    velocity_max, mit_limits = jc.load_actuator_limits(str(safety_path))
    log = RunFolder(
        label=label or "rt", source="ros", enabled=not args.no_log,
        meta={
            "tool": "arm_roundtrip.py", "arm_side": args.arm_side, "mapping": str(mapping),
            "actuators": str(safety_path), "rate_hz": args.rate,
            "frame": "command frame (degrees) for both sp_deg and pos_deg",
            "joints": {str(mid): info["name"] for mid, info in sorted(joint_map.items())},
            "moving": sorted(moving), "vel_dps": args.vel,
            "request": {"offset": rel, "pose": absolute},
            "limits": {
                "per_joint_deg": {info["name"]: [info["lower"], info["upper"]]
                                  for info in joint_map.values() if info["limit_range"]},
                "velocity_max_dps_per_joint": velocity_max,
                "velocity_max_dps": max(velocity_max.values()) if velocity_max else None,
                "mit": mit_limits,
                "max_torque_nm": min((v["max_torque_nm"] for v in mit_limits.values()
                                      if v.get("max_torque_nm") is not None), default=None),
                "max_torque_nm_by_joint": jc.max_torque_by_joint(str(safety_path)),
                "max_track_err_deg": args.max_track_err,
            },
        },
    )

    stops = {"n": 0}

    def on_signal(signum, _frame):
        stops["n"] += 1
        if stops["n"] == 1:
            print("\n[stop] returning to the origin (Ctrl-C again to stop streaming now)",
                  flush=True)
        else:
            print("\n[stop] stopping the stream now", flush=True)

    rclpy.init()
    node = RoundTripNode(joint_map, log, args.rate)
    period = 1.0 / args.rate
    outcome, reason = "completed", None
    try:
        # --- preflight: who else is on the topic, is joint_command there ------------------
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end and not all(m in node.feedback for m in slot_motor):
            rclpy.spin_once(node, timeout_sec=0.05)
        others = [i for i in node.get_publishers_info_by_topic("/arm/joint_targets")
                  if i.node_name != node.get_name()]
        if others:
            raise SystemExit("something else is publishing /arm/joint_targets ("
                             + ", ".join(sorted({i.node_name for i in others}))
                             + "); stop it (teleop, ros2 topic pub) first")
        consumers = [i for i in node.get_subscriptions_info_by_topic("/arm/joint_targets")
                     if i.node_name != node.get_name()]
        if not consumers:
            raise SystemExit("nobody subscribes to /arm/joint_targets -- start joint_command: "
                             "ros2 launch joint_command joint_command.launch.py")
        silent = {i for i, m in enumerate(slot_motor) if m not in node.feedback}
        silent_moving = [f"{infos[i]['name']} (motor {slot_motor[i]})" for i in sorted(silent)
                         if infos[i]["name"] in moving]
        if silent_moving:
            raise SystemExit("no feedback from " + ", ".join(silent_moving) + ": a joint this "
                             "run moves must report -- this tool cannot hold an origin it "
                             "cannot read.")
        for i in sorted(silent):
            print(f"WARNING: {infos[i]['name']} (motor {slot_motor[i]}) is not reporting -- "
                  f"joint_command excludes it; it is not moved.")
        powered = [m for i, m in enumerate(slot_motor) if i not in silent]

        # --- origin and target -------------------------------------------------------------
        # A silent slot's value is a placeholder: joint_command excludes that joint.
        origin = [0.0 if i in silent else node.measured(m) for i, m in enumerate(slot_motor)]
        target = list(origin)    # what is requested
        expected = list(origin)  # where joint_command should put it (target, clamped)
        refused, warned = [], []
        for i, name in enumerate(jc.ARM_POSE_NAMES):
            if i in silent:
                continue
            info = infos[i]
            lo, hi = ((info["lower"], info["upper"]) if info["limit_range"]
                      else (-math.inf, math.inf))
            if not (lo <= origin[i] <= hi):
                # Excluded by joint_command; only matters if this run moves it.
                (refused if name in moving and (rel is None or rel[i]) else warned).append(
                    f"{name} is at {origin[i]:+.1f}, outside its limits [{lo:g}, {hi:g}]: "
                    f"joint_command excludes it -- recalibrate it (calibrate_arm.py)")
                continue
            if name not in moving:
                continue
            target[i] = origin[i] + rel[i] if rel else absolute[i]
            expected[i] = min(max(target[i], lo), hi) if args.exercise_limits else target[i]
            if abs(expected[i] - origin[i]) > args.max_delta:
                refused.append(f"{name}: a {expected[i] - origin[i]:+.1f} deg move exceeds "
                               f"--max-delta {args.max_delta:g}")
            if args.exercise_limits:
                if expected[i] != target[i]:
                    warned.append(f"{name}: request {target[i]:+.1f} is past its limits "
                                  f"[{lo:g}, {hi:g}] -- joint_command should clamp it to "
                                  f"{expected[i]:+.1f}")
            elif target[i] != origin[i] and not (lo + args.limit_margin <= target[i]
                                                 <= hi - args.limit_margin):
                refused.append(f"{name}: target {target[i]:+.1f} is within "
                               f"{args.limit_margin:g} deg of (or past) its limits "
                               f"[{lo:g}, {hi:g}] (--exercise-limits to test the clamp)")
        for msg in warned:
            print("WARNING: " + msg)
        for msg in refused:
            print("REFUSED: " + msg)
        if refused:
            raise SystemExit("nothing was moved")
        if all(abs(t - o) < 1e-9 for t, o in zip(target, origin)):
            raise SystemExit("the requested move is zero for every selected joint")

        if args.exercise_limits:
            t_ramp = args.duration
        else:
            t_ramp = profile_time(origin, target, vel, max(args.duration, 0.5))
        # Exercise mode: time to arrive at velocity_max, plus low-pass slack.
        settle_timeout = t_ramp + 3.0 + 1.5 * max(
            abs(e - o) / v for e, o, v in zip(expected, origin, vmax_joint))
        print(f"mapping: {mapping}\nsafety:  {safety_path}\n{log.describe()}")
        if args.exercise_limits:
            print(f"{'joint':<16}{'origin':>9}{'request':>9}{'expect':>9}{'req peak':>10}"
                  f"{'vel max':>9}")
            for i, name in enumerate(jc.ARM_POSE_NAMES):
                if i in silent:
                    print(f"{name:<16}{'unpowered':>9}")
                    continue
                peak = math.pi / 2 * abs(target[i] - origin[i]) / t_ramp
                print(f"{name:<16}{origin[i]:>+9.2f}{target[i]:>+9.2f}{expected[i]:>+9.2f}"
                      f"{peak:>10.1f}{vmax_joint[i]:>9.1f}")
            print(f"request ramps over {t_ramp:g} s each way (deg/s above: the request's peak; "
                  f"joint_command caps the motion at vel max), dwell {args.dwell:g} s\n")
        else:
            print(f"{'joint':<16}{'origin':>9}{'target':>9}{'delta':>8}{'vel cap':>9}")
            for i, name in enumerate(jc.ARM_POSE_NAMES):
                if i in silent:
                    print(f"{name:<16}{'unpowered':>9}")
                    continue
                print(f"{name:<16}{origin[i]:>+9.2f}{target[i]:>+9.2f}"
                      f"{target[i] - origin[i]:>+8.2f}{vel[i]:>9.1f}")
            print(f"ramp {t_ramp:.1f} s each way, dwell {args.dwell:g} s\n")
        print("  HARDWARE E-STOP: physical cutoff on the motor supply within reach.\n")
        log.note(origin_deg=[round(v, 3) for v in origin],
                 target_deg=[round(v, 3) for v in target],
                 expected_deg=[round(v, 3) for v in expected], ramp_s=round(t_ramp, 3),
                 exercise_limits=args.exercise_limits, duration_s=args.duration)

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        # --- the run: one tick = publish + check --------------------------------------------
        state = {"phase": "hold", "t0": time.monotonic(), "from": list(origin),
                 "to": list(origin), "dur": args.hold, "pose": list(origin),
                 "seed_checked": False}

        def enter(phase: str, to: List[float], dur: float) -> None:
            state.update(phase=phase, t0=time.monotonic(), dur=dur, to=list(to),
                         **{"from": list(state["pose"])})
            node.phase = phase
            print(f"[{phase}] {dur:.1f} s", flush=True)

        def start_return(why: str) -> None:
            nonlocal outcome, reason
            if outcome == "completed":
                outcome, reason = "stopped", why
                log.note(outcome=outcome, abort_reason=why)
            print(f"returning to origin: {why}", flush=True)
            if args.exercise_limits:
                # Return from where joint_command has the arm, not the out-of-range request.
                state["pose"] = [node.setpoints.get(m, state["pose"][i])
                                 for i, m in enumerate(slot_motor)]
            enter("return", origin, profile_time(state["pose"], origin, vel, 0.5))

        def check_followed() -> None:
            """Did each moving joint cover at least half its move? Catches a drive that never
            enabled, which a small move would not trip the tracking limit for."""
            nonlocal outcome, reason
            reached, lagging = {}, []
            for i, m in enumerate(slot_motor):
                delta = expected[i] - origin[i]
                pos = node.measured(m)
                if abs(delta) < 0.5 or pos is None:
                    continue
                reached[infos[i]["name"]] = round(pos - expected[i], 3)
                if (pos - origin[i]) / delta < 0.5:
                    lagging.append(f"{infos[i]['name']} moved {pos - origin[i]:+.1f} of "
                                   f"{delta:+.1f} deg")
            log.note(reach_err_deg=reached)
            print("  at target, error (deg): " +
                  ", ".join(f"{k} {v:+.2f}" for k, v in reached.items()), flush=True)
            if lagging and outcome == "completed":
                outcome = "incomplete"
                reason = "did not follow: " + "; ".join(lagging)
                log.note(outcome=outcome, abort_reason=reason)
                print(f"  WARNING {reason} -- is the drive enabled (MIT_ENTER reached it? "
                      f"joint_command log), powered, and not excluded?", flush=True)

        node.phase = "hold"
        print("[hold] seeding joint_command at the measured pose", flush=True)
        t_next = time.monotonic()
        frozen = False
        while True:
            now = time.monotonic()
            # joint_command is commanding a silent slot (old build / powered on mid-run): stop.
            commanded_silent = [f"{infos[i]['name']} (motor {slot_motor[i]})"
                                for i in sorted(silent) if slot_motor[i] in node.setpoints]
            if commanded_silent:
                outcome = "aborted"
                reason = ("joint_command is commanding a joint that was silent at preflight: "
                          + ", ".join(commanded_silent) + " -- was it powered on mid-run, or "
                          "does joint_command predate the unpowered-joint exclusion (rebuild)?")
                log.note(outcome=outcome, abort_reason=reason)
                print(f"\nSTOPPED: {reason}", flush=True)
                raise Stop()
            # Feedback watchdog: never move blind.
            stale = [infos[i]["name"] for i, m in enumerate(slot_motor)
                     if m in powered and now - node.fb_time.get(m, 0.0) > args.feedback_timeout]
            if stale and not frozen:
                frozen = True
                outcome, reason = "aborted", f"feedback lost: {', '.join(stale)}"
                log.note(outcome=outcome, abort_reason=reason)
                node.phase = "frozen"
                print(f"\nFEEDBACK LOST from {', '.join(stale)}: holding the current pose. "
                      f"Support the arm, then Ctrl-C to stop streaming.", flush=True)
            if frozen:
                if stops["n"] >= 1:
                    raise Stop()
                node.publish_pose(state["pose"])
            else:
                phase = state["phase"]
                if stops["n"] >= 2 or (stops["n"] >= 1 and phase in ("done",)):
                    raise Stop()
                if stops["n"] >= 1 and phase in ("hold", "ramp", "dwell"):
                    start_return("stop requested")
                    phase = state["phase"]

                alpha = (now - state["t0"]) / state["dur"] if state["dur"] > 0 else 1.0
                s = cosine(alpha)
                state["pose"] = [a + (b - a) * s for a, b in zip(state["from"], state["to"])]
                node.publish_pose(state["pose"])

                # Tracking: measured vs what joint_command actually commanded.
                if phase in ("ramp", "dwell", "return"):
                    for i, m in enumerate(slot_motor):
                        sp, pos = node.setpoints.get(m), node.measured(m)
                        if sp is None or pos is None:
                            continue
                        err = abs(pos - sp)
                        if phase != "return" and err > args.max_track_err:
                            start_return(f"{infos[i]['name']} tracking error {err:.1f} deg > "
                                         f"{args.max_track_err:g}")
                            break
                        if phase == "return" and err > 1.5 * args.max_track_err:
                            frozen = True
                            outcome = "aborted"
                            reason = (f"{infos[i]['name']} tracking error {err:.1f} deg on the "
                                      f"way back")
                            log.note(outcome=outcome, abort_reason=reason)
                            node.phase = "frozen"
                            print(f"\n{reason}: holding here. Support the arm, then Ctrl-C.",
                                  flush=True)
                            break

                # Exercise mode: wait for joint_command's setpoint, not the request.
                arrived = True
                if args.exercise_limits and phase in ("ramp", "return") and \
                        now - state["t0"] < settle_timeout:
                    goal = expected if phase == "ramp" else origin
                    arrived = all(node.setpoints.get(m) is None or
                                  abs(node.setpoints[m] - goal[i]) <= 0.5
                                  for i, m in enumerate(slot_motor))
                if alpha >= 1.0 and state["phase"] == phase and arrived:
                    if phase == "hold":
                        if not state["seed_checked"]:
                            bad = []
                            for i, m in enumerate(slot_motor):
                                sp, pos = node.setpoints.get(m), node.measured(m)
                                if i in silent:
                                    continue
                                if sp is None:
                                    if infos[i]["name"] in moving:
                                        bad.append(f"{infos[i]['name']}: joint_command sent no "
                                                   f"command (excluded? see its log)")
                                    continue
                                if abs(sp - pos) > args.seed_tolerance:
                                    bad.append(f"{infos[i]['name']}: commanded {sp:+.2f} vs "
                                               f"measured {pos:+.2f} deg")
                            if bad:
                                outcome, reason = "aborted", "seed check: " + "; ".join(bad)
                                log.note(outcome=outcome, abort_reason=reason)
                                print("REFUSED before any motion -- joint_command is not "
                                      "commanding the measured pose:\n  " + "\n  ".join(bad),
                                      flush=True)
                                raise Stop()
                            state["seed_checked"] = True
                            print("  setpoints match the measured pose", flush=True)
                        enter("ramp", target, t_ramp)
                    elif phase == "ramp":
                        enter("dwell", target, args.dwell)
                    elif phase == "dwell":
                        check_followed()
                        enter("return", origin, t_ramp)
                    elif phase == "return":
                        enter("rest", origin, args.rest)
                    elif phase == "rest":
                        if not args.hold_at_end:
                            break
                        enter("done", origin, 0.0)
                        print("back at the origin; holding. Ctrl-C to stop streaming.",
                              flush=True)

            t_next += period
            while True:
                remaining = t_next - time.monotonic()
                if remaining <= 0:
                    break
                rclpy.spin_once(node, timeout_sec=remaining)
    except Stop:
        pass
    finally:
        node.phase = "end"
        final = {}
        for i, m in enumerate(slot_motor):
            pos = node.measured(m)
            if pos is not None and "origin" in locals() and i not in silent:
                final[infos[i]["name"]] = round(pos - origin[i], 3)
        if final:
            log.note(final_err_from_origin_deg=final)
            print("\nfinal error from origin (deg): " +
                  ", ".join(f"{k} {v:+.2f}" for k, v in final.items()))
        run_dir = log.close(outcome, **({"abort_reason": reason} if reason else {}))
        node.destroy_node()
        rclpy.shutdown()
        if outcome != "completed":
            print(f"outcome: {outcome} -- {reason}")
        if run_dir is not None:
            print(f"Telemetry: {run_dir}")
    return 0 if outcome == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
