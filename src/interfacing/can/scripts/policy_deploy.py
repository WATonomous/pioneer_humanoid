#!/usr/bin/env python3
"""Run the leg-only walking policy live on one leg's knee / ankle joints.

Every 20 ms: measured angles -> 50-number observation (the --joints joints real, everything else
constants) -> numpy MLP -> target = default + 0.25 * action -> safety box around the default
pose -> LegPose on /leg/<side>/joint_targets. joint_command's clamp, velocity limit and MIT
watchdog all apply, as for roundtrip.py; nothing here bypasses them.
Guide: explainers/RIGHT_LEG_POLICY_LIVE.md.

Run (joint_command container, ROS sourced; joint_command with this side in `legs` and exactly
the --joints joints `active: true` in leg_actuators.yaml)::

  policy_deploy.py --weights /outputs/policy/leg_only_flat_2999.npz --self-test   # no ROS
  policy_deploy.py --weights ... --side right --joints ankle.pitch --vx 0 --box 5 --check
  policy_deploy.py --weights ... --side right --joints ankle.pitch --vx 0 --box 5 \
      --seconds 20 --label s1a

Dry run: joint_command on motor_cmd_topic /dry_run/motorCMD, and --motor-cmd-topic
/dry_run/motorCMD here.

Phases: seed (2 s at the measured pose; joint_command's setpoint must match it) -> approach
(cosine, <= 10 deg/s, to the policy's default pose) -> hold 2 s -> policy (--seconds, eased in
and out over --ease s) -> hold 2 s -> retreat to the origin.

Logged in one run folder, same t_s clock: telemetry.csv (sp_raw_deg = sent, sp_deg =
joint_command's setpoint, pos_deg = measured) and policy.csv (obs, action, wanted_<URDF>,
sent_<URDF>).

Stopping: Ctrl-C or tracking error > --max-track-err ramps to the default pose, then to the
origin; a second Ctrl-C stops streaming. Feedback older than --feedback-timeout freezes at the
last sent pose (support the leg, then Ctrl-C).

HARDWARE E-STOP on the 48 V leg supply within reach for every run.
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time

import numpy as np

# Training joint order: observation slots 12-23, 24-35, 36-47 and the 12 actions.
JOINTS = ["Hip_F_L", "Hip_F_R", "Hip_A_L", "Hip_A_R", "Hip_R_L", "Hip_R_R",
          "Knee_L", "Knee_R", "Ankle_P_L", "Ankle_P_R", "Ankle_R_L", "Ankle_R_R"]
DEFAULT_RAD = np.array([{"Hip_F": 0.25, "Knee": -0.5, "Ankle_P": -0.25}.get(j[:-2], 0.0)
                        for j in JOINTS])
ACTION_SCALE = 0.25
GAIT_PERIOD = 1.0  # s
RIGHT_IDX = {"knee.pitch": 7, "ankle.pitch": 9, "ankle.roll": 11}  # left leg: one less
OBS_NAMES = ([f"base_lin_vel_{a}" for a in "xyz"] + [f"base_ang_vel_{a}" for a in "xyz"]
             + [f"gravity_{a}" for a in "xyz"] + ["cmd_vx", "cmd_vy", "cmd_wz"]
             + [f"q_rel_{j}" for j in JOINTS] + [f"qd_{j}" for j in JOINTS]
             + [f"last_action_{j}" for j in JOINTS] + ["clock_sin", "clock_cos"])
APPROACH_VEL = 10.0  # deg/s, peak of the approach and retreat ramps
HOLD_S = 2.0


def load_policy(path: str):
    """-> (mlp(obs) -> action, ref_obs, ref_act) from the A1 export."""
    data = np.load(path)
    weights = [(data[f"w{i}"], data[f"b{i}"]) for i in range(int(data["n"]))]

    def mlp(x):
        for i, (w, b) in enumerate(weights):
            x = w @ x + b
            if i < len(weights) - 1:
                x = np.where(x > 0, x, np.expm1(x))  # ELU
        return x

    return mlp, data["ref_obs"], data["ref_act"]


def build_obs(q_rel, qd, last_action, t, cmd):
    """q_rel, qd: 12 each in training order, 0 for a joint without a motor."""
    phase = (t % GAIT_PERIOD) / GAIT_PERIOD
    return np.concatenate([np.zeros(6), [0.0, 0.0, -1.0], cmd, q_rel, qd, last_action,
                           [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]])


def self_test(weights: str) -> int:
    mlp, ref_obs, ref_act = load_policy(weights)
    ok = True
    for k, vx in enumerate((0.0, 0.3)):
        obs = build_obs(np.zeros(12), np.zeros(12), np.zeros(12), 0.0, np.array([vx, 0.0, 0.0]))
        obs_err = float(np.abs(obs - ref_obs[k]).max())
        act_err = float(np.abs(mlp(obs) - ref_act[k]).max())
        good = obs_err <= 1e-6 and act_err <= 1e-6
        ok &= good
        print(f"vx {vx}: max |obs - ref_obs| = {obs_err:.2e}, max |action - ref_act| = "
              f"{act_err:.2e}  {'OK' if good else 'MISMATCH'}")
    print("\nobservation at vx 0.3 (the second reference):")
    for i, (name, v) in enumerate(zip(OBS_NAMES, obs)):
        print(f"  {i:2d} {name:<22}{v:+.3f}")
    print("\nself-test " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--weights", required=True, help="the .npz from the guide's step A1")
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--joints", help="comma-separated, from knee.pitch, ankle.pitch, ankle.roll: "
                                     "fed into the policy AND driven")
    ap.add_argument("--vx", type=float, default=0.0, help="command, m/s (default 0)")
    ap.add_argument("--vy", type=float, default=0.0, help="command, m/s (default 0)")
    ap.add_argument("--wz", type=float, default=0.0, help="command, rad/s (default 0)")
    ap.add_argument("--box", type=float, default=5.0,
                    help="deg: what is sent stays within default +- box (default 5)")
    ap.add_argument("--seconds", type=float, default=20.0, help="policy run length, s (default 20)")
    ap.add_argument("--ease", type=float, default=3.0,
                    help="s to ease the policy in at the start and out at the end (default 3)")
    ap.add_argument("--max-track-err", type=float, default=8.0,
                    help="measured vs joint_command setpoint, deg: above this the run stops "
                         "and the leg returns (default 8)")
    ap.add_argument("--seed-tolerance", type=float, default=2.0,
                    help="max |setpoint - measured| before moving, deg (default 2)")
    ap.add_argument("--feedback-timeout", type=float, default=0.5,
                    help="s without feedback before freezing (default 0.5)")
    ap.add_argument("--limit-margin", type=float, default=2.0,
                    help="deg kept clear of each joint limit (default 2)")
    ap.add_argument("--label", default="", help="run folder name (default: derived)")
    ap.add_argument("--motor-cmd-topic", default="/interfacing/motorCMD",
                    help="where joint_command publishes MotorCmd; /dry_run/motorCMD for a dry run")
    ap.add_argument("--self-test", action="store_true",
                    help="no ROS: rebuild the A1 reference observations, compare the actions, exit")
    ap.add_argument("--check", action="store_true",
                    help="preflight against live feedback, publish nothing, exit")
    ap.set_defaults(limb="leg", mapping=None, actuators=None)  # for roundtrip.resolve_config
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(args.weights)

    if not args.joints:
        sys.exit("--joints is required (e.g. ankle.pitch)")
    joints = [j.strip() for j in args.joints.split(",") if j.strip()]
    unknown = [j for j in joints if j not in RIGHT_IDX]
    if unknown or not joints:
        sys.exit(f"unknown joint(s) {unknown}; choose from {list(RIGHT_IDX)}")
    if args.box <= 0 or args.seconds <= 0 or args.ease < 0 or 2 * args.ease > args.seconds:
        sys.exit("need --box > 0, --seconds > 0 and 0 <= 2 * --ease <= --seconds")
    mlp, _, _ = load_policy(args.weights)
    idx = {j: RIGHT_IDX[j] - (args.side == "left") for j in joints}
    cmd = np.array([args.vx, args.vy, args.wz])

    # ROS only from here, so --self-test runs on the host.
    import rclpy
    import joint_config as jc
    from roundtrip import RoundTripNode, Stop, cosine, profile_time, resolve_config
    from telemetry import RunFolder, default_run_root

    mapping, safety_path = resolve_config(args)
    names = jc.LEG_POSE_NAMES
    joint_map = jc.load_joint_map(mapping, args.side, "leg")
    slot_motor = []
    for slot, name in enumerate(names):
        mid = next((m for m, info in joint_map.items() if info["slot"] == slot), None)
        if mid is None:
            sys.exit(f"{args.side} {name} has no can_id in {mapping}")
        slot_motor.append(mid)
    slot = {j: names.index(j) for j in joints}
    motor = {j: slot_motor[slot[j]] for j in joints}
    blocks = {name: jc.joint_safety(safety_path, args.side, name) for name in names}

    refused = []
    active = {name for name in names if blocks[name].get("active")}
    if active != set(joints):
        refused.append(f"{safety_path} has {sorted(active) or 'nothing'} active on the "
                       f"{args.side} leg; --joints is {sorted(joints)}. They must be the same "
                       f"(guide B1): an active joint outside --joints would be held at a "
                       f"fixed pose without being watched")
    vel = {}
    for j in joints:
        vmax = float(blocks[j].get("velocity_max") or 0.0)
        if vmax <= 0:
            refused.append(f"{j} has no velocity_max in {safety_path}")
        vel[j] = min(APPROACH_VEL, vmax) if vmax > 0 else APPROACH_VEL
    default_deg = {j: math.degrees(DEFAULT_RAD[idx[j]]) for j in joints}
    limits = {}
    for j in joints:
        info = joint_map[motor[j]]
        lo, hi = (info["lower"], info["upper"]) if info["limit_range"] else (-math.inf, math.inf)
        limits[j] = (lo, hi)
        if not (lo + args.limit_margin <= default_deg[j] - args.box and
                default_deg[j] + args.box <= hi - args.limit_margin):
            refused.append(f"{j}: default {default_deg[j]:+.2f} +- box {args.box:g} does not "
                           f"fit inside its limits [{lo:g}, {hi:g}] minus the "
                           f"{args.limit_margin:g} deg margin")
    if refused:
        for msg in refused:
            print("REFUSED: " + msg)
        sys.exit("nothing was moved")

    label = args.label or (f"policy-{args.side}-" + "-".join(j.replace(".", "") for j in joints)
                           + f"-vx{args.vx:g}-box{args.box:g}")
    meta = {
        "tool": "policy_deploy.py", "limb": "leg", "arm_side": args.side,
        "mapping": str(mapping), "actuators": str(safety_path),
        "motor_cmd_topic": args.motor_cmd_topic, "rate_hz": 50.0,
        "frame": "command frame (degrees) for both sp_deg and pos_deg",
        "joints": {str(mid): info["name"] for mid, info in sorted(joint_map.items())},
        "moving": sorted(joints), "weights": args.weights,
        "command": [args.vx, args.vy, args.wz], "box_deg": args.box,
        "seconds": args.seconds, "ease_s": args.ease,
        "default_deg": default_deg,
        "limits": {
            "per_joint_deg": {j: list(limits[j]) for j in joints},
            "velocity_max_dps_per_joint": {j: blocks[j].get("velocity_max") for j in joints},
            "max_track_err_deg": args.max_track_err,
        },
    }
    # No run folder until preflight passes (none at all for --check).
    log = RunFolder(label, "ros", enabled=False)

    stops = {"n": 0}

    def on_signal(signum, _frame):
        stops["n"] += 1
        print("\n[stop] " + ("returning to the origin (Ctrl-C again to stop streaming now)"
                             if stops["n"] == 1 else "stopping the stream now"), flush=True)

    rclpy.init()
    node = RoundTripNode(joint_map, log, 50.0, [args.side], "leg", args.motor_cmd_topic)
    topic = jc.pose_topic("leg", args.side)
    outcome, reason = "completed", None
    policy_file = None
    dts = []
    try:
        # --- preflight ---------------------------------------------------------------------
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end and not all(m in node.feedback for m in motor.values()):
            rclpy.spin_once(node, timeout_sec=0.05)
        others = [i.node_name for i in node.get_publishers_info_by_topic(topic)
                  if i.node_name != node.get_name()]
        if others:
            raise SystemExit(f"something else is publishing {topic} ({', '.join(sorted(set(others)))})"
                             "; stop it (teleop, ros2 topic pub) first")
        if not [i for i in node.get_subscriptions_info_by_topic(topic)
                if i.node_name != node.get_name()]:
            raise SystemExit(f"nobody subscribes to {topic}: start joint_command with "
                             f"`legs: [\"{args.side}\"]` and the --joints joints active")
        for j in joints:
            m = motor[j]
            if not joint_map[m]["calibrated"]:
                refused.append(f"{j} (motor {m}) is not calibrated (null direction / "
                               f"zero_offset in {mapping})")
            elif m not in node.feedback:
                refused.append(f"no feedback from {j} (motor {m})")
            else:
                lo, hi = limits[j]
                pos = node.measured(m)
                if not lo <= pos <= hi:
                    refused.append(f"{j} is at {pos:+.1f}, outside its limits [{lo:g}, {hi:g}]: "
                                   "recalibrate it")
        if refused:
            for msg in refused:
                print("REFUSED: " + msg)
            raise SystemExit("nothing was moved")

        # A silent or uncalibrated slot's value is a placeholder: it is inactive (checked above).
        origin = [node.measured(m) if (m in node.feedback and joint_map[m]["calibrated"])
                  else 0.0 for m in slot_motor]
        default_pose = list(origin)
        for j in joints:
            default_pose[slot[j]] = default_deg[j]
        slot_vel = [vel.get(name, APPROACH_VEL) for name in names]
        t_approach = profile_time(origin, default_pose, slot_vel, 0.5)

        print(f"mapping: {mapping}\nsafety:  {safety_path}")
        print(f"{'joint':<14}{'motor':>6}{'origin':>9}{'default':>9}{'sent range':>18}"
              f"{'limits':>18}")
        for j in joints:
            lo, hi = limits[j]
            d = default_deg[j]
            print(f"{j:<14}{motor[j]:>6}{origin[slot[j]]:>+9.2f}{d:>+9.2f}"
                  f"{f'[{d - args.box:+.1f}, {d + args.box:+.1f}]':>18}"
                  f"{f'[{lo:g}, {hi:g}]':>18}")
        print(f"command vx {args.vx:g} m/s, vy {args.vy:g} m/s, wz {args.wz:g} rad/s; "
              f"approach {t_approach:.1f} s, policy {args.seconds:g} s (ease {args.ease:g} s)")
        print("\n  HARDWARE E-STOP: physical cutoff on the 48 V leg supply within reach.\n")
        if args.check:
            print("--check: preflight passed; nothing was published.")
            return 0

        # --- the run --------------------------------------------------------------------------
        log = node.log = RunFolder(label=label, source="ros", meta=meta,
                                   root=default_run_root("policy"))
        print(log.describe(), flush=True)
        log.note(origin_deg=[round(v, 3) for v in origin], approach_s=round(t_approach, 3))
        policy_file = (log.path / "policy.csv").open("w", newline="", encoding="utf-8")
        policy_csv = csv.writer(policy_file)
        urdf = {j: JOINTS[idx[j]] for j in joints}
        policy_csv.writerow(["t_s", "dt_ms"] + OBS_NAMES + [f"action_{n}" for n in JOINTS]
                            + [f"wanted_{urdf[j]}" for j in joints]
                            + [f"sent_{urdf[j]}" for j in joints])
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        state = {"phase": "seed", "t0": time.monotonic(), "from": list(origin),
                 "to": list(origin), "dur": HOLD_S, "pose": list(origin)}
        pol = {}

        def enter(phase, to, dur):
            state.update(phase=phase, t0=time.monotonic(), dur=dur, to=list(to),
                         **{"from": list(state["pose"])})
            node.phase = phase
            print(f"[{phase}] {dur:.1f} s", flush=True)
            if phase == "policy":
                now = time.monotonic()
                pol.update(t_prev=now, last_action=np.zeros(12), qd=np.zeros(12),
                           q_prev={j: math.radians(node.measured(motor[j])) - DEFAULT_RAD[idx[j]]
                                   for j in joints})

        def stop(why):
            nonlocal outcome, reason
            if outcome == "completed":
                outcome, reason = "stopped", why
                log.note(outcome=outcome, abort_reason=why)
            print(f"stopping: {why}", flush=True)
            if state["phase"] in ("policy", "hold_end"):
                enter("stop_default", default_pose,
                      profile_time(state["pose"], default_pose, slot_vel, 0.5))
            else:
                enter("retreat", origin, profile_time(state["pose"], origin, slot_vel, 0.5))

        node.phase = "seed"
        print("[seed] holding the measured pose; joint_command must match it", flush=True)
        t_next = time.monotonic()
        frozen = False
        while True:
            now = time.monotonic()
            stale = [j for j in joints
                     if now - node.fb_time.get(motor[j], 0.0) > args.feedback_timeout]
            if stale and not frozen:
                frozen = True
                outcome, reason = "aborted", f"feedback lost: {', '.join(stale)}"
                log.note(outcome=outcome, abort_reason=reason)
                node.phase = "frozen"
                print(f"\nFEEDBACK LOST from {', '.join(stale)}: holding the last sent pose. "
                      "Support the leg, then Ctrl-C to stop streaming.", flush=True)
            if frozen:
                if stops["n"] >= 1:
                    raise Stop()
                node.publish_pose(state["pose"])
            else:
                if stops["n"] >= 2:
                    raise Stop()
                if stops["n"] >= 1 and state["phase"] not in ("stop_default", "retreat"):
                    stop("stop requested")
                phase = state["phase"]
                elapsed = now - state["t0"]

                if phase == "policy":
                    t = elapsed
                    dt = now - pol["t_prev"]
                    pol["t_prev"] = now
                    q_rel = np.zeros(12)
                    for j in joints:
                        i = idx[j]
                        q = math.radians(node.measured(motor[j])) - DEFAULT_RAD[i]
                        raw = (q - pol["q_prev"][j]) / dt if dt > 0 else 0.0
                        pol["qd"][i] = 0.5 * pol["qd"][i] + 0.5 * raw
                        pol["q_prev"][j] = q
                        q_rel[i] = q
                    obs = build_obs(q_rel, pol["qd"], pol["last_action"], t, cmd)
                    action = mlp(obs)
                    pol["last_action"] = action
                    if args.ease > 0:
                        env = min(cosine(t / args.ease), cosine((args.seconds - t) / args.ease))
                    else:
                        env = 1.0
                    wanted, sent = {}, {}
                    for j in joints:
                        d = default_deg[j]
                        wanted[j] = math.degrees(DEFAULT_RAD[idx[j]] + ACTION_SCALE * action[idx[j]])
                        s = d + env * min(max(wanted[j] - d, -args.box), args.box)
                        lo, hi = limits[j]
                        sent[j] = min(max(s, lo + args.limit_margin), hi - args.limit_margin)
                        state["pose"][slot[j]] = sent[j]
                    dts.append(1000.0 * dt)
                    policy_csv.writerow([f"{now - log._t0:.4f}", f"{1000.0 * dt:.2f}"]  # telemetry.csv's clock
                                        + [f"{v:.5f}" for v in obs] + [f"{v:.5f}" for v in action]
                                        + [f"{wanted[j]:.4f}" for j in joints]
                                        + [f"{sent[j]:.4f}" for j in joints])
                else:
                    alpha = elapsed / state["dur"] if state["dur"] > 0 else 1.0
                    s = cosine(alpha)
                    state["pose"] = [a + (b - a) * s for a, b in zip(state["from"], state["to"])]
                node.publish_pose(state["pose"])

                # Tracking: measured vs what joint_command actually commanded.
                if phase != "seed":
                    for j in joints:
                        sp, pos = node.setpoints.get(motor[j]), node.measured(motor[j])
                        if sp is None or pos is None:
                            continue
                        err = abs(pos - sp)
                        if phase != "retreat" and phase != "stop_default" and \
                                err > args.max_track_err:
                            stop(f"{j} tracking error {err:.1f} deg > {args.max_track_err:g}")
                            break
                        if phase in ("retreat", "stop_default") and err > 1.5 * args.max_track_err:
                            frozen = True
                            outcome = "aborted"
                            reason = f"{j} tracking error {err:.1f} deg on the way back"
                            log.note(outcome=outcome, abort_reason=reason)
                            node.phase = "frozen"
                            print(f"\n{reason}: holding here. Support the leg, then Ctrl-C.",
                                  flush=True)
                            break

                if state["phase"] == phase and elapsed >= state["dur"]:
                    if phase == "seed":
                        bad = []
                        for j in joints:
                            sp, pos = node.setpoints.get(motor[j]), node.measured(motor[j])
                            if sp is None:
                                bad.append(f"{j}: joint_command sent no command (inactive? "
                                           "excluded? see its log)")
                            elif abs(sp - pos) > args.seed_tolerance:
                                bad.append(f"{j}: commanded {sp:+.2f} vs measured {pos:+.2f} deg")
                        if bad:
                            outcome, reason = "aborted", "seed check: " + "; ".join(bad)
                            log.note(outcome=outcome, abort_reason=reason)
                            print("REFUSED before any motion: joint_command is not commanding "
                                  "the measured pose:\n  " + "\n  ".join(bad), flush=True)
                            raise Stop()
                        print("  setpoints match the measured pose", flush=True)
                        enter("approach", default_pose, t_approach)
                    elif phase == "approach":
                        enter("hold", default_pose, HOLD_S)
                    elif phase == "hold":
                        enter("policy", default_pose, args.seconds)
                    elif phase == "policy":
                        state["pose"] = list(default_pose)  # the ease-out ends at the default
                        enter("hold_end", default_pose, HOLD_S)
                    elif phase == "hold_end":
                        enter("retreat", origin, t_approach)
                    elif phase == "stop_default":
                        enter("retreat", origin, profile_time(default_pose, origin, slot_vel, 0.5))
                    elif phase == "retreat":
                        break

            t_next += 0.02
            while True:
                remaining = t_next - time.monotonic()
                if remaining <= 0:
                    break
                rclpy.spin_once(node, timeout_sec=remaining)
    except Stop:
        pass
    finally:
        node.phase = "end"
        if policy_file is not None:
            policy_file.close()
        if dts:
            a = np.array(dts[1:] or dts)
            stats = {"median_ms": round(float(np.median(a)), 2),
                     "p99_ms": round(float(np.percentile(a, 99)), 2),
                     "over_25ms": int((a > 25.0).sum()), "ticks": int(a.size)}
            log.note(policy_dt=stats)
            print(f"\npolicy tick dt: median {stats['median_ms']:.1f} ms, 99% "
                  f"{stats['p99_ms']:.1f} ms, {stats['over_25ms']} of {stats['ticks']} ticks "
                  "> 25 ms")
        run_dir = log.close(outcome, **({"abort_reason": reason} if reason else {}))
        node.destroy_node()
        rclpy.shutdown()
        if outcome != "completed":
            print(f"outcome: {outcome} -- {reason}")
        if run_dir is not None:
            print(f"Run folder: {run_dir}")
    return 0 if outcome == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
