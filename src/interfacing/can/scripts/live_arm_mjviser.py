#!/usr/bin/env python3
"""Live browser view (MuJoCo + mjviser) of the real arm, driven by CAN feedback.

READ-ONLY: subscribes to /interfacing/motorFeedback and never publishes, so it cannot move
the arm. Use it after calibrate_arm.py to check the calibration looks right before commanding
anything (src/interfacing/ARM_BRINGUP.md step 2).

Run in the `simulation_mj` container while `interfacing` is up and the arm is powered:

    ./watod -t simulation_mj
    python3 /workspace/humanoid/src/interfacing/can/scripts/live_arm_mjviser.py --arm-side left
    # open http://localhost:8080

Angle shown for each joint, in degrees -- the command frame IS the URDF frame:

    q_urdf = q_cmd = zero_offset + motor / direction       (arm_calibration.yaml)

Check the calibration against pioneer_bimanual_arm.urdf: move each joint by hand and confirm
the on-screen joint turns the same way and stops at the same angle. Try corrections with
--flip / --offset (viewer-only). A joint that needs --flip has the wrong `direction` in
arm_calibration.yaml: flip it and re-run calibrate_arm.py (zero_offset and the limits depend on
it). One that needs --offset was not zeroed hanging: re-run calibrate_arm.py.

The gripper's fingers follow its position, as joint_command maps it: 0 = open (its calibrated
zero) .. 1 = closed (arm_calibration.yaml upper_limit). Open and close it by hand: the fingers
must open and close with it (calibrate it open, direction so that closing is positive).

The side panel plots each joint's angle over the last --plot-window seconds (default 20): the
calibrated command-frame angle, the same numbers arm_roundtrip and joint_command use (the
viewer-only --flip / --offset only change the 3D view). Dashed red lines are the joint limits;
they appear once the angle comes near them. Below them, each motor's current draw (A, as the drive
reports it in MotorFeedback.current; signed) over the same window.

Check the angle math without ROS or MuJoCo:  python3 live_arm_mjviser.py --self-test
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

from joint_config import find_calibration, gripper_position, load_joint_map, motor_to_cmd_deg

# pioneer_humanoid is not installed in the simulation_mj image; scripts add src/ paths themselves.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pioneer_humanoid"))

STALE_AFTER_S = 1.0
PLOT_HZ = 10.0  # plot refresh; each refresh resends the visible history to the browser
CURRENT_MIN_SPAN_A = 2.0  # current plots are at least this tall, so a 0.3 A wobble stays readable


def plot_y_range(values: list[float], lower: float, upper: float,
                 min_span: float = 10.0) -> tuple[float, float]:
    """Y range for one plot: the visible data plus a margin, at least min_span units tall, so small
    moves stay visible. A limit line shows only once the data comes within the margin of it."""
    if not values:
        return (lower, upper) if math.isfinite(lower) and math.isfinite(upper) else (-min_span, min_span)
    lo, hi = min(values), max(values)
    span = max(hi - lo, min_span)
    mid = (lo + hi) / 2
    return mid - 0.6 * span, mid + 0.6 * span


def urdf_deg(info: dict, motor_deg: float) -> float:
    """Motor-frame feedback (deg) -> URDF joint angle (deg), with the viewer-only --flip /
    --offset applied. Never clamped: this only displays."""
    q_cmd = motor_to_cmd_deg(info, motor_deg)
    return info["view_flip"] * q_cmd + info["view_offset_deg"]


def build_joint_map(mapping: Path, arm_side: str, urdf_joints: list[str],
                    flip: set[str], offset: dict[str, float]) -> dict[int, dict]:
    """{motor_id: joint info + urdf_joint, view_flip, view_offset_deg} for the six arm joints."""
    out = {}
    for motor_id, info in load_joint_map(mapping, arm_side).items():
        if info["slot"] is None:  # gripper
            continue
        out[motor_id] = dict(
            info,
            urdf_joint=urdf_joints[info["slot"]],
            view_flip=-1 if info["name"] in flip else 1,
            view_offset_deg=offset.get(info["name"], 0.0),
        )
    return out


def self_test() -> None:
    info = {"direction": -1.0, "zero_offset": 10.0, "view_flip": 1, "view_offset_deg": 0.0}
    assert urdf_deg(info, 0.0) == 10.0
    assert urdf_deg(info, -20.0) == 30.0  # direction -1: motor -20 is +20 in the command frame
    assert urdf_deg(dict(info, view_flip=-1, view_offset_deg=90.0), -20.0) == 60.0

    mapping = find_calibration(None)
    names = ["j0", "j1", "j2", "j3", "j4", "j5"]
    joints = build_joint_map(mapping, "left", names, set(), {})
    assert sorted(j["urdf_joint"] for j in joints.values()) == names, joints
    assert all(j["view_flip"] == 1 and j["view_offset_deg"] == 0.0 for j in joints.values())
    flipped = build_joint_map(mapping, "left", names, {"shoulder.roll"}, {"shoulder.roll": 5.0})
    for motor_id, joint in joints.items():
        other = flipped[motor_id]
        if joint["name"] == "shoulder.roll":
            assert other["view_flip"] == -1 and other["view_offset_deg"] == 5.0
        else:
            assert other == joint
    grip = next(info for info in load_joint_map(mapping, "left").values() if info["slot"] is None)
    assert gripper_position(grip, 0.0) == 0.0 and gripper_position(grip, grip["upper"]) == 1.0

    assert plot_y_range([], -90.0, 90.0) == (-90.0, 90.0)
    lo, hi = plot_y_range([1.0, 4.0], -85.0, 6.0)  # a 3 deg move gets a 10+ deg tall plot
    assert lo < 1.0 and hi > 4.0 and hi - lo >= 10.0
    lo, hi = plot_y_range([0.2, 0.5], -math.inf, math.inf, min_span=CURRENT_MIN_SPAN_A)
    assert lo < 0.2 and hi > 0.5 and abs((hi - lo) - 1.2 * CURRENT_MIN_SPAN_A) < 1e-9
    print(f"self-test ok ({len(joints)} joints + gripper from {mapping})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-side", default="left", choices=["left", "right"],
                        help="side in arm_calibration.yaml; drives the same side of the URDF")
    parser.add_argument("--flip", nargs="*", default=[], metavar="JOINT",
                        help="viewer-only: invert these joints (e.g. shoulder.roll); a fix belongs in "
                             "arm_calibration.yaml's direction")
    parser.add_argument("--offset", nargs="*", default=[], metavar="JOINT=DEG",
                        help="viewer-only: degrees added to these joints (e.g. shoulder.yaw=90); a fix "
                             "is a re-run of calibrate_arm.py")
    parser.add_argument("--calibration", "--mapping", dest="mapping", default=None,
                        help="arm_calibration.yaml (default: auto)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--hz", type=float, default=30.0, help="scene refresh rate")
    parser.add_argument("--plot-window", type=float, default=20.0, metavar="S",
                        help="seconds of history shown in the joint plots (default 20)")
    parser.add_argument("--self-test", action="store_true", help="check the angle math and exit")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    import mujoco
    import rclpy
    import viser
    from mjviser import ViserMujocoScene
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    from common_msgs.msg import MotorFeedback
    from pioneer_humanoid import arm_params
    from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
    from pioneer_humanoid.mujoco_bimanual_arm import arm_spec

    offset = {name.strip(): float(deg) for name, _, deg in (o.partition("=") for o in args.offset)}
    mapping = find_calibration(args.mapping)
    urdf_joints = LEFT_ARM_JOINTS if args.arm_side == "left" else RIGHT_ARM_JOINTS
    joints = build_joint_map(mapping, args.arm_side, urdf_joints, set(args.flip), offset)
    unknown = (set(args.flip) | set(offset)) - {j["name"] for j in joints.values()}
    if unknown or not joints:
        raise SystemExit(f"no such joint(s) {sorted(unknown)} for arm side {args.arm_side!r} in "
                         f"{mapping} (have {sorted(j['name'] for j in joints.values())})")

    model = arm_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos_adr = {j["urdf_joint"]: int(model.joint(j["urdf_joint"]).qposadr[0]) for j in joints.values()}
    # The gripper (slot None): position 0..1 -> finger travel in metres, as in sim.
    side = args.arm_side.upper()
    grip = next(({**info, "motor_id": mid} for mid, info in load_joint_map(mapping, args.arm_side).items()
                 if info["slot"] is None), None)
    fingers = [(int(model.joint(f).qposadr[0]), getattr(arm_params, f"{side}_GRIPPER_OPEN")[f],
                getattr(arm_params, f"{side}_GRIPPER_CLOSED")[f])
               for f in getattr(arm_params, f"{side}_GRIPPER_JOINTS")]

    print(f"{args.arm_side} arm, read-only. mapping: {mapping}")
    for motor_id, j in sorted(joints.items()):
        view = ""
        if j["view_flip"] != 1 or j["view_offset_deg"]:
            view = f"  (viewer-only: flip {j['view_flip']:+d}, offset {j['view_offset_deg']:+g})"
        print(f"  id {motor_id:>3} {j['name']:<15} -> {j['urdf_joint']:<8} direction {j['direction']:+g}{view}")
    print("NOT VERIFIED on hardware: move each joint by hand and compare before trusting this view.")

    lock = threading.Lock()
    latest_deg: dict[str, float] = {}
    last_seen: dict[int, float] = {}
    latest_grip: list[float] = []
    # Plot history per motor: (monotonic time, value). Feedback is ~50 Hz; 100/s leaves headroom.
    history: dict[int, deque] = {}
    current_history: dict[int, deque] = {}  # same, for MotorFeedback.current (A)
    history_len = int(args.plot_window * 100) + 1

    def record(motor_id: int, value: float, current: float) -> None:
        now = time.monotonic()
        history.setdefault(motor_id, deque(maxlen=history_len)).append((now, value))
        current_history.setdefault(motor_id, deque(maxlen=history_len)).append((now, current))
        last_seen[motor_id] = now

    def on_feedback(msg: MotorFeedback) -> None:
        motor_id = int(msg.motor_id)
        if grip is not None and motor_id == grip["motor_id"]:
            pos = gripper_position(grip, motor_to_cmd_deg(grip, float(msg.position)))
            with lock:
                latest_grip[:] = [pos]
                record(motor_id, pos, float(msg.current))
            return
        j = joints.get(motor_id)
        if j is None:
            return
        with lock:
            latest_deg[j["urdf_joint"]] = urdf_deg(j, float(msg.position))
            record(motor_id, motor_to_cmd_deg(j, float(msg.position)), float(msg.current))

    rclpy.init()
    node = Node("live_arm_mjviser")
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=50)
    node.create_subscription(MotorFeedback, "/interfacing/motorFeedback", on_feedback, qos)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    server = viser.ViserServer(port=args.port)
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.update_from_mjdata(data)

    # One plot per joint (slot order), then the gripper. x = seconds before now.
    plotted = [(mid, j["name"], "deg", j["lower"], j["upper"]) if j["limit_range"]
               else (mid, j["name"], "deg", -math.inf, math.inf)
               for mid, j in sorted(joints.items(), key=lambda kv: kv[1]["slot"])]
    if grip is not None:
        plotted.append((grip["motor_id"], "gripper (0 open .. 1 closed)", "", 0.0, 1.0))
    import numpy as np
    plots = {}
    with server.gui.add_folder(f"Joint angles, last {args.plot_window:g} s (command frame)"):
        for mid, name, unit, lower, upper in plotted:
            plots[mid] = server.gui.add_uplot(
                data=(np.array([-args.plot_window, 0.0]), np.full(2, np.nan),
                      np.full(2, lower if math.isfinite(lower) else np.nan),
                      np.full(2, upper if math.isfinite(upper) else np.nan)),
                series=({"label": "s ago"},
                        {"label": name, "stroke": "#2f6fdf", "width": 2},
                        {"label": "lower limit", "stroke": "#d33", "dash": [6, 4], "width": 1},
                        {"label": "upper limit", "stroke": "#d33", "dash": [6, 4], "width": 1}),
                scales={"x": {"time": False, "range": [-args.plot_window, 0.0]}},
                title=f"{name}  (no feedback yet)", aspect=2.5)
    current_plots = {}
    with server.gui.add_folder(f"Motor current, last {args.plot_window:g} s (A)"):
        for mid, name, _, _, _ in plotted:
            label = name.split(" (")[0]  # "gripper (0 open .. 1 closed)" -> "gripper"
            current_plots[mid] = server.gui.add_uplot(
                data=(np.array([-args.plot_window, 0.0]), np.full(2, np.nan)),
                series=({"label": "s ago"}, {"label": f"{label} A", "stroke": "#e08a1e", "width": 2}),
                scales={"x": {"time": False, "range": [-args.plot_window, 0.0]}},
                title=f"{label}  (no feedback yet)", aspect=2.5)
    print(f"open http://localhost:{args.port}")

    def refresh_plots(now: float, hist: dict[int, list], cur_hist: dict[int, list],
                      seen: dict[int, float]) -> None:
        for mid, name, unit, lower, upper in plotted:
            pts = [(t - now, v) for t, v in hist.get(mid, []) if now - t <= args.plot_window]
            xs = np.array([p[0] for p in pts] or [-args.plot_window, 0.0])
            ys = np.array([p[1] for p in pts] or [np.nan, np.nan])
            lo, hi = plot_y_range([p[1] for p in pts], lower, upper)
            plots[mid].data = (xs, ys, np.full(len(xs), lower if math.isfinite(lower) else np.nan),
                               np.full(len(xs), upper if math.isfinite(upper) else np.nan))
            plots[mid].scales = {"x": {"time": False, "range": [-args.plot_window, 0.0]},
                                 "y": {"range": [lo, hi]}}
            if now - seen.get(mid, -math.inf) > STALE_AFTER_S:
                plots[mid].title = f"{name}  NO FEEDBACK"
            else:
                plots[mid].title = f"{name}  {pts[-1][1]:+.1f}{'°' if unit else ''}"
            label = name.split(" (")[0]
            cpts = [(t - now, v) for t, v in cur_hist.get(mid, []) if now - t <= args.plot_window]
            cx = np.array([p[0] for p in cpts] or [-args.plot_window, 0.0])
            cy = np.array([p[1] for p in cpts] or [np.nan, np.nan])
            clo, chi = plot_y_range([p[1] for p in cpts], -math.inf, math.inf, min_span=CURRENT_MIN_SPAN_A)
            current_plots[mid].data = (cx, cy)
            current_plots[mid].scales = {"x": {"time": False, "range": [-args.plot_window, 0.0]},
                                         "y": {"range": [clo, chi]}}
            if now - seen.get(mid, -math.inf) > STALE_AFTER_S:
                current_plots[mid].title = f"{label}  NO FEEDBACK"
            elif cpts:
                peak = max(abs(p[1]) for p in cpts)
                current_plots[mid].title = f"{label}  {cpts[-1][1]:+.2f} A  (peak {peak:.2f})"

    next_warn = time.monotonic() + STALE_AFTER_S
    next_plot = time.monotonic()
    try:
        while rclpy.ok():
            now = time.monotonic()
            with lock:
                shown = dict(latest_deg)
                grip_pos = list(latest_grip)
                stale = sorted(j["name"] for motor_id, j in joints.items()
                               if now - last_seen.get(motor_id, -math.inf) > STALE_AFTER_S)
                if now >= next_plot:
                    hist = {mid: list(h) for mid, h in history.items()}
                    cur_hist = {mid: list(h) for mid, h in current_history.items()}
                    seen = dict(last_seen)
            if now >= next_plot:
                refresh_plots(now, hist, cur_hist, seen)
                next_plot = now + 1.0 / PLOT_HZ
            # A frozen joint looks the same as a still one, so say when the picture is not live.
            if stale and now >= next_warn:
                print(f"NO FEEDBACK for >{STALE_AFTER_S:g}s (view is not live): {', '.join(stale)}")
                next_warn = now + STALE_AFTER_S
            for urdf_joint, deg in shown.items():
                data.qpos[qpos_adr[urdf_joint]] = math.radians(deg)
            if grip_pos:
                c = min(max(grip_pos[0], 0.0), 1.0)
                for adr, opened, closed in fingers:
                    data.qpos[adr] = opened + c * (closed - opened)
            mujoco.mj_forward(model, data)
            scene.update_from_mjdata(data)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        server.stop()


if __name__ == "__main__":
    main()
