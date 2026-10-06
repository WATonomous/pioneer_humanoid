#!/usr/bin/env python3
"""Live browser view (MuJoCo + mjviser) of the real arm, driven by CAN feedback.

READ-ONLY: subscribes to /interfacing/motorFeedback and never publishes, so it cannot move
the arm. Use it after calibrate_arm.py to check the calibration looks right before commanding
anything (src/interfacing/README.md step 2).

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

The gripper is not driven: its 0-100 command units have no confirmed scale to the finger
joints' travel in metres.

Check the angle math without ROS or MuJoCo:  python3 live_arm_mjviser.py --self-test
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

from joint_config import find_calibration, load_joint_map, motor_to_cmd_deg

# pioneer_humanoid is not installed in the simulation_mj image; scripts add src/ paths themselves.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pioneer_humanoid"))

STALE_AFTER_S = 1.0


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
    print(f"self-test ok ({len(joints)} joints from {mapping})")


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

    def on_feedback(msg: MotorFeedback) -> None:
        j = joints.get(int(msg.motor_id))
        if j is None:
            return
        with lock:
            latest_deg[j["urdf_joint"]] = urdf_deg(j, float(msg.position))
            last_seen[int(msg.motor_id)] = time.monotonic()

    rclpy.init()
    node = Node("live_arm_mjviser")
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=50)
    node.create_subscription(MotorFeedback, "/interfacing/motorFeedback", on_feedback, qos)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    server = viser.ViserServer(port=args.port)
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.update_from_mjdata(data)
    print(f"open http://localhost:{args.port}")

    next_warn = time.monotonic() + STALE_AFTER_S
    try:
        while rclpy.ok():
            now = time.monotonic()
            with lock:
                shown = dict(latest_deg)
                stale = sorted(j["name"] for motor_id, j in joints.items()
                               if now - last_seen.get(motor_id, -math.inf) > STALE_AFTER_S)
            # A frozen joint looks the same as a still one, so say when the picture is not live.
            if stale and now >= next_warn:
                print(f"NO FEEDBACK for >{STALE_AFTER_S:g}s (view is not live): {', '.join(stale)}")
                next_warn = now + STALE_AFTER_S
            for urdf_joint, deg in shown.items():
                data.qpos[qpos_adr[urdf_joint]] = math.radians(deg)
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
