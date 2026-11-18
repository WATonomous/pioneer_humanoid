"""Real-time Isaac Sim viewer for the physical bimanual test stand (Isaac Lab
counterpart to live_arm_mjviser.py, same directory).

Drives one of BIMANUAL_ARM_CFG's two joint chains from live /interfacing/motorFeedback.
--arm-side picks which hardware_mapping.yaml motors to read (only "left" is
wired/calibrated today); --urdf-side picks which URDF chain to animate. Defaults to
--urdf-side right (unsuffixed joint1..joint6) -- confirmed against the real test stand
that this is the visually correct chain, NOT joint1L..joint6l as bimanual_arm_cfg.py's
docstring claims. Read-only: force-writes joint state via write_joint_state_to_sim()
each tick (no PD lag, exact mirror); never commands the real arm.

Zero position: zero_offset + direction*motor_deg per joint, matching
live_arm_mjviser.py and BIMANUAL_ARM_CFG's rest pose, so unwired joints (e.g.
wrist_pitch) sit at calibrated zero rather than a stale snapshot.

For better compatibility with everyone's system, feedback comes via UDP from feedback_to_udp_bridge.py (system ROS
Python) instead of a direct subscription.

Terminal 1 (system python, ROS sourced):
    source /opt/ros/jazzy/setup.bash
    source /home/rwahib/wato/humanoid/autonomy/install/setup.bash
    /usr/bin/python3 autonomy/behaviour/joint_command/scripts/feedback_to_udp_bridge.py

Terminal 2 (env_isaaclab):
    conda activate env_isaaclab
    cd /home/rwahib/wato/humanoid
    python autonomy/simulation/Humanoid_Wato/wato_bimanual_arm/live_arm_isaacsim.py
"""


import argparse
import os
import socket
import struct
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Live Isaac Sim mirror of the real bimanual arm.")
parser.add_argument("--arm-side", default="left", choices=["left", "right"],
                     help="hardware side: which hardware_mapping.yaml section / real motors to read "
                          "(only 'left' is wired/calibrated today)")
parser.add_argument("--urdf-side", default="right", choices=["left", "right"],
                     help="which URDF chain to drive with that feedback: 'right' -> unsuffixed "
                          "joint1..joint6 (default, matches live_arm_mjviser.py's validated "
                          "ARM_BRINGUP.md invocation), 'left' -> suffixed joint1L..joint6l "
                          "(the chain task_space_real.py drives for real hardware output).")
parser.add_argument("--flip", nargs="*", default=[], metavar="LABEL",
                     help="hardware_mapping labels whose sign to invert (e.g. shoulder_roll) -- "
                          "same as live_arm_mjviser.py's --flip.")
parser.add_argument("--host", type=str, default="127.0.0.1", help="feedback_to_udp_bridge.py host")
parser.add_argument("--port", type=int, default=5006, help="feedback_to_udp_bridge.py port")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import bimanual_arm_cfg from keyboard teleoperation (same robot model/zero convention
# as task_space_real.py and live_arm_mjviser.py).
_KEYBOARD_TELEOP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../Teleop/keyboard_based_teleoperation")
)
sys.path.insert(0, _KEYBOARD_TELEOP_DIR)

import yaml  # noqa: E402

from bimanual_arm_cfg import (  # noqa: E402
    BIMANUAL_ARM_CFG,
    _HARDWARE_MAPPING_PATH,
    apply_joint_limits,
    resolve_joint_name,
)
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
import math  # noqa: E402


# hardware_mapping.yaml label -> BIMANUAL_ARM_CFG URDF joint, per side. Same dict as
# live_arm_mjviser.py's LABEL_TO_URDF_JOINT (duplicated rather than imported from that
# script, since it pulls in rclpy/mujoco/viser at import time).
LABEL_TO_URDF_JOINT = {
    "left": {
        "shoulder_pitch": "joint1L",
        "shoulder_roll": "joint2l",
        "shoulder_yaw": "joint3l",
        "elbow_pitch": "joint4l",
        "elbow_roll": "joint5l",
        "wrist_pitch": "joint6l",
    },
    "right": {
        "shoulder_pitch": "joint1",
        "shoulder_roll": "joint2",
        "shoulder_yaw": "joint3",
        "elbow_pitch": "joint4",
        "elbow_roll": "joint5",
        "wrist_pitch": "joint6",
    },
}


def load_can_id_map(
    mapping_path: str,
    hw_side: str,
    urdf_side: str,
    flip_labels: set = frozenset(),
) -> dict:
    """hardware_mapping.yaml -> {can_id: {label, urdf_joint, direction, zero_offset,
    lower_limit, upper_limit}}.

    Same fields, --flip semantics, and hw_side/urdf_side split as
    live_arm_mjviser.py's load_can_id_map.
    """
    with open(mapping_path) as f:
        config = yaml.safe_load(f)[hw_side]

    label_to_joint = LABEL_TO_URDF_JOINT[urdf_side]
    can_id_map = {}
    for group, joints in config.items():
        for name, cfg in joints.items():
            label = f"{group}_{name}"
            urdf_joint = label_to_joint.get(label)
            if urdf_joint is None:
                continue
            # Must negate zero_offset together with direction, not direction alone: zero_offset was
            # derived as -home_pos/direction so zero_offset + direction*home_pos == 0 at the real
            # zero pose. Flipping direction alone breaks that and shifts the zero pose by
            # 2*zero_offset. Negating both preserves joint_deg(home_pos)==0 while correctly
            # reversing motion sense (verified: joint_deg_new(raw) == -joint_deg_old(raw)).

            flip = label in flip_labels
            direction = int(cfg["direction"]) * (-1 if flip else 1)
            zero_offset = float(cfg["zero_offset"]) * (-1 if flip else 1)
            can_id_map[int(cfg["can_id"])] = {
                "label": label,
                "urdf_joint": urdf_joint,
                "direction": direction,
                "zero_offset": zero_offset,
                "lower_limit": float(cfg["lower_limit"]),
                "upper_limit": float(cfg["upper_limit"]),
            }
    return can_id_map


@configclass
class BareSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = BIMANUAL_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene,
                   can_id_map: dict, sock: socket.socket) -> None:
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    scene.update(sim_dt)
    apply_joint_limits(robot)

    name_to_id = {name: i for i, name in enumerate(robot.data.joint_names)}
    urdf_joint_ids = {
        cfg["urdf_joint"]: name_to_id[resolve_joint_name(robot, cfg["urdf_joint"])]
        for cfg in can_id_map.values()
    }

    print(f"Tracking {len(can_id_map)} motors from {_HARDWARE_MAPPING_PATH} "
          f"(hardware={args_cli.arm_side} arm -> driving URDF {args_cli.urdf_side} arm):")
    for can_id, cfg in sorted(can_id_map.items()):
        flip = "  (flipped)" if cfg["label"] in set(args_cli.flip) else ""
        print(f"  0x{can_id:02X} -> {cfg['label']:<14} -> {cfg['urdf_joint']}{flip}")

    joint_position = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_position, joint_vel)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    print("[INFO] Live Isaac Sim mirror running -- move the real arm to see it track here.")

    # Diagnostic only (no filtering/rejection -- this stays a raw, unfiltered mirror by
    # design, same as live_arm_mjviser.py). Logs a jump's RAW inputs so a future glitch
    # can be traced to its actual source: if the raw `position` printed here is already
    # anomalous, it's upstream (CAN/encoder); if raw looks sane but joint_deg doesn't,
    # it's a bug in this script's own transform.
    _JUMP_WARN_DEG = 15.0
    _last_deg = {}

    while simulation_app.is_running():
        while True:
            try:
                data, _ = sock.recvfrom(1024)
            except BlockingIOError:
                break
            if len(data) != struct.calcsize("=id"):
                continue
            motor_id, position = struct.unpack("=id", data)
            cfg = can_id_map.get(motor_id)
            if cfg is None:
                continue
            joint_deg = cfg["zero_offset"] + cfg["direction"] * position
            prev = _last_deg.get(motor_id)
            if prev is not None and abs(joint_deg - prev) > _JUMP_WARN_DEG:
                print(f"[WARN] {cfg['label']} (motor {motor_id}) jumped {prev:.2f}deg -> "
                      f"{joint_deg:.2f}deg (raw position={position:.3f}, packet_bytes={len(data)})")
            _last_deg[motor_id] = joint_deg
            joint_id = urdf_joint_ids[cfg["urdf_joint"]]
            joint_position[0, joint_id] = math.radians(joint_deg)

        # Direct state overwrite (not set_joint_position_target): matches
        # live_arm_mjviser.py's `data.qpos[...] = ...` -- an exact live mirror with no
        # PD lag, since this is read-only visualization, not a commanded target.
        robot.write_joint_state_to_sim(joint_position, joint_vel)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


def main() -> None:
    can_id_map = load_can_id_map(
        _HARDWARE_MAPPING_PATH, args_cli.arm_side, args_cli.urdf_side,
        set(args_cli.flip)
    )
    if not can_id_map:
        raise RuntimeError(
            f"No joints resolved for arm_side={args_cli.arm_side!r} in {_HARDWARE_MAPPING_PATH}"
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args_cli.host, args_cli.port))
    sock.setblocking(False)
    print(f"[UDP] Listening for motor feedback on {args_cli.host}:{args_cli.port} "
          f"(run feedback_to_udp_bridge.py to feed this)")

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 0.8])

    scene_cfg = BareSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    print("[INFO]: Setup complete...")
    run_simulator(sim, scene, can_id_map, sock)


if __name__ == "__main__":
    main()
    simulation_app.close()
