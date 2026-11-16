"""Isaac Sim backend of pioneer_leader_arm_teleop.py: the 8-servo leader drives the pioneer_bimanual_arm_8dof arms (no IK),
one leader board per arm (--arms right (default), left or bimanual; leader_arm_8dof.json).

    A (servo ID 2) -> {side}_shoulder_pitch      E (servo ID 4) -> {side}_forearm_roll
    B (servo ID 3) -> {side}_shoulder_roll       F (servo ID 7) -> {side}_wrist_pitch
    C (servo ID 1) -> {side}_shoulder_yaw        G (servo ID 8) -> {side}_wrist_yaw
    D (servo ID 5) -> {side}_elbow               H (servo ID 6) -> {side}_gripper (finger B follows by its mimic joint)

Same mapping as the MuJoCo backend (leader_mapping.py): leader angles map 1:1 from its calibrated hanging pose,
clamped to the URDF limits. Each arm starts at home (elbow bent 90 deg) and follows once its leader is within 15 deg
of home with the gripper open. Leader torque is always off; it is an input device only.

  R   reset the arms and every object in the scene; bring the leader back to home to resume

Recording (--record, src/robot_learning/config/pioneer_8dof_dataset_schema.yaml), 25 fps = every 4th physics step,
once every leader is at home. Keys S start, N save (then auto-reset), D discard, Esc stop. Per arm, left then right:
  observation.state  7 joints (rad) + gripper closure (0 open .. 1 closed)
  action             7 joint targets (rad) + leader gripper closure (0..1, continuous)
  observation.images.<name>  ego + the driven arms' wrist cameras, or --cameras ego,wrist_left / none
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

# pioneer_humanoid (canonical arm config) and humanoid_robot_learning (recording). Editable-installed
# in the image; this fallback keeps a bare bind-mounted checkout working.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pioneer_humanoid"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "robot_learning"))

from humanoid_robot_learning.recording.schema import ARM_SETS  # noqa: E402
from humanoid_robot_learning.sim.sim_teleop_record import (  # noqa: E402
    add_record_args,
    load_record_schema,
    make_sim_recorder,
)

from leader_mapping import (  # noqa: E402
    LeaderInput,
    LeaderMapping,
    WallClock,
    add_leader_args,
    check_leader_args,
)

parser = argparse.ArgumentParser(description="8-servo leader teleoperation of the Pioneer 8dof arms in Isaac Sim.")
add_leader_args(
    parser,
    scene_help="scene registered in humanoid_isaac_scenes (validated after launch; pass an unknown name to list them)",
)
parser.add_argument("--arms", choices=tuple(ARM_SETS), default="right",
                    help="arms to drive and record, one leader board each (leader_arm_8dof.json)")
parser.add_argument("--lock-wrist", action=argparse.BooleanOptionalAction, default=None,
                    help="pin the wrist (F and G) at home by its joint limits, leader's wrist ignored "
                         "(default: each arm's lock_wrist in leader_arm_8dof.json)")
add_record_args(parser, task_description="sim leader teleop demonstration")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
_sides = list(ARM_SETS[args_cli.arms])  # in ARM_ORDER: a recorded frame is the left arm's values, then the right's
if args_cli.record and args_cli.cameras is None:
    args_cli.cameras = ",".join(["ego"] + [f"wrist_{side}" for side in _sides])
_boards = check_leader_args(parser, args_cli, _sides)
_record = load_record_schema(parser, args_cli, arms=_sides, robot=args_cli.robot)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import numpy as np
import omni.appwindow
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from pioneer_humanoid import arm_8dof_params as params
from pioneer_humanoid.bimanual_arm_8dof import BIMANUAL_ARM_8DOF_CFG, make_camera_cfg
from humanoid_isaac_scenes import list_scenes, make_scene_cfg, scene_camera, scene_post_init


@dataclass
class Arm:
    """One leader driving one arm: its mapping and the robot's joint ids it reads and writes."""

    side: str
    leader: Any  # LeaderInput, opened inside run_simulator's try so a failed open still closes the others
    mapping: LeaderMapping
    arm_ids: list[int]
    grip_ids: list[int]
    grip_open: torch.Tensor  # finger position at closure 0
    grip_closed: torch.Tensor  # finger position at closure 1


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    recorder, record_every = make_sim_recorder(args_cli, _record, device=sim.device, sim_dt=sim_dt)
    if recorder is not None:
        print("[RECORD] Keys: S=start, N=save episode (then reset), D=discard")
        recorder.start_keyboard()

    scene.update(sim_dt)
    default_pos = robot.data.default_joint_pos.clone()
    default_vel = robot.data.default_joint_vel.clone()
    side_joints = {
        "left": (params.LEFT_ARM_JOINTS, params.LEFT_GRIPPER_JOINTS, params.LEFT_GRIPPER_OPEN, params.LEFT_GRIPPER_CLOSED),
        "right": (params.RIGHT_ARM_JOINTS, params.RIGHT_GRIPPER_JOINTS, params.RIGHT_GRIPPER_OPEN,
                  params.RIGHT_GRIPPER_CLOSED),
    }
    arms: list[Arm] = []
    for side in _sides:
        joints, fingers, grip_open, grip_closed = side_joints[side]
        board = _boards[side]
        lock_wrist = board.lock_wrist if args_cli.lock_wrist is None else args_cli.lock_wrist
        arm_ids = robot.find_joints(joints, preserve_order=True)[0]
        mapping = LeaderMapping(
            board.signs, args_cli.filter_alpha,
            home_rad=default_pos[0, arm_ids].tolist(),
            limits_rad=robot.data.joint_pos_limits[0, arm_ids].tolist(),
            locked=board.layout.wrist_servos if lock_wrist else (),
            layout=board.layout,
        )
        if lock_wrist:
            # Frozen wrist: its limits closed onto home, so PhysX holds it there rigidly.
            wrist_ids = arm_ids[len(arm_ids) - len(board.layout.wrist_servos):]
            home = default_pos[:, wrist_ids]
            robot.write_joint_position_limit_to_sim(torch.stack([home, home], dim=-1), joint_ids=wrist_ids)
        arms.append(Arm(
            side=side, leader=None, mapping=mapping, arm_ids=arm_ids,
            grip_ids=robot.find_joints(fingers, preserve_order=True)[0],
            grip_open=torch.tensor([[grip_open[j] for j in fingers]], device=sim.device),
            grip_closed=torch.tensor([[grip_closed[j] for j in fingers]], device=sim.device),
        ))
        print(f"[{side.upper()}] " * (len(_sides) > 1) + mapping.describe(joints), flush=True)

    # Called by tick() only on recorded frames.
    def read_images():
        return {name: scene[f"record_cam_{name}"].data.output["rgb"][0, ..., :3] for name in _record.images}

    reset_requested = {"v": False}

    def _on_kb(event, *_):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input == carb.input.KeyboardInput.R:
            reset_requested["v"] = True
        return True

    # Held in a variable: the subscription stops when it is garbage-collected.
    kb_sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), _on_kb
    )

    def reset_all():
        """Arms and every rigid object back to their defaults; the arms wait for the leaders at home."""
        robot.write_joint_state_to_sim(default_pos, default_vel)
        robot.reset()
        # Joints no leader drives (the other arm) hold their default pose, gripper open.
        robot.set_joint_position_target(default_pos)
        for obj in scene.rigid_objects.values():
            root = obj.data.default_root_state.clone()
            root[:, :3] += scene.env_origins
            obj.write_root_pose_to_sim(root[:, :7])
            obj.write_root_velocity_to_sim(root[:, 7:])
            obj.reset()
        for arm in arms:
            arm.mapping.reset()

    print("[INFO] Leader calibrated (hanging = 0). Move it to home: elbow bent 90 deg, forearm forward, gripper open. R = reset.", flush=True)

    reset_all()  # the arms spawn straight (every joint 0): start at home
    physics_step = 0
    clock = WallClock(sim_dt)
    try:
        for arm in arms:
            arm.leader = LeaderInput(_boards[arm.side], args_cli.baud, label=arm.side.upper() if len(arms) > 1 else "")
        while simulation_app.is_running():
            if recorder is not None and recorder.is_complete:
                print("[RECORD] Session complete.")
                break
            if reset_requested["v"]:
                reset_requested["v"] = False
                if recorder is not None and recorder.num_buffered_frames > 0:
                    # Frames either side of a reset are not one demo: drop the take, keep recording.
                    recorder.cancel_recording()
                    print("[RECORD] Reset mid-episode: take discarded, recording restarts from home.")
                reset_all()
                print("\n[LEADER] Arm and scene reset; bring the leader back to home.", flush=True)

            actions = []
            for arm in arms:
                target_list, grip = arm.mapping.update(arm.leader.read())
                robot.set_joint_position_target(torch.tensor([target_list], device=sim.device), joint_ids=arm.arm_ids)
                robot.set_joint_position_target(torch.lerp(arm.grip_open, arm.grip_closed, grip), joint_ids=arm.grip_ids)
                actions.extend([*target_list, grip])

            # Record one synchronized frame only once every active leader is at home: a take starts from home.
            if recorder is not None and all(arm.mapping.engaged for arm in arms) and physics_step % record_every == 0:
                states = []
                for arm in arms:
                    closure = ((robot.data.joint_pos[:, arm.grip_ids] - arm.grip_open) / (arm.grip_closed - arm.grip_open))
                    states.extend([*robot.data.joint_pos[0, arm.arm_ids].tolist(), closure.mean().clamp(0.0, 1.0).item()])
                saved = recorder.tick(np.asarray(actions, dtype=np.float32), np.asarray(states, dtype=np.float32),
                                      read_images)
                if saved:
                    reset_all()
                    print("[RECORD] Episode saved; arm and scene reset.", flush=True)

            scene.write_data_to_sim()
            sim.step()
            physics_step += 1
            scene.update(sim_dt)

            for arm in arms:
                if len(arms) == 1:
                    arm.leader.report(arm.mapping)
                elif not arm.mapping.engaged:
                    # Two leaders: a line each while homing (a shared \r status line would be overwritten).
                    arm.leader.report(arm.mapping, end="\n")
            clock.wait()
    finally:
        for arm in arms:
            if arm.leader is not None:
                arm.leader.close()
        print("\n[INFO] Stopped. Leader torque is OFF.", flush=True)
        # Also on errors: flushes episodes still being written in the background.
        if recorder is not None:
            recorder.finalize()
            print(f"[RECORD] Saved under {recorder.dataset_root}")


def main():
    if args_cli.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args_cli.scene!r}; available: {list_scenes()}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim.set_camera_view(*(scene_camera(args_cli.scene) or ([2.5, 2.5, 2.0], [0.0, 0.0, 0.8])))

    scene_cfg = make_scene_cfg(args_cli.scene, BIMANUAL_ARM_8DOF_CFG, num_envs=1, env_spacing=2.0)
    # Added after the robot: cameras are parented under it, and entities are created in order.
    unknown = sorted(set(_record.images) - set(params.CAMERAS))
    if unknown:
        raise SystemExit(f"{_record.path}: unknown images {unknown}; available: {list(params.CAMERAS)}")
    for name, spec in _record.images.items():
        setattr(scene_cfg, f"record_cam_{name}", make_camera_cfg(name, int(spec["height"]), int(spec["width"])))
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    post_init = scene_post_init(args_cli.scene)
    if post_init is not None:
        post_init(scene, sim)
    print(f"[INFO]: Setup complete. Move the leader arm to drive the {args_cli.arms} arm(s).")
    run_simulator(sim, scene)


def run():
    try:
        main()
    finally:
        simulation_app.close()
