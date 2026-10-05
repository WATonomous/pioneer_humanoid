"""Isaac Sim backend of pioneer_leader_arm_teleop.py: the 7-servo leader arm drives the Pioneer left arm (no IK).

    A (servo ID 2) -> joint1L (shoulder flexion)
    B (servo ID 3) -> joint2l (shoulder abduction)
    C (servo ID 1) -> joint3l (shoulder rotation)
    D (servo ID 5) -> joint4l (elbow flexion)
    E (servo ID 4) -> joint5l (forearm rotation)
    F (servo ID 7) -> joint6l (wrist)
    G (servo ID 6) -> gripper (starts open at 41.5 deg, closes toward 0)

Leader angles map 1:1 from its calibrated hanging pose (calibrate_leader.py), clamped to the URDF
limits. There is no home gate: the simulated arm follows every valid leader reading immediately.
Leader torque is always off; it is an input device only.

  R   reset task objects and snap the simulated left arm to the leader's current physical pose

Recording (--record, src/robot_learning/config/dataset_schema_pioneer_v1.yaml), 25 fps = every
4th physics step. Keys S start, N save (then auto-reset), D discard:
  observation.state  6 joints (rad) + gripper closure (0 open .. 1 closed, mean of both fingers)
  action             6 joint targets (rad) + leader gripper closure (0..1, continuous)
  observation.images.<name>  cameras enabled in the schema, or --cameras ego,wrist_left / none
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# pioneer_humanoid (canonical arm config) and humanoid_robot_learning (recording). Editable-installed
# in the image; this fallback keeps a bare bind-mounted checkout working.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pioneer_humanoid"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "robot_learning"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "simulation" / "isaac_scenes"))

from humanoid_robot_learning.sim_teleop_record import (  # noqa: E402
    add_record_args,
    load_record_schema,
    make_sim_recorder,
)

from leader_mapping import (  # noqa: E402
    WRIST_DAMPING,
    LeaderInput,
    LeaderMapping,
    WallClock,
    add_leader_args,
    check_leader_args,
)

parser = argparse.ArgumentParser(description="7-servo leader teleoperation of the Pioneer left arm in Isaac Sim.")
add_leader_args(
    parser,
    scene_help="scene registered in humanoid_isaac_scenes (validated after launch; pass an unknown name to list them)",
)
add_record_args(parser, task_description="sim leader teleop demonstration")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
check_leader_args(parser, args_cli)
_record = load_record_schema(parser, args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni.appwindow
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from pioneer_humanoid.bimanual_arm import (
    BIMANUAL_ARM_CFG,
    CAMERA_NAMES,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_CLOSED,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRIPPER_OPEN,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_GRIPPER_OPEN,
    apply_joint_limits,
    make_camera_cfg,
    resolve_joint_name,
)
from humanoid_isaac_scenes import list_scenes, make_scene_cfg, scene_camera, scene_post_init
from leader_ui import LeaderControlWindow, RecordingStatusWindow, WristCameraWindow


def _joint_ids(robot, names: list[str]) -> list[int]:
    name_to_id = {name: i for i, name in enumerate(robot.data.joint_names)}
    return [name_to_id[resolve_joint_name(robot, name)] for name in names]


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    recorder, record_every = make_sim_recorder(args_cli, _record, device=sim.device, sim_dt=sim_dt)
    if recorder is not None:
        print("[RECORD] Keys: S=start, N=save episode (then reset), D=discard")
        recorder.start_keyboard()

    # Populate robot buffers, then write the URDF limits the targets are clamped to.
    scene.update(sim_dt)
    apply_joint_limits(robot)

    arm_ids = _joint_ids(robot, LEFT_ARM_JOINTS)
    gripper_ids = _joint_ids(robot, LEFT_GRIPPER_JOINTS)
    held_arm_ids = _joint_ids(robot, RIGHT_ARM_JOINTS)
    held_gripper_ids = _joint_ids(robot, RIGHT_GRIPPER_JOINTS)

    default_pos = robot.data.default_joint_pos.clone()
    default_vel = robot.data.default_joint_vel.clone()
    held_arm_default = default_pos[:, held_arm_ids].clone()
    mapping = LeaderMapping(
        args_cli,
        home_rad=default_pos[0, arm_ids].tolist(),
        limits_rad=robot.data.joint_pos_limits[0, arm_ids].tolist(),
    )

    gripper_open = torch.tensor([[LEFT_GRIPPER_OPEN[j] for j in LEFT_GRIPPER_JOINTS]], device=sim.device)
    gripper_closed = torch.tensor([[LEFT_GRIPPER_CLOSED[j] for j in LEFT_GRIPPER_JOINTS]], device=sim.device)
    held_gripper_open = torch.tensor([[RIGHT_GRIPPER_OPEN[j] for j in RIGHT_GRIPPER_JOINTS]], device=sim.device)
    zero_gripper_vel = torch.zeros(1, len(gripper_ids), device=sim.device)

    print(mapping.describe(LEFT_ARM_JOINTS), flush=True)

    # Called by tick() only on recorded frames.
    def read_images():
        return {name: scene[f"record_cam_{name}"].data.output["rgb"][0, ..., :3] for name in _record.images}

    reset_requested = {"v": False}

    def _on_kb(event, *_):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input == carb.input.KeyboardInput.R:
            reset_requested["v"] = True
        return True

    leader = LeaderInput(args_cli)
    # Held in a variable: the subscription stops when it is garbage-collected.
    kb_sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), _on_kb
    )

    def reset_all(angles=None):
        """Reset scene objects while keeping the simulated left arm matched to the leader."""
        target_list, grip = mapping.snap(leader.read() if angles is None else angles)
        matched_pos = default_pos.clone()
        matched_pos[:, arm_ids] = torch.tensor([target_list], device=sim.device)
        matched_pos[:, gripper_ids] = torch.lerp(gripper_open, gripper_closed, grip)
        robot.write_joint_state_to_sim(matched_pos, default_vel)
        robot.reset()
        for obj in scene.rigid_objects.values():
            root = obj.data.default_root_state.clone()
            root[:, :3] += scene.env_origins
            obj.write_root_pose_to_sim(root[:, :7])
            obj.write_root_velocity_to_sim(root[:, 7:])
            obj.reset()
        return target_list, grip

    # Do not spawn at the URDF's bent-elbow default.  At launch, the simulated
    # left arm starts at the current calibrated physical leader pose.
    initial_angles = leader.read()
    initial_targets, initial_grip = reset_all(initial_angles)
    controls = LeaderControlWindow(LEFT_ARM_JOINTS, mapping.signs, initial_targets, initial_grip)
    controls.update(initial_angles, initial_targets, initial_grip)
    record_status = RecordingStatusWindow() if recorder is not None else None
    if record_status is not None:
        record_status.update(recording=False, saved=recorder.num_recorded_episodes, frames=0, pending=0)
    wrist_preview = None
    wrist_spec = _record.images.get("wrist_left")
    if wrist_spec is not None:
        wrist_preview = WristCameraWindow(int(wrist_spec["width"]), int(wrist_spec["height"]))
    print(
        "[INFO] Direct leader matching active (straight/resting calibration = 0). "
        "R resets the scene and keeps the arm matched to the current physical pose. "
        "Live direction/default controls are open in a separate window.",
        flush=True,
    )

    physics_step = 0
    clock = WallClock(sim_dt)
    try:
        while simulation_app.is_running():
            angles = leader.read()
            if reset_requested["v"]:
                reset_requested["v"] = False
                if recorder is not None and recorder.num_buffered_frames > 0:
                    # Frames either side of a reset are not one demo: drop the take, keep recording.
                    recorder.cancel_recording()
                    print("[RECORD] Reset mid-episode: take discarded.")
                reset_all(angles)
                print("\n[LEADER] Scene reset; arm matched to the current physical leader pose.", flush=True)

            action = controls.pop_action()
            while action is not None:
                kind = action[0]
                try:
                    if kind == "directions":
                        mapping.set_directions(angles, action[1])
                        controls.set_status("Directions applied. Current simulated pose was preserved.")
                    elif kind == "defaults":
                        mapping.set_directions(angles, action[1])
                        mapping.set_current_pose_defaults(angles, action[2], action[3])
                        controls.set_status("Held physical pose now maps to the displayed defaults.")
                    elif kind == "clear":
                        mapping.set_directions(angles, action[1])
                        mapping.clear_runtime_zero(angles)
                        controls.set_status("Live zero cleared; using the saved encoder calibration.")
                except ValueError as exc:
                    controls.set_status(f"Could not apply settings: {exc}")
                action = controls.pop_action()

            target_list, grip = mapping.update(angles)
            target = torch.tensor([target_list], device=sim.device)

            robot.set_joint_position_target(target, joint_ids=arm_ids)
            # Coupled fingers: high stiffness + zero velocity target stops bounce on the move.
            robot.set_joint_position_target(torch.lerp(gripper_open, gripper_closed, grip), joint_ids=gripper_ids)
            robot.set_joint_velocity_target(zero_gripper_vel, joint_ids=gripper_ids)
            # Hold the other arm at its default pose, its gripper open.
            robot.set_joint_position_target(held_arm_default, joint_ids=held_arm_ids)
            robot.set_joint_position_target(held_gripper_open, joint_ids=held_gripper_ids)
            robot.set_joint_velocity_target(zero_gripper_vel, joint_ids=held_gripper_ids)

            # Record at the schema rate while an episode is active.
            if recorder is not None and mapping.engaged and physics_step % record_every == 0:
                finger_q = robot.data.joint_pos[:, gripper_ids]
                closure = (
                    ((finger_q - gripper_open) / (gripper_closed - gripper_open))
                    .mean(dim=-1)
                    .clamp(0.0, 1.0)
                )
                obs = torch.cat([robot.data.joint_pos[0, arm_ids], closure])
                act = torch.cat([target[0], torch.tensor([grip], device=sim.device)])
                saved = recorder.tick(
                    act.detach().cpu().numpy().astype("float32"),
                    obs.detach().cpu().numpy().astype("float32"),
                    read_images,
                )
                if saved:
                    reset_all()
                    print("[RECORD] Episode saved; scene reset and arm matched to leader.", flush=True)

            scene.write_data_to_sim()
            sim.step()
            physics_step += 1
            scene.update(sim_dt)

            # Update the lightweight UI at 25 Hz.  The wrist image reuses the recording camera's
            # existing tensor, so this does not create a second renderer or alter saved frames.
            if physics_step % 4 == 0:
                controls.update(angles, target_list, grip)
                if record_status is not None:
                    record_status.update(
                        recording=recorder.is_recording,
                        saved=recorder.num_recorded_episodes,
                        frames=recorder.num_buffered_frames,
                        pending=recorder.num_pending_episodes,
                    )
                if wrist_preview is not None:
                    wrist_preview.update(scene["record_cam_wrist_left"].data.output["rgb"][0])

            leader.report(mapping)
            clock.wait()
    finally:
        controls.close()
        if record_status is not None:
            record_status.close()
        if wrist_preview is not None:
            wrist_preview.close()
        leader.close()
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

    robot_cfg = BIMANUAL_ARM_CFG.replace(
        actuators={
            **BIMANUAL_ARM_CFG.actuators,
            "left_wrist": BIMANUAL_ARM_CFG.actuators["left_wrist"].replace(damping=WRIST_DAMPING),
        }
    )
    scene_cfg = make_scene_cfg(args_cli.scene, robot_cfg, num_envs=1, env_spacing=2.0)
    # Added after the robot: cameras are parented under it, and entities are created in order.
    unknown = sorted(set(_record.images) - set(CAMERA_NAMES))
    if unknown:
        raise SystemExit(f"{_record.path}: unknown images {unknown}; available: {list(CAMERA_NAMES)}")
    for name, spec in _record.images.items():
        setattr(scene_cfg, f"record_cam_{name}", make_camera_cfg(name, int(spec["height"]), int(spec["width"])))
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    post_init = scene_post_init(args_cli.scene)
    if post_init is not None:
        post_init(scene, sim)
    print("[INFO]: Setup complete. Move the leader arm to drive the left arm.")
    run_simulator(sim, scene)


def run():
    try:
        main()
    finally:
        simulation_app.close()
