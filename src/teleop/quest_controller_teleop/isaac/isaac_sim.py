"""Quest Touch controller teleoperation in Isaac Sim: both arms, any humanoid_isaac_scenes scene.

The same controller as mujoco/mujoco_sim.py (scene_driver.py: mink IK, input filter, buttons), on Kit's asyncio loop.
Open http://localhost:8444/ in Meta Browser after `adb reverse tcp:8444 tcp:8444` on the host (the simulation_isaac
container shares the host network). Left Y (or 6 in the viewport) resets the arms and the scene. --record: right B
starts / saves a take (or 7 / 8; 9 discards).

The local window is Isaac's viewport (25 fps); status lines print on the terminal. ``--headless`` shows nothing.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR.parents[2]  # src/teleop/quest_controller_teleop/isaac -> src
# pioneer_humanoid, humanoid_isaac_scenes and humanoid_robot_learning; this fallback keeps an uninstalled checkout working.
sys.path.insert(0, str(_SRC / "pioneer_humanoid"))
sys.path.insert(0, str(_SRC / "simulation" / "isaac_scenes"))
sys.path.insert(0, str(_SRC / "robot_learning"))
sys.path.insert(0, str(_THIS_DIR.parent))  # the simulator-independent controller modules

from scene_driver import ControllerSceneDriver, add_controller_args, check_controller_args  # noqa: E402
from humanoid_robot_learning.sim.sim_teleop_record import (  # noqa: E402
    add_record_args, load_record_schema, make_sim_recorder)

CONTROL_DT = 0.01        # s, IK solve, actuator command and physics step period
CAMERAS = ("ego", "wrist_left", "wrist_right")
SIZE = (480, 640)        # (height, width) of the ego camera when only its frame is needed (--controller-frame camera)


@dataclass
class ArmIds:
    """Where one arm's joints sit in the Isaac articulation."""

    joints: list[int]
    grip: list[int]
    grip_open: "torch.Tensor"
    grip_closed: "torch.Tensor"


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Quest Touch controller teleoperation in Isaac Sim (both arms).")
    parser.add_argument("--scene", default="bare",
                        help="scene registered in humanoid_isaac_scenes (an unknown name lists them)")
    add_controller_args(parser)
    add_record_args(parser, task_description="Quest VR teleop demonstration")
    parser.set_defaults(controller_frame="base")   # as mujoco_sim.py: the ego camera looks 50 deg down
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    check_controller_args(parser, args)
    if args.record and args.cameras is None:
        args.cameras = ",".join(CAMERAS)   # both arms move: record both wrists
    record = load_record_schema(parser, args, arms=["left", "right"], robot="8dof")
    if args.controller_frame == "camera":
        args.enable_cameras = True   # ego_frame reads the ego camera prim
    return args, record


class Recording:
    """Right B (or 7) starts a take and saves it (or 8); left Y, 6 or 9 discards it. Frames: 25 Hz of sim time."""

    def __init__(self, recorder, every, driver):
        self.recorder, self.every, self.driver = recorder, every, driver
        self.flags = recorder.start_flags()
        self.presses = self.saved_takes = 0

    def toggle(self) -> None:
        self.save() if self.flags.start else self.start()

    def start(self) -> None:
        if not self.flags.start:
            self.flags.start = True
            self.driver.status = "Recording: right B saves, left Y discards"

    def save(self) -> None:
        if self.flags.start:
            self.flags.success = True   # recorder.tick saves at the next sample, then the loop resets the scene

    def discard(self) -> None:
        if self.flags.start:
            self.flags.start = False
            self.recorder.cancel_recording()
            self.driver.status = "Take discarded. Right B records a take."

    def saved(self) -> None:
        self.saved_takes += 1
        self.driver.status = f"Saved take {self.saved_takes}. Right B records the next."


def arm_ids(robot, side: str) -> ArmIds:
    import torch
    from pioneer_humanoid import arm_8dof_params as p

    joints, fingers, grip_open, grip_closed = {
        "left": (p.LEFT_ARM_JOINTS, p.LEFT_GRIPPER_JOINTS, p.LEFT_GRIPPER_OPEN, p.LEFT_GRIPPER_CLOSED),
        "right": (p.RIGHT_ARM_JOINTS, p.RIGHT_GRIPPER_JOINTS, p.RIGHT_GRIPPER_OPEN, p.RIGHT_GRIPPER_CLOSED),
    }[side]
    return ArmIds(
        joints=robot.find_joints(joints, preserve_order=True)[0],
        grip=robot.find_joints(fingers, preserve_order=True)[0],
        grip_open=torch.tensor([[grip_open[j] for j in fingers]], device=robot.device),
        grip_closed=torch.tensor([[grip_closed[j] for j in fingers]], device=robot.device),
    )


def arm_targets(robot, ids: ArmIds, command, velocity, bias) -> None:
    """Position, velocity and torque targets that make the implicit PD act like the motor's MIT mode.

    PhysX's drive applies kp (q* - q) + kd (qd* - qdot) and takes the velocity target as given, so unlike MuJoCo's
    position actuator no shift of the position target is needed for the velocity feedforward. It has no gravity
    compensation either: `bias` (gravity and Coriolis torques, all joints) goes in as the effort target, as the MIT
    mode's torque feedforward.
    """
    import torch

    q = torch.as_tensor(np.asarray([command]), dtype=torch.float32, device=robot.device)
    qd = torch.as_tensor(np.asarray([velocity]), dtype=torch.float32, device=robot.device)
    robot.set_joint_position_target(q, joint_ids=ids.joints)
    robot.set_joint_velocity_target(qd, joint_ids=ids.joints)
    robot.set_joint_effort_target(bias[:, ids.joints], joint_ids=ids.joints)


def ego_frame(robot, camera) -> np.ndarray:
    """Ego camera orientation in base_link: the frame the headset image is seen in.

    USD (OpenGL) and WebXR cameras share axes (+X right, +Y up, looking along -Z), as MuJoCo's: the same calibration
    as mujoco_sim.ego_frame.
    """
    from isaaclab.utils.math import matrix_from_quat

    base = matrix_from_quat(robot.data.body_quat_w[0, robot.find_bodies("base_link")[0][0]])
    return (base.T @ matrix_from_quat(camera.data.quat_w_opengl[0])).cpu().double().numpy()


def make_driver(args, robot, scene) -> ControllerSceneDriver:
    from pioneer_humanoid.arm_8dof_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS

    frame = ego_frame(robot, scene["cam_ego"]) if args.controller_frame == "camera" else None
    return ControllerSceneDriver.from_args(args, {"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS},
                                           frame_calibration=frame)


def run() -> None:
    args, record = parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    try:
        main(args, record, app)
    finally:
        app.close()


def main(args, record, app) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from humanoid_isaac_scenes import list_scenes, make_scene_cfg, scene_camera, scene_post_init
    from pioneer_humanoid.bimanual_arm_8dof import BIMANUAL_ARM_8DOF_CFG, make_camera_cfg

    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=CONTROL_DT, device=args.device))
    sim.set_camera_view(*(scene_camera(args.scene) or ([2.5, 2.5, 2.0], [0.0, 0.0, 0.8])))
    scene_cfg = make_scene_cfg(args.scene, BIMANUAL_ARM_8DOF_CFG)
    sizes = {name: (int(spec["height"]), int(spec["width"])) for name, spec in record.images.items()}
    if args.controller_frame == "camera":
        sizes.setdefault("ego", SIZE)
    for name, (height, width) in sizes.items():   # after the robot: cameras are parented under it
        setattr(scene_cfg, f"cam_{name}", make_camera_cfg(name, height, width))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    post_init = scene_post_init(args.scene)
    if post_init is not None:
        post_init(scene, sim)
    robot = scene["robot"]
    arms = {side: arm_ids(robot, side) for side in ("left", "right")}

    def reset() -> None:
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.set_joint_velocity_target(robot.data.default_joint_vel)
        for obj in scene.rigid_objects.values():
            root = obj.data.default_root_state.clone()
            root[:, :3] += scene.env_origins
            obj.write_root_state_to_sim(root)
        scene.reset()
        scene.write_data_to_sim()
        sim.step(render=True)   # the cameras see the reset scene
        scene.update(CONTROL_DT)

    reset()
    driver = make_driver(args, robot, scene)
    driver.start()   # the web server on Kit's asyncio loop; driver.update pumps it every tick
    print(f"[Quest] Isaac scene {args.scene}: open http://localhost:{args.controller_port}/ in Meta Browser "
          f"(after adb reverse tcp:{args.controller_port} tcp:{args.controller_port}). Left Y or 6 in the viewport resets.",
          flush=True)
    # CPU buffers as mujoco_sim.py: 120 s of three 640x480 cameras is 8 GB, more than the GPU has beside Isaac
    recorder, every = make_sim_recorder(args, record, device="cpu", sim_dt=CONTROL_DT)
    recording = Recording(recorder, every, driver) if recorder is not None else None
    reset_key = {"set": False}
    keys = {"KEY_6": lambda: reset_key.update(set=True)}
    if recording is not None:
        keys.update({"KEY_7": recording.start, "KEY_8": recording.save, "KEY_9": recording.discard})
        print(f"[RECORD] Right B (or 7) records a take, again (or 8) saves it; left Y, 6 or 9 discards. "
              f"Saving to {recorder.dataset_root}", flush=True)
    keyboard = None if args.headless else subscribe_keys(keys)   # held: the subscription ends when collected
    try:
        loop(app, sim, scene, robot, driver, arms, recording, reset, reset_key, every or 4, sizes)
    except KeyboardInterrupt:
        pass
    finally:
        del keyboard
        driver.close()
        if recorder is not None:
            recorder.finalize()
            print(f"[RECORD] {recorder.num_recorded_episodes} takes saved under {recorder.dataset_root}", flush=True)
        print("\n[Quest] Stopped.", flush=True)


def subscribe_keys(keys):
    """Viewport key presses: {carb KeyboardInput name: callback}."""
    import carb
    import omni.appwindow

    def on_key(event, *_):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name in keys:
            keys[event.input.name]()
        return True

    return carb.input.acquire_input_interface().subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), on_key)


def loop(app, sim, scene, robot, driver, arms, recording, reset, reset_key, every, sizes) -> None:
    """Control ticks at CONTROL_DT until the app closes (Ctrl+C): input, IK, physics; rendering every `every` ticks."""
    import torch

    def joint_positions():
        q = robot.data.joint_pos[0].cpu().double().numpy()
        return {side: q[ids.joints].copy() for side, ids in arms.items()}

    def images():
        return {name: scene[f"cam_{name}"].data.output["rgb"][0, ..., :3] for name in sizes}

    next_tick, shown, tick = time.monotonic(), None, 0
    previous = joint_positions()
    while app.is_running():
        if recording is not None and recording.recorder.is_complete:
            print("[RECORD] Session complete.", flush=True)
            break
        reset_now = driver.update(joint_positions()) or reset_key["set"]   # left Y, or 6 in the viewport
        if recording is not None and driver.record_presses != recording.presses:
            recording.presses = driver.record_presses
            recording.toggle()
        if reset_now:
            reset_key["set"] = False
            if recording is not None:
                recording.discard()   # frames either side of a reset are not one demo
            reset()
            driver.reset(joint_positions())
            previous = joint_positions()
            tick = 0
            print("[Quest] Scene reset; arms back to home.", flush=True)
        commands = driver.solve(joint_positions(), CONTROL_DT)
        view = robot.root_physx_view
        bias = view.get_gravity_compensation_forces() + view.get_coriolis_and_centrifugal_compensation_forces()
        for side, ids in arms.items():
            arm_targets(robot, ids, commands[side], (commands[side] - previous[side]) / CONTROL_DT, bias)
            previous[side] = commands[side]
            robot.set_joint_position_target(torch.lerp(ids.grip_open, ids.grip_closed, driver.grippers[side]),
                                            joint_ids=ids.grip)
        if recording is not None and tick % every == 0:
            # Before the step: the images were rendered at the last step, so image, state and action are of the same
            # instant. Left arm first, as the schema.
            q = robot.data.joint_pos[0].cpu().double().numpy()
            action, state = [], []
            for side, ids in arms.items():
                closure = (q[ids.grip] - ids.grip_open[0].cpu().numpy()) / (ids.grip_closed - ids.grip_open)[0].cpu().numpy()
                action.extend([*commands[side], driver.grippers[side]])
                state.extend([*q[ids.joints], float(np.clip(closure.mean(), 0., 1.))])
            if recording.recorder.tick(np.asarray(action, np.float32), np.asarray(state, np.float32), images):
                recording.saved()
                reset_key["set"] = True
        tick += 1
        scene.write_data_to_sim()
        sim.step(render=tick % every == 0)
        scene.update(CONTROL_DT)
        if driver.status != shown:
            shown = driver.status
            print(f"[Quest] {shown}", flush=True)
        now = time.monotonic()
        next_tick = max(next_tick + CONTROL_DT, now - 0.1)  # catch up at most 0.1 s after a slow tick
        time.sleep(max(0., next_tick - time.monotonic()))


if __name__ == "__main__":
    run()
