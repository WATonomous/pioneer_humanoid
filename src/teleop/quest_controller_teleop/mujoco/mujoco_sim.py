"""Quest Touch controller teleoperation in MuJoCo: both arms, any humanoid_mujoco_scenes scene.

Same headset page, IK and controls as the Isaac backend (scene_driver.py): open
http://localhost:8444/ in Meta Browser after `adb reverse tcp:8444 tcp:8444`. The headset
shows the ego camera and both wrist cameras (mono; the stereo eye pair is Isaac-only).
Left Y (or 6 with the MuJoCo window focused) resets the arms and the scene.

Runs on a Mac without a GPU. The MuJoCo viewer window needs ``mjpython`` on macOS;
``--headless`` skips it and runs under plain ``python``.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR.parents[2]  # src/teleop/quest_controller_teleop/mujoco -> src
# pioneer_humanoid and humanoid_mujoco_scenes; this fallback keeps an uninstalled checkout working.
sys.path.insert(0, str(_SRC / "pioneer_humanoid"))
sys.path.insert(0, str(_SRC / "simulation" / "mujoco_scenes"))
sys.path.insert(0, str(_THIS_DIR.parent))  # the simulator-independent controller modules
sys.path.insert(0, str(_THIS_DIR))

from mujoco_cameras import CameraStream  # noqa: E402
from mujoco_overlay import draw_targets, show_text  # noqa: E402
from scene_driver import ControllerSceneDriver, add_controller_args, check_controller_args  # noqa: E402

CONTROL_DT = 0.01        # s, IK solve and actuator command period
CAMERA_PERIOD = 1 / 15   # s, headset camera refresh (rendered on its own thread, ~30 ms for all three)
CAMERA_SIZE = (360, 480)  # (height, width) px, 4:3 like the real ego and wrist cameras
HEADSET_CAMERAS = ("ego", "wrist_left", "wrist_right")
# Letters, 0-5, Esc and F1-F5 are MuJoCo viewer shortcuts; 6 matches the leader-arm teleop's reset key.
KEY_RESET = ord("6")  # GLFW key code


@dataclass
class ArmIds:
    """Where one arm's joints and actuators sit in the MuJoCo model."""

    qpos: list[int]
    arm_acts: list[int]
    grip_acts: list[int]
    grip_open: np.ndarray
    grip_closed: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quest Touch controller teleoperation in MuJoCo (both arms).")
    parser.add_argument("--scene", default="bimanual_sorting",
                        help="scene registered in humanoid_mujoco_scenes (an unknown name lists them)")
    parser.add_argument("--headless", action="store_true", help="no MuJoCo viewer window; the headset is the only view")
    add_controller_args(parser)
    # The ego camera looks 50 deg down, so its frame turns "hand up" into "wrist forward and up" and "hand back"
    # into "wrist back and up" (scripted-motion check). The upright base frame keeps up as up, forward as forward.
    parser.set_defaults(controller_frame="base")
    args = parser.parse_args()
    check_controller_args(parser, args)
    return args


def arm_ids(model, side: str) -> ArmIds:
    from pioneer_humanoid import arm_params as p

    joints, fingers, grip_open, grip_closed = {
        "left": (p.LEFT_ARM_JOINTS, p.LEFT_GRIPPER_JOINTS, p.LEFT_GRIPPER_OPEN, p.LEFT_GRIPPER_CLOSED),
        "right": (p.RIGHT_ARM_JOINTS, p.RIGHT_GRIPPER_JOINTS, p.RIGHT_GRIPPER_OPEN, p.RIGHT_GRIPPER_CLOSED),
    }[side]
    return ArmIds(
        qpos=[model.joint(j).qposadr[0] for j in joints],
        arm_acts=[model.actuator(j).id for j in joints],
        grip_acts=[model.actuator(j).id for j in fingers],
        grip_open=np.array([grip_open[j] for j in fingers]),
        grip_closed=np.array([grip_closed[j] for j in fingers]),
    )


def ego_frame(model, data) -> np.ndarray:
    """Ego camera orientation in base_link: the frame the headset image is seen in.

    MuJoCo and WebXR cameras share axes (+X right, +Y up, looking along -Z), so this is the
    same calibration the Isaac backend reads from its stereo camera prim.
    """
    base = data.xmat[model.body("base_link").id].reshape(3, 3)
    camera = data.cam_xmat[model.camera("ego").id].reshape(3, 3)
    return base.T @ camera


def make_driver(args, model, data) -> ControllerSceneDriver:
    from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS

    frame = ego_frame(model, data) if args.controller_frame == "camera" else None
    return ControllerSceneDriver.from_args(args, {"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS},
                                           frame_calibration=frame, scene="mujoco")


def run() -> None:
    args = parse_args()
    import mujoco
    from humanoid_mujoco_scenes import list_scenes, make_model, scene_camera, scene_reset, scene_step
    from pioneer_humanoid.mujoco_bimanual_arm import set_home

    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")
    model = make_model(args.scene, cameras={name: CAMERA_SIZE for name in HEADSET_CAMERAS})
    data = mujoco.MjData(model)
    scene_hook, randomise = scene_step(args.scene), scene_reset(args.scene)
    rng = np.random.default_rng()
    arms = {side: arm_ids(model, side) for side in ("left", "right")}

    def reset() -> None:
        mujoco.mj_resetData(model, data)
        set_home(model, data)
        if randomise is not None:
            randomise(model, data, rng)
        mujoco.mj_forward(model, data)

    reset()
    driver = make_driver(args, model, data)
    substeps = max(1, round(CONTROL_DT / model.opt.timestep))
    dt = substeps * model.opt.timestep
    cameras = CameraStream(model, driver, HEADSET_CAMERAS, CAMERA_SIZE)
    driver.start_thread()
    cameras.start()
    print(f"[Quest] MuJoCo scene {args.scene}: open http://localhost:{args.controller_port}/ in Meta Browser "
          f"(after adb reverse tcp:{args.controller_port} tcp:{args.controller_port}). Left Y or 6 in the window resets.", flush=True)
    if not args.headless:
        import mujoco.viewer
    reset_key = threading.Event()  # set by the viewer's key thread, consumed by the control loop

    def on_key(key: int) -> None:
        if key == KEY_RESET:
            reset_key.set()

    try:
        viewer_ctx = nullcontext() if args.headless else mujoco.viewer.launch_passive(model, data, key_callback=on_key)
        with viewer_ctx as viewer:
            if viewer is not None:
                for key, value in (scene_camera(args.scene) or {}).items():
                    setattr(viewer.cam, key, value)
            loop(model, data, driver, arms, cameras, viewer, reset, reset_key, scene_hook, substeps, dt)
    except KeyboardInterrupt:
        pass
    finally:
        cameras.close()
        driver.close()
        print("\n[Quest] Stopped.", flush=True)


def loop(model, data, driver, arms, cameras, viewer, reset, reset_key, scene_hook, substeps, dt) -> None:
    """Control ticks at dt until the viewer closes (or Ctrl+C): input, IK, physics, camera frames."""
    import mujoco

    def joint_positions():
        return {side: data.qpos[ids.qpos].copy() for side, ids in arms.items()}

    next_tick, next_camera, shown = time.monotonic(), 0., None
    if viewer is not None:
        show_text(viewer, "Press 6 to reset")
    while viewer is None or viewer.is_running():
        if driver.update(joint_positions()) or reset_key.is_set():  # left Y, or 6 in the viewer window
            reset_key.clear()
            reset()
            driver.reset(joint_positions())
            print("[Quest] Scene reset; arms back to home.", flush=True)
        commands = driver.solve(joint_positions(), dt)
        for side, ids in arms.items():
            data.ctrl[ids.arm_acts] = commands[side]
            data.ctrl[ids.grip_acts] = ids.grip_open + driver.grippers[side] * (ids.grip_closed - ids.grip_open)
        if scene_hook is not None:
            scene_hook(model, data)
        mujoco.mj_step(model, data, nstep=substeps)
        now = time.monotonic()
        if viewer is not None:
            with viewer.lock():
                draw_targets(viewer, model, data, driver, joint_positions())
            viewer.sync()
        if now >= next_camera:
            cameras.push(data)
            next_camera = now + CAMERA_PERIOD
        if driver.status != shown:
            shown = driver.status
            print(f"[Quest] {shown}", flush=True)
        next_tick = max(next_tick + dt, now - 0.1)  # catch up at most 0.1 s after a slow tick
        time.sleep(max(0., next_tick - time.monotonic()))


if __name__ == "__main__":
    run()
