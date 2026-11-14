"""Quest Touch controller teleoperation in MuJoCo: both arms, any humanoid_mujoco_scenes scene.

Open http://localhost:8444/ in Meta Browser after `adb reverse tcp:8444 tcp:8444` (start_quest.sh does it all).
The headset shows the ego camera and both wrist cameras (mono); --no-headset-video makes it only track.
Left Y (or 6 with the window focused) resets the arms and the scene. --record: right B starts / saves a take;
--record_arms left|right records one arm (8 values) for single-arm tasks.

This machine shows one camera window (ego on top, both wrist cameras below; plain ``python``);
``--headless`` shows nothing. Runs on a Mac without a GPU.
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(_SRC / "robot_learning"))
sys.path.insert(0, str(_THIS_DIR.parent))  # the simulator-independent controller modules
sys.path.insert(0, str(_THIS_DIR))

from mujoco_cameras import CameraStream, CameraWindow  # noqa: E402
from scene_driver import ControllerSceneDriver, add_controller_args, check_controller_args  # noqa: E402
from humanoid_robot_learning.recording.schema import ARM_SETS  # noqa: E402
from humanoid_robot_learning.sim.sim_teleop_record import (  # noqa: E402
    add_record_args, load_record_schema, make_sim_recorder)

CONTROL_DT = 0.01        # s, IK solve and actuator command period
CAMERA_PERIOD = 1 / 15   # s, headset camera refresh (rendered on its own thread, ~30 ms for all three)
CAMERA_SIZE = (360, 480)  # (height, width) px, 4:3 like the real ego and wrist cameras
HEADSET_CAMERAS = ("ego", "wrist_left", "wrist_right")


@dataclass
class ArmIds:
    """Where one arm's joints and actuators sit in the MuJoCo model."""

    qpos: list[int]
    dofs: list[int]
    arm_acts: list[int]
    grip_acts: list[int]
    grip_qpos: list[int]
    grip_open: np.ndarray
    grip_closed: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(description="Quest Touch controller teleoperation in MuJoCo (both arms).")
    parser.add_argument("--scene", default="bimanual_sorting",
                        help="scene registered in humanoid_mujoco_scenes (an unknown name lists them)")
    parser.add_argument("--headless", action="store_true", help="no window on this machine; the headset is the only view")
    parser.add_argument("--no-headset-video", action="store_true",
                        help="headset only tracks the controllers: no camera rendering, so the GPU is the window's")
    add_controller_args(parser)
    add_record_args(parser, task_description="Quest VR teleop demonstration")
    parser.add_argument("--record_arms", choices=tuple(ARM_SETS), default="bimanual",
                        help="arms to record (both are always driven): left or right records that arm's 8 values "
                             "and its wrist camera into datasets/pioneer_8dof_<arm>_arm, for single-arm tasks")
    # The ego camera looks 50 deg down, so its frame turns "hand up" into "wrist forward and up" and "hand back"
    # into "wrist back and up" (scripted-motion check). The upright base frame keeps up as up, forward as forward.
    parser.set_defaults(controller_frame="base")
    args = parser.parse_args()
    check_controller_args(parser, args)
    sides = ARM_SETS[args.record_arms]
    if args.record and args.cameras is None:
        args.cameras = ",".join(["ego", *(f"wrist_{side}" for side in sides)])   # the recorded arms' wrists
    return args, load_record_schema(parser, args, arms=list(sides), robot="8dof")


class Recording:
    """Right B (or 7) starts a take and saves it (or 8); left Y, 6 or 9 discards it. Frames: 25 Hz of sim time.

    `sides`: the recorded arms (--record_arms), left first as the schema."""

    def __init__(self, recorder, window, every, driver, sides):
        self.recorder, self.window, self.every, self.driver, self.sides = recorder, window, every, driver, sides
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
            self.flags.success = True   # the render thread saves at the next sample, then resets the scene

    def discard(self) -> None:
        if self.flags.start:
            self.flags.start = False
            self.window.discard()
            self.driver.status = "Take discarded. Right B records a take."

    def saved(self) -> None:
        self.saved_takes += 1
        self.driver.status = f"Saved take {self.saved_takes}. Right B records the next."

    def status(self) -> str:
        if self.flags.start:
            return f"REC {self.recorder.num_buffered_frames / self.recorder.fps:.1f} s"
        return f"{self.saved_takes} saved"


def arm_ids(model, side: str) -> ArmIds:
    from pioneer_humanoid import arm_8dof_params as p

    joints, fingers, grip_open, grip_closed = {
        "left": (p.LEFT_ARM_JOINTS, p.LEFT_GRIPPER_JOINTS, p.LEFT_GRIPPER_OPEN, p.LEFT_GRIPPER_CLOSED),
        "right": (p.RIGHT_ARM_JOINTS, p.RIGHT_GRIPPER_JOINTS, p.RIGHT_GRIPPER_OPEN, p.RIGHT_GRIPPER_CLOSED),
    }[side]
    return ArmIds(
        qpos=[model.joint(j).qposadr[0] for j in joints],
        dofs=[model.joint(j).dofadr[0] for j in joints],
        arm_acts=[model.actuator(j).id for j in joints],
        grip_acts=[model.actuator(j).id for j in fingers],
        grip_qpos=[model.joint(j).qposadr[0] for j in fingers],
        grip_open=np.array([grip_open[j] for j in fingers]),
        grip_closed=np.array([grip_closed[j] for j in fingers]),
    )


def arm_ctrl(model, data, ids: ArmIds, command, velocity) -> np.ndarray:
    """Position targets that make MuJoCo's PD act like the motor's MIT mode with velocity and torque feedforward.

    A position actuator applies kp (ctrl - q) - kv qdot: it damps against zero velocity, so a moving target lags by
    kv/kp (~80 ms on these gains) and gravity sags it ~2 cm. Shifting ctrl by (kv velocity + bias) / kp gives
    kp (q* - q) + kv (qd* - qdot) + gravity/Coriolis, still clipped to the actuator's effort limit.
    """
    kp = model.actuator_gainprm[ids.arm_acts, 0]
    kv = -model.actuator_biasprm[ids.arm_acts, 2]
    return command + (kv * velocity + data.qfrc_bias[ids.dofs]) / kp


def ego_frame(model, data) -> np.ndarray:
    """Ego camera orientation in base_link: the frame the headset image is seen in.

    MuJoCo and WebXR cameras share axes (+X right, +Y up, looking along -Z), so this is the
    same calibration the Isaac backend reads from its stereo camera prim.
    """
    base = data.xmat[model.body("base_link").id].reshape(3, 3)
    camera = data.cam_xmat[model.camera("ego").id].reshape(3, 3)
    return base.T @ camera


def make_driver(args, model, data) -> ControllerSceneDriver:
    from pioneer_humanoid.arm_8dof_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS

    frame = ego_frame(model, data) if args.controller_frame == "camera" else None
    return ControllerSceneDriver.from_args(args, {"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS},
                                           frame_calibration=frame, scene="mujoco")


def run() -> None:
    args, record = parse_args()
    import mujoco
    from humanoid_mujoco_scenes import list_scenes, make_model, scene_progress, scene_reset, scene_step
    from pioneer_humanoid.mujoco_bimanual_arm_8dof import set_home

    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")
    # The model's offscreen buffer must fit the window's (and recording's) 640x480 renders; the headset's are smaller.
    model = make_model(args.scene, cameras={name: CameraWindow.SIZE for name in CameraWindow.CAMERAS}, robot="8dof")
    data = mujoco.MjData(model)
    scene_hook, randomise = scene_step(args.scene), scene_reset(args.scene)
    steps = scene_progress(args.scene)   # multi-step task: (index, total, instruction), or None
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
    cameras = None if args.no_headset_video else CameraStream(model, driver, HEADSET_CAMERAS, CAMERA_SIZE)
    if cameras is not None:
        cameras.start()
    print(f"[Quest] MuJoCo scene {args.scene}: open http://localhost:{args.controller_port}/ in Meta Browser "
          f"(after adb reverse tcp:{args.controller_port} tcp:{args.controller_port}). Left Y or 6 in the window resets.", flush=True)
    reset_key = threading.Event()  # set by the window's key handler, consumed by the control loop
    stop = threading.Event()
    # Multi-step scenes record the current step's instruction as each frame's task, plus its index (as the leader
    # teleop does): sim evaluation prompts the policy with the same per-step instruction.
    extra = {"subtask_index": ["subtask_index"]} if steps is not None else None
    recorder, every = make_sim_recorder(args, record, device="cpu", sim_dt=dt, extra_features=extra)
    sizes = {name: (int(spec["height"]), int(spec["width"])) for name, spec in record.images.items()}
    window = None if args.headless and recorder is None else CameraWindow(model, sizes, recorder)
    recording = Recording(recorder, window, every, driver, ARM_SETS[args.record_arms]) if recorder is not None else None
    keys = {"6": reset_key.set}
    if recording is not None:
        window.on_saved = lambda: (recording.saved(), reset_key.set())
        keys.update({"7": recording.start, "8": recording.save, "9": recording.discard})
        print(f"[RECORD] Right B (or 7) records a take, again (or 8) saves it; left Y, 6 or 9 discards. "
              f"Saving to {recorder.dataset_root}", flush=True)

    def control() -> None:
        driver.start_thread()   # the driver is owned by the thread that starts it: start, loop and close here
        try:
            loop(model, data, driver, arms, cameras, window, recording, reset, reset_key, scene_hook, steps, substeps,
                 dt, stop)
        finally:
            stop.set()
            driver.close()

    # The window must live on the main thread (macOS), so the control loop runs on its own thread.
    worker = threading.Thread(target=control, name="control", daemon=True)
    worker.start()
    try:
        if not args.headless:
            window.show(stop, worker.is_alive, keys)
            stop.set()   # window closed: stop the sim
        while worker.is_alive():
            worker.join(timeout=.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(timeout=2.)
        if cameras is not None:
            cameras.close()
        if window is not None:
            window.close()   # renders any queued recorded frames first
        if recorder is not None:
            recorder.finalize()
            print(f"[RECORD] {recorder.num_recorded_episodes} takes saved under {recorder.dataset_root}", flush=True)
        print("\n[Quest] Stopped.", flush=True)


def step_status(steps, model, data) -> str:
    """The operator's view of a multi-step task: the current step, or that all are done."""
    if steps is None:
        return ""
    index, total, instruction = steps(model, data)
    return "all steps done: save the take" if index >= total else f"step {index + 1}/{total}: {instruction}"


def loop(model, data, driver, arms, cameras, window, recording, reset, reset_key, scene_hook, steps, substeps, dt,
         stop) -> None:
    """Control ticks at dt until `stop` is set (window closed, Ctrl+C): input, IK, physics, camera frames.

    `steps`: the scene's progress function (multi-step tasks), or None."""
    import mujoco

    def joint_positions():
        return {side: data.qpos[ids.qpos].copy() for side, ids in arms.items()}

    next_tick, next_camera, shown, shown_step, tick = time.monotonic(), 0., None, None, 0
    previous = {side: data.qpos[ids.qpos].copy() for side, ids in arms.items()}
    while not stop.is_set():
        if recording is not None and recording.recorder.is_complete:
            print("[RECORD] Session complete.", flush=True)
            break
        reset_now = driver.update(joint_positions()) or reset_key.is_set()   # left Y, or 6 in the window
        if recording is not None and driver.record_presses != recording.presses:
            recording.presses = driver.record_presses
            recording.toggle()
        if reset_now:
            reset_key.clear()
            if recording is not None:
                recording.discard()   # frames either side of a reset are not one demo
            reset()
            driver.reset(joint_positions())
            previous = joint_positions()
            print("[Quest] Scene reset; arms back to home.", flush=True)
        commands = driver.solve(joint_positions(), dt)
        for side, ids in arms.items():
            data.ctrl[ids.arm_acts] = arm_ctrl(model, data, ids, commands[side], (commands[side] - previous[side]) / dt)
            previous[side] = commands[side]
            data.ctrl[ids.grip_acts] = ids.grip_open + driver.grippers[side] * (ids.grip_closed - ids.grip_open)
        if recording is not None and recording.flags.start and tick % recording.every == 0:
            # Before the step: image, state and action are of the same instant. Left arm first, as the schema.
            action, state = [], []
            for side in recording.sides:
                ids = arms[side]
                closure = (data.qpos[ids.grip_qpos] - ids.grip_open) / (ids.grip_closed - ids.grip_open)
                action.extend([*commands[side], driver.grippers[side]])
                state.extend([*data.qpos[ids.qpos], float(np.clip(closure.mean(), 0., 1.))])
            task, extras = None, None
            if steps is not None:
                index, _, task = steps(model, data)
                extras = {"subtask_index": np.array([index], np.float32)}
            window.push_sample(data, np.asarray(action, np.float32), np.asarray(state, np.float32), task, extras)
        step = step_status(steps, model, data)
        if step != shown_step:
            shown_step = step
            print(f"[TASK] {step}", flush=True)
        if window is not None:
            parts = [recording.status() if recording is not None else "", step]
            window.status = "  |  ".join(p for p in parts if p)
        tick += 1
        if scene_hook is not None:
            scene_hook(model, data)
        mujoco.mj_step(model, data, nstep=substeps)
        now = time.monotonic()
        if window is not None:
            window.push(data)
        if cameras is not None and now >= next_camera:
            cameras.push(data)
            next_camera = now + CAMERA_PERIOD
        if driver.status != shown:
            shown = driver.status
            print(f"[Quest] {shown}", flush=True)
        next_tick = max(next_tick + dt, now - 0.1)  # catch up at most 0.1 s after a slow tick
        time.sleep(max(0., next_tick - time.monotonic()))


if __name__ == "__main__":
    run()
