"""The MuJoCo side of the Quest teleop, shared by its entry points (mujoco_sim.py: plain teleop; resume_teleop.py: RECAP
corrections from rollout snapshots).

`MujocoSession` owns the scene: model, arms, reset, motor commands, recorded samples, state snapshots. `open_rig` builds
the driver, cameras, window and recorder around it; `serve` runs the control loop on its own thread (the window needs
the main thread on macOS). What starts an episode (left Y, 6) is the entry point's `start_episode(driver)`: a fresh
reset for teleop, a restored snapshot for corrections.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR.parents[2]  # src/teleop/quest_controller_teleop/mujoco -> src
# pioneer_humanoid and humanoid_mujoco_scenes; this fallback keeps an uninstalled checkout working.
sys.path.insert(0, str(_SRC / "pioneer_humanoid"))
sys.path.insert(0, str(_SRC / "simulation" / "mujoco_scenes"))
sys.path.insert(0, str(_SRC / "robot_learning"))
sys.path.insert(0, str(_THIS_DIR.parent))  # the simulator-independent controller modules
sys.path.insert(0, str(_THIS_DIR))

from mujoco_cameras import CameraWindow  # noqa: E402
from scene_driver import ControllerSceneDriver, add_controller_args  # noqa: E402
from humanoid_robot_learning.recap import sidecar  # noqa: E402
from humanoid_robot_learning.sim.sim_teleop_record import make_sim_recorder  # noqa: E402

CONTROL_DT = 0.01        # s, IK solve and actuator command period
SIDES = ("left", "right")


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


def add_session_args(parser, scene_default: str | None = "bimanual_sorting") -> None:
    """Flags every MuJoCo Quest entry point takes: scene, window, controller tuning."""
    parser.add_argument("--scene", default=scene_default,
                        help="scene registered in humanoid_mujoco_scenes (an unknown name lists them)")
    parser.add_argument("--headless", action="store_true", help="no window (e.g. recording only)")
    add_controller_args(parser)
    # The ego camera looks 50 deg down, so its frame turns "hand up" into "wrist forward and up" and "hand back"
    # into "wrist back and up" (scripted-motion check). The upright base frame keeps up as up, forward as forward.
    parser.set_defaults(controller_frame="base")


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
                                           frame_calibration=frame)


class MujocoSession:
    """One scene with both 8dof arms: reset, motor commands, recorded samples and state snapshots."""

    def __init__(self, scene: str):
        import mujoco
        from humanoid_mujoco_scenes import list_scenes, make_model, scene_progress, scene_reset, scene_step, scene_success

        if scene not in list_scenes():
            raise SystemExit(f"unknown --scene {scene!r}; available: {list_scenes()}")
        self.model = make_model(scene, cameras={name: CameraWindow.SIZE for name in CameraWindow.CAMERAS}, robot="8dof")
        self.data = mujoco.MjData(self.model)
        self.hook, self.randomise = scene_step(scene), scene_reset(scene)
        self.steps = scene_progress(scene)   # multi-step task: (index, total, instruction), or None
        self.success = scene_success(scene)
        self.arms = {side: arm_ids(self.model, side) for side in SIDES}
        self.substeps = max(1, round(CONTROL_DT / self.model.opt.timestep))
        self.dt = self.substeps * self.model.opt.timestep
        self.rng = np.random.default_rng()

    def reset(self) -> None:
        """Arms home, scene laid out afresh."""
        import mujoco
        from pioneer_humanoid.mujoco_bimanual_arm_8dof import set_home

        mujoco.mj_resetData(self.model, self.data)
        set_home(self.model, self.data)
        if self.randomise is not None:
            self.randomise(self.model, self.data, self.rng)
        mujoco.mj_forward(self.model, self.data)

    def capture(self) -> sidecar.FrameMeta:
        """Snapshot of the full sim state (and the scene's progress) for the RECAP sidecar."""
        return sidecar.capture(self.model, self.data, self.steps, self.success)

    def restore(self, state: np.ndarray) -> None:
        sidecar.restore(self.model, self.data, state)

    def joint_positions(self) -> dict[str, np.ndarray]:
        return {side: self.data.qpos[ids.qpos].copy() for side, ids in self.arms.items()}

    def gripper_commands(self) -> dict[str, float]:
        """Each gripper's current command as a 0 (open) .. 1 (closed) fraction, read back from ctrl."""
        return {side: float(np.clip(np.mean((self.data.ctrl[ids.grip_acts] - ids.grip_open)
                                            / (ids.grip_closed - ids.grip_open)), 0., 1.))
                for side, ids in self.arms.items()}

    def apply(self, commands, grippers, previous) -> None:
        """Joint commands (velocity feedforward from `previous`, updated in place) and gripper fractions to ctrl."""
        for side, ids in self.arms.items():
            velocity = (commands[side] - previous[side]) / self.dt
            self.data.ctrl[ids.arm_acts] = arm_ctrl(self.model, self.data, ids, commands[side], velocity)
            previous[side] = commands[side]
            self.data.ctrl[ids.grip_acts] = ids.grip_open + grippers[side] * (ids.grip_closed - ids.grip_open)

    def sample(self, sides, commands, grippers):
        """(action, state, task, extras) of the recorded arms `sides`, left first as the schema."""
        action, state = [], []
        for side in sides:
            ids = self.arms[side]
            closure = (self.data.qpos[ids.grip_qpos] - ids.grip_open) / (ids.grip_closed - ids.grip_open)
            action.extend([*commands[side], grippers[side]])
            state.extend([*self.data.qpos[ids.qpos], float(np.clip(closure.mean(), 0., 1.))])
        task, extras = None, None
        if self.steps is not None:
            index, _, task = self.steps(self.model, self.data)
            extras = {"subtask_index": np.array([index], np.float32)}
        return np.asarray(action, np.float32), np.asarray(state, np.float32), task, extras

    def step(self) -> None:
        """One control period: the scene's hook, then physics."""
        import mujoco

        if self.hook is not None:
            self.hook(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)

    def step_status(self) -> str:
        """The operator's view of a multi-step task: the current step, or that all are done."""
        if self.steps is None:
            return ""
        index, total, instruction = self.steps(self.model, self.data)
        return "all steps done: save the take" if index >= total else f"step {index + 1}/{total}: {instruction}"


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


def open_rig(args, record, session: MujocoSession, sides) -> SimpleNamespace:
    """Driver, window and recorder around `session` (whose state must be set: the driver reads the
    ego camera). `sides`: the arms recorded. Returns them with the keys and events serve() uses."""
    driver = make_driver(args, session.model, session.data)
    # Multi-step scenes record the current step's instruction as each frame's task, plus its index (as the leader
    # teleop does): sim evaluation prompts the policy with the same per-step instruction.
    extra = {"subtask_index": ["subtask_index"]} if session.steps is not None else None
    recorder, every = make_sim_recorder(args, record, device="cpu", sim_dt=session.dt, extra_features=extra)
    sizes = {name: (int(spec["height"]), int(spec["width"])) for name, spec in record.images.items()}
    window = None if args.headless and recorder is None else CameraWindow(session.model, sizes, recorder)
    recording = Recording(recorder, window, every, driver, sides) if recorder is not None else None
    rig = SimpleNamespace(driver=driver, recorder=recorder, window=window, recording=recording,
                          reset_key=threading.Event(), stop=threading.Event())
    rig.keys = {"6": rig.reset_key.set}   # set by the window's key handler, consumed by the control loop
    if recording is not None:
        rig.keys.update({"7": recording.start, "8": recording.save, "9": recording.discard})
        print(f"[RECORD] Right B (or 7) records a take, again (or 8) saves it; left Y, 6 or 9 discards. "
              f"Saving to {recorder.dataset_root}", flush=True)
    return rig


def serve(args, rig, session: MujocoSession, start_episode) -> None:
    """Run the control loop on its own thread and the window on this one until either stops; then save and close.

    `start_episode(driver) -> message`: what left Y or 6 does (sets the scene's state and re-anchors the driver)."""
    def control() -> None:
        rig.driver.start_thread()   # the driver is owned by the thread that starts it: start, loop and close here
        try:
            loop(session, rig, start_episode)
        finally:
            rig.stop.set()
            rig.driver.close()

    # The window must live on the main thread (macOS), so the control loop runs on its own thread.
    worker = threading.Thread(target=control, name="control", daemon=True)
    worker.start()
    try:
        if not args.headless:
            rig.window.show(rig.stop, worker.is_alive, rig.keys)
            rig.stop.set()   # window closed: stop the sim
        while worker.is_alive():
            worker.join(timeout=.5)
    except KeyboardInterrupt:
        pass
    finally:
        close_rig(rig, worker)


def close_rig(rig, worker) -> None:
    rig.stop.set()
    worker.join(timeout=2.)
    if rig.window is not None:
        rig.window.close()   # renders any queued recorded frames first
    if rig.recorder is not None:
        rig.recorder.finalize()
        print(f"[RECORD] {rig.recorder.num_recorded_episodes} takes saved under {rig.recorder.dataset_root}", flush=True)
    print("\n[Quest] Stopped.", flush=True)


def loop(session: MujocoSession, rig, start_episode) -> None:
    """Control ticks at dt until `rig.stop` is set (window closed, Ctrl+C): input, IK, physics, window frames."""
    driver, recording, window, dt = rig.driver, rig.recording, rig.window, session.dt
    next_tick, shown, shown_step, tick = time.monotonic(), None, None, 0
    previous = session.joint_positions()
    while not rig.stop.is_set():
        if recording is not None and recording.recorder.is_complete:
            print("[RECORD] Session complete.", flush=True)
            break
        reset_now = driver.update(session.joint_positions()) or rig.reset_key.is_set()   # left Y, or 6 in the window
        if recording is not None and driver.record_presses != recording.presses:
            recording.presses = driver.record_presses
            recording.toggle()
        if reset_now:
            rig.reset_key.clear()
            if recording is not None:
                recording.discard()   # frames either side of a reset are not one demo
            print(f"[Quest] {start_episode(driver)}", flush=True)
            previous = session.joint_positions()
        commands = driver.solve(session.joint_positions(), dt)
        session.apply(commands, driver.grippers, previous)
        if recording is not None and recording.flags.start and tick % recording.every == 0:
            # Before the step: image, state and action are of the same instant.
            action, state, task, extras = session.sample(recording.sides, commands, driver.grippers)
            window.push_sample(session.data, action, state, task, extras, meta=session.capture())
        shown_step = show_status(session, rig, shown_step)
        tick += 1
        session.step()
        now = time.monotonic()
        if window is not None:
            window.push(session.data)
        if driver.status != shown:
            shown = driver.status
            print(f"[Quest] {shown}", flush=True)
        next_tick = max(next_tick + dt, now - 0.1)  # catch up at most 0.1 s after a slow tick
        time.sleep(max(0., next_tick - time.monotonic()))


def show_status(session: MujocoSession, rig, shown_step: str | None) -> str:
    """Put the recording state and current step in the window; print the step when it changes."""
    step = session.step_status()
    if step != shown_step:
        print(f"[TASK] {step}", flush=True)
    if rig.window is not None:
        parts = [rig.recording.status() if rig.recording is not None else "", step]
        rig.window.status = "  |  ".join(p for p in parts if p)
    return step
