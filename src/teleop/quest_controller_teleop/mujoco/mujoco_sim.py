"""Quest Touch controller teleoperation in MuJoCo: both arms, any humanoid_mujoco_scenes scene.

The headset only tracks (index.html streams its poses): open http://localhost:8444/ in Meta Browser after
`adb reverse tcp:8444 tcp:8444` (start_quest.sh does it all).
Left Y (or 6 with the window focused) resets the arms and the scene. --record: right B starts / saves a take.

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

from mujoco_cameras import CameraWindow  # noqa: E402
from scene_driver import ControllerSceneDriver, add_controller_args, check_controller_args  # noqa: E402
from humanoid_robot_learning.sim.sim_teleop_record import (  # noqa: E402
    add_record_args, load_record_schema, make_sim_recorder)

CONTROL_DT = 0.01        # s, IK solve and actuator command period


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Quest Touch controller teleoperation in MuJoCo (both arms).")
    parser.add_argument("--scene", default="bimanual_sorting",
                        help="scene registered in humanoid_mujoco_scenes (an unknown name lists them)")
    parser.add_argument("--headless", action="store_true", help="no window (e.g. recording only)")
    add_controller_args(parser)
    add_record_args(parser, task_description="Quest VR teleop demonstration")
    parser.add_argument("--hg-dagger", action="store_true", help="Run a policy and collect accepted human corrections")
    parser.add_argument("--policy", type=Path, help="Complete pretrained checkpoint for --hg-dagger")
    parser.add_argument("--corrections", type=Path, help="Correction directory for --hg-dagger")
    parser.add_argument("--bind-dataset", type=Path, help="Attach this scene's contract to existing data and policy, then exit")
    parser.add_argument("--device", default="cuda", help="Policy inference device for --hg-dagger")
    parser.add_argument("--seed", type=int, default=0, help="Initial HG-DAgger scene reset seed")
    parser.add_argument("--max-frames", type=int, default=1500, help="Hold at this many unsaved correction frames")
    # The ego camera looks 50 deg down, so its frame turns "hand up" into "wrist forward and up" and "hand back"
    # into "wrist back and up" (scripted-motion check). The upright base frame keeps up as up, forward as forward.
    parser.set_defaults(controller_frame="base")
    args = parser.parse_args(argv)
    check_controller_args(parser, args)
    if args.hg_dagger:
        from humanoid_robot_learning.sim.sim_teleop_record import RecordSchema

        if args.record:
            parser.error("--record and --hg-dagger are mutually exclusive")
        if args.policy is None or (args.corrections is None and args.bind_dataset is None):
            parser.error("--hg-dagger requires --policy and either --corrections or --bind-dataset")
        if args.corrections is not None and args.bind_dataset is not None:
            parser.error("Use --bind-dataset separately from correction collection")
        if args.max_frames < 1:
            parser.error("--max-frames must be positive")
        if args.bind_dataset is None and args.task_description == "Quest VR teleop demonstration":
            parser.error("Supply --task_description matching the policy's task")
        if args.cameras is not None or args.image_size is not None or args.schema is not None:
            parser.error("HG-DAgger uses the checkpoint's camera sizes and joint contract")
        if args.dataset_root is not None:
            parser.error("Use --corrections for HG-DAgger output; --dataset_root belongs to --record")
        return args, RecordSchema()
    if args.policy is not None or args.corrections is not None or args.bind_dataset is not None:
        parser.error("--policy, --corrections and --bind-dataset require --hg-dagger")
    if args.record and args.cameras is None:
        args.cameras = ",".join(CameraWindow.CAMERAS)   # both arms move: record both wrists
    return args, load_record_schema(parser, args, arms=["left", "right"], robot="8dof")


class Recording:
    """Right B (or 7) starts a take and saves it (or 8); left Y, 6 or 9 discards it. Frames: 25 Hz of sim time."""

    def __init__(self, recorder, window, every, driver):
        self.recorder, self.window, self.every, self.driver = recorder, window, every, driver
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


# Attach human-gated collection to this script's existing scene and controller loop.
class HGCollection:
    def __init__(self, args):
        from humanoid_robot_learning.hg_dagger.core import Contract
        from humanoid_robot_learning.hg_dagger.learning import checkpoint_path, load_contract, LeRobotPolicy
        from humanoid_robot_learning.hg_dagger.storage import CorrectionStore
        from pioneer_humanoid import arm_8dof_params as params

        self.args = args
        path = checkpoint_path(args.policy)
        if args.bind_dataset is not None:
            import json

            info = json.loads((args.bind_dataset / "meta/info.json").read_text())
            features = info["features"]
            names = tuple(features["action"]["names"])
            state_names = tuple(features["observation.state"]["names"])
            unit = lambda name: "closure_0_1" if name in ("left_gripper", "right_gripper") else "rad"
            self.contract = Contract(args.scene, "pioneer_bimanual_arm_8dof", info["fps"], state_names, names,
                                     tuple(map(unit, state_names)), tuple(map(unit, names)),
                                     {k: tuple(v["shape"]) for k, v in features.items() if k.startswith("observation.images.")},
                                     "absolute_joint_radians_and_gripper_0_1")
        else:
            self.contract = load_contract(path)
        joints = {"left": params.LEFT_ARM_JOINTS, "right": params.RIGHT_ARM_JOINTS}
        self.sides = [s for s in joints if f"{s}_gripper" in self.contract.action_names]
        if not self.sides:
            raise ValueError("HG-DAgger requires a checkpoint for the current 8dof arm, not the legacy six-joint arm")
        names = tuple(n for s in self.sides for n in [*joints[s], f"{s}_gripper"])
        units = tuple(u for _ in self.sides for u in [*["rad"] * 7, "closure_0_1"])
        expected = Contract(args.scene, "pioneer_bimanual_arm_8dof", self.contract.fps,
                            names, names, units, units, self.contract.images,
                            "absolute_joint_radians_and_gripper_0_1")
        if self.contract != expected:
            raise ValueError("Checkpoint must match this scene and the current 8dof arm contract")
        self.sizes = {k.removeprefix("observation.images."): v[:2] for k, v in self.contract.images.items()}
        if not set(self.sizes) <= set(CameraWindow.CAMERAS):
            raise ValueError("Checkpoint requests cameras unavailable in the Quest scene")
        self.every = round(1 / (self.contract.fps * CONTROL_DT))
        if self.every < 1 or not np.isclose(self.every * CONTROL_DT, 1 / self.contract.fps):
            raise ValueError("Checkpoint fps must divide the 100 Hz control rate")
        if args.bind_dataset is not None:
            from humanoid_robot_learning.hg_dagger.learning import bind_contract

            bind_contract(args.bind_dataset, path, self.contract)
            print("[HG-DAgger] Contract verified and attached to the dataset and checkpoint", flush=True)
            return
        self.policy = LeRobotPolicy(path, self.contract, args.device)
        self.store = CorrectionStore(args.corrections, self.contract)

    # Reuse the owner's scene, driver, camera thread and reset callback.
    def bind(self, model, data, driver, arms, window, reset, stop):
        from types import SimpleNamespace
        from humanoid_robot_learning.hg_dagger.core import Session

        self.model, self.data, self.driver, self.arms = model, data, driver, arms
        self.window, self.stop = window, stop
        self.grippers = {s: 0. for s in arms}
        self.presses, self.previous_status = 0, None
        self.reset_callback = reset
        scene = SimpleNamespace(contract=self.contract, reset=self.reset)
        self.session = Session(scene, SimpleNamespace(seed=self.seed), self.policy, self.store,
                               self.args.task_description, self.args.seed, self.args.max_frames)
        print("[HG-DAgger] Waiting for VR and grip release. Right grip: human; release: policy; "
              "right B/8: save and reset; left Y/6/9: discard and reset", flush=True)

    def reset(self, seed):
        self.reset_callback(seed)
        self.grippers = {s: 0. for s in self.arms}

    def seed(self):
        self.driver.reset({s: self.data.qpos[ids.qpos].copy() for s, ids in self.arms.items()})
        self.driver.grippers.update(self.grippers)
        self.command, self.prediction, self.next_sample = None, None, 0

    # Handle decisions at 100 Hz, including while physics is held.
    def prepare(self, discard, accept=False):
        from humanoid_robot_learning.hg_dagger.core import Signals

        with self.driver.shared.lock:
            shared = self.driver.shared
            packet, received, presses = shared.packet, shared.received, shared.record_presses
        accept = accept or presses != self.presses
        self.presses = presses
        ready = (packet is not None and time.monotonic() - received <= .2 and packet["head"] is not None
                 and all(packet[s] is not None for s in {"right", *self.sides}))
        takeover = bool(ready and packet["right"].clutch)
        return self.session.decide(Signals(bool(ready), takeover, accept, discard))

    def state(self):
        values = []
        for side in self.sides:
            ids = self.arms[side]
            closure = (self.data.qpos[ids.grip_qpos] - ids.grip_open) / (ids.grip_closed - ids.grip_open)
            values.extend([*self.data.qpos[ids.qpos], float(np.clip(closure.mean(), 0, 1))])
        return np.asarray(values, dtype=np.float32)

    # Sample exact pre-action images at policy fps while IK and physics keep their 100 Hz cadence.
    def commands(self, positions, previous, dt, tick):
        self.observation = None
        if self.command is None or tick >= self.next_sample:
            images = self.window.capture(self.data, self.stop) if self.sizes else {}
            self.observation = {"observation.state": self.state(),
                                **{f"observation.images.{k}": images[k] for k in self.sizes}}
            self.contract.check_observation(self.observation)
            self.next_sample = tick + self.every
            if self.session.mode == "POLICY":
                self.prediction = self.contract.action(self.policy.action(self.observation, self.args.task_description))
        commands = {s: q.copy() for s, q in previous.items()}
        human = self.driver.solve(positions, dt) if self.session.mode == "HUMAN" else None
        for i, side in enumerate(self.sides):
            solver = self.driver.solvers[side]
            target = human[side] if human is not None else self.prediction[8*i:8*i+7]
            target = np.clip(target, solver.physical_limits[:, 0], solver.physical_limits[:, 1])
            commands[side] = np.clip(target, previous[side] - solver.velocity * dt,
                                    previous[side] + solver.velocity * dt)
            self.grippers[side] = float(np.clip(self.driver.grippers[side] if human is not None
                                               else self.prediction[8*i+7], 0, 1))
        self.command = np.asarray([v for s in self.sides for v in [*commands[s], self.grippers[s]]], dtype=np.float32)
        if self.observation is not None:
            self.session.record(self.observation, self.command)
        return commands

    def status(self):
        session = self.session
        mode = session.mode if session.started and not session.wait_release else "WAITING FOR VR / GRIP RELEASE"
        status = f"{mode} | Unsaved {session.pending_frames} | Saved segments {session.accepted}"
        if session.pending_frames >= session.max_frames:
            status = "BUFFER FULL - accept or discard | " + status
        if status != self.previous_status:
            print("[HG-DAgger] " + status, flush=True)
            self.previous_status = status
        if self.window is not None:
            self.window.status = status
        with self.driver.shared.lock:
            self.driver.shared.feedback["status"] = status
            self.driver.shared.feedback["hg_ready"] = session.started and not session.wait_release


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


def run() -> None:
    args, record = parse_args()
    import mujoco
    from humanoid_mujoco_scenes import list_scenes, make_model, scene_reset, scene_step
    from pioneer_humanoid.mujoco_bimanual_arm_8dof import set_home

    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")
    hg = HGCollection(args) if args.hg_dagger else None
    if args.bind_dataset is not None:
        return
    camera_sizes = {name: CameraWindow.SIZE for name in CameraWindow.CAMERAS}
    if hg is not None:
        camera_sizes.update(hg.sizes)
    model = make_model(args.scene, cameras=camera_sizes, robot="8dof")
    data = mujoco.MjData(model)
    scene_hook, randomise = scene_step(args.scene), scene_reset(args.scene)
    rng = np.random.default_rng()
    arms = {side: arm_ids(model, side) for side in ("left", "right")}

    def reset(seed=None) -> None:
        mujoco.mj_resetData(model, data)
        set_home(model, data)
        if randomise is not None:
            randomise(model, data, rng if seed is None else np.random.default_rng(seed))
        mujoco.mj_forward(model, data)

    reset()
    driver = make_driver(args, model, data)
    substeps = max(1, round(CONTROL_DT / model.opt.timestep))
    dt = substeps * model.opt.timestep
    print(f"[Quest] MuJoCo scene {args.scene}: open http://localhost:{args.controller_port}/ in Meta Browser "
          f"(after adb reverse tcp:{args.controller_port} tcp:{args.controller_port}). Left Y or 6 in the window resets.", flush=True)
    reset_key = threading.Event()  # set by the window's key handler, consumed by the control loop
    stop = threading.Event()
    recorder, every = make_sim_recorder(args, record, device="cpu", sim_dt=dt)
    sizes = {name: (int(spec["height"]), int(spec["width"])) for name, spec in record.images.items()}
    if hg is not None:
        sizes.update(hg.sizes)
    needs_window = not args.headless or recorder is not None or (hg is not None and bool(hg.sizes))
    window = CameraWindow(model, sizes, recorder) if needs_window else None
    recording = Recording(recorder, window, every, driver) if recorder is not None else None
    keys = {"6": reset_key.set}
    accept_key = threading.Event()
    if hg is not None:
        keys.update({"8": accept_key.set, "9": reset_key.set})
    if recording is not None:
        window.on_saved = lambda: (recording.saved(), reset_key.set())
        keys.update({"7": recording.start, "8": recording.save, "9": recording.discard})
        print(f"[RECORD] Right B (or 7) records a take, again (or 8) saves it; left Y, 6 or 9 discards. "
              f"Saving to {recorder.dataset_root}", flush=True)

    failures = []

    def control() -> None:
        try:
            driver.start_thread()
            if hg is not None:
                hg.bind(model, data, driver, arms, window, reset, stop)
            loop(model, data, driver, arms, window, recording, reset, reset_key, scene_hook, substeps, dt,
                 stop, hg, accept_key)
        except InterruptedError as error:
            if not stop.is_set():
                failures.append(error)
        except Exception as error:
            failures.append(error)
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
        if window is not None:
            window.close()   # renders any queued recorded frames first
        if recorder is not None:
            recorder.finalize()
            print(f"[RECORD] {recorder.num_recorded_episodes} takes saved under {recorder.dataset_root}", flush=True)
        print("\n[Quest] Stopped.", flush=True)
    if failures:
        raise RuntimeError("Quest control stopped because of an error") from failures[0]


def loop(model, data, driver, arms, window, recording, reset, reset_key, scene_hook, substeps, dt,
         stop, hg=None, accept_key=None) -> None:
    """Control ticks at dt until `stop` is set (window closed, Ctrl+C): input, IK, physics, camera frames."""
    import mujoco

    def joint_positions():
        return {side: data.qpos[ids.qpos].copy() for side, ids in arms.items()}

    next_tick, shown, tick = time.monotonic(), None, 0
    previous = {side: data.qpos[ids.qpos].copy() for side, ids in arms.items()}
    while not stop.is_set():
        if window is not None and window.error is not None:
            raise RuntimeError("Camera rendering failed") from window.error
        if recording is not None and recording.recorder.is_complete:
            print("[RECORD] Session complete.", flush=True)
            break
        reset_now = driver.update(joint_positions()) or reset_key.is_set()   # left Y, or 6 in the window
        if recording is not None and driver.record_presses != recording.presses:
            recording.presses = driver.record_presses
            recording.toggle()
        if hg is not None:
            accept = accept_key.is_set() if accept_key is not None else False
            if accept_key is not None:
                accept_key.clear()
            reset_key.clear()
            if not hg.prepare(reset_now, accept):
                hg.status()
                previous = joint_positions()
                if window is not None:
                    window.push(data)
                next_tick = max(next_tick + dt, time.monotonic() - .1)
                time.sleep(max(0., next_tick - time.monotonic()))
                continue
        elif reset_now:
            reset_key.clear()
            if recording is not None:
                recording.discard()   # frames either side of a reset are not one demo
            reset()
            driver.reset(joint_positions())
            previous = joint_positions()
            print("[Quest] Scene reset; arms back to home.", flush=True)
        commands = (hg.commands(joint_positions(), previous, dt, tick) if hg is not None
                    else driver.solve(joint_positions(), dt))
        if stop.is_set():
            break
        grippers = hg.grippers if hg is not None else driver.grippers
        for side, ids in arms.items():
            data.ctrl[ids.arm_acts] = arm_ctrl(model, data, ids, commands[side], (commands[side] - previous[side]) / dt)
            previous[side] = commands[side]
            data.ctrl[ids.grip_acts] = ids.grip_open + grippers[side] * (ids.grip_closed - ids.grip_open)
        if recording is not None and recording.flags.start and tick % recording.every == 0:
            # Before the step: image, state and action are of the same instant. Left arm first, as the schema.
            action, state = [], []
            for side, ids in arms.items():
                closure = (data.qpos[ids.grip_qpos] - ids.grip_open) / (ids.grip_closed - ids.grip_open)
                action.extend([*commands[side], driver.grippers[side]])
                state.extend([*data.qpos[ids.qpos], float(np.clip(closure.mean(), 0., 1.))])
            window.push_sample(data, np.asarray(action, np.float32), np.asarray(state, np.float32))
        if recording is not None:
            window.status = recording.status()
        if hg is not None:
            hg.status()
        tick += 1
        if scene_hook is not None:
            scene_hook(model, data)
        mujoco.mj_step(model, data, nstep=substeps)
        now = time.monotonic()
        if window is not None:
            window.push(data)
        if hg is None and driver.status != shown:
            shown = driver.status
            print(f"[Quest] {shown}", flush=True)
        next_tick = max(next_tick + dt, now - 0.1)  # catch up at most 0.1 s after a slow tick
        time.sleep(max(0., next_tick - time.monotonic()))


if __name__ == "__main__":
    run()
