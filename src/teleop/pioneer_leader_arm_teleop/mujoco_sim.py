"""MuJoCo leader teleop: one leader per arm, left arm by default (--arms left, right or bimanual).

Same leader mapping as the Isaac backend (leader_mapping.py). Scenes: humanoid_mujoco_scenes.

  6   reset the arm and every object in the scene; bring the leader back to home to resume

Recording (--record): same schema and features as the Isaac backend (7 start, 8 save then
auto-reset, 9 discard); output under <repo>/datasets/<schema record.root>/sim. Needs a display for
the MuJoCo viewer (on macOS run with ``mjpython``).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[2]
# pioneer_humanoid, humanoid_mujoco_scenes, humanoid_robot_learning; this fallback keeps an
# uninstalled checkout working.
sys.path.insert(0, str(_SRC / "pioneer_humanoid"))
sys.path.insert(0, str(_SRC / "simulation" / "mujoco_scenes"))
sys.path.insert(0, str(_SRC / "robot_learning"))

from humanoid_robot_learning.recording.schema import ARM_SETS  # noqa: E402
from humanoid_robot_learning.sim.sim_teleop_record import (  # noqa: E402
    add_record_args,
    load_record_schema,
    make_sim_recorder,
)

from leader_mapping import (  # noqa: E402
    CONTROL_DT,
    WRIST_DAMPING,
    LeaderBoard,
    LeaderInput,
    LeaderMapping,
    WallClock,
    add_leader_args,
    check_leader_args,
)

# Every letter, 0-5, Esc and the F1-F5 keys are MuJoCo viewer shortcuts; 6-9 are free.
_KEY_RESET = ord("6")  # GLFW key code
RECORD_KEYS = dict(start="7", save="8", discard="9", esc_aborts=False)  # Esc: viewer's free camera


@dataclass
class Arm:
    """One leader driving one arm: its mapping and the model ids it reads and writes."""

    side: str
    leader: Any  # LeaderInput, opened inside run()'s try so a failed open still closes the others
    board: LeaderBoard
    mapping: LeaderMapping
    arm_acts: list[int]
    grip_acts: list[int]
    arm_qpos: list[int]
    grip_qpos: list[int]
    grip_open: Any  # np.ndarray, finger positions at closure 0
    grip_closed: Any  # np.ndarray, finger positions at closure 1


class RecordStatus:
    """What the recorder is doing and the key to press next, in the MuJoCo window's corner, where you are looking.

    The record keys (7/8/9) set `flags` from the recorder's keyboard thread; poll() turns their changes into messages.
    """

    EVENT_S = 3.0  # how long a "saved" / "discarded" message stays up

    def __init__(self, recorder: Any, flags: Any):
        self.recorder = recorder
        self.flags = flags  # the recorder's EpisodeFlags
        self._was = dict(start=False, remove=False)
        self._event, self._event_t = "", -math.inf
        self._shown = None

    def event(self, text: str) -> None:
        print(f"\n[RECORD] {text}", flush=True)
        self._event, self._event_t = text, time.monotonic()

    def poll(self, started: bool) -> None:
        """Once per loop, before tick(): announce 7 and 9, and drop an 8 / 9 pressed with nothing recorded."""
        flags = self.flags
        if (flags.success or flags.remove) and self.recorder.num_buffered_frames == 0:
            # Left queued, a stale 8 / 9 would fire on the next take's first frame and throw that take away.
            flags.success = flags.remove = False
            self.event("nothing recorded yet" + ("" if flags.start else ": press 7 first"))
        if flags.start and not self._was["start"]:
            self.event("recording started" if started else "recording armed: it starts once the leader is at home")
        if flags.remove and not self._was["remove"]:
            self.event("take discarded, recording again")
        self._was = dict(start=flags.start, remove=flags.remove)

    def _lines(self, started: bool, recording: bool) -> tuple[str, str]:
        if recording and not started:
            head, keys = "REC armed", "bring the leader to home to start"
        elif recording:
            head, keys = f"RECORDING {self.recorder.num_buffered_frames / self.recorder.fps:.1f} s", "8 save   9 discard"
        else:
            head, keys = f"not recording ({self.recorder.num_recorded_episodes} saved)", "7 record"
        if time.monotonic() - self._event_t < self.EVENT_S:
            head, keys = f"{self._event}\n{head}", f"\n{keys}"
        return head, keys

    def show(self, viewer: Any, started: bool) -> None:
        import mujoco

        text = self._lines(started, self.flags.start)
        if text != self._shown:  # set_texts only on change: the seconds count ticks at 10 Hz
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_200, mujoco.mjtGridPos.mjGRID_TOPLEFT, *text))
            self._shown = text


def print_camera(cam: Any, scene_name: str) -> None:
    """Print the free camera the window closed on, as the @scene(camera=...) that opens the scene there next time."""
    import mujoco

    if cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
        print("\n[CAMERA] The window closed on a fixed camera; switch back to the free camera to capture a view.")
        return
    view = dict(lookat=[round(float(v), 3) for v in cam.lookat], distance=round(float(cam.distance), 3),
                azimuth=round(float(cam.azimuth), 1), elevation=round(float(cam.elevation), 1))
    print(f"\n[CAMERA] To open {scene_name} on this view, use in its @scene(...): camera={view}")


def run() -> None:
    parser = argparse.ArgumentParser(description="Pioneer MuJoCo leader teleoperation, one leader per arm.")
    add_leader_args(parser, scene_help="scene registered in humanoid_mujoco_scenes (an unknown name lists them)")
    parser.add_argument("--arms", choices=tuple(ARM_SETS), default="left",
                        help="arms to drive and record, one leader board each (leader_arm.json)")
    add_record_args(parser, task_description="sim leader teleop demonstration")
    args = parser.parse_args()
    sides = list(ARM_SETS[args.arms])  # in ARM_ORDER: a recorded frame is the left arm's values, then the right's
    boards = check_leader_args(parser, args, sides)
    record = load_record_schema(parser, args, arms=sides)

    import mujoco
    import mujoco.viewer
    import numpy as np
    from humanoid_mujoco_scenes import list_scenes, make_model, scene_camera, scene_progress, scene_reset, scene_step
    from pioneer_humanoid import arm_params
    from pioneer_humanoid.mujoco_bimanual_arm import set_home

    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")
    unknown = sorted(set(record.images) - set(arm_params.CAMERA_NAMES))
    if unknown:
        raise SystemExit(f"{record.path}: unknown images {unknown}; available: {list(arm_params.CAMERA_NAMES)}")
    cameras = {name: (int(spec["height"]), int(spec["width"])) for name, spec in record.images.items()}
    model = make_model(args.scene, cameras=cameras)
    scene_hook = scene_step(args.scene)  # per-step scene mechanics (e.g. zip_tie's ratchet), or None
    scene_randomise = scene_reset(args.scene)  # new layout per episode, or None
    scene_steps = scene_progress(args.scene)  # multi-step task: (index, total, instruction), or None
    rng = np.random.default_rng()
    data = mujoco.MjData(model)
    set_home(model, data)
    if scene_randomise is not None:
        scene_randomise(model, data, rng)
    mujoco.mj_forward(model, data)
    # side -> (arm joints, gripper fingers, finger positions open, closed)
    side_joints = {
        "left": (arm_params.LEFT_ARM_JOINTS, arm_params.LEFT_GRIPPER_JOINTS,
                 arm_params.LEFT_GRIPPER_OPEN, arm_params.LEFT_GRIPPER_CLOSED),
        "right": (arm_params.RIGHT_ARM_JOINTS, arm_params.RIGHT_GRIPPER_JOINTS,
                  arm_params.RIGHT_GRIPPER_OPEN, arm_params.RIGHT_GRIPPER_CLOSED),
    }
    arms: list[Arm] = []
    for side in sides:
        joints, fingers, grip_open, grip_closed = side_joints[side]
        arm_acts = [model.actuator(j).id for j in joints]
        mapping = LeaderMapping(
            boards[side].signs, args.filter_alpha, home_rad=[data.ctrl[a] for a in arm_acts],
            limits_rad=[tuple(model.joint(j).range) for j in joints],
        )
        arms.append(Arm(
            side=side, leader=None, board=boards[side], mapping=mapping, arm_acts=arm_acts,
            grip_acts=[model.actuator(j).id for j in fingers],
            arm_qpos=[model.joint(j).qposadr[0] for j in joints],
            grip_qpos=[model.joint(j).qposadr[0] for j in fingers],
            grip_open=np.array([grip_open[j] for j in fingers]),
            grip_closed=np.array([grip_closed[j] for j in fingers]),
        ))
        model.actuator_biasprm[model.actuator(joints[-1]).id, 2] = -WRIST_DAMPING
        print(f"[{side.upper()}] " * (len(sides) > 1) + mapping.describe(joints), flush=True)

    reset_requested = [False]

    def on_key(key: int) -> None:
        if key == _KEY_RESET:
            reset_requested[0] = True

    substeps = max(1, round(CONTROL_DT / model.opt.timestep))
    control_dt = substeps * model.opt.timestep
    # Multi-step scenes record the current step's instruction as each frame's task, plus its index.
    extra = {"subtask_index": ["subtask_index"]} if scene_steps is not None else None
    recorder, record_every = make_sim_recorder(args, record, device="cpu", sim_dt=control_dt, extra_features=extra)
    if recorder is not None:
        print("[RECORD] Keys: 7=start, 8=save episode (then reset), 9=discard, 6=reset")
        recorder.start_keyboard(**RECORD_KEYS)
    # The Quest teleop reads the recorder's flags the same way; SimLeRobotRecorder has no public accessor.
    status = RecordStatus(recorder, recorder._flags) if recorder is not None else None
    renderers = {name: mujoco.Renderer(model, h, w) for name, (h, w) in cameras.items()}

    # Called by tick() only on recorded frames.
    def read_images():
        images = {}
        for name, renderer in renderers.items():
            renderer.update_scene(data, name)
            images[name] = renderer.render()
        return images

    def reset_all():
        """Arm and every object back to their defaults; the arm waits for the leader at home."""
        mujoco.mj_resetData(model, data)
        set_home(model, data)
        if scene_randomise is not None:
            scene_randomise(model, data, rng)
        mujoco.mj_forward(model, data)
        for arm in arms:
            arm.mapping.reset()

    clock = WallClock(control_dt)
    step = 0
    shown_step = None  # last task step announced
    print("[INFO] Leader calibrated (hanging = 0). Move it to home: elbow bent 90 deg, forearm forward, gripper open. 6 = reset.", flush=True)
    try:
        for arm in arms:
            arm.leader = LeaderInput(arm.board, args.baud, label=arm.side.upper() if len(arms) > 1 else "")
        with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
            for key, value in (scene_camera(args.scene) or {}).items():
                setattr(viewer.cam, key, value)
            while viewer.is_running():
                if recorder is not None and recorder.is_complete:
                    print("\n[RECORD] Session complete.")
                    break
                if reset_requested[0]:
                    reset_requested[0] = False
                    if recorder is not None and recorder.num_buffered_frames > 0:
                        # Frames either side of a reset are not one demo: drop the take, keep recording.
                        recorder.cancel_recording()
                        status.event("reset mid-take: take discarded, recording again once started")
                    reset_all()
                    print("\n[LEADER] Arm and scene reset; bring the leader back to home.", flush=True)
                if status is not None:
                    status.poll(all(arm.mapping.engaged for arm in arms))

                actions = []
                for arm in arms:
                    target, grip = arm.mapping.update(arm.leader.read())
                    data.ctrl[arm.arm_acts] = target
                    data.ctrl[arm.grip_acts] = arm.grip_open + grip * (arm.grip_closed - arm.grip_open)
                    actions.extend([*target, grip])

                # Record one synchronized frame only once every active leader is at home.
                if recorder is not None and all(arm.mapping.engaged for arm in arms) and step % record_every == 0:
                    states = []
                    for arm in arms:
                        closure = ((data.qpos[arm.grip_qpos] - arm.grip_open) / (arm.grip_closed - arm.grip_open)).mean()
                        states.extend([*data.qpos[arm.arm_qpos], np.clip(closure, 0, 1)])
                    state = np.asarray(states, dtype=np.float32)
                    action = np.asarray(actions, dtype=np.float32)
                    task, extras = None, None
                    if scene_steps is not None:
                        index, _, task = scene_steps(model, data)
                        extras = {"subtask_index": np.array([index], dtype=np.float32)}
                    seconds = recorder.num_buffered_frames / recorder.fps
                    with viewer.lock():
                        saved = recorder.tick(action, state, read_images, task=task, extras=extras)
                    if saved:
                        reset_all()
                        status.event(f"SAVED {seconds:.1f} s take; arm and scene reset, 7 to record the next")

                if scene_hook is not None:
                    scene_hook(model, data)
                mujoco.mj_step(model, data, nstep=substeps)
                step += 1
                if scene_steps is not None:
                    index, total, instruction = scene_steps(model, data)
                    if (index, instruction) != shown_step:
                        shown_step = (index, instruction)
                        msg = "all steps done -- 8 to save" if index >= total else f"step {index + 1}/{total}: {instruction}"
                        print(f"\n[TASK] {msg}", flush=True)
                if status is not None:
                    status.show(viewer, all(arm.mapping.engaged for arm in arms))
                viewer.sync()

                for arm in arms:
                    if len(arms) == 1:
                        arm.leader.report(arm.mapping)
                    elif not arm.mapping.engaged:
                        # Two leaders: a line each while homing (a shared \r status line would be overwritten).
                        arm.leader.report(arm.mapping, end="\n")
                clock.wait()
            print_camera(viewer.cam, args.scene)
    finally:
        for arm in arms:
            if arm.leader is not None:
                arm.leader.close()
        print("\n[INFO] Stopped. Leader torque is OFF.", flush=True)
        for renderer in renderers.values():
            renderer.close()
        # Also on errors: flushes episodes still being written in the background.
        if recorder is not None:
            recorder.finalize()
            print(f"[RECORD] Saved under {recorder.dataset_root}")
