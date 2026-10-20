"""MuJoCo leader teleop: left arm by default, both arms with --bimanual.

Same leader mapping as the Isaac backend (leader_mapping.py). Scenes: humanoid_mujoco_scenes.

  R   reset the arm and every object in the scene; bring the leader back to home to resume

Recording (--record): same schema, keys and features as the Isaac backend (S start, N save then
auto-reset, D discard); output under <repo>/datasets/<schema record.root>/sim. Needs a display for
the MuJoCo viewer (on macOS run with ``mjpython``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
# pioneer_humanoid, humanoid_mujoco_scenes, humanoid_robot_learning; this fallback keeps an
# uninstalled checkout working.
sys.path.insert(0, str(_SRC / "pioneer_humanoid"))
sys.path.insert(0, str(_SRC / "simulation" / "mujoco_scenes"))
sys.path.insert(0, str(_SRC / "robot_learning"))

from humanoid_robot_learning.sim_teleop_record import (  # noqa: E402
    BIMANUAL_SCHEMA,
    BIMANUAL_RECORD_JOINTS,
    DEFAULT_SCHEMA,
    LEFT_ARM_RECORD_JOINTS,
    add_record_args,
    load_record_schema,
    make_sim_recorder,
)

from leader_mapping import (  # noqa: E402
    CONTROL_DT,
    WRIST_DAMPING,
    LeaderInput,
    LeaderMapping,
    WallClock,
    add_leader_args,
    check_leader_args,
)

_KEY_R = 82  # GLFW key code


class TeleopMode:
    """Leader configuration and recording format for single-arm or bimanual control."""

    def __init__(self, parser: argparse.ArgumentParser, args: argparse.Namespace):
        self._leader_args = [("LEFT", args)]
        self._record_joints = LEFT_ARM_RECORD_JOINTS
        self._default_schema = DEFAULT_SCHEMA
        if args.bimanual:
            if not all((args.right_port, args.right_calibration, args.right_signs)):
                parser.error("--bimanual requires --right-port, --right-calibration and --right-signs")
            if args.right_port == args.port:
                parser.error("Left and right leaders must use different serial ports")
            right_args = argparse.Namespace(**vars(args))
            right_args.port = args.right_port
            right_args.calibration = args.right_calibration
            right_args.signs = args.right_signs
            self._leader_args.append(("RIGHT", right_args))
            self._record_joints = BIMANUAL_RECORD_JOINTS
            self._default_schema = BIMANUAL_SCHEMA
        for _, leader_args in self._leader_args:
            check_leader_args(parser, leader_args)

    def get_leader_args(self) -> list[tuple[str, argparse.Namespace]]:
        return self._leader_args

    def get_record_joints(self) -> list[str]:
        return self._record_joints

    def get_default_schema(self) -> Path:
        return self._default_schema


def run() -> None:
    parser = argparse.ArgumentParser(description="Pioneer MuJoCo leader teleoperation (left arm, or two leaders with --bimanual).")
    add_leader_args(parser, scene_help="scene registered in humanoid_mujoco_scenes (an unknown name lists them)",
                    bimanual=True)
    add_record_args(parser, task_description="sim leader teleop demonstration")
    args = parser.parse_args()
    mode = TeleopMode(parser, args)
    record = load_record_schema(parser, args, default_schema=mode.get_default_schema())

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
    arms = []
    for side, leader_args in mode.get_leader_args():
        joints = getattr(arm_params, f"{side}_ARM_JOINTS")
        fingers = getattr(arm_params, f"{side}_GRIPPER_JOINTS")
        arm_acts = [model.actuator(j).id for j in joints]
        mapping = LeaderMapping(
            leader_args, home_rad=[data.ctrl[a] for a in arm_acts],
            limits_rad=[tuple(model.joint(j).range) for j in joints],
        )
        arms.append(dict(
            args=leader_args, mapping=mapping, arm_acts=arm_acts,
            grip_acts=[model.actuator(j).id for j in fingers],
            arm_qpos=[model.joint(j).qposadr[0] for j in joints],
            grip_qpos=[model.joint(j).qposadr[0] for j in fingers],
            grip_open=np.array([getattr(arm_params, f"{side}_GRIPPER_OPEN")[j] for j in fingers]),
            grip_closed=np.array([getattr(arm_params, f"{side}_GRIPPER_CLOSED")[j] for j in fingers]),
        ))
        model.actuator_biasprm[model.actuator(joints[-1]).id, 2] = -WRIST_DAMPING
        print(mapping.describe(joints), flush=True)

    reset_requested = [False]

    def on_key(key: int) -> None:
        if key == _KEY_R:
            reset_requested[0] = True

    substeps = max(1, round(CONTROL_DT / model.opt.timestep))
    control_dt = substeps * model.opt.timestep
    # Multi-step scenes record the current step's instruction as each frame's task, plus its index.
    extra = {"subtask_index": ["subtask_index"]} if scene_steps is not None else None
    recorder, record_every = make_sim_recorder(args, record, device="cpu", sim_dt=control_dt, extra_features=extra,
                                               joint_names=mode.get_record_joints())
    if recorder is not None:
        print("[RECORD] Keys: S=start, N=save episode (then reset), D=discard")
        recorder.start_keyboard()
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
            arm["mapping"].reset()

    leaders = []
    clock = WallClock(control_dt)
    step = 0
    shown_step = None  # last task step announced
    print("[INFO] Leader calibrated (hanging = 0). Move it to home: elbow bent 90 deg, forearm forward, gripper open. R = reset.", flush=True)
    try:
        for arm in arms:
            leaders.append(LeaderInput(arm["args"]))
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
                        print("\n[RECORD] Reset mid-episode: take discarded, recording restarts from home.")
                    reset_all()
                    print("\n[LEADER] Arm and scene reset; bring the leader back to home.", flush=True)

                actions = []
                for arm, leader in zip(arms, leaders):
                    target, grip = arm["mapping"].update(leader.read())
                    data.ctrl[arm["arm_acts"]] = target
                    data.ctrl[arm["grip_acts"]] = arm["grip_open"] + grip * (arm["grip_closed"] - arm["grip_open"])
                    actions.extend([*target, grip])

                # Record one synchronized frame only once every active leader is at home.
                if recorder is not None and all(arm["mapping"].engaged for arm in arms) and step % record_every == 0:
                    states = []
                    for arm in arms:
                        closure = ((data.qpos[arm["grip_qpos"]] - arm["grip_open"])
                                   / (arm["grip_closed"] - arm["grip_open"])).mean()
                        states.extend([*data.qpos[arm["arm_qpos"]], np.clip(closure, 0, 1)])
                    state = np.asarray(states, dtype=np.float32)
                    action = np.asarray(actions, dtype=np.float32)
                    task, extras = None, None
                    if scene_steps is not None:
                        index, _, task = scene_steps(model, data)
                        extras = {"subtask_index": np.array([index], dtype=np.float32)}
                    with viewer.lock():
                        saved = recorder.tick(action, state, read_images, task=task, extras=extras)
                    if saved:
                        reset_all()
                        print("\n[RECORD] Episode saved; arm and scene reset.", flush=True)

                if scene_hook is not None:
                    scene_hook(model, data)
                mujoco.mj_step(model, data, nstep=substeps)
                step += 1
                if scene_steps is not None:
                    index, total, instruction = scene_steps(model, data)
                    if (index, instruction) != shown_step:
                        shown_step = (index, instruction)
                        msg = "all steps done -- N to save" if index >= total else f"step {index + 1}/{total}: {instruction}"
                        print(f"\n[TASK] {msg}", flush=True)
                viewer.sync()

                for arm, leader in zip(arms, leaders):
                    leader.report(arm["mapping"])
                clock.wait()
    finally:
        for leader in leaders:
            leader.close()
        print("\n[INFO] Stopped. Leader torque is OFF.", flush=True)
        for renderer in renderers.values():
            renderer.close()
        # Also on errors: flushes episodes still being written in the background.
        if recorder is not None:
            recorder.finalize()
            print(f"[RECORD] Saved under {recorder.dataset_root}")
