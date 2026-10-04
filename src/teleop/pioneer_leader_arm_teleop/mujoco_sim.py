"""MuJoCo backend of pioneer_leader_arm_teleop.py: the leader arm drives the Pioneer left arm in plain MuJoCo (CPU).

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


def run() -> None:
    parser = argparse.ArgumentParser(description="7-servo leader teleoperation of the Pioneer left arm in MuJoCo.")
    add_leader_args(parser, scene_help="scene registered in humanoid_mujoco_scenes (an unknown name lists them)")
    add_record_args(parser, task_description="sim leader teleop demonstration")
    args = parser.parse_args()
    check_leader_args(parser, args)
    record = load_record_schema(parser, args)

    import mujoco
    import mujoco.viewer
    import numpy as np
    from humanoid_mujoco_scenes import list_scenes, make_model, scene_camera, scene_step
    from pioneer_humanoid.arm_params import (
        LEFT_ARM_JOINTS,
        LEFT_GRIPPER_CLOSED,
        LEFT_GRIPPER_JOINTS,
        LEFT_GRIPPER_OPEN,
    )
    from pioneer_humanoid.arm_params import CAMERA_NAMES
    from pioneer_humanoid.mujoco_bimanual_arm import set_home

    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")
    unknown = sorted(set(record.images) - set(CAMERA_NAMES))
    if unknown:
        raise SystemExit(f"{record.path}: unknown images {unknown}; available: {list(CAMERA_NAMES)}")
    cameras = {name: (int(spec["height"]), int(spec["width"])) for name, spec in record.images.items()}
    model = make_model(args.scene, cameras=cameras)
    scene_hook = scene_step(args.scene)  # per-step scene mechanics (e.g. zip_tie's ratchet), or None
    data = mujoco.MjData(model)
    # Position actuator bias is [0, -kp, -kv]: lower the wrist's kv (see WRIST_DAMPING).
    model.actuator_biasprm[model.actuator("joint6l").id, 2] = -WRIST_DAMPING

    arm_acts = [model.actuator(j).id for j in LEFT_ARM_JOINTS]
    grip_acts = [model.actuator(j).id for j in LEFT_GRIPPER_JOINTS]
    grip_open = [LEFT_GRIPPER_OPEN[j] for j in LEFT_GRIPPER_JOINTS]
    grip_closed = [LEFT_GRIPPER_CLOSED[j] for j in LEFT_GRIPPER_JOINTS]
    set_home(model, data)
    mapping = LeaderMapping(
        args,
        home_rad=[data.ctrl[a] for a in arm_acts],
        limits_rad=[tuple(model.jnt_range[model.joint(j).id]) for j in LEFT_ARM_JOINTS],
    )
    print(mapping.describe(LEFT_ARM_JOINTS), flush=True)

    reset_requested = [False]

    def on_key(key: int) -> None:
        if key == _KEY_R:
            reset_requested[0] = True

    substeps = max(1, round(CONTROL_DT / model.opt.timestep))
    control_dt = substeps * model.opt.timestep
    recorder, record_every = make_sim_recorder(args, record, device="cpu", sim_dt=control_dt)
    if recorder is not None:
        print("[RECORD] Keys: S=start, N=save episode (then reset), D=discard")
        recorder.start_keyboard()
    renderers = {name: mujoco.Renderer(model, h, w) for name, (h, w) in cameras.items()}
    arm_qpos = [model.joint(j).qposadr[0] for j in LEFT_ARM_JOINTS]
    grip_qpos = [model.joint(j).qposadr[0] for j in LEFT_GRIPPER_JOINTS]

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
        mujoco.mj_forward(model, data)
        mapping.reset()

    leader = LeaderInput(args)
    clock = WallClock(control_dt)
    step = 0
    print("[INFO] Leader calibrated (hanging = 0). Move it to home: elbow bent 90 deg, forearm forward, gripper open. R = reset.", flush=True)
    try:
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

                target, grip = mapping.update(leader.read())
                data.ctrl[arm_acts] = target
                data.ctrl[grip_acts] = [o + grip * (c - o) for o, c in zip(grip_open, grip_closed)]

                # No frames until the leader is at home: a take starts from home.
                if recorder is not None and mapping.engaged and step % record_every == 0:
                    # Same features as the Isaac backend: 6 joints + mean finger closure / 6 targets + leader grip.
                    closure = sum(
                        (data.qpos[q] - o) / (c - o) for q, o, c in zip(grip_qpos, grip_open, grip_closed)
                    ) / len(grip_qpos)
                    state = np.append(data.qpos[arm_qpos], min(max(closure, 0.0), 1.0)).astype(np.float32)
                    action = np.append(target, grip).astype(np.float32)
                    with viewer.lock():
                        saved = recorder.tick(action, state, read_images)
                    if saved:
                        reset_all()
                        print("\n[RECORD] Episode saved; arm and scene reset.", flush=True)

                if scene_hook is not None:
                    scene_hook(model, data)
                mujoco.mj_step(model, data, nstep=substeps)
                step += 1
                viewer.sync()

                leader.report(mapping)
                clock.wait()
    finally:
        leader.close()
        print("\n[INFO] Stopped. Leader torque is OFF.", flush=True)
        for renderer in renderers.values():
            renderer.close()
        # Also on errors: flushes episodes still being written in the background.
        if recorder is not None:
            recorder.finalize()
            print(f"[RECORD] Saved under {recorder.dataset_root}")
