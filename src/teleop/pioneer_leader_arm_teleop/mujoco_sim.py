"""MuJoCo backend of pioneer_leader_arm_teleop.py: the leader arm drives the Pioneer left arm in plain MuJoCo (CPU).

Same leader mapping as the Isaac backend (leader_mapping.py). Scenes: humanoid_mujoco_scenes.

Keys go to the scene window (pioneer_humanoid.mujoco_scene_window, not mujoco.viewer, whose letter keys
toggle display options):

  R   reset the arm and every object in the scene; the arm then waits for the leader again
  Esc quit

  S   start: the arm follows once the leader is at home, as in Isaac. With --record, S also starts a take.

Recording (--record): same schema and features as the Isaac backend. When the scene's last step is done the
arm glides home by itself (the leader is ignored meanwhile) and the take is saved, then a new layout comes up
and the arm waits again.
  S start a take    N save now    B discard
Output under <repo>/datasets/<schema record.root>/sim.
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

from servo_leader import SERVO_IDS  # noqa: E402

from leader_mapping import (  # noqa: E402
    CONTROL_DT,
    WRIST_DAMPING,
    LeaderInput,
    LeaderMapping,
    WallClock,
    add_leader_args,
    check_leader_args,
)

GLIDE_S = 1.2            # a finished take glides home over this long ...
HOME_HOLD_S = 0.5        # ... and records this long at home before it is saved


def run() -> None:
    parser = argparse.ArgumentParser(description="7-servo leader teleoperation of the Pioneer left arm in MuJoCo.")
    add_leader_args(parser, scene_help="scene registered in humanoid_mujoco_scenes (an unknown name lists them)")
    add_record_args(parser, task_description="sim leader teleop demonstration")
    args = parser.parse_args()
    check_leader_args(parser, args)
    record = load_record_schema(parser, args)

    import mujoco
    import numpy as np
    from humanoid_mujoco_scenes import (
        list_scenes, make_model, scene_camera, scene_condition, scene_progress, scene_reset, scene_step,
    )
    from pioneer_humanoid.arm_params import (
        LEFT_ARM_JOINTS,
        LEFT_GRIPPER_CLOSED,
        LEFT_GRIPPER_JOINTS,
        LEFT_GRIPPER_OPEN,
    )
    from pioneer_humanoid.arm_params import CAMERA_NAMES
    from pioneer_humanoid.mujoco_bimanual_arm import set_home
    from pioneer_humanoid.mujoco_scene_window import SceneWindow
    from humanoid_mujoco_scenes.fallen import off_table

    if args.scene not in list_scenes():
        raise SystemExit(f"unknown --scene {args.scene!r}; available: {list_scenes()}")
    cameras = {name: (int(spec["height"]), int(spec["width"])) for name, spec in record.images.items()}
    # Arm cameras are mounted on the robot; any other image must be a camera the scene has (tidy_table's "top").
    model = make_model(args.scene, cameras={n: hw for n, hw in cameras.items() if n in CAMERA_NAMES})
    unknown = sorted(n for n in cameras
                     if n not in CAMERA_NAMES and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) < 0)
    if unknown:
        raise SystemExit(f"{record.path}: images {unknown} are neither arm cameras {list(CAMERA_NAMES)} "
                         f"nor cameras of scene {args.scene!r}")
    scene_hook = scene_step(args.scene)  # per-step scene mechanics (e.g. zip_tie's ratchet), or None
    scene_randomise = scene_reset(args.scene)  # new layout per episode, or None
    scene_steps = scene_progress(args.scene)  # multi-step task: (index, total, instruction), or None
    scene_cond = scene_condition(args.scene)  # (names, fn): the instruction as numbers, or None
    rng = np.random.default_rng()
    data = mujoco.MjData(model)
    # Position actuator bias is [0, -kp, -kv]: lower the wrist's kv (see WRIST_DAMPING).
    model.actuator_biasprm[model.actuator("joint6l").id, 2] = -WRIST_DAMPING

    arm_acts = [model.actuator(j).id for j in LEFT_ARM_JOINTS]
    grip_acts = [model.actuator(j).id for j in LEFT_GRIPPER_JOINTS]
    grip_open = [LEFT_GRIPPER_OPEN[j] for j in LEFT_GRIPPER_JOINTS]
    grip_closed = [LEFT_GRIPPER_CLOSED[j] for j in LEFT_GRIPPER_JOINTS]
    set_home(model, data)
    if scene_randomise is not None:
        scene_randomise(model, data, rng)
        mujoco.mj_forward(model, data)
    mapping = LeaderMapping(
        args,
        home_rad=[data.ctrl[a] for a in arm_acts],
        limits_rad=[tuple(model.jnt_range[model.joint(j).id]) for j in LEFT_ARM_JOINTS],
    )
    print(mapping.describe(LEFT_ARM_JOINTS), flush=True)

    substeps = max(1, round(CONTROL_DT / model.opt.timestep))
    control_dt = substeps * model.opt.timestep
    # Leader encoders every frame: angles (rad; calibrated, hanging = 0) and raw counts, in servo order.
    # Not "observation.*" -- LeRobot would feed those to the policy, and the leader IS the action.
    extra = {"leader_angles": list(SERVO_IDS), "leader_counts": list(SERVO_IDS)}
    # Multi-step scenes record the current step's instruction as each frame's task, plus its index.
    if scene_steps is not None:
        extra["subtask_index"] = ["subtask_index"]
    # ... and as numbers (e.g. tidy_table's one-hot target), a policy input: ACT has no language input.
    if scene_cond is not None:
        extra["observation.environment_state"] = scene_cond[0]
    recorder, record_every = make_sim_recorder(args, record, device="cpu", sim_dt=control_dt, extra_features=extra)
    renderers = {name: mujoco.Renderer(model, h, w) for name, (h, w) in cameras.items()}
    arm_qpos = [model.joint(j).qposadr[0] for j in LEFT_ARM_JOINTS]
    grip_qpos = [model.joint(j).qposadr[0] for j in LEFT_GRIPPER_JOINTS]

    def read_images():
        images = {}
        for name, renderer in renderers.items():
            renderer.update_scene(data, name)
            images[name] = renderer.render()
        return images

    leader = LeaderInput(args)
    window = SceneWindow(model, data, f"Pioneer leader teleop: {args.scene}", scene_camera(args.scene), one_shot="rsnb")
    clock = WallClock(control_dt)
    recording, saved = False, 0
    finish = None                                # task done while recording: [targets then, grip then, steps since]
    step, shown, msg = 0, None, ""

    mujoco.mj_forward(model, data)
    grounded = off_table(model, data)      # loose bodies that start below the table: not "fallen"

    def reset_all():
        """Arm and every object back to their defaults; the arm waits for the leader again."""
        nonlocal recording, finish, shown
        with window.lock:
            mujoco.mj_resetData(model, data)
            set_home(model, data)
            if scene_randomise is not None:
                scene_randomise(model, data, rng)
            mujoco.mj_forward(model, data)
            grounded.clear()
            grounded.update(off_table(model, data))
        mapping.reset()
        recording, finish, shown = False, None, None

    def save_take():
        nonlocal saved
        recorder.save_episode()
        saved += 1
        print(f"\n[RECORD] Take {saved} saved; new layout.", flush=True)
        reset_all()

    wait_for = "move the leader to home: elbow bent 90 deg, forearm forward, gripper open"
    print(f"[INFO] To start, {wait_for}. Keys (in the scene window): S start, R reset, Esc quit"
          + (", N save, B discard." if recorder is not None else "."), flush=True)
    try:
        while window.is_running():
            if recorder is not None and recorder.is_complete:
                print("\n[RECORD] Session complete.")
                break
            while window.events:
                ev = window.events.popleft()
                if ev == "esc":
                    raise KeyboardInterrupt
                if ev == "r":
                    if recording:
                        # Frames either side of a reset are not one demo: drop the take.
                        recorder.cancel_recording()
                        print("\n[RECORD] Reset mid-take: take discarded.", flush=True)
                    reset_all()
                    print("\n[LEADER] Arm and scene reset.", flush=True)
                elif ev == "s" and finish is None:
                    if not mapping.engaged:
                        print("\n[LEADER] Not started: the leader is not at home yet.", flush=True)
                        continue
                    if recorder is not None and not recording:
                        recording = True
                        print("\n[RECORD] Recording a take (N save, B discard).", flush=True)
                elif recorder is None:
                    pass
                elif ev == "n" and recording:
                    save_take()
                elif ev == "n":
                    print("\n[RECORD] Nothing to save: no take is recording (S starts one).", flush=True)
                elif ev == "b" and recording:
                    recorder.cancel_recording()
                    recording, finish = False, None
                    print("\n[RECORD] Take discarded (S to start again, R for a new layout).", flush=True)

            angles = leader.read()                 # every step, also while gliding: the encoders unwrap step by step
            if finish is None:
                target, grip = mapping.update(angles)
            else:
                # The task is done: glide home and hold there, whatever the leader does, so every take ends
                # the same way (a policy trained on takes that stop over the bins never learns to finish).
                finish[2] += 1
                s = min(finish[2] * control_dt / GLIDE_S, 1.0)
                ease = s * s * (3 - 2 * s)
                target = [q + ease * (h - q) for q, h in zip(finish[0], mapping.home)]
                grip = finish[1] * (1 - ease)

            if recording and step % record_every == 0:
                # Same features as the Isaac backend: 6 joints + mean finger closure / 6 targets + leader grip.
                closure = sum(
                    (data.qpos[q] - o) / (c - o) for q, o, c in zip(grip_qpos, grip_open, grip_closed)
                ) / len(grip_qpos)
                state = np.append(data.qpos[arm_qpos], min(max(closure, 0.0), 1.0)).astype(np.float32)
                action = np.append(target, grip).astype(np.float32)
                task = None
                extras = {
                    "leader_angles": np.asarray(leader.angles, dtype=np.float32),
                    "leader_counts": np.asarray(leader.counts(), dtype=np.float32),
                }
                if scene_steps is not None:
                    index, _, task = scene_steps(model, data)
                    extras["subtask_index"] = np.array([index], dtype=np.float32)
                if scene_cond is not None:
                    extras["observation.environment_state"] = scene_cond[1](model, data)
                recorder.push_frame_to_buffer(action, state, read_images(), extras=extras, task=task)

            with window.lock:
                data.ctrl[arm_acts] = target
                data.ctrl[grip_acts] = [o + grip * (c - o) for o, c in zip(grip_open, grip_closed)]
                if scene_hook is not None:
                    scene_hook(model, data)
                mujoco.mj_step(model, data, nstep=substeps)
            step += 1
            fell = off_table(model, data) - grounded
            if fell:
                # Nothing can pick it up from the floor: this layout is over.
                print(f"\n[RESET] {', '.join(sorted(fell))} fell off the table: new layout"
                      + (" (take discarded)." if recording else "."), flush=True)
                if recording:
                    recorder.cancel_recording()
                reset_all()
            if scene_steps is not None:
                index, total, instruction = scene_steps(model, data)
                if (index, instruction) != shown:
                    shown = (index, instruction)
                    done_msg = "going home, then saving" if recording else "R for a new layout"
                    msg = f"all steps done -- {done_msg}" if index >= total else f"step {index + 1}/{total}: {instruction}"
                    print(f"\n[TASK] {msg}", flush=True)
                if recording and index >= total and finish is None:
                    finish = [list(target), grip, 0]
            if finish is not None and finish[2] * control_dt >= GLIDE_S + HOME_HOLD_S:
                save_take()

            state_text = "" if recorder is None else "RECORDING" if recording else "not recording (S)"
            waiting = "" if mapping.engaged else f"waiting: {wait_for.split(':')[0]}"
            window.sync("\n".join(s for s in (state_text, waiting, msg) if s))
            leader.report(mapping)
            clock.wait()
    except KeyboardInterrupt:
        pass
    finally:
        leader.close()
        print("\n[INFO] Stopped. Leader torque is OFF.", flush=True)
        window.close()
        for renderer in renderers.values():
            renderer.close()
        # Also on errors: flushes episodes still being written in the background.
        if recorder is not None:
            if recording:
                recorder.cancel_recording()
                print("[RECORD] The take in progress was not saved (N saves).", flush=True)
            # Interrupted here, the dataset folder is left half-written and can't be re-opened.
            if saved:
                print("[RECORD] Finishing the saved takes -- wait, don't press Ctrl+C.", flush=True)
            recorder.finalize()
            print(f"[RECORD] {saved} take(s) saved this session under {recorder.dataset_root}")
