"""Isaac cargo HG-DAgger and synthetic expert entry point; see assets/nasa_cargo/HG_DAGGER.md."""
import argparse
import json
import os
import sys
import traceback
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for package in ("src/robot_learning", "src/pioneer_humanoid", "src/simulation/isaac_scenes"):
    sys.path.insert(0, str(ROOT / package))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("probe", "generate", "collect", "evaluate", "teleop"), default="probe")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-attempts", type=int, default=200)
parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/nasa_cargo/synthetic_50")
parser.add_argument("--policy", type=Path)
parser.add_argument("--no-cameras", action="store_true", help="Physics probe only")
parser.add_argument("--resume", action="store_true", help="Append to a matching dataset; never overwrite")
parser.add_argument("--round-id", default="round_001")
parser.add_argument("--max-seconds", type=float, default=45.)
parser.add_argument("--max-segment-frames", type=int, default=750,
                    help="Human frame buffer: VR waits for B/Y at the limit; keyboard saves at the boundary")
parser.add_argument("--report", type=Path, default=ROOT / "outputs/nasa_cargo/evaluation.json")
parser.add_argument("--debug-images", action="store_true")
parser.add_argument("--randomization", choices=("baseline", "varied", "wide", "tray"), default="baseline",
                    help="varied: XY +/-1.5cm; wide: bag X=0.475..0.525m, Y=0.115..0.15m, larger pickup support. "
                         "tray: fixed cargo, moving green tray. Body colour and arm starts also vary.")
parser.add_argument("--teleop", choices=("keyboard", "vr"), default="keyboard")
parser.add_argument("--vr-port", type=int, default=8444)
parser.add_argument("--vr-gain", type=float, default=1.0)
parser.add_argument("--vr-calibration", type=Path)
parser.add_argument("--single-task-thread", action="store_true",
                    help="Use the VR branch's Carbonite task-worker startup workaround")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.no_cameras and args.mode != "probe":
    parser.error("--no-cameras is only supported for a physics probe")
if args.episodes <= 0 or args.max_attempts <= 0 or args.max_segment_frames <= 0:
    parser.error("Episode, attempt and segment counts must be positive")
if not 0 < args.max_seconds < float("inf"):
    parser.error("--max-seconds must be positive and finite")
if args.mode in ("collect", "evaluate") and args.policy is None:
    parser.error("--policy is required for collection/evaluation")
if args.mode in ("collect", "teleop") and args.headless and args.teleop == "keyboard":
    parser.error("Keyboard collection needs the Isaac window; omit --headless")
if args.teleop == "vr" and args.mode not in ("collect", "teleop"):
    parser.error("--teleop vr is for collect/teleop; evaluate runs the policy autonomously")
if not 1 <= args.vr_port <= 65535 or not 0 < args.vr_gain < float("inf"):
    parser.error("VR port must be 1..65535 and VR gain positive and finite")
if args.single_task_thread:
    args.kit_args = f"{args.kit_args or ''} --/plugins/carb.tasking.plugin/threadCount=1".strip()
if not args.no_cameras:
    args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
from humanoid_robot_learning.sim.cargo_env import CargoEnv, FPS, TASK
from humanoid_robot_learning.sim.cargo_expert import ArmKinematics, CargoExpert
from humanoid_robot_learning.sim.cargo_data import EpisodeWriter


def synthetic(env):
    kin = ArmKinematics()
    writer = EpisodeWriter(args.dataset_root, env.contract(), args.resume) if args.mode == "generate" else None
    if writer:
        profile_path = writer.root / "randomization_profile.json"
        previous = json.loads(profile_path.read_text()) if profile_path.exists() else "baseline"
        if writer.episodes and previous != args.randomization:
            raise ValueError("Resume profile differs; use a new dataset directory")
        profile_path.write_text(json.dumps(args.randomization) + "\n")
    successes = writer.episodes if writer else 0
    used_seeds = set()
    if writer and (writer.root / "collection_manifest.jsonl").exists():
        used_seeds = {json.loads(line)["seed"] for line in (writer.root / "collection_manifest.jsonl").read_text().splitlines()}
    for attempt in range(args.max_attempts):
        if successes >= args.episodes:
            break
        seed = args.seed + attempt
        if seed in used_seeds:
            continue
        rng = env.reset(seed)
        q = env.robot.data.joint_pos[0, env.arm_ids].cpu().numpy()
        tip, quat = env.tip_pose()
        predicted, _ = kin.fk(q)
        mismatch = np.linalg.norm(predicted + env.robot.data.root_pos_w[0].cpu().numpy() - tip)
        print(f"[Expert] seed={seed} initial FK error={mismatch:.5f}m q={q.round(3)} tip={tip.round(3)}", flush=True)
        expert = CargoExpert(kin, env.robot.data.root_pos_w[0].cpu().numpy(), q,
                             env.cargo.data.root_pos_w[0].cpu().numpy(),
                             env.cargo.data.root_quat_w[0].cpu().numpy(), env.target_xy, rng,
                             retreat_height=.10 if env.randomization == "tray" else .15,
                             target_yaw=env.reset_metadata.get("expert_place_yaw_rad", 0.))
        maximum = env.cargo.data.root_pos_w[0, 2].item()
        previous = None
        ok, failure = False, None
        frames = []
        phases = []
        try:
            for frame in range(int(expert.duration * FPS) + 1):
                obs = env.observe() if writer else None
                action = expert.action(1/FPS, env.cargo.data.root_pos_w[0].cpu().numpy(), env.tip_pose()[0])
                actual = env.command(action)
                if writer:
                    frames.append({**obs, "action": actual, "task": TASK})
                    phases.append(expert.phase)
                ok = env.advance()
                maximum = max(maximum, env.cargo.data.root_pos_w[0, 2].item())
                if previous != expert.phase:
                    tip, _ = env.tip_pose()
                    print(f"[Expert] {expert.phase}: tip={tip.round(3)} cargo={env.cargo.data.root_pos_w[0].cpu().numpy().round(3)}", flush=True)
                    print(f"[Expert] cargo_quat={env.cargo.data.root_quat_w[0].cpu().numpy().round(3)} tool_quat={env.tip_pose()[1].round(3)}", flush=True)
                    if args.debug_images:
                        from PIL import Image
                        debug = ROOT / "outputs/nasa_cargo/expert_debug"
                        debug.mkdir(parents=True, exist_ok=True)
                        for key, value in env.observe().items():
                            if key.startswith("observation.images."):
                                Image.fromarray(value).save(debug / f"seed{seed}_{expert.phase}_{key.rsplit('.',1)[-1]}.png")
                    previous = expert.phase
            ok = ok and env.last[6] < .05
        except RuntimeError as exc:
            ok = False
            failure = str(exc)
        print(f"[Expert] seed={seed} success={ok} max_cargo_z={maximum:.3f} max_ik_error={expert.max_error:.4f} failure={failure}", flush=True)
        print(f"[Expert] final status={ {k: v.cpu().tolist() for k,v in env.status.items()} } quat={env.cargo.data.root_quat_w[0].cpu().numpy().round(3)}", flush=True)
        provenance = {
            "source": "synthetic_expert", "seed": seed, "success": bool(ok),
            "lifted": bool(env.task.lifted[0].item()), "released": bool(env.last[6] < .05),
            "cargo_motion": "robot_contact_only", "maximum_cargo_z": maximum,
            "max_ik_error": expert.max_error, "failure": failure,
            "duration_s": len(frames) / FPS if writer else expert.elapsed,
            "reset": env.reset_metadata,
        }
        if writer:
            with (writer.root / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(provenance) + "\n")
            if ok:
                writer.save(frames, {**provenance, "phases": phases})
            frames.clear()
        successes += int(ok)
        if successes >= args.episodes:
            break
    if successes < args.episodes:
        if writer:
            writer.close()
        raise RuntimeError(f"Only {successes}/{args.episodes} successful demonstrations")
    if writer:
        writer.close()
        print(f"CARGO_GENERATION_COMPLETE episodes={successes} root={writer.root}", flush=True)


def load_actor(env):
    from humanoid_robot_learning.sim.cargo_policy import CargoPolicy
    actor = CargoPolicy(str(args.policy), args.device, env.contract())
    # Locate the original dataset contract from the training configuration when available.
    train_config = args.policy / "train_config.json"
    if train_config.exists():
        trained_root = json.loads(train_config.read_text()).get("dataset", {}).get("root")
        if trained_root and (Path(trained_root) / "cargo_contract.json").exists():
            if json.loads((Path(trained_root) / "cargo_contract.json").read_text()) != env.contract():
                raise ValueError("Checkpoint training dataset used a different cargo contract")
    return actor


def evaluate(env):
    actor = load_actor(env)
    results = []
    for index in range(args.episodes):
        seed = args.seed + index
        print(f"[Playback] Starting episode {index+1}/{args.episodes}, seed={seed}, "
              f"timeout={args.max_seconds:g} simulated seconds", flush=True)
        env.reset(seed)
        actor.reset()
        success = False
        start_z = env.cargo.data.root_pos_w[0, 2].item()
        maximum_z = start_z
        minimum_handle_distance = float("inf")
        for frame in range(int(args.max_seconds * FPS)):
            obs = env.observe()
            tip, _ = env.tip_pose()
            cargo_pos = env.cargo.data.root_pos_w[0].cpu().numpy()
            minimum_handle_distance = min(minimum_handle_distance, float(np.linalg.norm(
                tip - (cargo_pos + [0, .08002, .363]))))
            if frame % (5 * FPS) == 0:
                print(f"[Evaluation] seed={seed} t={frame/FPS:.1f}s cargo_z={cargo_pos[2]:.3f} "
                      f"command_gripper={env.last[6]:.3f}", flush=True)
            if args.debug_images and frame % (5 * FPS) == 0:
                from PIL import Image
                debug = args.report.parent / (args.report.stem + "_images")
                debug.mkdir(parents=True, exist_ok=True)
                for key, value in obs.items():
                    if key.startswith("observation.images."):
                        Image.fromarray(value).save(debug / f"seed{seed}_t{frame/FPS:05.1f}_{key.rsplit('.',1)[-1]}.png")
            env.command(actor.action(obs, TASK))
            success = env.advance() and env.last[6] < .05
            maximum_z = max(maximum_z, env.cargo.data.root_pos_w[0, 2].item())
            if success:
                measured_closure = ((env.robot.data.joint_pos[:, env.grip_ids] - env.g_open)
                                    / (env.g_closed - env.g_open)).mean().clamp(0, 1).item()
                if measured_closure < .1:
                    break
                success = False
        measured_closure = ((env.robot.data.joint_pos[:, env.grip_ids] - env.g_open)
                            / (env.g_closed - env.g_open)).mean().clamp(0, 1).item()
        success = success and measured_closure < .1
        row = {"seed": seed, "success": bool(success), "seconds": (frame + 1) / FPS,
               "ever_lifted": bool(env.task.lifted[0].item()), "maximum_cargo_rise_m": maximum_z-start_z,
               "inside_target": bool(env.status["inside_target"][0].item()),
               "minimum_nominal_handle_distance_m": minimum_handle_distance,
               "final_command_gripper": float(env.last[6]), "final_measured_gripper": measured_closure,
               "reset": env.reset_metadata,
               "final_cargo_position": env.cargo.data.root_pos_w[0].cpu().tolist()}
        results.append(row)
        print(f"[Playback] Finished episode {index+1}/{args.episodes}: success={bool(success)}; "
              + ("resetting for next episode" if index+1 < args.episodes else "playback complete"), flush=True)
        print(f"[Evaluation] {row}", flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"checkpoint": str(args.policy), "episodes": results,
                                           "in_progress": True}, indent=2) + "\n")
    report = {"checkpoint": str(args.policy), "episodes": results,
              "success_rate": sum(r["success"] for r in results) / len(results), "contract": env.contract(),
              "randomization": args.randomization, "max_seconds": args.max_seconds}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"CARGO_EVALUATION_COMPLETE success_rate={report['success_rate']:.3f} report={args.report}", flush=True)


def collect(env):
    import carb
    from humanoid_robot_learning.sim.cargo_control import HumanGate

    writer = EpisodeWriter(args.dataset_root, env.contract(), args.resume)
    actor = load_actor(env) if args.mode == "collect" else None
    vr = args.teleop == "vr"
    stale_input_errors = ()
    if vr:
        from humanoid_robot_learning.sim.cargo_vr import VrExpert, VrInputUnavailable
        stale_input_errors = (VrInputUnavailable,)
        expert = VrExpert(env, args.vr_port, args.vr_gain, args.vr_calibration)
        expert.start()
    else:
        from humanoid_robot_learning.sim.cargo_keyboard import KeyboardExpert
        expert = KeyboardExpert(ArmKinematics())
    seed = args.seed
    env.reset(seed)
    expert.seed(env)
    if actor:
        actor.reset()
    events = []
    segment = 0

    def save(frames, reason):
        nonlocal segment
        writer.save(frames, {"source": "human_intervention" if actor else "human_demo",
                                "seed": seed, "round": args.round_id, "checkpoint": str(args.policy),
                                "teleop": "quest_controller_v2" if vr else "keyboard",
                                "save_mode": "explicit_button" if vr else "hand_back",
                                "reset": env.reset_metadata,
                                "segment": segment, "boundary": reason})
        segment += 1

    gate = HumanGate(env, expert, actor, save, args.max_segment_frames, save_on_hand_back=not vr)
    gate.saved_segments = writer.episodes

    def reset_attempt(notice):
        nonlocal seed
        seed += 1
        env.reset(seed)
        gate.reset(resume_policy=vr)
        if vr:
            expert.waiting_for_release = True
        gate.notice = notice
        print(f"[HG-DAgger] {notice}; reset to seed={seed}", flush=True)

    def key(event, *unused):
        if event.input.name in ("LEFT_SHIFT", "RIGHT_SHIFT"):
            expert.fine = event.type != carb.input.KeyboardEventType.KEY_RELEASE
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in ("H", "P", "B", "N", "DELETE", "ESCAPE"):
                events.append(event.input.name)
        return True

    interface, keyboard, subscription = None, None, None
    if not args.headless:
        import omni.appwindow
        interface = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        subscription = interface.subscribe_to_keyboard_events(keyboard, key)
    if not args.headless:
        print("[HG-DAgger] H takeover/return | P pause | N save | Delete discard | B reset | Esc quit"
              + (" | K gripper" if not vr else ""), flush=True)
    if vr:
        print(f"[HG-DAgger VR] adb reverse tcp:{args.vr_port} tcp:{args.vr_port}; "
              f"Quest browser http://localhost:{args.vr_port}/ -> Enter VR", flush=True)
        print("[HG-DAgger VR] Hold RIGHT grip: human; release: return WITHOUT saving. "
              "RIGHT B: save then reset; LEFT Y: discard then reset. LEFT motion/trigger: arm/gripper.", flush=True)
        print("[HG-DAgger VR] Waiting before starting: Enter VR with head and both controllers tracked, "
              "and release both grip buttons. Physics and recording are held until then.", flush=True)
    else:
        print("[HG-DAgger] W/S A/D Q/E translation; Z/X T/G C/V rotation; Shift fine. Focus viewport.", flush=True)
    print(f"[HG-DAgger] mode={gate.mode}", flush=True)
    quit_requested = False
    last_stream = 0.
    try:
        while app.is_running() and not quit_requested and writer.episodes < args.episodes:
            started = time.monotonic()
            if vr:
                was_waiting = expert.waiting_for_start
                authority = expert.poll(env)
                if was_waiting and not expert.waiting_for_start:
                    print("[HG-DAgger VR] VR ready; starting control.", flush=True)
                if expert.reset_requested:
                    events.append("B")
                elif actor and authority is not None and authority != gate.mode:
                    gate.toggle()
                    print(f"[HG-DAgger] mode={gate.mode}", flush=True)
                if not expert.reset_requested and expert.save_requested:
                    print(f"[HG-DAgger VR] RIGHT B received; unsaved human frames={gate.pending_frames}", flush=True)
                    events.append("VR_SAVE")
            while events:
                event = events.pop(0)
                if event == "ESCAPE":
                    quit_requested = True
                elif event == "H" and actor:
                    gate.toggle()
                    print(f"[HG-DAgger] mode={gate.mode}", flush=True)
                elif event == "P":
                    gate.pause()
                    print(f"[HG-DAgger] paused={gate.paused}", flush=True)
                elif event in ("N", "VR_SAVE"):
                    count = gate.pending_frames
                    gate.save("vr_operator_save" if event == "VR_SAVE" else "operator_save")
                    gate.notice = (f"SAVED {count} human frames; total saved segments={writer.episodes}" if count
                                   else f"NO NEW CORRECTION TO SAVE; total saved segments={writer.episodes}. "
                                        "Hold RIGHT grip to record human actions")
                    if event == "VR_SAVE":
                        # Save completes before reset. A write error must never trigger a reset.
                        reset_attempt(gate.notice)
                    else:
                        print(f"[HG-DAgger] {gate.notice}", flush=True)
                elif event == "DELETE":
                    print(f"[HG-DAgger] Discarded {gate.discard()} unsaved human frames", flush=True)
                elif event == "B":
                    count = gate.pending_frames
                    reset_attempt(f"DISCARDED {count} unsaved human frames; total saved segments={writer.episodes}")
            if quit_requested:
                break
            if vr:
                expert.feedback(gate)
            if (gate.paused or gate.buffer_full or not env.sim.is_playing()
                    or (vr and (expert.waiting_for_start or expert.waiting_for_release
                                or (gate.mode == "HUMAN" and not expert.is_ready())))):
                if vr and time.monotonic() - last_stream >= .1:
                    expert.publish(env.observe(), gate)
                    last_stream = time.monotonic()
                else:
                    env.sim.render()
                time.sleep(.02)
                continue
            obs = env.observe()
            if vr and time.monotonic() - last_stream >= .1:
                expert.publish(obs, gate)
                last_stream = time.monotonic()
            if vr and gate.mode == "HUMAN" and not expert.is_ready():
                continue
            try:
                selected = gate.select(obs, TASK)
            except stale_input_errors:
                # Input can expire between the readiness check and action selection.
                # No command, physics step or training label is admitted here.
                continue
            actual = env.command(selected)
            gate.record(obs, actual, TASK)
            env.advance()
            time.sleep(max(0., 1/FPS - (time.monotonic() - started)))
    finally:
        # Exiting/discard/reset never implicitly saves the current segment.
        writer.close()
        if vr:
            expert.close()
        if subscription is not None:
            interface.unsubscribe_to_keyboard_events(keyboard, subscription)
        print(f"[HG-DAgger] Saved {writer.episodes} human segments; discarded {gate.pending_frames} pending frames", flush=True)


def main():
    env = CargoEnv(args.device, cameras=not args.no_cameras, randomization=args.randomization)
    if args.mode in ("probe", "generate"):
        synthetic(env)
    elif args.mode == "evaluate":
        evaluate(env)
    else:
        collect(env)


try:
    main()
except KeyboardInterrupt:
    print("[HG-DAgger] Stopped by operator; pending correction frames discarded", flush=True)
    os._exit(0)
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(1)
else:
    sys.stdout.flush()
    os._exit(0)
