# Run collection, correction validation, aggregation and policy evaluation.
import argparse
import importlib
import json
import math
from pathlib import Path
import time

from .core import Session
from .storage import CorrectionStore, open_store


# Load an adapter factory from its module:function path.
def factory(spec):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


# Read application settings as a JSON object.
def options(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"Expected a JSON object: {error.msg}") from error
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("Adapter options must be a JSON object")
    return value


# Add the shared flags for selecting a scene adapter.
def add_scene_arguments(parser):
    parser.add_argument("--scene-adapter", required=True, metavar="MODULE:FACTORY",
                        help="Factory implementing the Scene protocol; no built-in scene is selected")
    parser.add_argument("--scene-options", type=options, default={}, metavar="JSON",
                        help="Settings passed unchanged to your scene adapter")


# Measure policy success on fixed seeds without human interventions.
def evaluate(scene, policy, task, seeds, seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Episode seconds must be positive")
    # Check that the scene can report success before running evaluations.
    scene.reset(seeds[0])
    scene.success()
    rows = []
    for seed in seeds:
        scene.reset(seed)
        policy.reset()
        success = False
        for step in range(max(1, math.ceil(seconds * scene.contract.fps))):
            obs = scene.observe()
            scene.contract.check_observation(obs)
            scene.execute(scene.contract.action(policy.action(obs, task)))
            if scene.success():
                success = True
                break
        rows.append({"seed": seed, "success": success, "steps": step + 1})
        print(f"[HG-DAgger eval] seed={seed} success={success}", flush=True)
    return {"successes": sum(r["success"] for r in rows), "episodes": len(rows),
            "success_rate": sum(r["success"] for r in rows) / len(rows), "rollouts": rows,
            "contract": json.loads(scene.contract.json()), "seconds": seconds, "task": task}


# Run the requested collection, data or evaluation command.
def main():
    parser = argparse.ArgumentParser(description="Collection, local validation, aggregation and unassisted evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    add_scene_arguments(collect)
    collect.add_argument("--teleop-adapter", required=True, metavar="MODULE:FACTORY")
    collect.add_argument("--teleop-options", type=options, default={}, metavar="JSON")
    collect.add_argument("--task", required=True)
    collect.add_argument("--policy", required=True)
    collect.add_argument("--corrections", required=True)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--max-frames", type=int, default=1500)
    validate = commands.add_parser("validate")
    validate.add_argument("root")
    bind = commands.add_parser("bind")
    add_scene_arguments(bind)
    bind.add_argument("--dataset", required=True)
    bind.add_argument("--policy", help="Local pretrained_model directory trained on this dataset")
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--base", nargs="+", required=True)
    aggregate.add_argument("--corrections", nargs="+", required=True)
    aggregate.add_argument("--output", required=True)
    evaluation = commands.add_parser("evaluate")
    add_scene_arguments(evaluation)
    evaluation.add_argument("--task", required=True)
    evaluation.add_argument("--policy", required=True)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--seed", type=int, default=81000)
    evaluation.add_argument("--episodes", type=int, default=50)
    evaluation.add_argument("--seconds", type=float, default=30)
    evaluation.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        # Count and check saved corrections without starting a scene.
        segments = frames = 0
        for episode, _ in open_store(args.root).episodes():
            segments += 1
            frames += len(episode)
        print(json.dumps({"segments": segments, "frames": frames}))
        return
    if args.command == "aggregate":
        # Export original demonstrations and accepted corrections together.
        from .learning import aggregate as build
        print(json.dumps(build(args.base, args.corrections, args.output)))
        return
    scene = factory(args.scene_adapter)(args.scene_options)
    teleop = None
    try:
        if args.command == "bind":
            from .learning import bind_contract
            bind_contract(args.dataset, args.policy, scene.contract)
            print("Contract verified and attached to local dataset/checkpoint")
            return
        from .learning import LeRobotPolicy
        policy = LeRobotPolicy(args.policy, scene.contract, args.device)
        if args.command == "evaluate":
            # Save a new evaluation report without overwriting earlier results.
            if args.episodes <= 0:
                raise ValueError("episodes must be positive")
            output = Path(args.output)
            if output.exists():
                raise FileExistsError(output)
            result = evaluate(scene, policy, args.task, list(range(args.seed, args.seed + args.episodes)), args.seconds)
            result["policy"] = args.policy
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Success: {result['successes']}/{result['episodes']} ({result['success_rate']:.1%})")
            return
        store = CorrectionStore(args.corrections, scene.contract)
        teleop = factory(args.teleop_adapter)(scene, args.teleop_options)
        session = Session(scene, teleop, policy, store, args.task, args.seed, args.max_frames)
        print("[HG-DAgger] Waiting for teleoperation readiness. Accept: save/reset; "
              "discard: discard/reset; Ctrl+C: quit without saving", flush=True)
        previous, next_tick = None, time.monotonic()
        # Poll decisions and run control at the scene's declared frame rate.
        while session.tick():
            status = (f"{session.mode if session.started else 'WAITING'} | Unsaved {session.pending_frames} | "
                      f"Saved segments {session.accepted}")
            if session.pending_frames >= session.max_frames:
                status = "BUFFER FULL - " + status
            if status != previous:
                print("[HG-DAgger] " + status, flush=True)
                previous = status
            if hasattr(teleop, "feedback"):
                teleop.feedback(status)
            if hasattr(scene, "preview") and scene.preview(status):
                break
            next_tick += 1 / scene.contract.fps
            time.sleep(max(0, next_tick - time.monotonic()))
            next_tick = max(next_tick, time.monotonic())
    except KeyboardInterrupt:
        print("[HG-DAgger] Stopped; only explicitly accepted corrections are saved")
    finally:
        if teleop is not None:
            teleop.close()
        scene.close()


if __name__ == "__main__":
    main()
