"""Raw takes -> a LeRobot dataset in pioneer_umi_dataset_schema.yaml (TCP pose + gripper, wrist camera, 25 fps).

    python -m umi_gripper.postprocess.to_lerobot datasets/umi_raw/20261120_101500            # one session
    python -m umi_gripper.postprocess.to_lerobot datasets/umi_raw --dry-run                   # check every take
    python -m umi_gripper.postprocess.to_lerobot datasets/umi_raw --out datasets/pioneer_umi/real --overwrite

The rig's CURRENT config and calibration are applied (takes store the raw grip pose), so recalibrating and
re-running re-converts old takes. A report of every take (kept or why not) is written next to the dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from umi_gripper import config as umi_config
from umi_gripper._paths import ROBOT_LEARNING, add
from umi_gripper.postprocess.quality import check_episode
from umi_gripper.postprocess.sync import episode_vectors, sync_take
from umi_gripper.take import find_takes, load_take

SCHEMA = ROBOT_LEARNING / "config" / "pioneer_umi_dataset_schema.yaml"


def load_schema(no_images: bool = False) -> dict:
    add(ROBOT_LEARNING)
    from humanoid_robot_learning.recording.schema import load_yaml, select_cameras
    schema = load_yaml(SCHEMA)
    return select_cameras(schema, "none") if no_images else schema


def evaluate_takes(paths: list[Path], cfg: dict, require_camera: bool) -> list[dict]:
    """Sync and check every take; one report entry each, with the verdict's episode under "_verdict"."""
    entries = []
    for path in paths:
        entry = {"take": str(path)}
        try:
            take = load_take(path)
            verdict = check_episode(sync_take(take, cfg), cfg["postprocess"])
            reasons = list(verdict.reasons)
            if require_camera and verdict.episode.frame_index is None:
                reasons.append("take has no camera frames (recorded with --no-camera)")
            entry.update(task=take.meta.get("task", ""), ok=not reasons, reasons=reasons, **verdict.stats)
            entry["_verdict"] = verdict
        except (OSError, ValueError, KeyError) as exc:
            entry.update(ok=False, reasons=[f"unreadable: {exc}"])
        entries.append(entry)
    return entries


def write_dataset(entries: list[dict], schema: dict, out: Path) -> int:
    add(ROBOT_LEARNING)
    from humanoid_robot_learning.recording.schema import enabled_images
    from humanoid_robot_learning.recording.sinks.lerobot import LeRobotSink
    from humanoid_robot_learning.recording.snapshot import ObservationSnapshot

    images = list(enabled_images(schema))
    sink = LeRobotSink(schema, root=out)
    written = 0
    try:
        for entry in entries:
            if not entry["ok"]:
                continue
            episode = entry["_verdict"].episode
            state, action = episode_vectors(episode)
            for step in range(len(episode)):
                frames = {}
                if images:
                    frames[images[0]] = episode.take.load_frame(int(episode.frame_index[step]))
                sink.add_frame(ObservationSnapshot(state=state[step], action=action[step], images=frames),
                               task=entry["task"])
            sink.save_episode()
            written += 1
    finally:
        sink.finalize()
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="takes, sessions or the raw root")
    parser.add_argument("--out", type=Path, default=None, help="dataset root (default: the schema's record.root/real)")
    parser.add_argument("--config", type=Path, default=umi_config.DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=umi_config.DEFAULT_CALIBRATION)
    parser.add_argument("--dry-run", action="store_true", help="check and report only; no lerobot needed")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset at --out")
    parser.add_argument("--no-images", action="store_true", help="poses and gripper only (no camera feature)")
    args = parser.parse_args(argv)

    cfg = umi_config.load_config(args.config, args.calibration)
    paths = [take for root in args.inputs for take in find_takes(root)]
    if not paths:
        parser.error(f"no saved takes under {[str(p) for p in args.inputs]}")
    schema = load_schema(args.no_images)
    if int(schema["fps"]) != int(cfg["postprocess"]["fps"]):
        parser.error(f"schema fps {schema['fps']} != postprocess.fps {cfg['postprocess']['fps']}")
    entries = evaluate_takes(paths, cfg, require_camera=not args.no_images)
    kept = [e for e in entries if e["ok"]]
    for entry in entries:
        mark = "keep" if entry["ok"] else "drop"
        detail = f"{entry.get('duration_s', 0):6.1f} s" if entry["ok"] else "; ".join(entry["reasons"])
        print(f"[{mark}] {entry['take']}: {detail}")
    print(f"{len(kept)}/{len(entries)} takes pass")

    out = args.out or umi_config.resolve_repo_path(schema["record"]["root"]) / "real"
    report = [{k: v for k, v in e.items() if not k.startswith("_")} for e in entries]
    if args.dry_run or not kept:
        return 0 if kept else 1
    if out.exists():
        if not args.overwrite:
            parser.error(f"{out} exists: pass --overwrite to replace it, or --out elsewhere")
        shutil.rmtree(out)
    written = write_dataset(entries, schema, out)
    report_path = out.parent / f"{out.name}_umi_report.json"
    report_path.write_text(json.dumps({"dataset": str(out), "config": cfg, "takes": report}, indent=2, default=str))
    print(f"wrote {written} episodes to {out} (report: {report_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
