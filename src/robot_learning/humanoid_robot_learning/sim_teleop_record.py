"""Recording setup shared by the Isaac Sim teleop scripts (keyboard, leader arm)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from humanoid_robot_learning.record_utils import resolve_config_path, resolve_dataset_root
from humanoid_robot_learning.schema import enabled_images, load_yaml, select_cameras

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = PKG_ROOT / "config" / "dataset_schema_pioneer_v1.yaml"

# Recorded joint order (real-robot names, joint_command_core.cpp); must match the schema.
LEFT_ARM_RECORD_JOINTS = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_elbow_roll",
    "left_wrist_pitch",
    "left_gripper",
]


@dataclass
class RecordSchema:
    """Schema loaded for --record; empty when not recording."""

    path: Path | None = None
    cfg: dict[str, Any] | None = None
    images: dict[str, dict[str, Any]] = field(default_factory=dict)


def add_record_args(parser: argparse.ArgumentParser, *, task_description: str) -> None:
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record demonstrations (requires: pip install -e src/robot_learning[sim])",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=str(DEFAULT_SCHEMA),
        help="dataset_schema YAML (default: src/robot_learning/config/dataset_schema_pioneer_v1.yaml)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Output directory (default: <repo>/<schema record.root>/sim)",
    )
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--task_description", type=str, default=task_description)
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="cameras to record, e.g. 'ego,wrist_left' or 'none' (default: schema's enabled images)",
    )


def load_record_schema(parser: argparse.ArgumentParser, args: argparse.Namespace) -> RecordSchema:
    """Call before AppLauncher: recording cameras need --enable_cameras."""
    if args.cameras is not None and not args.record:
        parser.error("--cameras requires --record")
    if not args.record:
        return RecordSchema()
    path = resolve_config_path(args.schema, anchor=PKG_ROOT)
    try:
        cfg = select_cameras(load_yaml(path), args.cameras)
    except ValueError as exc:
        parser.error(f"--cameras: {exc}")
    images = enabled_images(cfg)
    if images:
        args.enable_cameras = True
    return RecordSchema(path, cfg, images)


def make_sim_recorder(
    args: argparse.Namespace,
    schema: RecordSchema,
    *,
    device: str,
    sim_dt: float,
    joint_names: list[str] = LEFT_ARM_RECORD_JOINTS,
    extra_features: dict[str, list[str]] | None = None,
):
    """Return (recorder, record_every): record one frame every `record_every` physics steps.

    ``extra_features``: extra float32 per-frame features, {name: [component names]} (see SimLeRobotRecorder).
    """
    if not args.record:
        return None, 0
    try:
        from humanoid_robot_learning.sim_recorder import SimLeRobotRecorder
    except ImportError as exc:
        raise ImportError(
            "Recording requires humanoid-robot-learning. Install with:\n"
            "  pip install -e src/robot_learning[sim]"
        ) from exc

    cfg = schema.cfg
    if list(cfg["joint_names"]) != joint_names:
        raise ValueError(
            f"{schema.path}: joint_names must be {joint_names}, got {list(cfg['joint_names'])}"
        )
    fps = int(cfg.get("fps", 25))
    physics_hz = 1.0 / sim_dt
    record_every = round(physics_hz / fps)
    if record_every < 1 or abs(record_every * fps - physics_hz) > 1e-6:
        raise ValueError(f"{schema.path}: fps={fps} must divide the physics rate ({physics_hz:g} Hz)")
    dataset_root = resolve_dataset_root(cfg, args.dataset_root, subdir="sim")
    cameras = {name: {"height": spec["height"], "width": spec["width"]} for name, spec in schema.images.items()}
    recorder = SimLeRobotRecorder(
        task_name=args.task_description,
        repo_id=str(cfg.get("repo_id", "humanoid/sim")),
        dataset_root=dataset_root,
        fps=fps,
        device=device,
        joint_names=list(cfg["joint_names"]),
        cameras=cameras,
        num_episodes=args.num_episodes,
        robot_type=str(cfg.get("robot_id", "pioneer_v1_left_arm")),
        rate_limit=False,
        extra_features=extra_features,
    )
    recorder.init_dataset()
    print(f"[RECORD] Writing to {dataset_root} at {fps} fps (every {record_every} physics steps)")
    if cameras:
        print(f"[RECORD] Cameras: {sorted(cameras)}")
    return recorder, record_every
