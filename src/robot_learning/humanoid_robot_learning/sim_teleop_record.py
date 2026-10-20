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
BIMANUAL_SCHEMA = PKG_ROOT / "config" / "dataset_schema_pioneer_bimanual.yaml"

# Recorded joint order (real-robot names, joint_command_core.cpp); must match the schema.
JOINTS = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_pitch",
    "elbow_roll",
    "wrist_pitch",
    "gripper",
]
LEFT_ARM_RECORD_JOINTS = [f"left_{joint}" for joint in JOINTS]
RIGHT_ARM_RECORD_JOINTS = [f"right_{joint}" for joint in JOINTS]
BIMANUAL_RECORD_JOINTS = LEFT_ARM_RECORD_JOINTS + RIGHT_ARM_RECORD_JOINTS


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
        default=None,
        help="dataset_schema YAML override (default: selected by arm mode)",
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
    parser.add_argument(
        "--image_size",
        type=str,
        default=None,
        help="WIDTHxHEIGHT for every recorded camera, e.g. 320x240 (default: the schema's sizes). Rendering at the "
        "policy's input size is faster and lighter than recording large and downscaling later.",
    )


def load_record_schema(
    parser: argparse.ArgumentParser, args: argparse.Namespace, *, default_schema: Path = DEFAULT_SCHEMA
) -> RecordSchema:
    """Call before AppLauncher: recording cameras need --enable_cameras."""
    if args.cameras is not None and not args.record:
        parser.error("--cameras requires --record")
    if args.image_size is not None and not args.record:
        parser.error("--image_size requires --record")
    if not args.record:
        return RecordSchema()
    schema_path = default_schema if args.schema is None else args.schema
    path = resolve_config_path(schema_path, anchor=PKG_ROOT)
    try:
        cfg = select_cameras(load_yaml(path), args.cameras)
    except ValueError as exc:
        parser.error(f"--cameras: {exc}")
    images = enabled_images(cfg)
    if args.image_size is not None:
        parts = args.image_size.lower().split("x")
        if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
            parser.error(f"--image_size: expected WIDTHxHEIGHT such as 320x240, got {args.image_size!r}")
        width, height = int(parts[0]), int(parts[1])
        for spec in images.values():           # the same dicts as cfg["images"]: renderers and recorder both see it
            spec["width"], spec["height"] = width, height
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
    buffer_capacity_s: float = 120.0,
):
    """Return (recorder, record_every): record one frame every `record_every` physics steps.

    ``extra_features``: extra float32 per-frame features, {name: [component names]} (see SimLeRobotRecorder).
    ``buffer_capacity_s``: longest episode the frame buffers hold (they are allocated at that size).
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
    missing = [key for key in ("fps", "repo_id", "robot_id", "joint_names") if key not in cfg]
    if missing:
        raise ValueError(f"{schema.path}: missing {', '.join(missing)}")
    if list(cfg["joint_names"]) != joint_names:
        raise ValueError(
            f"{schema.path}: joint_names must be {joint_names}, got {list(cfg['joint_names'])}"
        )
    fps = int(cfg["fps"])
    physics_hz = 1.0 / sim_dt
    record_every = round(physics_hz / fps)
    if record_every < 1 or abs(record_every * fps - physics_hz) > 1e-6:
        raise ValueError(f"{schema.path}: fps={fps} must divide the physics rate ({physics_hz:g} Hz)")
    dataset_root = resolve_dataset_root(cfg, args.dataset_root, subdir="sim")
    cameras = {name: {"height": spec["height"], "width": spec["width"]} for name, spec in schema.images.items()}
    recorder = SimLeRobotRecorder(
        task_name=args.task_description,
        repo_id=str(cfg["repo_id"]),
        dataset_root=dataset_root,
        fps=fps,
        device=device,
        joint_names=list(cfg["joint_names"]),
        cameras=cameras,
        num_episodes=args.num_episodes,
        robot_type=str(cfg["robot_id"]),
        rate_limit=False,
        extra_features=extra_features,
        buffer_capacity_s=buffer_capacity_s,
    )
    recorder.init_dataset()
    print(f"[RECORD] Writing to {dataset_root} at {fps} fps (every {record_every} physics steps)")
    if cameras:
        print(f"[RECORD] Cameras: {sorted(cameras)}")
    return recorder, record_every
