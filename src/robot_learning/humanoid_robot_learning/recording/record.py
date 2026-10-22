"""Record teleoperation data from ROS 2 (real arm) or dry-run tests."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from humanoid_robot_learning.recording.record_loop import run_record_loop
from humanoid_robot_learning.recording.recorder import RecordSettings
from humanoid_robot_learning.recording.record_utils import resolve_config_path, resolve_dataset_root
from humanoid_robot_learning.recording.arm_pose_io import ARM_POSE_JOINT_NAMES
from humanoid_robot_learning.recording.schema import enabled_images, load_yaml, select_arms, select_cameras
from humanoid_robot_learning.recording.sinks import parse_sink_names
from humanoid_robot_learning.recording.snapshot import ObservationSnapshot

logger = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCHEMA = _PKG_ROOT / "config" / "pioneer_humanoid_dataset_schema.yaml"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record demonstrations to LeRobot and/or HDF5 (WATO humanoid arm, ROS 2)."
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=str(_DEFAULT_SCHEMA),
        help="dataset_schema YAML (default: config/pioneer_humanoid_dataset_schema.yaml)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="cameras to record, e.g. 'wrist_left' or 'none' (default: schema's enabled images)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Output directory (default: <repo>/<schema record.root>/real)",
    )
    parser.add_argument(
        "--sink",
        type=str,
        default="lerobot",
        help="Output sinks: lerobot, hdf5, or comma-separated (e.g. lerobot,hdf5)",
    )
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--task_description", type=str, default="humanoid demonstration")
    parser.add_argument(
        "--episode_time_s",
        type=float,
        default=None,
        help="Auto-finish each episode after N seconds (keyboard optional)",
    )
    parser.add_argument(
        "--reset_time_s",
        type=float,
        default=5.0,
        help="Pause between saved episodes",
    )
    parser.add_argument(
        "--auto_start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log frames immediately when an episode begins",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Synthetic data only; no ROS (tests write path)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _check_real_support(cfg: dict[str, Any]) -> None:
    """Fail fast on schema fields the real arm cannot supply yet."""
    problems = []
    names = list(cfg["joint_names"])
    if names != list(ARM_POSE_JOINT_NAMES):
        missing = [n for n in names if n not in ARM_POSE_JOINT_NAMES]
        problems.append(
            f"joint_names {missing or names} have no ROS source (ArmPose supplies {list(ARM_POSE_JOINT_NAMES)})"
        )
    no_topic = [k for k, spec in enabled_images(cfg).items() if not spec.get("topic")]
    if no_topic:
        problems.append(f"cameras {no_topic} have no ROS topic (use --cameras to pick others)")
    if problems:
        raise SystemExit("Real recording not supported for this schema yet:\n  " + "\n  ".join(problems))


class _SnapshotSource:
    def maybe_spin(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class _DryRunSource(_SnapshotSource):
    def __init__(self, image_shapes: dict[str, tuple[int, int]], dim: int) -> None:
        self._image_shapes = image_shapes
        self._dim = dim
        self._t0 = time.monotonic()

    def __call__(self) -> ObservationSnapshot:
        from humanoid_robot_learning.recording.record_loop import dry_snapshot

        return dry_snapshot(time.monotonic() - self._t0, self._image_shapes, self._dim)


class _RosSource(_SnapshotSource):
    def __init__(self, cfg: dict[str, Any]) -> None:
        import rclpy
        from rclpy.node import Node

        from humanoid_robot_learning.recording.ros_buffer import RosRecordBuffer

        self._rclpy = rclpy
        rclpy.init()
        self._node = Node("humanoid_record")
        self._buffer = RosRecordBuffer(self._node, cfg)

    def maybe_spin(self) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def __call__(self) -> ObservationSnapshot:
        return self._buffer.snapshot()

    def cleanup(self) -> None:
        self._node.destroy_node()
        self._rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    schema_path = resolve_config_path(args.schema, anchor=_PKG_ROOT)
    try:
        cfg = select_cameras(load_yaml(schema_path), args.cameras)
    except ValueError as exc:
        raise SystemExit(f"--cameras: {exc}") from exc
    cfg = select_arms(cfg, ["left"])  # ArmPose carries the left arm only
    if not args.dry_run:
        _check_real_support(cfg)
    image_shapes = {k: (int(v["height"]), int(v["width"])) for k, v in enabled_images(cfg).items()}
    dim = len(cfg["joint_names"])

    record_root = resolve_dataset_root(cfg, args.dataset_root, subdir="real")

    settings = RecordSettings(
        num_episodes=args.num_episodes,
        task_description=args.task_description,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        auto_start=args.auto_start,
    )
    sink_names = parse_sink_names(args.sink)

    if args.dry_run:
        source: _SnapshotSource = _DryRunSource(image_shapes, dim)
    else:
        source = _RosSource(cfg)

    output_dir = run_record_loop(
        cfg,
        source,
        settings,
        record_root=record_root,
        sink_names=sink_names,
        dry_run=args.dry_run,
    )
    print(f"Dataset saved under: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
