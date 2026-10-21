"""Build LeRobot feature dicts from dataset_schema.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def enabled_images(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    images = cfg.get("images") or {}
    out: dict[str, dict[str, Any]] = {}
    for key, spec in images.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("enabled", True):
            out[key] = spec
    return out


def select_cameras(cfg: dict[str, Any], cameras: str | None) -> dict[str, Any]:
    """Copy of cfg with only the chosen images enabled.

    cameras: None keeps the schema's `enabled` flags, "none" disables all,
    otherwise a comma list of image names (e.g. "ego,wrist_left").
    """
    if cameras is None:
        return cfg
    images = cfg.get("images") or {}
    names = [] if cameras == "none" else [n.strip() for n in cameras.split(",") if n.strip()]
    unknown = [n for n in names if n not in images]
    if unknown:
        raise ValueError(f"unknown cameras {unknown}; schema has {list(images)}")
    out = dict(cfg)
    out["images"] = {k: {**spec, "enabled": k in names} for k, spec in images.items()}
    return out


# Arm order in a recorded frame (the left arm's values, then the right's); the schema's `arms` must list them so.
ARM_ORDER = ("left", "right")
# Dataset name suffix per recorded arm set.
ARM_SET_SUFFIX = {("left",): "_left_arm", ("right",): "_right_arm", ("left", "right"): "_bimanual"}


def select_arms(cfg: dict[str, Any], arms: list[str]) -> dict[str, Any]:
    """Copy of cfg recording `arms`: joint_names from their joint lists in ARM_ORDER, and robot_id,
    repo_id and record.root with the arm set's suffix, so each arm set is its own dataset."""
    schema_arms = cfg["arms"]
    if tuple(schema_arms) != ARM_ORDER:
        raise ValueError(f"schema arms must be {list(ARM_ORDER)} in that order, got {list(schema_arms)}")
    unknown = [a for a in arms if a not in schema_arms]
    if unknown or not arms:
        raise ValueError(f"arms must be a non-empty subset of {list(schema_arms)}, got {arms}")
    arm_set = tuple(a for a in ARM_ORDER if a in arms)
    suffix = ARM_SET_SUFFIX[arm_set]
    out = dict(cfg)
    out["joint_names"] = [joint for arm in arm_set for joint in schema_arms[arm]]
    out["robot_id"] = cfg["robot_id"] + suffix
    out["repo_id"] = cfg["repo_id"] + suffix
    out["record"] = {**cfg["record"], "root": cfg["record"]["root"] + suffix}
    return out


def build_features(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """LeRobotDataset.create features block (pattern: lehome dataset_record.py)."""
    joint_names = list(cfg["joint_names"])
    dim = len(joint_names)
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (dim,),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (dim,),
            "names": joint_names,
        },
    }

    for key, spec in enabled_images(cfg).items():
        h = int(spec["height"])
        w = int(spec["width"])
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }

    return features


def create_dataset(cfg: dict[str, Any], *, root: Path):
    """Create a new LeRobotDataset on disk under the given root."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "lerobot is required for recording. Install with: "
            "pip install -e 'src/robot_learning[lerobot]'"
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    record_cfg = cfg.get("record") or {}
    return LeRobotDataset.create(
        repo_id=str(cfg.get("repo_id", "humanoid/local")),
        fps=int(cfg.get("fps", 30)),
        root=root,
        robot_type=str(cfg.get("robot_id", "unknown")),
        use_videos=bool(record_cfg.get("use_videos", True)),
        image_writer_threads=int(record_cfg.get("image_writer_threads", 4)),
        image_writer_processes=int(record_cfg.get("image_writer_processes", 0)),
        features=build_features(cfg),
    )
