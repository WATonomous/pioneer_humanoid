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
