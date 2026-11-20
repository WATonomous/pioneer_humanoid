"""Load config/umi_gripper.yaml merged with the rig's measured umi_calibration.json."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from umi_gripper import geometry
from umi_gripper._paths import REPO_ROOT

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_CONFIG = CONFIG_DIR / "umi_gripper.yaml"
DEFAULT_CALIBRATION = CONFIG_DIR / "umi_calibration.json"
HEADSETS = ("quest2", "quest3")
SIDES = ("left", "right")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Path = DEFAULT_CONFIG, calibration: Path | None = DEFAULT_CALIBRATION,
                overrides: dict | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    if calibration is not None and Path(calibration).is_file():
        cfg = deep_merge(cfg, json.loads(Path(calibration).read_text()))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    if cfg.get("headset") not in HEADSETS:
        raise ValueError(f"headset must be one of {HEADSETS}, got {cfg.get('headset')!r}")
    if cfg.get("controller") not in SIDES:
        raise ValueError(f"controller must be one of {SIDES}, got {cfg.get('controller')!r}")
    if cfg["headset"] not in cfg.get("mounts", {}):
        raise ValueError(f"no mount for headset {cfg['headset']!r} in mounts")
    if cfg["width"]["source"] not in ("encoder", "trigger"):
        raise ValueError("width.source must be encoder or trigger")
    grip_to_tcp(cfg)
    tcp_to_camera(cfg)


def _pose(entry: dict) -> np.ndarray:
    xyz = np.asarray(entry["xyz"], dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all() or np.linalg.norm(xyz) > 0.5:
        raise ValueError(f"invalid offset {entry['xyz']!r} (metres, |xyz| <= 0.5)")
    if "quat_xyzw" in entry:
        rotation = geometry.quat_to_matrix(entry["quat_xyzw"])
    else:
        rotation = geometry.rpy_deg_to_matrix(entry.get("rpy_deg", (0, 0, 0)))
    return geometry.make_pose(rotation, xyz)


def grip_to_tcp(cfg: dict) -> np.ndarray:
    """T_grip_tcp: the tcp frame expressed in the mounted controller's grip frame."""
    return _pose(cfg["mounts"][cfg["headset"]]["grip_to_tcp"])


def tcp_to_camera(cfg: dict) -> np.ndarray:
    """T_tcp_camera: the camera's OpenCV optical frame expressed in the tcp frame."""
    return _pose(cfg["tcp_to_camera"])


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def update_calibration(updates: dict, path: Path = DEFAULT_CALIBRATION) -> dict:
    """Merge `updates` into the rig's calibration file and write it back."""
    current = json.loads(path.read_text()) if path.is_file() else {}
    merged = deep_merge(current, updates)
    path.write_text(json.dumps(merged, indent=2) + "\n")
    return merged
