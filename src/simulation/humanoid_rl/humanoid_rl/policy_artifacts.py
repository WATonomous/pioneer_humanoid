"""Portable, fail-closed contracts for exported Pioneer policies.

This module uses only the Python standard library. Importing it must not launch
Isaac Sim or require Torch. Format 3 pins actual USD assets rather than the
converter's machine-specific ``config.yaml``. It does not establish a policy's
training provenance or locomotion quality.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any


DEFAULT_TASK = "Isaac-Locomotion-RoughNoStairsSelectiveKneeShape-PioneerHumanoid-Play-v0"
_STAIRS_TASK = "Isaac-Locomotion-Stairs-PioneerHumanoid-Play-v0"
SUPPORTED_TASKS = {
    DEFAULT_TASK: DEFAULT_TASK,
    DEFAULT_TASK.replace("Isaac-Locomotion-", "Isaac-Velocity-"): DEFAULT_TASK,
    _STAIRS_TASK: _STAIRS_TASK,
    _STAIRS_TASK.replace("Isaac-Locomotion-", "Isaac-Velocity-"): _STAIRS_TASK,
}
EXPECTED_OBSERVATION_TERMS = (
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
    ("height_scan", 187),
)
EXPECTED_JOINT_NAMES = (
    "Hip_F_L", "Hip_F_R", "Hip_A_L", "Hip_A_R", "Hip_R_L", "Hip_R_R",
    "Knee_L", "Knee_R", "Ankle_P_L", "Ankle_P_R", "Ankle_R_L", "Ankle_R_R",
)
ACTION_CONTRACT = {
    "scale": {
        "Hip_F_.*": 0.25,
        "Hip_A_.*": 0.06,
        "Hip_R_.*": 0.15,
        "Knee_.*": 0.30,
        "Ankle_P_.*": 0.12,
        "Ankle_R_.*": 0.08,
    },
    "use_default_offset": True,
    "clip": {"Knee_.*": [-0.95, -0.05]},
    "clip_actions": None,
    "step_dt": 0.02,
}
ASSET_DIRECTORY_RELATIVE = Path("assets/whole_body_humanoid/usd")
ASSET_USD_FILE = "whole_body_humanoid.usd"
_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
_FLAT_PROPORTIONS = {
    "plane": 1.0,
    "random_rough": 0.0,
    "boxes": 0.0,
    "hf_pyramid_slope": 0.0,
    "hf_pyramid_slope_inv": 0.0,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def canonical_task(task: str) -> str:
    """Accept only the supported play tasks and their equivalent aliases."""
    if not isinstance(task, str) or task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported exported-policy task: {task!r}.")
    return SUPPORTED_TASKS[task]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_hashes(directory: Path) -> dict[str, str]:
    """Hash the cached USD layers, ignoring converter logs/configuration."""
    directory = Path(directory)
    return {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in _USD_SUFFIXES
    }


def _safe_relative_path(value: Any, label: str) -> PurePosixPath:
    # Check both path dialects, even on Linux; manifests are moved to Windows.
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError(f"{label} must be a safe forward-slash relative path.")
    return PurePosixPath(value)


def _validate_asset_metadata(metadata: Mapping) -> dict[str, str]:
    directory = metadata.get("asset_directory_relative")
    _safe_relative_path(directory, "asset_directory_relative")
    if directory != ASSET_DIRECTORY_RELATIVE.as_posix():
        raise ValueError("asset_directory_relative does not name the canonical Pioneer USD cache.")
    filename = metadata.get("asset_usd_file")
    _safe_relative_path(filename, "asset_usd_file")
    if filename != ASSET_USD_FILE:
        raise ValueError("asset_usd_file does not name the canonical Pioneer USD.")
    hashes = metadata.get("asset_files_sha256")
    if not isinstance(hashes, dict) or not hashes or ASSET_USD_FILE not in hashes:
        raise ValueError("asset_files_sha256 must include the canonical USD and its payload layers.")
    for name, digest in hashes.items():
        path = _safe_relative_path(name, "asset_files_sha256 key")
        if path.suffix.lower() not in _USD_SUFFIXES:
            raise ValueError(f"asset_files_sha256 must contain only USD layers: {name!r}.")
        _validate_hash(digest, f"asset_files_sha256[{name!r}]")
    return hashes


def _validate_hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")


def validate_assets(repo_root: Path, metadata: Mapping) -> Path:
    """Verify cached USD identity and return its non-resolved native path.

    In particular, do not resolve a Windows mapped drive to a WSL UNC path:
    native Isaac Sim needs the mapped spelling for relative USD payloads.
    """
    expected = _validate_asset_metadata(metadata)
    directory = Path(repo_root) / Path(metadata["asset_directory_relative"])
    usd_path = directory / metadata["asset_usd_file"]
    if not usd_path.is_file():
        raise FileNotFoundError(f"Cached Pioneer USD does not exist: {usd_path}")
    actual = asset_hashes(directory)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
        raise ValueError(
            "Cached robot assets differ from the export: "
            f"missing={missing}, extra={extra}, changed={changed}."
        )
    return usd_path


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _exact(value: Any, expected: Any, label: str) -> None:
    # Numeric int/float spelling is immaterial to JSON; booleans must never
    # masquerade as dimensions, scales or tolerances.
    if isinstance(expected, dict):
        if not isinstance(value, Mapping) or set(value) != set(expected):
            raise ValueError(f"{label} has unexpected or missing fields.")
        for key, wanted in expected.items():
            _exact(value[key], wanted, f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(value, (list, tuple)) or len(value) != len(expected):
            raise ValueError(f"{label} has an unexpected layout.")
        for index, (actual, wanted) in enumerate(zip(value, expected)):
            _exact(actual, wanted, f"{label}[{index}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"{label}: expected {expected!r}, found {value!r}.")
    elif isinstance(expected, int):
        if type(value) is not int or value != expected:
            raise ValueError(f"{label}: expected integer {expected}, found {value!r}.")
    else:
        if _number(value, label) != expected:
            raise ValueError(f"{label}: expected {expected!r}, found {value!r}.")


def load_metadata(path: Path, task: str | None = None) -> dict[str, Any]:
    """Read strict format-3 metadata, without opening a policy or simulator.

    Source/version/checkpoint provenance may be recorded as extra fields. The
    required layout, normalization, action, asset and parity fields are checked
    here; actual policy/asset hashes are checked by the corresponding callers.
    """
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Policy metadata must be a JSON object.")
    _exact(metadata.get("format"), "wato_torchscript_policy", "format")
    _exact(metadata.get("format_version"), 3, "format_version")
    saved_task = canonical_task(metadata.get("task"))
    if task is not None and saved_task != canonical_task(task):
        raise ValueError(f"Task mismatch: viewer={task!r}, exported={metadata.get('task')!r}.")
    expected = {
        "observation_dim": 235,
        "critic_observation_dim": 235,
        "action_dim": 12,
        "observation_terms": [
            {"name": name, "dimension": size} for name, size in EXPECTED_OBSERVATION_TERMS
        ],
        "action_joint_names": list(EXPECTED_JOINT_NAMES),
        "actor_observation_normalized": True,
        "critic_observation_normalized": True,
        "actor_observation_normalizer": "EmpiricalNormalization",
        "critic_observation_normalizer": "EmpiricalNormalization",
        "clip_actions": None,
        "step_dt": ACTION_CONTRACT["step_dt"],
        "action_contract": ACTION_CONTRACT,
    }
    for label, wanted in expected.items():
        if label not in metadata:
            raise ValueError(f"Policy metadata is missing {label}.")
        _exact(metadata[label], wanted, label)
    _validate_hash(metadata.get("policy_sha256"), "policy_sha256")
    if "checkpoint_sha256" in metadata:
        _validate_hash(metadata["checkpoint_sha256"], "checkpoint_sha256")
    _validate_asset_metadata(metadata)
    probe = metadata.get("probe")
    if not isinstance(probe, dict):
        raise ValueError("Policy metadata is missing its parity probe.")
    for name, size in (("observation", 235), ("expected_action", 12)):
        values = probe.get(name)
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"probe.{name} must contain {size} finite numbers.")
        for index, value in enumerate(values):
            _number(value, f"probe.{name}[{index}]")
    for name in ("rtol", "atol"):
        tolerance = _number(probe.get(name), f"probe.{name}")
        if not 0.0 <= tolerance <= 1.0e-4:
            raise ValueError(f"probe.{name} must lie between 0 and 1e-4.")
    return metadata


def _get(value: Any, name: str, label: str) -> Any:
    try:
        return value[name] if isinstance(value, Mapping) else getattr(value, name)
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError(f"Missing runtime contract field {label}.") from error


def _at(value: Any, *names: str) -> Any:
    label = "env"
    for name in names:
        label += "." + name
        value = _get(value, name, label)
    return value


def configure_flat_terrain(env_cfg: Any) -> Any:
    """Change only the five existing rough-mixture weights, not the task."""
    terrains = _at(env_cfg, "scene", "terrain", "terrain_generator", "sub_terrains")
    if not isinstance(terrains, Mapping) or set(terrains) != set(_FLAT_PROPORTIONS):
        raise ValueError("Flat playback requires the five-term rough-no-stairs terrain generator.")
    for name, terrain in terrains.items():
        _get(terrain, "proportion", f"terrain.{name}.proportion")
    for name, proportion in _FLAT_PROPORTIONS.items():
        terrain = terrains[name]
        if isinstance(terrain, Mapping):
            terrain["proportion"] = proportion
        else:
            terrain.proportion = proportion
    return env_cfg


def validate_env_config(env_cfg: Any, metadata: Mapping) -> None:
    """Check the task's position-target mapping and simulation control period."""
    _exact(metadata.get("action_contract"), ACTION_CONTRACT, "action_contract")
    action = _at(env_cfg, "actions", "joint_pos")
    for name in ("scale", "use_default_offset", "clip"):
        label = f"actions.joint_pos.{name}"
        _exact(_get(action, name, label), ACTION_CONTRACT[name], label)
    _exact(_get(action, "asset_name", "actions.joint_pos.asset_name"), "robot", "actions.joint_pos.asset_name")
    _exact(_get(action, "joint_names", "actions.joint_pos.joint_names"), [".*"], "actions.joint_pos.joint_names")
    dt = _number(_at(env_cfg, "sim", "dt"), "sim.dt")
    decimation = _at(env_cfg, "decimation")
    if type(decimation) is not int or decimation <= 0 or dt <= 0.0:
        raise ValueError("Simulation dt and decimation must be positive.")
    if not math.isclose(dt * decimation, ACTION_CONTRACT["step_dt"], rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"Runtime control period differs: expected 0.02, got {dt * decimation}.")


def validate_runtime_layout(base_env: Any, metadata: Mapping) -> None:
    """Check resolved action/observation order and dimensions, never skip checks."""
    try:
        robot = _at(base_env, "scene")["robot"]
    except (KeyError, TypeError) as error:
        raise ValueError("Runtime scene must expose robot.") from error
    _exact(
        list(_get(robot, "joint_names", "robot.joint_names")),
        list(EXPECTED_JOINT_NAMES),
        "robot.joint_names",
    )
    manager = _at(base_env, "action_manager")
    try:
        action = manager.get_term("joint_pos")
    except (AttributeError, KeyError, ValueError) as error:
        raise ValueError("Runtime action manager must expose joint_pos.") from error
    action_names = getattr(action, "joint_names", getattr(action, "_joint_names", None))
    if action_names is None:
        raise ValueError("Runtime action term does not expose resolved joint order.")
    _exact(list(action_names), list(EXPECTED_JOINT_NAMES), "action.joint_names")
    obs_manager = _at(base_env, "observation_manager")
    names = _at(obs_manager, "active_terms", "policy")
    dimensions = _at(obs_manager, "group_obs_term_dim", "policy")
    _exact(names, [name for name, _ in EXPECTED_OBSERVATION_TERMS], "observations.policy.names")
    _exact(dimensions, [[size] for _, size in EXPECTED_OBSERVATION_TERMS], "observations.policy.dimensions")
    step_dt = _number(_at(base_env, "step_dt"), "step_dt")
    if not math.isclose(step_dt, ACTION_CONTRACT["step_dt"], rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"Runtime control period differs: expected 0.02, got {step_dt}.")
    # A direct call must not quietly validate against a drifted manifest.
    _exact(metadata.get("action_joint_names"), list(EXPECTED_JOINT_NAMES), "action_joint_names")
    _exact(
        metadata.get("observation_terms"),
        [{"name": name, "dimension": size} for name, size in EXPECTED_OBSERVATION_TERMS],
        "observation_terms",
    )
