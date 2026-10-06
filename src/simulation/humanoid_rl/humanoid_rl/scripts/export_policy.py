#!/usr/bin/env python3
"""Export a normalized Pioneer checkpoint on CPU, without launching Isaac Sim.

Only the registered rough-no-stairs knee-shape and stairs recipes are supported.
The task must be supplied from the training recipe: weights alone do not identify
their training environment or prove stair-climbing capability.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

try:
    from rsl_rl.networks import EmpiricalNormalization
except ImportError:
    from rsl_rl.modules import EmpiricalNormalization

from humanoid_rl.policy_artifacts import (
    ACTION_CONTRACT,
    ASSET_DIRECTORY_RELATIVE,
    ASSET_USD_FILE,
    EXPECTED_JOINT_NAMES,
    EXPECTED_OBSERVATION_TERMS,
    SUPPORTED_TASKS,
    asset_hashes,
    canonical_task,
    load_metadata,
    sha256,
)


class _TorchPolicyExporter(torch.nn.Module):
    """An actor with its saved observation normalizer embedded exactly once."""

    def __init__(self, actor: torch.nn.Module, normalizer: torch.nn.Module):
        super().__init__()
        self.actor = copy.deepcopy(actor)
        self.normalizer = copy.deepcopy(normalizer)

    def forward(self, observations):
        return self.actor(self.normalizer(observations))

    @torch.jit.export
    def reset(self):
        pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="RSL-RL model_*.pt checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="Destination TorchScript .pt file.")
    parser.add_argument("--metadata", type=Path, help="JSON manifest; defaults to OUTPUT with a .json suffix.")
    parser.add_argument(
        "--task", required=True, choices=SUPPORTED_TASKS,
        help="Play task matching the saved training recipe.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing export files.")
    return parser.parse_args(argv)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _expected_state_keys() -> set[str]:
    keys = {"log_std"}
    for prefix in ("actor", "critic"):
        for index in (0, 2, 4, 6):
            keys.update((f"{prefix}.{index}.weight", f"{prefix}.{index}.bias"))
        keys.update(f"{prefix}_obs_normalizer.{name}" for name in ("_mean", "_var", "_std", "count"))
    return keys


def _validate_state(state: dict[str, torch.Tensor]) -> None:
    expected = _expected_state_keys()
    if set(state) != expected:
        missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
        raise ValueError(f"Unsupported policy state keys; missing={missing}, unhandled={extra}.")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Policy state '{name}' is not a tensor.")
        if value.is_complex() or not torch.isfinite(value).all():
            raise ValueError(f"Policy tensor '{name}' is complex or contains NaN or infinity.")
    if tuple(state["log_std"].shape) != (12,) or state["log_std"].dtype != torch.float32:
        raise ValueError("Expected a float32 12-element log_std parameter.")
    for prefix, output_dim in (("actor", 12), ("critic", 1)):
        dimensions = (235, 512, 256, 128, output_dim)
        for index, input_dim, next_dim in zip((0, 2, 4, 6), dimensions, dimensions[1:]):
            weight, bias = state[f"{prefix}.{index}.weight"], state[f"{prefix}.{index}.bias"]
            if tuple(weight.shape) != (next_dim, input_dim) or tuple(bias.shape) != (next_dim,):
                raise ValueError(f"Unsupported {prefix} architecture: expected 235/[512,256,128]/{output_dim}.")
            if weight.dtype != torch.float32 or bias.dtype != torch.float32:
                raise ValueError(f"The supported {prefix} recipe requires float32 network tensors.")


def _build_mlp(state: dict[str, torch.Tensor], prefix: str) -> torch.nn.Sequential:
    dimensions = (235, 512, 256, 128, 12 if prefix == "actor" else 1)
    layers: list[torch.nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(torch.nn.Linear(input_dim, output_dim))
        if index < 3:
            layers.append(torch.nn.ELU())
    network = torch.nn.Sequential(*layers)
    network.load_state_dict(
        {name.removeprefix(f"{prefix}."): value for name, value in state.items() if name.startswith(f"{prefix}.")},
        strict=True,
    )
    return network.cpu().eval()


def _build_normalizer(state: dict[str, torch.Tensor], prefix: str) -> torch.nn.Module:
    saved = {name.removeprefix(f"{prefix}."): value for name, value in state.items() if name.startswith(f"{prefix}.")}
    normalizer = EmpiricalNormalization(235).cpu()
    expected = normalizer.state_dict()
    for name, value in saved.items():
        if tuple(value.shape) != tuple(expected[name].shape) or value.dtype != expected[name].dtype:
            raise ValueError(f"Unsupported normalizer tensor shape/dtype: {prefix}.{name}.")
    if (saved["_var"] < 0).any() or (saved["_std"] < 0).any() or (saved["count"] < 0).any():
        raise ValueError(f"Invalid variance/std/count in {prefix}.")
    if not torch.allclose(saved["_std"], saved["_var"].sqrt(), rtol=1.0e-5, atol=1.0e-6):
        raise ValueError(f"Inconsistent variance/std in {prefix}.")
    normalizer.load_state_dict(saved, strict=True)
    return normalizer.eval()


def _checkpoint_iteration(checkpoint: dict[str, Any]) -> int | None:
    for key in ("iter", "iteration", "current_learning_iteration"):
        value = checkpoint.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _publish(temporary: Path, destination: Path, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary, destination)
    else:
        # Do not clobber a file created after the initial destination checks.
        os.link(temporary, destination)
        temporary.unlink()


def export_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    metadata_path: Path,
    task: str,
    *,
    overwrite: bool = False,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the supported recipe, serialize on CPU, and publish a verified pair."""
    task = canonical_task(task)
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    metadata_path = metadata_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if output_path == checkpoint_path or metadata_path == checkpoint_path:
        raise ValueError("Export destinations must not overwrite the source checkpoint.")
    if output_path == metadata_path:
        raise ValueError("Policy and metadata destinations must be different files.")
    for destination in (output_path, metadata_path):
        if destination.exists() and (not overwrite or not destination.is_file()):
            raise FileExistsError(f"Export destination already exists: {destination}; use --overwrite for files.")

    repository_root = repository_root or Path(__file__).resolve().parents[5]
    asset_directory = repository_root / ASSET_DIRECTORY_RELATIVE
    if not (asset_directory / ASSET_USD_FILE).is_file():
        raise FileNotFoundError(f"Canonical robot USD is missing: {asset_directory / ASSET_USD_FILE}")
    hashes = asset_hashes(asset_directory)
    # No unsafe-pickle fallback: checkpoints with custom Python objects must be
    # converted to a tensor/primitives-only checkpoint in their trusted source environment.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint must contain a model_state_dict mapping.")
    state = checkpoint["model_state_dict"]
    _validate_state(state)
    actor, critic = _build_mlp(state, "actor"), _build_mlp(state, "critic")
    normalizer = _build_normalizer(state, "actor_obs_normalizer")
    critic_normalizer = _build_normalizer(state, "critic_obs_normalizer")
    with torch.inference_mode():
        critic_actions = critic(critic_normalizer(torch.zeros(1, 235)))
    if tuple(critic_actions.shape) != (1, 1) or not torch.isfinite(critic_actions).all():
        raise ValueError("Reconstructed critic failed its shape/finite probe.")

    for destination in (output_path, metadata_path):
        destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        for destination in (output_path, metadata_path):
            descriptor, filename = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
            os.close(descriptor)
            temporary_paths.append(Path(filename))
        temporary_policy, temporary_metadata = temporary_paths
        torch.jit.script(_TorchPolicyExporter(actor, normalizer).cpu().eval()).save(str(temporary_policy))
        observations = torch.cat((
            torch.zeros(1, 235),
            torch.randn(31, 235, generator=torch.Generator().manual_seed(20260728)),
        ))
        with torch.inference_mode():
            expected = actor(normalizer(observations))
            scripted = torch.jit.load(str(temporary_policy), map_location="cpu").eval()
            actual = scripted(observations)
            reloaded = torch.jit.load(str(temporary_policy), map_location="cpu").eval()(observations)
        if tuple(actual.shape) != (32, 12) or not torch.isfinite(actual).all():
            raise ValueError("Exported actor failed its shape/finite probe.")
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)
        torch.testing.assert_close(reloaded, expected, rtol=1.0e-6, atol=1.0e-6)
        metadata = {
            "format": "wato_torchscript_policy", "format_version": 3, "task": task,
            "checkpoint": checkpoint_path.name, "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_iteration": _checkpoint_iteration(checkpoint),
            "policy_file": output_path.name, "policy_sha256": sha256(temporary_policy),
            "observation_dim": 235, "critic_observation_dim": 235, "action_dim": 12,
            "observation_terms": [{"name": name, "dimension": size} for name, size in EXPECTED_OBSERVATION_TERMS],
            "action_joint_names": list(EXPECTED_JOINT_NAMES), "action_contract": copy.deepcopy(ACTION_CONTRACT),
            "actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128],
            "activation": "elu", "noise_std_type": "log",
            "actor_observation_normalizer": "EmpiricalNormalization", "actor_observation_normalized": True,
            "critic_observation_normalizer": "EmpiricalNormalization", "critic_observation_normalized": True,
            "clip_actions": None, "step_dt": 0.02,
            "asset_directory_relative": Path(ASSET_DIRECTORY_RELATIVE).as_posix(), "asset_usd_file": ASSET_USD_FILE,
            "asset_files_sha256": hashes,
            "probe": {
                "observation": observations[1].tolist(), "expected_action": actual[1].tolist(),
                "rtol": 1.0e-5, "atol": 1.0e-5,
            },
            "export_validation": {
                "sample_count": 32, "max_abs_error": float((actual - expected).abs().max()),
                "rtol": 1.0e-6, "atol": 1.0e-6,
            },
            "versions": {
                "python_torch": str(torch.__version__), "rsl_rl_lib": _package_version("rsl-rl-lib"),
                "isaaclab": _package_version("isaaclab"), "isaaclab_rl": _package_version("isaaclab-rl"),
            },
        }
        temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        load_metadata(temporary_metadata)
        _publish(temporary_policy, output_path, overwrite)
        _publish(temporary_metadata, metadata_path, overwrite)
        return metadata
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata = export_checkpoint(args.checkpoint, args.output, metadata_path, args.task, overwrite=args.overwrite)
    print(f"[export_complete] policy={args.output} metadata={metadata_path}", flush=True)
    print(
        "contract: 235 observations, 12 actions; actor normalizer embedded; "
        f"parity error={metadata['export_validation']['max_abs_error']:.9g}", flush=True,
    )


if __name__ == "__main__":
    main()
