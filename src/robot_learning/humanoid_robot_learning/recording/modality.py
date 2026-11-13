"""meta/modality.json: which slices of the flat state/action vectors are which arm and gripper (GR00T's format).

NVIDIA GR00T reads a LeRobot dataset plus this file; with it a Pioneer dataset can be fine-tuned on GR00T as is.
Built from the recorded joint names: ``left_*`` / ``right_*`` joints, the ``*_gripper`` ones as the grippers.
Datasets whose joints don't follow that naming (the old 6-DoF arms: ``joint1L`` ...) get no file.
Not yet checked against a GR00T training run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SIDES = ("left", "right")


def _groups(names: list[str]) -> dict[str, dict[str, int]] | None:
    """{"left_arm": {"start", "end"}, "left_gripper": ...} for contiguous runs of one side's arm / gripper joints."""
    groups: dict[str, dict[str, int]] = {}
    for i, name in enumerate(names):
        side = next((s for s in SIDES if name.startswith(f"{s}_")), None)
        if side is None:
            return None
        key = f"{side}_gripper" if name.endswith("gripper") else f"{side}_arm"
        if key in groups and groups[key]["end"] != i:
            return None   # not contiguous: no clean slice
        groups.setdefault(key, {"start": i, "end": i})["end"] = i + 1
    return groups


def build_modality(joint_names: list[str], cameras: list[str],
                   extra_features: dict[str, list[str]] | None = None) -> dict[str, Any] | None:
    groups = _groups(list(joint_names))
    if groups is None:
        return None
    state = {k: dict(v) for k, v in groups.items()}
    action = {k: dict(v) for k, v in groups.items()}
    # Hand poses (mujoco_arm_extras.py): 9 values per arm, [x y z, 6D rotation], arms in the state's order.
    for key, target in (("observation.ee_pose", state), ("action.ee_pose", action)):
        comps = (extra_features or {}).get(key)
        if not comps:
            continue
        for side in SIDES:
            idx = [i for i, c in enumerate(comps) if c.startswith(f"{side}_hand_")]
            if idx:
                target[f"{side}_eef_position"] = {"original_key": key, "start": idx[0], "end": idx[0] + 3}
                target[f"{side}_eef_rotation_6d"] = {"original_key": key, "start": idx[0] + 3, "end": idx[-1] + 1}
    return {
        "state": state,
        "action": action,
        "video": {name: {"original_key": f"observation.images.{name}"} for name in cameras},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def write_modality(root: Path, joint_names: list[str], cameras: list[str],
                   extra_features: dict[str, list[str]] | None = None) -> Path | None:
    """Write <root>/meta/modality.json (overwrites: it is derived, never edited). Returns the path, or None."""
    modality = build_modality(joint_names, cameras, extra_features)
    if modality is None:
        return None
    path = Path(root) / "meta" / "modality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(modality, indent=2) + "\n")
    return path
