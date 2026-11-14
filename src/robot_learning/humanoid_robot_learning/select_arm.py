"""One arm of a bimanual LeRobot dataset, as a single-arm dataset that ``lerobot-train`` and sim evaluation take as is.

    python -m humanoid_robot_learning.select_arm --src datasets/pioneer_8dof_bimanual/sim \\
        --out datasets/pioneer_8dof_left_arm_from_bimanual/sim --arm left

For single-arm tasks recorded with a bimanual teleop (the Quest teleop always records both arms): keeps the arm's
joints of ``action`` and ``observation.state`` (picked by name, ``<arm>_...``), drops the other arm's wrist camera
and keeps everything else (other cameras, per-frame task, ``subtask_index``). Built with LeRobot's own dataset tools,
then the numeric stats are recomputed. ``--relative_action_stats`` computes the action stats relative to the state, as
pi0.5 needs when trained with ``--policy.use_relative_actions=true`` (pass the same ``--chunk_size`` as the policy).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

VECTOR_KEYS = ("action", "observation.state")
ARMS = ("left", "right")


def arm_indices(names: list[str], arm: str) -> list[int]:
    """Positions of the arm's components (named ``<arm>_<joint>`` by the recorder)."""
    idx = [i for i, name in enumerate(names) if name.startswith(f"{arm}_")]
    if not idx:
        raise SystemExit(f"no '{arm}_*' components among {names}")
    return idx


def other_arm_cameras(video_keys: list[str], arm: str) -> list[str]:
    """Camera keys that belong to the other arm (e.g. observation.images.wrist_right when keeping the left)."""
    other = next(a for a in ARMS if a != arm)
    return [key for key in video_keys if key.endswith(f"_{other}")]


def sliced_columns(root: Path, idx: list[int]) -> dict[str, np.ndarray]:
    """The kept components of each vector key, over all frames in the order LeRobot's feature tools write them."""
    frames = pd.concat([pd.read_parquet(path, columns=list(VECTOR_KEYS)) for path in sorted((root / "data").glob("*/*.parquet"))])
    return {key: np.stack(frames[key].to_numpy())[:, idx].astype(np.float32) for key in VECTOR_KEYS}


def default_repo_id(src: Path, arm: str) -> str:
    """From the recorder's layout <root>/<dataset name>/sim: humanoid/pioneer_8dof_bimanual -> ..._left_arm."""
    name = (src.parent if src.name in ("sim", "real") else src).name
    name = name.replace("_bimanual", f"_{arm}_arm") if "_bimanual" in name else f"{name}_{arm}_arm"
    return f"humanoid/{name}"


def select_arm(src: Path, out: Path, arm: str, repo_id: str | None, relative_action_stats: bool, chunk_size: int):
    from lerobot.datasets.dataset_tools import add_features, recompute_stats, remove_feature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = repo_id or default_repo_id(src, arm)
    dataset = LeRobotDataset(repo_id=repo_id, root=src)   # local: the repo id only names it
    names = list(dataset.meta.features["action"]["names"])
    idx = arm_indices(names, arm)
    values = sliced_columns(src, idx)
    infos = {key: {**dataset.meta.features[key], "shape": (len(idx),), "names": [names[i] for i in idx]} for key in VECTOR_KEYS}

    # LeRobot can't change a feature's shape in place, and can't add a feature under a name the dataset already has:
    # drop the bimanual vectors (and the other wrist camera) first, then add the one-arm vectors under the same names.
    staging = out.with_name(out.name + ".staging")
    dropped = remove_feature(dataset, [*VECTOR_KEYS, *other_arm_cameras(dataset.meta.video_keys, arm)],
                             output_dir=staging, repo_id=repo_id)
    result = add_features(dropped, {key: (values[key], infos[key]) for key in VECTOR_KEYS}, output_dir=out, repo_id=repo_id)
    shutil.rmtree(staging)
    recompute_stats(result, relative_action=relative_action_stats, chunk_size=chunk_size)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", required=True, type=Path, help="bimanual LeRobot dataset folder (meta/, data/, videos/)")
    p.add_argument("--out", required=True, type=Path, help="new single-arm dataset folder (must not exist)")
    p.add_argument("--arm", choices=ARMS, default="left")
    p.add_argument("--repo_id", default=None, help="repo id of the new dataset (default: the source's, _bimanual -> _<arm>_arm)")
    p.add_argument("--relative_action_stats", action="store_true",
                   help="action stats relative to the state, for --policy.use_relative_actions=true")
    p.add_argument("--chunk_size", type=int, default=50, help="the policy's chunk size (relative stats only; pi0.5: 50)")
    args = p.parse_args()
    if args.out.exists():
        p.error(f"--out {args.out} exists: pick a new folder")
    result = select_arm(args.src, args.out, args.arm, args.repo_id, args.relative_action_stats, args.chunk_size)
    print(f"[SELECT_ARM] {args.arm} arm: {result.meta.total_episodes} episodes, {result.meta.total_frames} frames, "
          f"features {sorted(result.meta.features)} -> {args.out}")


if __name__ == "__main__":
    main()
