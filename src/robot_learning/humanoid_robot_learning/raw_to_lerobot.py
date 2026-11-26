"""Convert raw simulator episode collections (manifest.json + trajectory.npz + camera PNGs) into one
image-based LeRobot dataset. Only successful episodes are converted; images stay lossless.

Raw layout (written e.g. by ``humanoid_mujoco_scenes.peg_insert.recording``)::

    <raw>/manifest.json   {"fps", "joint_names", "task", "cameras": {name: [h, w]}, "results": [{"seed", "success", ...}]}
    <raw>/episode_<seed>/trajectory.npz   observation.state [T,D], action [T,D], optional phase [T]
    <raw>/episode_<seed>/<camera>/<frame>.png

Each raw episode stays one LeRobot episode, so action chunks never cross a correction boundary.
If trajectories carry ``phase``, ``<root>/phase_index.json`` records per-episode phase ranges (frame
indices within this dataset) plus each result's ``condition`` for ``phase_sampling``.

  python -m humanoid_robot_learning.raw_to_lerobot --raw datasets/peg_insert_raw --root datasets/pioneer_peg_insert
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_manifests(raws):
    manifests = [json.loads((raw / "manifest.json").read_text()) for raw in raws]
    for m in manifests[1:]:
        if any(m[k] != manifests[0][k] for k in ["fps", "joint_names", "cameras", "task"]):
            raise ValueError("raw dataset schemas/tasks differ")
    return manifests


def features(manifest):
    names = manifest["joint_names"]
    out = {key: {"dtype": "float32", "shape": (len(names),), "names": names} for key in ["observation.state", "action"]}
    for name, (h, w) in manifest["cameras"].items():
        out[f"observation.images.{name}"] = {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return out


def phase_ranges(phases, offset):
    ranges = []
    for start in range(len(phases)):
        if start == 0 or phases[start] != phases[start - 1]:
            if ranges:
                ranges[-1]["end"] = offset + start
            ranges.append({"start": offset + start, "end": offset + len(phases), "phase": str(phases[start])})
    return ranges


def add_episode(dataset, directory, manifest):
    from PIL import Image

    with np.load(directory / "trajectory.npz", allow_pickle=False) as trajectory:
        state, action = trajectory["observation.state"], trajectory["action"]
        phases = trajectory["phase"] if "phase" in trajectory else None
    for index in range(len(action)):
        frame = {"observation.state": state[index], "action": action[index], "task": manifest["task"]}
        for name in manifest["cameras"]:
            with Image.open(directory / name / f"{index:06d}.png") as image:
                frame[f"observation.images.{name}"] = np.asarray(image.convert("RGB")).copy()
        dataset.add_frame(frame)
    dataset.save_episode()
    return len(action), phases


def convert(raws, root, repo_id, robot_type, repeat=1):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifests = load_manifests(raws)
    dataset = LeRobotDataset.create(repo_id, fps=manifests[0]["fps"], root=root, features=features(manifests[0]),
                                    robot_type=robot_type, use_videos=False)
    cursor, episodes = 0, []
    for raw, manifest in zip(raws, manifests):
        for result in (r for r in manifest["results"] if r["success"]):
            for copy in range(repeat):
                length, phases = add_episode(dataset, raw / f"episode_{result['seed']:06d}", manifest)
                episodes.append({"raw": str(raw.resolve()), "seed": result["seed"], "repeat": copy,
                                 "condition": result.get("condition", ""), "expert_only": manifest.get("expert_only", False),
                                 "phase_ranges": phase_ranges(phases, cursor) if phases is not None else []})
                cursor += length
            print(f"Converted {raw.name} seed {result['seed']}", flush=True)
    if not episodes:
        raise ValueError("no successful episodes to convert")
    dataset.finalize()
    (root / "phase_index.json").write_text(json.dumps({"version": 2, "num_frames": cursor, "episodes": episodes}, indent=2))
    return len(episodes), cursor


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, action="append", required=True, help="raw collection; repeat to merge several")
    p.add_argument("--root", type=Path, required=True, help="new LeRobot dataset directory")
    p.add_argument("--repo-id", default="humanoid/pioneer_peg_insert")
    p.add_argument("--robot-type", default="pioneer_left_arm")
    p.add_argument("--repeat", type=int, default=1, help="duplicate each episode (weights short corrections without a phase sampler)")
    args = p.parse_args()
    if args.root.exists() or args.repeat < 1:
        p.error("--root must be new and --repeat positive")
    episodes, frames = convert(args.raw, args.root, args.repo_id, args.robot_type, args.repeat)
    print(f"Saved {episodes} episodes / {frames} frames to {args.root}")


if __name__ == "__main__":
    main()
