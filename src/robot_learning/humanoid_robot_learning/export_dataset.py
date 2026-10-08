"""Summarise a recorded LeRobot dataset and export every per-frame value to CSV, one file per episode.

    python -m humanoid_robot_learning.export_dataset <dataset root> [--out DIR]

Reads the parquet files directly (no video decoding, no torch). Each vector feature becomes one column per
component, named after the schema (``observation.state.left_shoulder_pitch``, ``leader_counts.A``, ...),
plus the frame's task text. The camera streams stay where the recorder put them (``videos/<key>/...``);
the summary lists them. Needs pandas + pyarrow (both come with lerobot).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root: Path):
    import pandas as pd

    info = json.loads((root / "meta" / "info.json").read_text())
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"{root}: no data/*/*.parquet -- not a LeRobot v3 dataset, or nothing saved yet")
    frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_of = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])}
    frames["task"] = frames["task_index"].map(task_of)
    return info, frames.sort_values(["episode_index", "frame_index"], ignore_index=True)


def flatten(info: dict, frames):
    """One column per vector component, named <feature>.<component>."""
    import pandas as pd

    cols = {k: frames[k] for k in ("episode_index", "frame_index", "timestamp", "task")}
    for key, ft in info["features"].items():
        if ft["dtype"] in ("video", "image") or key not in frames or key in cols:
            continue
        values = frames[key]
        if len(ft["shape"]) == 1 and ft["shape"][0] >= 1 and hasattr(values.iloc[0], "__len__"):
            names = ft.get("names") or [str(i) for i in range(ft["shape"][0])]
            stacked = pd.DataFrame(values.tolist(), index=frames.index)
            for i, name in enumerate(names):
                cols[f"{key}.{name}"] = stacked[i]
        elif key not in ("index", "task_index"):
            cols[key] = values
    return pd.DataFrame(cols)


def summary(root: Path, info: dict, frames) -> str:
    lines = [f"{root}", f"  {info['total_episodes']} episodes, {len(frames)} frames at {info['fps']} fps, "
                        f"robot {info.get('robot_type')}"]
    lines.append("  features:")
    for key, ft in info["features"].items():
        if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            continue
        shape = "x".join(str(s) for s in ft["shape"])
        if ft["dtype"] in ("video", "image"):
            videos = sorted(str(p.relative_to(root)) for p in (root / "videos" / key).glob("*/*.mp4"))
            lines.append(f"    {key:34s} {ft['dtype']} {shape}  {len(videos)} file(s), e.g. {videos[:1]}")
            continue
        import numpy as np

        arr = np.stack(frames[key].to_numpy()) if key in frames else None
        rng = f"min {arr.min():.4g} max {arr.max():.4g}" if arr is not None and arr.size else "(not in data)"
        names = ft.get("names") or []
        lines.append(f"    {key:34s} {ft['dtype']} {shape}  {rng}  [{', '.join(map(str, names))}]")
    lengths = frames.groupby("episode_index").size()
    lines.append(f"  episode length (frames): min {lengths.min()} mean {lengths.mean():.0f} max {lengths.max()}")
    lines.append(f"  tasks ({frames['task'].nunique()}): " + "; ".join(sorted(frames["task"].unique())[:6])
                 + (" ..." if frames["task"].nunique() > 6 else ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dataset folder (the one holding meta/, data/, videos/)")
    parser.add_argument("--out", type=Path, default=None, help="CSV folder (default: <root>/csv); 'none' to skip")
    args = parser.parse_args()
    info, frames = load(args.root)
    print(summary(args.root, info, frames))
    if args.out is not None and str(args.out) == "none":
        return
    out = args.out or args.root / "csv"
    out.mkdir(parents=True, exist_ok=True)
    table = flatten(info, frames)
    for ep, rows in table.groupby("episode_index"):
        rows.to_csv(out / f"episode_{int(ep):06d}.csv", index=False)
    print(f"  wrote {table['episode_index'].nunique()} CSV file(s), {table.shape[1]} columns, to {out}")


if __name__ == "__main__":
    main()
