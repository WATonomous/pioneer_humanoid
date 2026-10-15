"""A synthetic_teleop LeRobot dataset -> per-episode arrays for the Patch Policy authors' trainer (datasets/pioneer.py).

    python convert_lerobot.py <lerobot_root> <out_dir> [--size 224]

Writes obs_XXX.npy (T, V, H, W, 3 uint8; views in the order ego, wrist_left; area-resized to size x size, the input the
frozen DINOv2 encoder takes), and actions.npy / states.npy / seq_lengths.npy (padded to the longest episode).
"""
import argparse
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

CAMERAS = ("observation.images.ego", "observation.images.wrist_left")


def frames(path, n, size):
    out = np.empty((n, size, size, 3), np.uint8)
    with av.open(str(path)) as c:
        i = -1
        for i, f in enumerate(c.decode(video=0)):
            if i >= n:
                break
            out[i] = cv2.resize(f.to_ndarray(format="rgb24"), (size, size), interpolation=cv2.INTER_AREA)
    if i + 1 < n:
        raise SystemExit(f"{path}: {i + 1} frames, expected {n}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--size", type=int, default=224)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = pd.concat([pd.read_parquet(f) for f in sorted((a.root / "data").rglob("*.parquet"))])
    episodes = pd.concat([pd.read_parquet(f) for f in sorted((a.root / "meta/episodes").rglob("*.parquet"))]).set_index("episode_index")
    lengths, actions, states = [], [], []
    for ep, rows in data.groupby("episode_index"):
        rows = rows.sort_values("frame_index")
        n = len(rows)
        obs = np.lib.format.open_memmap(a.out / f"obs_{ep:03d}.npy", "w+", np.uint8, (n, len(CAMERAS), a.size, a.size, 3))
        for v, cam in enumerate(CAMERAS):
            e = episodes.loc[ep]
            if e[f"videos/{cam}/from_timestamp"] != 0:
                raise SystemExit("expects one video file per episode")
            obs[:, v] = frames(a.root / "videos" / cam / f"chunk-{e[f'videos/{cam}/chunk_index']:03d}" / f"file-{e[f'videos/{cam}/file_index']:03d}.mp4", n, a.size)
        obs.flush()
        lengths.append(n)
        actions.append(np.stack(rows["action"].to_numpy()).astype(np.float32))
        states.append(np.stack(rows["observation.state"].to_numpy()).astype(np.float32))
        print(f"episode {ep}: {n} frames", flush=True)
    T = max(lengths)
    pad = lambda xs: np.stack([np.concatenate([x, np.zeros((T - len(x), x.shape[1]), np.float32)]) for x in xs])  # noqa: E731
    np.save(a.out / "actions.npy", pad(actions)); np.save(a.out / "states.npy", pad(states)); np.save(a.out / "seq_lengths.npy", np.array(lengths))
    print(f"{len(lengths)} episodes, {sum(lengths)} frames -> {a.out}")


if __name__ == "__main__":
    main()
