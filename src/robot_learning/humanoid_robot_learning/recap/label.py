"""RECAP advantage labels: binarise each frame's N-step advantage under the value function, for the next training round.

    python -m humanoid_robot_learning.recap.label --dataset datasets/recap/drawer_stow_r1 \\
        --value outputs/recap/value_r1/value.pt --round 1 --training_copy datasets/recap/drawer_stow_r1_train

pi*0.6 (arXiv 2511.14759): A_t = r_t + ... + r_{t+N-1} + V(o_{t+N}) - V(o_t), in the value's normalised units
(reward -1/T a step, T = the time limit in steps). Within N steps of the end, the rest of the Monte Carlo return
replaces the bootstrap: A_t = G_t - V(o_t). A frame is positive when A_t is above the threshold eps, the
(1 - --positive_frac) quantile of all frames' advantages: --positive_frac 0.7 makes 70% positive. Frames of
--force_positive sources (default: corrections, which are expert actions) are positive whatever their advantage and
leave the quantile.

Writes <dataset>/recap/labels_r<round>.parquet: index, episode_index, frame_index, source, value, advantage, positive.

--training_copy OUT writes a copy of the dataset (LeRobot dataset_tools.modify_features, which also adds the per-frame
`advantage` feature: 1 positive, 0 negative, -1 dropped) whose task text is the step instruction plus " Advantage:
positive" or " Advantage: negative". For --dropout of the frames (default 30%, drawn once with --seed) the token is
left out, so the policy also learns the unconditioned prompt (classifier-free guidance). The dropout is static: LeRobot
has no per-sample hook to redraw it every epoch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from humanoid_robot_learning.recap.data import Frames, load_frames
from humanoid_robot_learning.recap.rollouts import ADVANTAGES
from humanoid_robot_learning.recap.sidecar import SOURCES, recap_dir

DROPPED = -1    # `advantage` feature of a frame trained without the token


def episode_advantages(values: np.ndarray, returns: np.ndarray, horizon: int, max_steps: int) -> np.ndarray:
    """N-step advantages of one episode (module docstring); `values`, `returns`: per frame, normalised."""
    advantage = returns - values                    # within `horizon` of the end: Monte Carlo
    head = len(values) - horizon
    if head > 0:
        advantage[:head] = -horizon / max_steps + values[horizon:] - values[:head]
    return advantage


def advantages(frames: Frames, values: np.ndarray, horizon: int, max_steps: int) -> np.ndarray:
    out = np.empty(len(values))
    for episode in np.unique(frames.episode):
        rows = np.flatnonzero(frames.episode == episode)
        out[rows] = episode_advantages(values[rows], frames.returns[rows], horizon, max_steps)
    return out


def binarise(adv: np.ndarray, source: np.ndarray, positive_frac: float, force_positive: list[str]) -> tuple[np.ndarray, float]:
    """(positive per frame, eps): above the (1 - positive_frac) quantile, or from a forced source."""
    forced = np.isin(source, force_positive)
    free = adv[~forced]
    eps = float(np.quantile(free, 1.0 - positive_frac)) if len(free) else 0.0
    return forced | (adv > eps), eps


def label_frames(frames: Frames, values: np.ndarray, args):
    """The labels table (module docstring) and eps."""
    import pandas as pd

    max_steps = int(round(args.max_s * args.fps))
    adv = advantages(frames, values, args.horizon, max_steps)
    positive, eps = binarise(adv, frames.source, args.positive_frac, args.force_positive)
    table = pd.DataFrame({"index": frames.index, "episode_index": frames.episode, "frame_index": frames.frame,
                          "source": frames.source.astype(str), "value": values.astype(np.float32),
                          "advantage": adv.astype(np.float32), "positive": positive})
    return table, eps


def advantage_codes(labels, dropout: float, seed: int) -> np.ndarray:
    """Per frame, in labels order: 1 positive, 0 negative, DROPPED for a `dropout` fraction drawn with `seed`."""
    codes = labels["positive"].to_numpy().astype(np.int64)
    codes[np.random.default_rng(seed).random(len(codes)) < dropout] = DROPPED
    return codes


def retask(root: Path, codes_by_index: dict[int, int]) -> int:
    """Append the advantage token to each frame's task (per frame, unlike dataset_tools.modify_tasks, which sets one
    task per episode and would erase the step instructions); returns the new task count."""
    import pandas as pd
    from lerobot.datasets.io_utils import load_info, load_tasks, write_info, write_tasks

    old = load_tasks(root)
    names = {int(i): t for t, i in zip(old.index, old["task_index"])}
    suffix = {1: ADVANTAGES["positive"], 0: ADVANTAGES["negative"], DROPPED: ""}
    new_names: dict[str, int] = {}
    episode_tasks: dict[int, set] = {}
    for path in sorted((root / "data").glob("*/*.parquet")):
        df = pd.read_parquet(path)
        texts = [names[t] + suffix[codes_by_index[i]] for t, i in zip(df["task_index"], df["index"])]
        df["task_index"] = [new_names.setdefault(t, len(new_names)) for t in texts]
        for e, t in zip(df["episode_index"], texts):
            episode_tasks.setdefault(int(e), set()).add(t)
        df.to_parquet(path, index=False)
    for path in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        df = pd.read_parquet(path)
        df["tasks"] = [sorted(episode_tasks[int(e)]) for e in df["episode_index"]]
        df.to_parquet(path, index=False)
    write_tasks(pd.DataFrame({"task_index": list(new_names.values())}, index=pd.Index(list(new_names), name="task")), root)
    info = load_info(root)
    info.total_tasks = len(new_names)
    write_info(info, root)
    return len(new_names)


def training_copy(dataset: str, labels, out: Path, dropout: float, seed: int) -> None:
    """Copy `dataset` to `out` with the per-frame `advantage` feature and the advantage token in its tasks."""
    from lerobot.datasets.dataset_tools import modify_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    labels = labels.sort_values("index")
    codes = advantage_codes(labels, dropout, seed)
    src = LeRobotDataset(repo_id=f"local/{Path(dataset).name}", root=dataset)
    if not np.array_equal(labels["index"].to_numpy(), np.arange(src.meta.total_frames)):
        raise ValueError("labels do not cover the dataset's frames 0..N-1: relabel this dataset")
    feature = {"dtype": "int64", "shape": [1], "names": None}
    modify_features(src, add_features={"advantage": (codes, feature)}, output_dir=out, repo_id=f"local/{out.name}")
    n_tasks = retask(out, dict(zip(labels["index"].to_numpy().tolist(), codes.tolist())))
    print(f"[LABEL] training copy {out}: {n_tasks} tasks, token dropped on {np.mean(codes == DROPPED):.0%} of frames",
          flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__.split("\n\n", 2)[2],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="LeRobot dataset with its recap/ sidecar")
    p.add_argument("--value", required=True, help="value.pt from recap.value (its scene, robot, fps, time limit apply)")
    p.add_argument("--round", type=int, required=True, help="RECAP round k: writes recap/labels_r<k>.parquet")
    p.add_argument("--horizon", type=int, default=50, help="N, steps of the advantage (50 = 2 s at 25 fps)")
    p.add_argument("--positive_frac", type=float, default=0.7, help="fraction of (non-forced) frames labelled positive")
    p.add_argument("--force_positive", nargs="*", default=["correction"], choices=SOURCES,
                   help="sources whose frames are always positive")
    p.add_argument("--training_copy", default=None, metavar="OUT", help="also write the token-conditioned training copy here")
    p.add_argument("--dropout", type=float, default=0.3, help="fraction of frames trained without the token")
    p.add_argument("--seed", type=int, default=0, help="seed of the token dropout")
    return p.parse_args(argv)


def main(argv=None) -> None:
    from humanoid_robot_learning.recap.value import ValueFunction

    args = parse_args(argv)
    model = ValueFunction.load(args.value)
    cfg = model.config
    args.fps, args.max_s = cfg.fps, cfg.max_s
    frames = load_frames(args.dataset, cfg.scene, cfg.robot, cfg.fps, cfg.max_s)
    labels, eps = label_frames(frames, model.value(frames.inputs), args)
    path = recap_dir(args.dataset) / f"labels_r{args.round}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(path, index=False)
    by_source = labels.groupby("source")["positive"].agg(["mean", "size"]).round(3).to_dict("index")
    print(f"[LABEL] eps {eps:.4f}; positive {labels['positive'].mean():.1%} of {len(labels)} frames; by source {by_source}")
    print(f"[LABEL] wrote {path}", flush=True)
    if args.training_copy:
        training_copy(args.dataset, labels, Path(args.training_copy), args.dropout, args.seed)


if __name__ == "__main__":
    main()
