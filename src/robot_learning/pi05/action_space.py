"""Delta (relative) joint actions for the Pioneer pi0.5 policy.

The dataset stores ABSOLUTE joint targets. In delta mode the policy instead predicts each step
of its 50-step chunk as an offset from the arm's measured joint state when the chunk was planned:

    delta[t, k] = action[t + k] - state[t]        for the arm joints (joint1L..joint6l)

The gripper fingers stay absolute -- "open" / "closed" is a target, not a motion. This follows
openpi's DeltaActions transform (pi0/pi0.5 on joint-space robots such as ALOHA).

Why: an absolute policy that hasn't learned to read the state falls back to a common pose (the
start pose); a delta policy in the same spot predicts ~0 and holds where the arm is.

It can't be baked into the dataset: the same recorded frame sits in 50 different chunks, each
relative to a different start state. So the training scripts convert every batch with to_delta(),
and pioneer_eval_offline.py (and any rollout script) convert predicted chunks back with
to_absolute(), using the state the chunk was planned from. Each checkpoint folder gets an
action_space.json whose action_space_delta says whether it was trained on deltas; no file
means absolute.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DELTA_DIMS = 6                 # joint1L..joint6l; joint7l/joint8l (fingers) stay absolute
MARKER = "action_space.json"


def to_delta(action, state):
    """Absolute chunk (..., K, D) + the state it was planned from (..., D) -> delta chunk."""
    out = action.clone()
    out[..., :DELTA_DIMS] -= state[..., None, :DELTA_DIMS]
    return out


def to_absolute(action, state):
    """Inverse of to_delta: delta chunk (..., K, D) + its planning state (..., D) -> absolute chunk."""
    out = action.clone()
    out[..., :DELTA_DIMS] += state[..., None, :DELTA_DIMS]
    return out


def delta_action_stats(root, episodes, chunk_size, base_stats):
    """Normalization stats for delta actions, over every (start frame, look-ahead) pair in training.

    The arm dims' stats are recomputed for action[t + k] - state[t]; the finger dims keep the
    dataset's own. Past an episode's end LeRobot pads the chunk by repeating the last frame, so
    the same is done here.
    """
    df = pd.read_parquet(Path(root) / "data", columns=["episode_index", "action", "observation.state"])
    deltas = []
    for _, ep in df[df["episode_index"].isin(episodes)].groupby("episode_index"):
        action = np.stack(ep["action"].to_numpy())[:, :DELTA_DIMS]
        state = np.stack(ep["observation.state"].to_numpy())[:, :DELTA_DIMS]
        T = len(ep)
        idx = np.minimum(np.arange(T)[:, None] + np.arange(chunk_size), T - 1)   # (T, K) frame indices
        deltas.append((action[idx] - state[:, None]).reshape(-1, DELTA_DIMS))
    d = np.concatenate(deltas)

    stats = {k: np.array(v, copy=True) for k, v in base_stats.items()}
    stats["min"][:DELTA_DIMS] = d.min(axis=0)
    stats["max"][:DELTA_DIMS] = d.max(axis=0)
    stats["mean"][:DELTA_DIMS] = d.mean(axis=0)
    stats["std"][:DELTA_DIMS] = d.std(axis=0)
    for key in stats:
        if key.startswith("q"):                       # q01, q10, q50, q90, q99
            stats[key][:DELTA_DIMS] = np.quantile(d, int(key[1:]) / 100, axis=0)
    return stats


def save_action_space_delta(checkpoint_dir, action_space_delta):
    with open(Path(checkpoint_dir) / MARKER, "w") as f:
        json.dump({"action_space_delta": action_space_delta, "delta_dims": DELTA_DIMS}, f, indent=1)


def load_action_space_delta(checkpoint_dir):
    """True if the checkpoint predicts delta actions. Checkpoints from before delta support have
    no marker -> False (absolute)."""
    path = Path(checkpoint_dir) / MARKER
    return json.load(open(path))["action_space_delta"] if path.exists() else False
