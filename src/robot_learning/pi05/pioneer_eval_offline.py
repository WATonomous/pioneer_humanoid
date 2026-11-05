"""Offline evaluation: the policy's predicted actions vs. the recorded ones, on dataset episodes.

No Isaac Sim needed -- recorded camera frames + joint states go in, predicted 50-step action
chunks come out, and each prediction is compared with what the teleoperator actually did.

Run from the repo root (in a normal terminal, not VS Code's -- loading needs ~15 GB RAM):
    python src/robot_learning/pi05/pioneer_eval_offline.py
Redraw the plots from the last run's predictions, without loading the model:
    python src/robot_learning/pi05/pioneer_eval_offline.py --replot

Writes to eval/ next to this script:
    report_card.png       per-joint error over the 0.4 s the robot executes, vs. a "freeze" baseline
    error_vs_horizon.png  how the error grows with how far ahead the policy predicts
    heldout_rollout.png   what the robot would execute on the held-out episode vs. the operator
    training_curves.png   train/val loss per update, from checkpoints/train_log.csv
    metrics.json          the numbers behind the plots + a rule-of-thumb verdict
    predictions.npz       raw predictions, for --replot

Reading it -- the "freeze" baseline keeps commanding the arm's current position:
    train error >= freeze error      -> underfitting: hasn't learned even the demos it trained on
    held-out error >> train error    -> overfitting: memorised the training demos
    both well below freeze           -> fits; next check is closed-loop, in sim or on the arm
With a single held-out episode the held-out numbers are noisy; the training curves are the
steadier over/underfitting signal.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from action_space import load_action_space_delta, to_absolute

matplotlib.use("Agg")  # headless: plots are only written to PNG

HERE = Path(__file__).resolve().parent
REPO_ID = "notmaxxx/pioneer_vla_pick_place"   # dataset on the Hugging Face Hub
ROOT = None                                     # LeRobot caches it under ~/.cache/huggingface/lerobot/
FPS = 25
N_ARM = 6                     # joint1L..joint6l (rad); joint7l/joint8l are gripper fingers (m)
REPLAN_STEPS = 10             # actions executed per predicted chunk before replanning

# ---- 1. Arguments ----
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None,
                    help="checkpoint folder or Hub repo id (default: newest checkpoints/step_*)")
parser.add_argument("--train_episodes", type=int, default=5, help="training episodes to score for comparison")
parser.add_argument("--out", default=str(HERE / "eval"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--replot", action="store_true", help="redraw from <out>/predictions.npz, skip the model")
parser.add_argument("--train_log", default=str(HERE / "checkpoints" / "train_log.csv"),
                    help="per-update losses written by pioneer_train_cloud.py")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def to_display_units(a):
    """rad -> deg for the arm, m -> mm for the fingers."""
    a = np.array(a, dtype=np.float32)
    a[..., :N_ARM] *= 180.0 / math.pi
    a[..., N_ARM:] *= 1000.0
    return a


# ---- 2. Predict a chunk at every frame of the held-out + a few training episodes ----
def run_policy():
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    torch.manual_seed(args.seed)
    # the same train/val split the training scripts made
    ds = LeRobotDataset(REPO_ID, root=ROOT)
    episodes = list(range(ds.meta.total_episodes))
    random.seed(42)
    random.shuffle(episodes)
    heldout_ep, train_eps = episodes[0], episodes[1:]
    train_eps = random.Random(args.seed).sample(train_eps, min(args.train_episodes, len(train_eps)))
    starts, ends = ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"]

    if args.checkpoint is None:
        steps = sorted((HERE / "checkpoints").glob("step_*"))
        if not steps:
            raise SystemExit(f"no checkpoints in {HERE / 'checkpoints'}; train first or pass --checkpoint")
        args.checkpoint = str(steps[-1])
    if not Path(args.checkpoint).exists():   # a Hub repo id: fetch it so action_space.json is on disk too
        from huggingface_hub import snapshot_download
        args.checkpoint = snapshot_download(args.checkpoint)
    # pioneer_train_cloud.py saves fp32 weights (14.5 GB), more than a 12 GB GPU holds; evaluation
    # needs no fp32, so build the model in bf16 (the weights are cast as they load)
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.dtype = "bfloat16"
    policy = PI05Policy.from_pretrained(args.checkpoint, config=config).eval()
    preprocess, postprocess = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)
    action_space_delta = load_action_space_delta(args.checkpoint)
    print(f"[EVAL] checkpoint predicts {'delta' if action_space_delta else 'absolute'} actions", flush=True)

    stats = ds.meta.stats["action"]
    saved = {
        "names": np.array(ds.meta.features["action"]["names"]),
        "heldout_ep": heldout_ep, "train_eps": np.array(train_eps), "action_space_delta": action_space_delta,
        # each joint's range of motion over the whole dataset, to put errors in context
        "action_range": to_display_units(np.asarray(stats["max"]) - np.asarray(stats["min"])),
    }
    for ep in [heldout_ep] + train_eps:
        print(f"[EVAL] episode {ep} ({'held-out' if ep == heldout_ep else 'train'}), "
              f"{int(ends[ep]) - int(starts[ep])} frames", flush=True)
        preds, recorded, state = [], [], []
        for idx in range(int(starts[ep]), int(ends[ep])):
            item = ds[idx]
            obs = {k: v for k, v in item.items() if k.startswith("observation.")}
            obs["task"] = item["task"]
            with torch.inference_mode():
                chunk = postprocess(policy.predict_action_chunk(preprocess(obs))).float().cpu()
            if action_space_delta:   # back to absolute targets, relative to the state it planned from
                chunk = to_absolute(chunk, item["observation.state"].float())
            preds.append(chunk[0].numpy())                               # (50, 8)
            recorded.append(item["action"].float().numpy())               # (8,)
            state.append(item["observation.state"].float().numpy())       # (8,)
        saved[f"preds_{ep}"] = to_display_units(np.stack(preds))
        saved[f"recorded_{ep}"] = to_display_units(np.stack(recorded))
        saved[f"state_{ep}"] = to_display_units(np.stack(state))
    np.savez_compressed(out / "predictions.npz", **saved)
    return saved


saved = dict(np.load(out / "predictions.npz")) if args.replot else run_policy()
names = [str(n) for n in saved["names"]]
heldout_ep, train_eps = int(saved["heldout_ep"]), [int(e) for e in saved["train_eps"]]
all_eps = [heldout_ep] + train_eps
joint_range = saved["action_range"]
action_space_delta = bool(saved.get("action_space_delta", False))
unit = ["deg"] * N_ARM + ["mm"] * (len(names) - N_ARM)


# ---- 3. Error vs. look-ahead, for the policy and for a do-nothing baseline ----
def error_by_horizon(preds, recorded):
    """mean |pred[t, k] - recorded[t + k]| for each look-ahead k -> (50, 8)."""
    T, K, _ = preds.shape
    return np.stack([np.abs(preds[: T - k, k] - recorded[k:]).mean(axis=0) if k < T else
                     np.full(preds.shape[-1], np.nan) for k in range(K)])


K = saved[f"preds_{heldout_ep}"].shape[1]
policy_err = {ep: error_by_horizon(saved[f"preds_{ep}"], saved[f"recorded_{ep}"]) for ep in all_eps}
# "freeze": every step of the chunk is the arm's current position
freeze_err = {ep: error_by_horizon(np.repeat(saved[f"state_{ep}"][:, None], K, axis=1), saved[f"recorded_{ep}"])
              for ep in all_eps}
held = policy_err[heldout_ep]
train = np.nanmean([policy_err[e] for e in train_eps], axis=0)
freeze = np.nanmean([freeze_err[e] for e in all_eps], axis=0)

# the part the robot actually runs: the first REPLAN_STEPS of each chunk
executed = {name: np.nanmean(e[:REPLAN_STEPS], axis=0) for name, e in
            (("train", train), ("heldout", held), ("freeze", freeze))}
pct = {name: 100 * e / joint_range for name, e in executed.items()}
train_vs_freeze = pct["train"].mean() / pct["freeze"].mean()
heldout_vs_train = pct["heldout"].mean() / pct["train"].mean()
if train_vs_freeze >= 1:
    verdict = "UNDERFITTING - worse than freezing in place, even on episodes it trained on"
elif heldout_vs_train >= 1.5:
    verdict = "OVERFITTING - good on training episodes, much worse on the held-out one"
elif train_vs_freeze >= 0.7:
    verdict = "WEAK FIT - only slightly better than freezing in place"
else:
    verdict = "FITS - clearly beats freezing and holds up on the held-out episode"

# ---- 4. Plots ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.8,
})


def fmt(v, j):
    return f"{v:.1f}°" if j < N_ARM else f"{v:.1f} mm"


def header(fig, title, subtitle):
    """Bold title + gray how-to-read lines, top-left. Returns the plot area's top for tight_layout."""
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.14 / h, title, ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.012, 1 - 0.46 / h, subtitle, ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.5)
    return 1 - (0.7 + 0.2 * (subtitle.count("\n") + 1)) / h


def label_line_ends(ax, x, ys, labels, colors):
    """Name each line at its right end, nudging labels apart so they don't overlap."""
    min_gap = 0.07 * ax.get_ylim()[1]
    placed = []
    for i in np.argsort([y[-1] for y in ys]):
        y = ys[i][-1] if not placed else max(ys[i][-1], placed[-1] + min_gap)
        placed.append(y)
        ax.plot(x[-1], ys[i][-1], "o", ms=5, color=colors[i], mec=SURFACE, mew=1.5, clip_on=False)
        ax.text(x[-1] + 0.03 * (x[-1] - x[0]), y, labels[i], va="center", fontsize=9, color=INK2)


# 4a. report card: one bar per joint, against the freeze baseline
fig, ax = plt.subplots(figsize=(12, 5.6))
top = header(fig, f"How far off is the policy over the {REPLAN_STEPS / FPS:.1f} s it actually executes?",
             f"Bars: mean |predicted − recorded action| over the next {REPLAN_STEPS} steps, as % of that joint's "
             "range of motion in the dataset (lower is better).\n"
             "Gray line: a do-nothing policy that keeps commanding the arm's current position. "
             "A bar above its gray line is worse than doing nothing.\n"
             f"Rule-of-thumb verdict: {verdict}.")
x = np.arange(len(names))
w = 0.36
ax.bar(x - w / 2, pct["train"], w, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
       label=f"training episodes ({len(train_eps)})")
ax.bar(x + w / 2, pct["heldout"], w, color=ORANGE, edgecolor=SURFACE, linewidth=1.5,
       label=f"held-out episode {heldout_ep}")
ax.hlines(pct["freeze"], x - w - 0.04, x + w + 0.04, color=INK2, lw=2.2)
ax.axvline(N_ARM - 0.5, ymax=0.85, color=AXIS, lw=1)
ax.set_xticks(x, [f"{n}\nrange {joint_range[j]:.0f}{'°' if j < N_ARM else ' mm'}" for j, n in enumerate(names)])
ax.set_ylabel("mean |error|, % of joint range")
ax.set_ylim(0, max(pct["train"].max(), pct["heldout"].max(), pct["freeze"].max()) * 1.2)
ax.grid(axis="x", visible=False)
ax.legend(handles=ax.get_legend_handles_labels()[0] + [Line2D([], [], color=INK2, lw=2.2)],
          labels=ax.get_legend_handles_labels()[1] + ["freeze baseline"], loc="upper right", ncol=3)
fig.tight_layout(rect=(0, 0, 1, top))
fig.savefig(out / "report_card.png", dpi=150)

# 4b. error vs. look-ahead
ahead = np.arange(K) / FPS
fig, (ax_arm, ax_grip) = plt.subplots(1, 2, figsize=(12, 5))
top = header(fig, "Error vs. how far ahead the policy is predicting",
             f"Each plan covers {K} steps ({K / FPS:.0f} s), but only the shaded first {REPLAN_STEPS / FPS:.1f} s "
             "is executed before the policy replans.\n"
             "Training ≈ held-out → no overfitting.  Policy above the gray freeze line → doing nothing "
             "would be more accurate at that look-ahead.")
for ax, cols, title, u in ((ax_arm, slice(0, N_ARM), "Arm joints (mean of 6)", "degrees"),
                           (ax_grip, slice(N_ARM, None), "Gripper fingers (mean of 2)", "mm")):
    ax.axvspan(0, REPLAN_STEPS / FPS, color=GRID, alpha=0.6, lw=0)
    lines = ((f"train ({len(train_eps)} eps)", BLUE, train), ("held-out", ORANGE, held), ("freeze", MUTED, freeze))
    ys = [e[:, cols].mean(axis=1) for _, _, e in lines]
    for (label, color, _), y in zip(lines, ys):
        ax.plot(ahead, y, color=color, label=label)
    ax.set_title(title, loc="left")
    ax.set_xlabel("seconds ahead")
    ax.set_ylabel(f"mean |error| ({u})")
    ax.set_ylim(bottom=0)
    label_line_ends(ax, ahead, ys, [l for l, _, _ in lines], [c for _, c, _ in lines])
    ax.text(REPLAN_STEPS / FPS / 2, ax.get_ylim()[1], "executed", ha="center", va="top", fontsize=8.5, color=INK2)
ax_arm.legend(loc="lower right")
fig.tight_layout(rect=(0, 0, 0.97, top))
fig.savefig(out / "error_vs_horizon.png", dpi=150)

# 4c. held-out rollout: stitch together the part of each plan the robot would execute
preds, recorded = saved[f"preds_{heldout_ep}"], saved[f"recorded_{heldout_ep}"]
T = len(recorded)
t = np.arange(T) / FPS
replans = range(0, T, REPLAN_STEPS)
stitched = np.concatenate([preds[s, :REPLAN_STEPS] for s in replans])[:T]
fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
top = header(fig, f"Held-out episode {heldout_ep}: what the robot would execute vs. what the operator did",
             f"Every {REPLAN_STEPS / FPS:.1f} s (dots) the policy sees the recorded camera images + joint state and "
             f"plans; orange is the first {REPLAN_STEPS / FPS:.1f} s of each plan, the part the robot executes.\n"
             "The policy never trained on this episode. Jumps at the dots = consecutive plans disagree; "
             "flat orange while blue moves = the policy wants to stay put.")
for j, ax in enumerate(axes.flat):
    ax.plot(t, recorded[:, j], color=BLUE, label="recorded (operator)")
    for s in replans:
        seg = slice(s, min(s + REPLAN_STEPS, T))
        ax.plot(t[seg], stitched[seg, j], color=ORANGE, label="executed (policy)" if s == 0 else None)
        ax.plot(t[s], stitched[s, j], "o", ms=5, color=ORANGE, mec=SURFACE, mew=1.5)
    gap = np.abs(stitched[:, j] - recorded[:, j]).mean()
    ax.set_title(f"{names[j]}   gap {fmt(gap, j)} ({100 * gap / joint_range[j]:.0f}% of range)",
                 loc="left", fontsize=9.5)
    ax.set_ylabel("deg" if j < N_ARM else "mm")
for ax in axes[1]:
    ax.set_xlabel("time in episode (s)")
fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper right", ncol=2,
           bbox_to_anchor=(0.99, 1 - 0.1 / fig.get_figheight()))
fig.tight_layout(rect=(0, 0, 1, top))
fig.savefig(out / "heldout_rollout.png", dpi=150)

# 4d. training curves, if the training run logged them
log_path = Path(args.train_log)
if log_path.exists():
    rows = list(csv.DictReader(open(log_path)))
    steps = np.array([int(r["step"]) for r in rows])
    loss = np.array([float(r["train_loss"]) for r in rows])
    val = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    top = header(fig, "Training curves",
                 "Flow-matching loss. Noise is resampled every batch, so the raw train loss is jumpy - "
                 "read the smoothed line.\n"
                 "Both still falling → undertrained, train longer.  Train falling while validation rises → "
                 "overfitting.  Both flat and high → underfitting.")
    ax.plot(steps, loss, color=BLUE, alpha=0.25, lw=1)
    win = max(1, min(20, len(loss) // 5))
    smooth = np.convolve(loss, np.ones(win) / win, mode="valid")
    ax.plot(steps[win - 1:], smooth, color=BLUE, label=f"train (smoothed over {win} updates)")
    if val:
        vs, vl = zip(*val)
        ax.plot(vs, vl, "o-", color=ORANGE, ms=7, mec=SURFACE, mew=2, label="validation (held-out episode)")
    ax.set_xlabel("weight update")
    ax.set_ylabel("loss")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out / "training_curves.png", dpi=150)
else:
    print(f"(no {log_path} - rerun training to log losses, then --replot for training_curves.png)")

# ---- 5. Numbers ----
metrics = {
    "verdict": verdict,
    "action_space_delta": action_space_delta,
    "train_vs_freeze": round(float(train_vs_freeze), 3),
    "heldout_vs_train": round(float(heldout_vs_train), 3),
    "heldout_episode": heldout_ep, "train_episodes": train_eps, "joints": names, "units": unit,
    "action_range_in_data": joint_range.round(3).tolist(),
    "executed_window_mae": {k: v.round(3).tolist() for k, v in executed.items()},
    "executed_window_pct_of_range": {k: v.round(2).tolist() for k, v in pct.items()},
    "one_step_mae": {"heldout": held[0].round(3).tolist(), "train": train[0].round(3).tolist(),
                     "freeze": freeze[0].round(3).tolist()},
}
json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
print(f"\nerror over the executed {REPLAN_STEPS} steps (% of each joint's range):")
print(f"{'joint':9s} {'train':>7s} {'held-out':>9s} {'freeze':>7s}")
for j, n in enumerate(names):
    print(f"{n:9s} {pct['train'][j]:6.1f}% {pct['heldout'][j]:8.1f}% {pct['freeze'][j]:6.1f}%")
print(f"\ntrain / freeze   = {train_vs_freeze:.2f}   (< 1 beats doing nothing)")
print(f"held-out / train = {heldout_vs_train:.2f}   (~1 generalizes, >> 1 overfits)")
print(f"verdict: {verdict}")
print(f"\nwrote plots + metrics.json to {out}")
