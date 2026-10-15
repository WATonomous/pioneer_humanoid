"""Fine-tune pi0.5 (lerobot/pi05_base) on a local LeRobot dataset, on one 16 GB GPU.

Same recipe as pioneer_train_cloud.py (VLM frozen, only the action expert trains), sized down:
the frozen VLM is held in bf16 and the trained weights in fp32, batch 8 peaks near 13.5 GB.
Nothing leaves the machine (no Hub push, no wandb).

    python pioneer_train_local.py --dataset_root <repo>/datasets/pioneer_v1_left_arm/sim_synthetic/peg_insert_320 \
        --output_dir <repo>/outputs/train/pi05_peg

Writes <output_dir>/step_XXXXXX/ (a pretrained_model directory: sim_policy_eval.py and
pioneer_eval_offline.py load it) and <output_dir>/train_log.csv. --resume continues from the newest step.
"""
import argparse
import csv
import math
import os
import random
import shutil
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import lerobot.policies.pi05.modeling_pi05 as pi05_modeling
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors

from action_space import delta_action_stats, save_action_space_delta, to_delta

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--dataset_root", required=True)
parser.add_argument("--repo_id", default="humanoid/pioneer_v1_left_arm", help="name only; the data is read from --dataset_root")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--model", default="lerobot/pi05_base")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--accum_steps", type=int, default=2, help="batches per weight update")
parser.add_argument("--steps", type=int, default=6000, help="weight updates")
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--decay_steps", type=int, default=1000, help="cosine decay over the last N updates")
parser.add_argument("--min_lr_frac", type=float, default=0.1)
parser.add_argument("--val_episodes", type=int, default=4)
parser.add_argument("--val_every", type=int, default=250)
parser.add_argument("--save_every", type=int, default=1000)
parser.add_argument("--keep_last", type=int, default=2, help="checkpoints kept (each ~9 GB)")
parser.add_argument("--num_workers", type=int, default=6)
parser.add_argument("--video_backend", default="pyav", help="pyav needs no system FFmpeg; torchcodec is faster where it loads")
parser.add_argument("--delta", action="store_true",
                    help="train on delta joint actions (action_space.py); the rollout must convert back. Default: absolute")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

# The frozen VLM only feeds the action expert: detach its output at every layer so its activations
# are not kept for a backward pass that would update nothing (see pioneer_train_cloud.py).
_compute_layer_complete = pi05_modeling.compute_layer_complete


def _compute_layer_frozen_prefix(*a, **k):
    prefix_out, suffix_out = _compute_layer_complete(*a, **k)
    return [prefix_out.detach(), suffix_out]


pi05_modeling.compute_layer_complete = _compute_layer_frozen_prefix

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "train_log.csv"
saved = sorted(d for d in os.listdir(out_dir) if d.startswith("step_"))
resume_from = out_dir / saved[-1] if args.resume and saved else None

meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
cfg = PreTrainedConfig.from_pretrained(resume_from or args.model)
cfg.pretrained_path = str(resume_from or args.model)
cfg.device = "cuda"
cfg.dtype = "bfloat16"            # the frozen VLM; the trained weights go back to fp32 below
cfg.train_expert_only = True
cfg.gradient_checkpointing = True
if resume_from is None:
    cfg.input_features, cfg.output_features = {}, {}   # take the cameras, state and action from the dataset, not pi05_base's

policy = make_policy(cfg, ds_meta=meta)
policy.model.paligemma_with_expert.gemma_expert.lm_head.to("cpu")   # never used by the action expert

TRAIN_PREFIXES = (
    "model.paligemma_with_expert.gemma_expert.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.time_mlp_in.",
    "model.time_mlp_out.",
)
trainable_params = []
for name, p in policy.named_parameters():
    p.requires_grad = name.startswith(TRAIN_PREFIXES) and "lm_head" not in name
    if p.requires_grad:
        p.data = p.data.float()   # fp32 so small AdamW updates aren't rounded away; autocast does the math in bf16
        trainable_params.append(p)
print(f"trainable {sum(p.numel() for p in trainable_params) / 1e6:.0f} M of "
      f"{sum(p.numel() for p in policy.parameters()) / 1e6:.0f} M parameters")

episodes = list(range(meta.total_episodes))
random.shuffle(episodes)
val_eps, train_eps = sorted(episodes[:args.val_episodes]), sorted(episodes[args.val_episodes:])

delta_timestamps = resolve_delta_timestamps(policy.config, meta)
# One dataset over every episode, split by frame. LeRobotDataset(episodes=[...]) is not used: in LeRobot 0.4.3 a
# subset returns the action chunk of the wrong frames (the chunk indices are absolute, the filtered table is not),
# which trains the policy to hold the start pose.
dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, delta_timestamps=delta_timestamps,
                         video_backend=args.video_backend)
frame_episode = torch.as_tensor(dataset.hf_dataset.with_format(None)["episode_index"])
is_val = torch.isin(frame_episode, torch.tensor(val_eps))
train_ds = torch.utils.data.Subset(dataset, torch.nonzero(~is_val).flatten().tolist())
val_frames = torch.nonzero(is_val).flatten().tolist()
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=args.num_workers, persistent_workers=True)
# a fixed, evenly spread subset of the held-out frames: enough to see over/underfitting, fast enough to run often
val_idx = val_frames[::max(1, len(val_frames) // (16 * args.batch_size))]
val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=args.batch_size, num_workers=2)
print(f"{len(train_ds)} training frames, {len(val_frames)} held-out frames (episodes {val_eps}), {len(val_idx)} used for val")

stats = dataset.meta.stats
if args.delta:
    stats = {**stats, "action": delta_action_stats(args.dataset_root, train_eps, policy.config.chunk_size, stats["action"])}
preprocess, postprocess = make_pre_post_processors(policy.config, dataset_stats=stats)

optimizer = AdamW(trainable_params, lr=policy.config.optimizer_lr, foreach=False)


def lr_factor(step):
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    decay_start = args.steps - args.decay_steps
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / args.decay_steps
    return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))


scheduler = LambdaLR(optimizer, lr_factor)

step = 0
if resume_from is not None:
    state = torch.load(resume_from / "training_state.pt", map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    step = state["step"]
    print(f"resumed from {resume_from} at step {step}")


def to_action_space(batch):
    if args.delta:
        batch["action"] = to_delta(batch["action"], batch["observation.state"])
    return batch


@torch.no_grad()
def validate():
    policy.eval()
    losses = []
    # same noise + flow times every call, so a change in val loss comes from the weights, not the dice
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0)
        for batch in val_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(preprocess(to_action_space(batch)))
            losses.append(loss.item())
    policy.train()
    return sum(losses) / len(losses)


def save_checkpoint(step):
    old = sorted(d for d in os.listdir(out_dir) if d.startswith("step_"))
    for d in old[:max(len(old) - args.keep_last + 1, 0)]:
        shutil.rmtree(out_dir / d)
    path = out_dir / f"step_{step:06d}"
    policy.save_pretrained(path)
    preprocess.save_pretrained(path)
    postprocess.save_pretrained(path)
    save_action_space_delta(path, args.delta)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step},
               path / "training_state.pt")
    print(f"step {step} | saved {path}", flush=True)


new_log = step == 0 or not log_path.exists()
log_file = open(log_path, "w" if new_log else "a", newline="")
log = csv.writer(log_file)
if new_log:
    log.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm"])

policy.train()
running_loss, step_loss, micro = 0.0, 0.0, 0
while step < args.steps:
    for batch in train_loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(preprocess(to_action_space(batch)))
        (loss / args.accum_steps).backward()
        loss = loss.item() / args.accum_steps
        running_loss += loss
        step_loss += loss
        micro += 1
        if micro % args.accum_steps != 0:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        if step % 10 == 0:
            print(f"step {step} | loss {running_loss / 10:.4f} | grad {grad_norm:.2f} | lr {scheduler.get_last_lr()[0]:.2e} "
                  f"| peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            running_loss = 0.0
        val_loss = validate() if step % args.val_every == 0 else None
        if val_loss is not None:
            print(f"step {step} | val_loss {val_loss:.4f}", flush=True)
        log.writerow([step, step_loss, val_loss, scheduler.get_last_lr()[0], grad_norm.item()])
        log_file.flush()
        step_loss = 0.0
        if step % args.save_every == 0:
            save_checkpoint(step)
        if step >= args.steps:
            break

if step % args.save_every != 0:
    save_checkpoint(step)
