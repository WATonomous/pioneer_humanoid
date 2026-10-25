"""Fine-tune pi0.5 (lerobot/pi05_base) on notmaxxx/pioneer_vla_pick_place, the Pioneer left arm.

Sized for one 48 GB GPU (A40 / A6000); the VLM is frozen and only the action expert trains.
Before running: `hf auth login` (write token: the final checkpoint is pushed to HUB_REPO_ID),
`wandb login` (or USE_WANDB = False), and ~40 GB of disk (pi05_base + one fp32 checkpoint).

    python pioneer_train_cloud.py         # from this folder: action_space.py sits next to it

Writes checkpoints/step_XXXXXX/ (keeps the newest KEEP_LAST) and checkpoints/train_log.csv,
which pioneer_eval_offline.py evaluates and plots.
"""
import csv
import math
import random
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import os

from huggingface_hub import HfApi, snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors

from action_space import delta_action_stats, save_action_space_delta, to_delta
import lerobot.policies.pi05.modeling_pi05 as pi05_modeling

"""
in joint vla models like pi0.5, vlm layer attends to the action expert which passes all 2b param on VLmM
causes pytorch to cache all vlm activations (calculations of neurons passing through) and trace through frozen weights
BUT SINCE we freeze the backbone this is all redundant!! Attention mask blocks later tokens to affect current tokens, so 
we can run .detach on all prefix tokens stops vlm from caching values to save VRAM
also skips backpropgation through VLM 2b backbone model

"""
_compute_layer_complete = pi05_modeling.compute_layer_complete


def _compute_layer_frozen_prefix(*args, **kwargs):
    prefix_out, suffix_out = _compute_layer_complete(*args, **kwargs)
    return [prefix_out.detach(), suffix_out]


pi05_modeling.compute_layer_complete = _compute_layer_frozen_prefix

torch.manual_seed(42)
random.seed(42)

MODEL = "lerobot/pi05_base"
REPO_ID = "notmaxxx/pioneer_vla_pick_place"
ROOT = snapshot_download(REPO_ID, repo_type="dataset", revision="v3.0",
                         local_dir=HF_LEROBOT_HOME / REPO_ID)
BATCH_SIZE = 16       # samples per forward pass (limited by GPU memory)
ACCUM_STEPS = 2      # 16 * 2 = 32 samples per weight update
TOTAL_STEPS = 6000  # number of weight updates
WARMUP_STEPS = 100   # ramp LR linear and slowly to not wreck weights
DECAY_STEPS = 1000   # LR holds constant, then decays over the final DECAY_STEPS updates
MIN_LR_FRAC = 0.1    # LR at the last update, as a fraction of the peak
VAL_EVERY = 250       # often enough to see where train and val loss diverge
SAVE_EVERY = 3000    # weight updates between checkpoints
KEEP_LAST = 1        # each checkpoint is ~14.5 GB (fp32 weights), so only keep the newest
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
LOG_PATH = os.path.join(CKPT_DIR, "train_log.csv")  # pioneer_eval_offline.py plots this as training_curves.png
# True: arm actions relative to the state the chunk was planned from; False: the dataset's raw
# (absolute) joint targets. See action_space.py. Eval/inference read it back from the checkpoint.
ACTION_SPACE_DELTA = True
USE_WANDB = True     # live loss curves at wandb.ai/<you>/pi05-pioneer; train_log.csv is written either way
# the final checkpoint is uploaded here (private) when training ends; None keeps it local only
HUB_REPO_ID = "notmaxxx/pi05_pioneer_pick_place" + ("_delta" if ACTION_SPACE_DELTA else "")

if HUB_REPO_ID:
    # create the repo now so a read-only token fails before training, not after it
    HfApi().create_repo(HUB_REPO_ID, private=True, exist_ok=True)

if USE_WANDB:
    import wandb
    # check the login now, not after the model loads; under nohup/tmux an unauthenticated
    # wandb.init would wait on an API-key prompt nobody can answer
    if not wandb.login(anonymous="never", timeout=30):
        raise SystemExit("wandb is not logged in: run `wandb login` (or set WANDB_API_KEY) first")

# dataset info of training data
meta = LeRobotDatasetMetadata(REPO_ID,
                              root=ROOT)
cfg = PreTrainedConfig.from_pretrained(MODEL)  # gets pretrained config of pi0.5
cfg.pretrained_path = MODEL
cfg.device = "cuda"
cfg.dtype = "float32"  # weights stored in fp32 so small AdamW updates aren't rounded away; autocast still does the math in bf16

policy = make_policy(cfg, ds_meta=meta)  # matches the policy's inputs to this dataset
policy.model.paligemma_with_expert.gemma_expert.lm_head.to("cpu")

# shuffle and split eps
episodes = list(range(meta.total_episodes))  # gets total # of ep
random.shuffle(episodes)
val_eps, train_eps = episodes[:1], episodes[1:]  # "slicing syntax", val gets x eps, train gets the rest

# creates global index mapping of frames across all episodes, concacenates videos together
delta_timestamps = resolve_delta_timestamps(policy.config, meta)
train_ds = LeRobotDataset(REPO_ID,
                          episodes=train_eps,
                          delta_timestamps=delta_timestamps,
                          root=ROOT)
val_ds = LeRobotDataset(REPO_ID,
                        episodes=val_eps,
                        delta_timestamps=delta_timestamps,
                        root=ROOT)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=6)  # 8 vCPUs: leave 2 for the main process
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)   # runs only every VAL_EVERY updates

stats = train_ds.meta.stats
if ACTION_SPACE_DELTA:
    # normalize by the spread of the deltas, not of the absolute joint angles
    stats = {**stats, "action": delta_action_stats(ROOT, train_eps, policy.config.chunk_size, stats["action"])}
preprocess, postprocess = make_pre_post_processors(policy.config, dataset_stats=stats)

# freeze backbone
TRAIN_PREFIXES = (
    "model.paligemma_with_expert.gemma_expert.",  # action expert transformer
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.time_mlp_in.",
    "model.time_mlp_out.",
)

trainable_params = []
for name, p in policy.named_parameters():
    # train only action action heads EXCLUDING lm_head
    if name.startswith(TRAIN_PREFIXES) and "lm_head" not in name:
        p.requires_grad = True
        trainable_params.append(p)
    else:
        p.requires_grad = False

optimizer = AdamW(trainable_params, lr=policy.config.optimizer_lr, foreach=False)

# define scheduler for learning rate, cosine decay


def lr_factor(step):
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS   # 0 -> 1
    decay_start = TOTAL_STEPS - DECAY_STEPS
    if step < decay_start:
        return 1.0                         # constant through the middle
    progress = (step - decay_start) / DECAY_STEPS                                     # 0 -> 1
    return MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * progress))  # 1 -> MIN_LR_FRAC


scheduler = LambdaLR(optimizer, lr_factor)


def to_action_space(batch):
    """Dataset batch (absolute targets) -> the action space the policy trains in."""
    if ACTION_SPACE_DELTA:
        batch["action"] = to_delta(batch["action"], batch["observation.state"])
    return batch

# validation function


@torch.no_grad()
def validate():
    policy.eval()
    losses = []
    # same noise + flow times every call, so a change in val loss comes from the weights, not the dice
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0)
        for batch in val_loader:   # the whole val set, not just its first few frames
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(preprocess(to_action_space(batch)))
            losses.append(loss.item())
    policy.train()   # easy to forget
    return sum(losses) / len(losses)

# save weights + processors so any checkpoint can be loaded and tested on its own


def save_checkpoint(step):
    os.makedirs(CKPT_DIR, exist_ok=True)
    # delete old ones BEFORE writing so the disk never holds more than KEEP_LAST at once
    old = sorted(d for d in os.listdir(CKPT_DIR) if d.startswith("step_"))
    for d in old[:max(len(old) - KEEP_LAST + 1, 0)]:
        shutil.rmtree(os.path.join(CKPT_DIR, d))
    path = os.path.join(CKPT_DIR, f"step_{step:06d}")
    policy.save_pretrained(path)
    preprocess.save_pretrained(path)
    postprocess.save_pretrained(path)
    save_action_space_delta(path, ACTION_SPACE_DELTA)
    print(f"step {step} | saved {path}")


os.makedirs(CKPT_DIR, exist_ok=True)
log_file = open(LOG_PATH, "w", newline="")
log = csv.writer(log_file)
log.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm"])

if USE_WANDB:
    wandb.init(
        project="pi05-pioneer",
        name=f"{'delta' if ACTION_SPACE_DELTA else 'absolute'}-{TOTAL_STEPS}steps",
        config={
            "batch_size": BATCH_SIZE,
            "accum_steps": ACCUM_STEPS,
            "total_steps": TOTAL_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "decay_steps": DECAY_STEPS,
            "min_lr_frac": MIN_LR_FRAC,
            "lr": policy.config.optimizer_lr,
            "val_eps": val_eps,
            "action_space_delta": ACTION_SPACE_DELTA,
        },
    )

policy.train()
step = 0        # weight updates
running_loss = 0
step_loss = 0   # loss of the current weight update, for the CSV
micro = 0       # batches seen
done = False

while not done:
    for batch in train_loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(preprocess(to_action_space(batch)))

        (loss / ACCUM_STEPS).backward()
        loss = loss.item() / ACCUM_STEPS   # one GPU->CPU sync per batch instead of two
        running_loss += loss
        step_loss += loss
        micro += 1
        if micro % ACCUM_STEPS != 0:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        # just for coms lol
        if step % 10 == 0:
            print(f"step {step} | loss { running_loss/10:.4f} | grad {grad_norm:.2f} | lr {scheduler.get_last_lr()[0]:.2e}")
            running_loss = 0
        val_loss = validate() if step % VAL_EVERY == 0 else None
        if val_loss is not None:
            print(f"step {step} | val_loss {val_loss:.4f}")
        log.writerow([step, step_loss, val_loss, scheduler.get_last_lr()[0], grad_norm.item()])
        log_file.flush()
        if USE_WANDB:
            metrics = {"loss/train": step_loss, "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm.item()}
            if val_loss is not None:
                metrics["loss/val"] = val_loss
            wandb.log(metrics, step=step)
        step_loss = 0
        if step % SAVE_EVERY == 0:
            save_checkpoint(step)
        if step >= TOTAL_STEPS:
            done = True
            break

if step % SAVE_EVERY != 0:
    save_checkpoint(step)
if USE_WANDB:
    wandb.finish()

if HUB_REPO_ID:
    final = os.path.join(CKPT_DIR, f"step_{step:06d}")
    try:
        HfApi().upload_folder(repo_id=HUB_REPO_ID, folder_path=final,
                              commit_message=f"step {step} ({'delta' if ACTION_SPACE_DELTA else 'absolute'})")
        print(f"uploaded {final} -> https://huggingface.co/{HUB_REPO_ID}")
    except Exception as e:   # the checkpoint is already on disk, so a network error doesn't lose the run
        print(f"upload failed: {e}\nretry with: hf upload {HUB_REPO_ID} {final} . --repo-type model --private")
