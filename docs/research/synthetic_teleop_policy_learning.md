# Policies from synthetic teleop demos (MuJoCo): status and handoff

State as of 2026-10-12 (late evening). Everything here runs headless on the Pioneer left arm in the MuJoCo scenes
(`src/simulation/mujoco_scenes`), on demos recorded by the scripted operator
(`src/teleop/pioneer_leader_arm_teleop/synthetic_teleop.py`). All code is in
`src/teleop/pioneer_leader_arm_teleop/`. Datasets and checkpoints are git-ignored: re-record or copy them.

## Setup

```bash
conda create -n synth_teleop python=3.12 -y && conda activate synth_teleop
conda install -c conda-forge ffmpeg -y                      # needs libsvtav1
pip install mujoco torch "lerobot @ git+https://github.com/huggingface/lerobot.git@e670ac5daf9b76" pynput   # LeRobot 0.4.3
pip install transformers                                    # only for patch_backbone.py (DINOv2)
cd src/teleop/pioneer_leader_arm_teleop && MUJOCO_GL=egl python synthetic_teleop.py --self-test
```

The repo's own packages are found through the scripts' `sys.path` fallback; no editable install is needed.
Versions this was run with: mujoco 3.15, torch 2.7.1, lerobot 0.4.3.

Data contract: 7-value state and action (6 joints in rad + gripper 0–1), 25 fps, cameras `ego` and `wrist_left`.

## Pipeline

```bash
D=$PWD/datasets/pioneer_v1_left_arm/sim_synthetic; cd src/teleop/pioneer_leader_arm_teleop

# 1. Demos. --num_episodes is the TOTAL the dataset should reach; a rerun resumes. Keep eval seeds (1000-1199) free.
MUJOCO_GL=egl python synthetic_teleop.py --scene peg_insert --record --dataset_root $D/peg_insert --num_episodes 180 --workers 2

# 2. 320x240 copy: 3.3x faster ACT training. (Resolution was never isolated: the 640x480 runs had fewer demos and steps.)
python downscale_dataset.py $D/peg_insert $D/peg_insert_320 --height 240 --width 320 --jobs 6

# 3. Multi-step scenes only (drawer_stow, duplo, matcha): put the current instruction into the state (see Findings 5).
python add_subtask_state.py $D/duplo_320 words

# 4. Train: 40k-50k steps at the default lr, then 10k steps at lr 1e-6 (the "low-LR finish").
lerobot-train --dataset.repo_id=humanoid/pioneer_v1_left_arm --dataset.root=$D/peg_insert_320 --policy.type=act \
  --policy.push_to_hub=false --policy.device=cuda --wandb.enable=false --steps=40000 --save_freq=40000 \
  --batch_size=8 --num_workers=3 --output_dir=outputs/train/act_peg
lerobot-train ... --policy.path=outputs/train/act_peg/checkpoints/040000/pretrained_model \
  --policy.optimizer_lr=1e-6 --policy.optimizer_lr_backbone=1e-6 --steps=10000 --output_dir=outputs/train/act_peg_lowlr

# 5. Evaluate on unseen layouts. Fast (renders only when the policy replans; ACT only, ~1 s per episode):
MUJOCO_GL=egl python qplan_collect.py --scene peg_insert --policy <ckpt>/pretrained_model --out eval.pt \
  --seed 1000 --episodes 200 --select first --n 1 --z_std 0
#    Any LeRobot policy, with videos (~10 s per episode):
MUJOCO_GL=egl python sim_policy_eval.py --scene peg_insert --policy <ckpt>/pretrained_model --episodes 100 --max_s 40
```

For a multi-step scene, copy `<dataset>/meta/subtask_state.json` into the checkpoint's `pretrained_model/` before
evaluating (the rollout reads it to build the same state), and use a longer `--max_s` (duplo 60, drawer_stow 90, matcha 150).
If the demos were recorded at 320x240 directly (matcha: `--schema` with 240x320 cameras), set `QPLAN_RENDER=240x320`.

## Results (success on unseen layouts)

| scene | recipe | result |
|---|---|---|
| peg_insert | ACT, 60 demos, 640x480, 24k steps | 13/30 |
| peg_insert | ACT, 180 clean demos, 320x240, 40k | ~79% (3 runs, 500 layouts) |
| peg_insert | same + low-LR finish | 84.5% (3 runs, 600 layouts) |
| peg_insert | ACT, 180 clean + 250 DAgger takeovers, 50k + low-LR finish | **~90%** (181/200; 177–182/200 on re-runs) |
| drawer_stow | ACT, 40 demos, no instruction | 3/12 |
| drawer_stow | ACT, 84 demos, 40k + low-LR (instruction in state, unused: see Findings 5) | 55/100 by the scene check |
| drawer_stow | ACT, 116 demos, same | **61/100** by the scene check; announced order followed in 9/65 |
| duplo | ACT, 40 demos, no instruction | 0/12 |
| duplo | ACT, 110 demos one-hot / 271 demos `words` | 6/100 / 4/100 (stacks bricks, ignores the order) |
| zip_tie | ACT, 10 demos, 1.5k steps | 20/20 (layout barely varies) |
| matcha | ACT, 60 demos (recorded at 320x240), instruction one-hot, 40k + low-LR | **11/40**; mean 2.0 of 4 steps, most stop at the water pour |

## Findings

1. **Data volume and training steps were the main levers** on peg_insert (60 → 180 demos, 6k → 40k steps).
   Past 40k steps at a constant lr the score only swings between checkpoints (77–90%).
2. **A low-LR finish (10k steps at 1e-6) is a cheap, robust gain**: +5 points for clean-only runs, and it removes the
   57–90% run-to-run spread of the DAgger runs.
3. **DAgger with the scripted expert adds little** (+1 to +6 points, not separable from noise). The expert only
   takes over from states the policy already handles; it gives up on the real failures (fallen or jammed peg).
   Takeovers that begin with a pause taught the policy to hover: fixed in `peg_plan(resume=True)`.
4. **Inference settings**: 100-step chunks executed open-loop are best. Replanning every 50 / 25 steps: 72% / 35%.
   Temporal ensembling and checkpoint weight averaging: worse.
5. **ACT does not use the instruction, even when it is in the state.** drawer_stow and duplo announce a new order
   every episode as the frame's task string. `add_subtask_state.py` appends it to the state (one-hot or `words`), but
   swapping the instruction changes the predicted chunk by <= 0.08 rad in both scenes. drawer_stow still scores
   55-61/100 because its check does not enforce order (step k = "the k-th announced block is in the drawer"; blocks put
   in early fail nothing): logging the order blocks enter, the first block matches the announcement in 27/84 rollouts
   (chance 1/3) and the full order in 9/65 (chance 1/6). duplo, where the order is physical, stays at 4-6/100.
   Hypothesis under test: ACT's CVAE latent carries "which brick" in training (`--policy.use_vae=false`).
6. **peg_insert failures are fragile, not systematic.** Of 21 failures in 200 layouts: 11 jams (peg tilted ~10° in
   the grip, on the rim, 1–4 mm off), 7 fallen pegs, 3 other. On failing layouts a sub-milliradian change to the
   action chunk makes it succeed 60–75% of the time (38% unchanged). Evals are therefore not exactly repeatable.
7. **Q-Planning (arXiv 2608.21204) on frozen ACT does not work here.**
   - ACT's latent gives no candidates (spread ~0.0001 rad); candidates need added noise.
   - A simulator-lookahead oracle choosing among 6 near-identical candidates per planning step reaches 299/300,
     so selection has full headroom.
   - A Q-function trained on 2400 exploration rollouts is near chance (AUC 0.55–0.64 on early chunks). Trained on the
     oracle's paired labels (300 layouts) its top pick is no better than the policy's own chunk (92.8% vs 92.8%;
     oracle 96.2%); it orders a success above a failure 62% of the time within a state.
   - On a frozen **Diffusion Policy** (same data, 40k steps; 11% alone on 200 layouts) it does work: Q trained on the
     planner's own rollouts lifts it to 37-45% (Q-weighted average of 16 samples; `qplan_dp.py`), then plateaus.
8. **Rewind-and-retry on stall does not help** (178–179/200 vs 177–182 without): the peg is already tilted in the
   grip, so lowering again jams again. A jam needs a regrasp.

Running logs with every number: `outputs/train/SYNTH_ACT_LOG.md` and `outputs/qplan/RESEARCH_LOG.md` on the
machine that ran them (git-ignored).

## Open work

- **Diffusion Policy on peg_insert, then Q-Planning on top of it** (real multi-modal candidates). Config that trains
  (78M params): `--policy.type=diffusion --policy.horizon=64 --policy.n_action_steps=32 --policy.n_obs_steps=2
  --policy.drop_n_last_frames=31 --policy.crop_shape='[216,288]' --policy.down_dims='[256,512,1024]'
  --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 --batch_size=16`. Not yet evaluated.
- **Patch Policy idea (arXiv 2607.18236)**: DINOv2 ViT-S patch tokens as ACT's backbone. `train_patch_act.py` with
  `--policy.vision_backbone=resnet_dinov2s --policy.pretrained_backbone_weights=null`; trains (62M params,
  ~2.3x slower per step than ResNet-18). Full run not done.
- **Grounded Action Model idea (arXiv 2609.23863)**: upper-bound test by feeding true object poses in the state. Not started.
- **duplo / drawer_stow**: retrain with the `words` encoding (`add_subtask_state.py <root> words`, the default) on 271 / 116 demos. Queued on the laptop.
- **pi0.5** (reads the instruction natively) on peg_insert and the multi-step scenes: being run on the desktop session.
- **matcha**: more demos (recording towards 150); the scripted plan succeeds on roughly half to all seeds depending on the batch.
- **A policy with observation history or a regrasp behaviour** for peg jams.

## Machine notes (8 GB GPU laptop, 31 GB RAM)

- One GPU training plus one CPU-heavy job at a time. A recording take holds its frames in RAM (drawer_stow takes
  peak near 18 GB at 640x480). Run long jobs detached (`setsid nohup`), since a crash of the host app kills children.
- ACT, batch 8: 0.10 s/step at 320x240 (0.33 at 640x480). Each checkpoint is ~200 MB plus ~400 MB of
  `training_state` (delete it once a run is done).

## Later additions (2026-10-13)

- **Out-of-range layouts** (best ACT peg_insert, 200 layouts each, `qplan_collect.py --jitter_scale`): training range ~89%,
  1.5x 62%, 2x 33.5%, 3x 6.5%. The policy is fit to a narrow position range; nothing else (lighting, cameras, physics) varies.
- **Where the peg's tilt comes from** (`peg_tilt_probe.py`): the peg arrives over the hole upright in demos and rollouts alike
  (median 0.8 deg, max 4.5 deg); the ~10 deg tilt of a jam is made when it lands on the rim. The scripted operator itself
  succeeds on 178/200 seeds, but all 22 failures are grasp give-ups (three missed grasps; such takes are discarded and never
  become demos). Once it holds the peg it inserts almost every time, by feel: push a little, ease off if the peg stops, nudge.
  The policy's failures are different (11 jams, 7 fallen pegs of 21): it does not reproduce that reactive loop when it
  replays 4 s chunks open-loop. So the equal 89% is a coincidence, not a teacher ceiling.
- **Replication on a second machine** (own demos, H.264 copy): ACT 154/200 at 40k, 175/200 after the low-LR finish.
- **pi0.5** (frozen VLM, action expert trained 6000 updates) on peg_insert: 19/200; underfit.
- **matcha**: 150 demos recorded; retrain in progress.
- **Out-of-range test on valid layouts** (`PEG_INSERT_RANGE` env var in the scene scales the shifts and redraws layouts where
  peg and block would touch): 1x 179/200, 1.5x 124/200, 2x 71/200. The scripted operator still succeeds on 17/20 takes at 2x.
- **`PEG_PLAN=wiggle`**: variant of `peg_plan` that works the peg in with a small outward spiral while easing down, so the
  demonstrated motion tolerates a millimetre or two of misalignment without feeling the rim. Operator success unchanged
  (58/60 with and without, +0.6 s per take). Whether a policy trained on it jams less is untested.
- **ACT with `use_vae=false` on duplo**: the policy starts to respond to the instruction on about half the layouts
  (0.18-0.46 rad chunk change vs <= 0.03 with the latent) but success is unchanged (8/100).
