# Encoder bench (PioneerNet)

This bench tests which vision encoder gives a manipulation policy the most robust input. It covers our distilled
RGB-D encoder, **PioneerNet**, and 2024–2026 foundation encoders. The paper write-up and result tables are in the
Claude doc "PioneerNet: A Robust RGB-D Encoder for Robot Manipulation" (https://claude.ai/artifact/Sr6UT6JVsPQWuMy4FZsRFg).
This file covers running the bench.

Everything runs on the shared workstation (16 GB GPU, 62 GB RAM, other users' jobs on it), from this directory, with
`.venv/bin/python` (a link to the `vipe` conda env). Main robot: the 8-DoF bimanual arm (`BENCH_ROBOT=8dof`), peg
insertion task. Other scenes: `BENCH_SCENE=zip_tie` and `BENCH_SCENE=drawer_stow`.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Demos and images (scripted expert, MuJoCo) | `gen_data.py --split train --seeds 0:160` | `data_8dof/` |
| 2. Distil PioneerNet (RGB teacher + depth teacher → one RGB-D ResNet-18 student) | `distill.py` | `results_8dof/pioneernet_vN.pt` |
| 3. Probe: frozen encoder → attentive probe → peg/hole/TCP positions (mm) | `probe.py --tag <tag> --configs <enc>@ego+wrist_left` | `results_8dof/<tag>.json` |
| 4. Policy: ACT-style policy on the encoder, closed loop on 100 held-out layouts plus shift tests | `bc.py --delta --tag <tag> --streams <enc>@ego+wrist_left --shift_eval [--finetune]` | `results_8dof/bc/<tag>.json`, saved policy |
| 5. Weather and scene tests on a saved policy (50 layouts each) | `scenario_eval.sh <tag>`, and `eval_policy.py --tag <tag> --variant table` for the held-out tablecloth | `results_8dof/bc_exec/<tag>__ensemble_m0.6_<scenario>.json` |
| 6. Pages | `build_summary.py` → `report_summary/index.html`; `build_report.py` → `report/index.html` | |

Encoders are defined in `encoders.py` (`E.load(name)`). Shift tests: light, colour, camera moved, dark colour camera,
depth-sensor noise (`bc.py` flags), and fog, glare, wet lens, shadows, clutter, new background, tablecloth
(`scenarios.py`).

## PioneerNet versions

All versions use `--rgb_teacher dinov2_s14 --depth_teacher defm_s14 --steps 45000 --moddrop 0.4 --stride 8 --dim 256
--hw 256 320 --geo 1.0 --cam_aug train_camjit --hue_mix 0` (the exact commands are in `jobs.txt`), plus:

| Version | Changes from v5 | Status |
|---|---|---|
| v5 | none | **The paper's model** (best clean, light, fog and lens) |
| v6 | `--extra_scenes zip_tie:2,drawer_stow:4 --scene_aug 0.5` (heavy scene-content corruption) | Better on backgrounds and tablecloth, worse on clean |
| v7 | extra scenes, `--scene_aug 0.3 --scene_keep_clean --scene_box_max 0.08` (light corruption) | About even with v5, not better |
| v8 | v7's recipe with `--rgb_teacher dinov3_b16` | Best on new backgrounds (37%), but 67% clean |

## Job queues

Three runners run in the background, each re-reading its job file after every job:

| Runner | Job file | Use | Memory cap |
|---|---|---|---|
| `runner.sh` | `jobs.txt` | Main queue: distillation, probes, policy training (including fine-tunes) | 56 GB |
| `runner_b.sh` | `jobs_b.txt` | Second training queue, small frozen-encoder policy runs only | 36 GB |
| `runner_eval.sh` | `jobs_eval.txt` | Scenario evals (light on the GPU) | 56 GB |

- A job is one line, `<tag> <command...>`. `#` lines are comments. A job is skipped once `results/jobs/<tag>.done`
  or `.failed` exists, and its output goes to `results/jobs/<tag>.log`. The runners' own logs are
  `results/runner*.log`.
- **Every job file must end with a newline.** The runners read lines with `read`, which drops a last line that
  has no newline. On 2026-11-16 this silently stopped `b8_siglino` from running.
- To re-run a job, delete its `.done` or `.failed` file. To add a job, append a line (not while the runner is
  reading the file). A job the runner has already started keeps its old command even if you edit the line.
- `./gate.sh <RAM GB> <GPU GB>` waits until the machine has that much free memory. Heavy jobs start with it.
  Typical gates are 32 GB / 5 GB for a frozen policy, 32 / 8 for a fine-tune, 36 / 4 for a probe and 16 / 3 for an
  eval.
- Most eval jobs wait on `pgrep -f "[b]c.py .*--finetune"`: a DINOv3 ConvNeXt fine-tune (about 10 GB), an eval
  and another user's job don't fit together on the 16 GB GPU. The PioneerNet fine-tune needs only 6.4 GB, so its
  evals (`sc_v5ft*`) skip that wait.
- Pause: `touch results/PAUSE` stops new jobs after the current one (the `pausechk_*` jobs and eval jobs wait on
  it). `rm results/PAUSE` resumes. `disk_guard.sh` sets PAUSE when the disk drops below 600 MB.
- `bc.py` is resumable. It writes `results_8dof/bc/<tag>.resume` during and after training, and the same command
  re-run continues from it, or goes straight to evaluation once training has finished.

## Known issues

- **Two heavy jobs at once crashed the workstation (2026-11-11).** Keep memory caps and gates on everything.
- **Big-token encoders run out of memory in `probe.py`** when every test condition is cached up front
  (C-RADIOv4-SO400M, 1152-d tokens, went over 56 GB). Use `--lazy_test`, which extracts, evaluates and frees one
  test condition at a time. It gives identical numbers.
- **The GPU is shared.** While another user's training holds it at 100%, our jobs run 2–3 times slower. On
  2026-11-16 a SigLino-70M evaluation managed about 25 episodes in 27 minutes beside such a job, but finished all
  250 in 28 minutes once the GPU was free. Check `nvidia-smi` before estimating times.

## Status (2026-11-16)

- Done: v5 to v8 (3 policy runs each, plus all weather and scene tests), DINOv3 ConvNeXt fine-tuned (3 runs),
  DeFM depth-only, C-RADIOv4-SO400M policy, SigLino-30M and 70M probes.
- Done: PioneerNet v5 fine-tuned end to end, 3 runs (`pioneer_v5_ft`, `_r2`, `_r3`): 85% clean (same as frozen),
  but light 14% and glare 9% against frozen v5's 71% and 45%. A run takes about 2.5 hours on a shared GPU and
  needs 6.4 GB of GPU memory and 22 GB of RAM.
- Done: C-RADIOv4-SO400M probe (`probe_2026_cradio4`, run with `--lazy_test`): 2.4 mm clean, 7.6 under light
  and 7.2 under colour. SigLino-70M policy: 10% clean, 0–6% under every shift.
- All queues are empty.
