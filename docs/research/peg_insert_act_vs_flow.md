# Peg insertion in MuJoCo: ACT vs flow-matching transformer

**Summary.** The flow transformer was trained on 400 of the ~605 unique episodes, for 225k
updates. It beats the final ACT (all episodes, 347k updates, seven correction rounds) on the
same 220 evaluation layouts: **139/220 vs 106/220** (Fisher p = 0.002). Most of the gap comes
from layouts whose failures supplied the corrections (68 vs 42). On fresh layouts the two are
level (56 vs 52, p = 0.67). Neither policy fails at grasping. Both fail at aligning with a hole
that has moved 3–7 cm (fresh: flow 8/30, ACT 9/30). The cheapest next step is to train flow on
corrections from *its own* failures, which it has never had. Plan at the end.

> **Scene version.** Every result below was collected in the original `peg_insert` scene: **1 mm
> total clearance** (0.5 mm per side), plain table at z = 0.705 m. `main` has since changed that
> scene to 4 mm clearance in the lightbox workcell (table top 0.780 m). See "Main's 4 mm workcell
> scene" for how these policies transfer and adapt.

Task: pick up a 40 mm square peg and insert it into a block with 1 mm total clearance, left arm,
25 Hz absolute joint targets, ego + wrist 640×480 RGB and 7-D proprioception. All numbers are
unassisted closed-loop success (seated **and** released for 0.5 s within 60 s) from the
evaluation reports copied to `outputs/eval/` (NAS paths in each report). Tooling:
[`peg_insert/README.md`](../../src/simulation/mujoco_scenes/humanoid_mujoco_scenes/peg_insert/README.md).

## Evaluation sets

| Set | Layouts | Notes |
|---|---|---|
| nominal | 20 seeds from 16000, both objects jittered ±1 cm | the original demo distribution |
| fresh | 100 new expert-feasible jitters inside mapped 0–7 cm workspace cells (50 peg-moved, 50 hole-moved) | no trajectory from these layouts was ever trained on |
| exposed | the 100 workspace held-out layouts | failures here supplied the moved-hole corrections, so they are *not* held out after that round |
| insertion cases | prepared held peg: aligned, ±3 mm offset, ±5° tilt, 10 seeds each | diagnoses insertion separately from grasping |

## ACT history

ACT (LeRobot, ResNet18, chunk 50) was trained on expert demos and continued through rounds of
expert corrections. Every round resumed the previous optimizer and kept its normalization.

| Round (job) | Added data | Steps | Result |
|---|---|---|---|
| 10k → 100k (703611, 703885) | 50 demos, ±1 cm | 100k, batch 4 | seeds 1000–1009, 5-step execution: 0/10, never grasped |
| approach takeovers (704251) + accuracy loop | 25 + 10/round expert takeovers within 3 cm of the hole | 150k | 150k diagnostics (706499): full 2–3/10; insertion cases 0–4/10, tilts 0–2/10 |
| phase-balanced diagnostic corrections (706909) | 56 corrections of failed diagnostic seeds; quotas 4/2/1/1 | 174k | held-out full 1/20; insertion 0–1/10. No gain |
| broad starts (707328) | 80 held starts up to 8 cm away, ±2–5° tilt | 198k | 50-step chunks: full 8/20, insertion 5–7/10 (25-step: 5/20, 10-step: 2/20) |
| rim-jam relift (707503) | 32 real early-jam states + 24 large-tilt starts; quotas 3/1/3/1 | 222k | **fresh ±1 cm: 17/20 full, 40/50 insertion cases** (50-step chunks) |
| independent workspace (708647, 709633) | 200 demos with peg *or* hole moved 0–7 cm, 2 shuffled passes | 323k | workspace held-out 7 → 18/100; nominal fell to 3/20 |
| moved-hole trace corrections (709670) | expert takeovers restored from 323k failure traces; quotas 3/2/2/1 | 347k | **fresh 21 → 52/100, exposed 18 → 42/100, nominal 3 → 12/20** |

What mattered, in order:

1. **Chunk execution length.** Executing 50 actions per chunk beat 25- and 10-step execution
   wherever it was compared (198k: 8/20 vs 5/20 vs 2/20; 222k: 18/20 vs 12/20). At 150k,
   temporal ensembling was no better than 10-step chunks (3/10 each). The
   expert's insertion is slow and smooth; frequent replanning introduced dithering at the rim.
2. **Corrections from the policy's *own* failure states** (rim jams, moved-hole traces) gave the
   two large jumps. Corrections from expert-prepared starts (diagnostic seeds) did not help.
3. **Phase-balanced batches** keep a few hundred correction frames visible next to ~50k demo
   frames, without duplicating episodes.
4. Wider placement data alone (workspace round) hurt the nominal layouts badly (17/20 → 3/20)
   until failure-driven corrections were added back on top.

Every learned-policy failure in the final 347k evaluation is after a successful grasp and lift
(100% grasp, 100% lift on fresh/exposed/nominal). The remaining problem is purely
hole alignment and insertion, worst when the *hole* is moved 3–7 cm (fresh: 9/30).

The 150k hole-shift audit (706907) showed why: with the robot state frozen and only the rendered
hole moved 3 mm, an ACT chunk moved the peg 0.3–0.5× as far as the expert (closed-loop
ensembling ~0.6–0.9× in x, 0.6× in y). The policy under-uses visual hole position.

## Expert ablation (this cleanup)

The moved-hole round used a separate `SeatingExpert` with early-jam aborts and a seating check
before release. As an option of the main expert (`guarded=True`), on full expert runs:

| N = 40 | nominal ±1 cm (20) | hole/peg moved 3–7 cm (20) | mean episode |
|---|---|---|---|
| plain | 20/20 | 15/20 | 42.2 s |
| guarded | 20/20 | 15/20 | 47.4 s |

The guard therefore stays off for demos and on for corrections that start from policy states.
The rewritten expert reproduces the old plain expert exactly: same outcome and simulated time
on 40/40 layouts.

## Main's 4 mm workcell scene

`main` changed `peg_insert` to 4 mm clearance inside the lightbox workcell. The table top is
7.5 cm higher (0.780 m), and the background and lighting are completely different: a white
lightbox instead of a wooden table on a checkered floor. All runs below are unassisted, with
50-action chunks and the same success gate (seated and released for 0.5 s within 60 s). They use
code snapshots of `main` plus these tools (`runs/` on NAS).

**Zero-shot (no new data).** The two scene variants separate clearance from everything else:

| Variant | ACT 347k nominal / fresh | Flow 225k nominal / fresh |
|---|---|---|
| Old scene, clearance only 1 → 4 mm | 16/20 · 53/100 | **20/20 · 90/100** |
| Main's workcell (4 mm + table + visuals) | 0/20 · 0/5 | 0/20 · 0/89 |

The 1 mm-trained policies transfer to 4 mm on their own scene; flow almost fully. In the
workcell neither grasps. Flow still reaches the correct grasp height for the higher table (lowest
finger centre 0.888 m, against 0.890 m needed), but it hovers 2.5–4 cm off the peg, never closes
the gripper, and eventually knocks the peg over. Coarse reaching transfers; the image-driven final
centring and the decision to close do not.

**Grasp corrections.** 60 expert grasps were handed off from each policy's own approach
(`collect_corrections --trigger pre-grasp --skill grasp`), followed by 10k updates with half of
each batch from old data and half from the corrections
(`config/peg_insert_grasp_phases.json`).

| | grasped / lifted, nominal 20 | grasped / lifted, fresh 99 | inserted |
|---|---|---|---|
| Flow 235k | 20 / 20 | 85 / 84 | 0 |
| ACT 357k | 20 / 10 | 88 / 49 | 0 |

**Round 2: full demos plus insertion corrections.** The round used:

- 75 full expert demos in the workcell: 25 nominal ±1.5 cm, and 50 with the peg or the hole
  moved 0–7 cm, taken from the training cells of the workspace map.
- Expert insertions restored from each policy's own failed rollouts (flow 287, ACT 178).
- 15k updates with batch quotas old 2 / grasp 2 / alignment 2 / recovery 1 / insertion 1
  (`config/peg_insert_workcell_phases.json`).

Evaluation is on held-out layouts: 20 new nominal seeds (17000+) and 99 new jitters in the fresh
set's cells.

| Held-out | Flow 250k | ACT 372k |
|---|---|---|
| nominal | 9/20 | 9/20 |
| fresh: hole moved 0–3 / 3–7 cm | 5/20 · 8/30 | 4/20 · 1/30 |
| fresh: peg moved 0–3 / 3–7 cm | 6/20 · 10/29 | 3/20 · 9/29 |
| **fresh total** | **29/99** | 17/99 (p = 0.06) |
| grasped / lifted (fresh) | 99 / 99 | 99 / 99 |

Grasp and transport now work for both policies. Almost every failure is a held peg near the hole
that never seats (flow 69/70, ACT 79/82), which is the same residual-alignment problem as in the
1 mm scene. Flow's lead comes mostly from holes moved 3–7 cm.

**Round 3: corrections from round-2 failures.** For each policy, 40 expert insertions were
restored from its own failed round-2 rollouts. Training continued 15k updates with the same mix
(flow 250k → 265k, ACT 372k → 387k). Each job took about 2 h. Both policies were evaluated on new
held-out layouts, identical for the two: 20 nominal seeds from 18000, and 99 new jitters in the
same cells. Round 2's held-out layouts had become correction sources.

| Held-out (new layouts) | Flow 265k | ACT 387k | p |
|---|---|---|---|
| nominal | 10/20 | 6/20 | |
| fresh: hole moved 0–3 / 3–7 cm | 10/20 · 7/30 | 4/20 · 3/30 | |
| fresh: peg moved 0–3 / 3–7 cm | 10/20 · 17/29 | 8/20 · 22/29 | |
| **fresh total** | **44/99** | **37/99** | 0.39 |
| all held-out | 54/119 | 43/119 | 0.19 |
| failures: held, not seated / seated, not released in time | 50 / 5 | 57 / 5 | |

From round 2 to round 3, fresh success rose for both policies: flow 29 → 44/99 (p = 0.04) and
ACT 17 → 37/99 (p = 0.002). The fresh sets in the two rounds are different jitters of the same
cells. Flow still leads, but the gap is no longer significant. ACT now does better when the peg
moves far (22/29); flow does better when the hole moves (17/50 vs 7/50). The hole moved 3–7 cm
is still the weakest case for both (7/30 and 3/30). Most failures are still a peg held near the
hole that never seats.

## Reproducing a round with the current tools

The per-round scripts (`phase_correction_round`, `broad_correction_round`,
`retry_correction_round`, `workspace_round`, `moved_hole_round`, `iterate_corrections`,
`diagnose_policy`) were replaced by the general verbs. Their source is in each run's
`code.tgz` on NAS.

| Old round | Now |
|---|---|
| approach / targeted takeovers | `collect_corrections --source takeover --trigger approach\|stall`, or `dagger` |
| broad starts | `collect_corrections --source tasks --tasks broad` |
| rim-jam relift + large tilt | `--source takeover --trigger rim-jam --hold-seconds 0` and `--source tasks --tasks large-tilt --hold-seconds 0` |
| workspace map + 200 demos + held-out layouts | `workspace map`, `workspace sample --split train --record ...`, `workspace sample --split test` |
| moved-hole trace corrections | `eval --trace-dir`, then `collect_corrections --source traces` |
| fine-tune with phase quotas | `humanoid_robot_learning.lerobot_finetune --extra-dataset ... --phase-config ...` |
| diagnostics (5 cases × 3 modes) | `eval --start-stage insertion --insertion-case <case>` with `--n-action-steps` / `--temporal-ensemble-coeff` |
| action audit | `audit hole-shift` |

Not carried over, because they were experiment bookkeeping rather than method: hard-coded
episode/frame-count assertions, the 10-correction early-stop loop's fixed seed arithmetic, and
the per-round failure subsets used to pick traces (filter the eval report instead).

## Flow transformer

Code: [`humanoid_il/flow_transformer`](../../src/robot_learning/humanoid_il/flow_transformer/README.md).
This is the architecture the user proposed, trained from scratch:

- ResNet-18 spatial tokens from both cameras, a state token and a flow-time token
- a 6-layer noncausal action decoder
- linear flow matching, 10 Euler steps, EMA weights
- 50-action chunks executed open-loop at 25 Hz, the same as the best ACT setting

It reuses the ACT datasets in place. Its corrections are the ones collected from ACT's failures,
because flow has not yet had a correction round of its own.

| Stage (job) | Training episodes | Updates | fresh /100 | exposed /100 | nominal /20 | interpolation |
|---|---|---|---|---|---|---|
| half (710016) | 291 of 582, stratified | 50k | 19 | 23 | 2 | — |
| half (710016) | 291 | 100k (cosine) | 39 | 33 | 7 | — |
| continue (710105) | 291 | 125k (LR 1e-5) | 47 | 29 | 13 | 31/100 |
| continue (710105) | 291 | 150k | 34 | 36 | 12 | — |
| spatial350 (710126), from 125k | 350 farthest-point curated, 16 cells reserved | 175k | 55 | 60 | 15 | 48/100 |
| paired400 (711188), from 175k | 377 + 23 paired-hole corrections | **225k** | **56** | **68** | **15** | 52/83 clean cells |

Interpolation = layouts jittered ≤1 mm inside workspace cells that no training placement touched
(83 remain clean after the 377-episode expansion exposed 4 cells).

## Head-to-head on identical layouts

Same simulator, cameras, 25 Hz, 50-action execution, 60 s limit and success gate.

| | Flow 225k | ACT 347k | Fisher p |
|---|---|---|---|
| fresh, peg moved 0–3 / 3–7 cm | 15/20 · 21/30 | 13/20 · 17/30 | |
| fresh, hole moved 0–3 / 3–7 cm | 12/20 · 8/30 | 13/20 · 9/30 | |
| **fresh total** | **56/100** | **52/100** | 0.67 |
| exposed, peg moved 0–3 / 3–7 | 16/20 · 23/30 | 11/20 · 20/30 | |
| exposed, hole moved 0–3 / 3–7 | 12/20 · 17/30 | 8/20 · 3/30 | |
| **exposed total** | **68/100** | **42/100** | 0.0004 |
| **nominal ±1 cm** | **15/20** | **12/20** | 0.50 |
| **all 220** | **139 (63%)** | **106 (48%)** | **0.002** |
| grasped / lifted | 220 / 219 | 220 / 220 | |
| training episodes, updates | 400 unique, 225k × 8 | ~605 unique, 347k × 8 (+ earlier batch 4/16) | |
| chunk latency, RTX 3090 | 42 ms | 12 ms | |

The comparison is not architecture-only. The two policies had different budgets and data, and
flow's curriculum was branched from the best intermediate checkpoint. Every evaluation is one
seed per layout; N = 100 cannot resolve differences under ~13 points.

## Where flow stands

1. **Insertion is the only failure mode.** Grasp and lift succeed in 439/440 rollouts across both
   policies. Every failure is a held peg that never seats.
2. **The weak spot is the moved hole.** On clean interpolation cells, peg-moved layouts reach
   42/50, while hole-moved reach 10/33 (inner band 6/25). For fresh layouts, flow's advantage over
   ACT comes entirely from peg-moved layouts. Both are at ~40% when the hole moves.
3. **It sees the hole, but under-corrects.** In the frozen-state hole-shift audit (711283) the
   policy moves the peg the right way 100% of the time. Median gain is 0.85 for a 3 mm shift and
   0.92 for 10 mm, against the expert's 1.0–1.09. The 150k ACT reached 0.3–0.5. A 15% undershoot
   of a 3 mm error leaves ~0.45 mm, close to the 0.5 mm per-side clearance. This looks like
   regression toward the training-mean hole position, not blindness.
4. **Corrections transfer well, but they were ACT's.** On the exposed layouts the same correction
   data gave flow 68/100 and ACT 42/100. The hole-moved 3–7 cm cell went 17/30 vs 3/30. Flow fits
   correction data better, and has never been given corrections from its own failure states.
5. **Checkpoints are noisy.** With the same data, going from 125k to 150k moved fresh from 47 to
   34 (p ≈ 0.08). Picking the last checkpoint is a gamble.
6. **Chunk length was never tuned for flow.** The 50-action open-loop execution was inherited
   from ACT. Flow's chunks are stochastic samples, so naive replanning switches modes. Shorter
   horizons need prefix-consistent sampling.

## How to make flow clearly better than ACT, cheapest first

| # | Change | Cost | Why |
|---|---|---|---|
| 1 | **Flow DAgger round from its own failures.** `eval --trace-dir` on the 225k fresh and exposed failures (76 episodes), then `collect_corrections --source traces`. Branch 225k → +25k at LR 1e-5 with phase quotas 3/2/2/1. Re-evaluate the same 220 layouts plus interpolation. | ~3 h collection (CPU sim) + ~4 h train + ~3 h eval, one 3090 | Policy-state corrections produced ACT's two largest jumps (×2.5 fresh in the moved-hole round). Flow already fits corrections better (finding 4). |
| 2 | **More paired-hole corrections.** `collect_corrections --source tasks --tasks paired` with more pairs or new seeds over hole cells. | expert-only, ~1 h CPU per 50 pairs | The 23 pairs moved clean hole-moved interpolation from 3 → 6/25. Pairs (same robot state, different hole) target the undershoot directly. |
| 3 | **Inference-only sweep.** `--n-action-steps` 50/25/10 × `--refinement-steps` 10/20 on the 50 hole-moved fresh layouts. Then real-time chunking (guided inpainting of the next chunk onto the executing prefix, [Black et al. 2025](https://arxiv.org/abs/2506.07339); `humanoid_robot_learning/rtc_driver.py` already does it for LeRobot policies) if shorter horizons jitter. | no training, ~1 h GPU per cell | Closed-loop correction during the last 6 cm of descent; latency (42 ms) allows replanning every 10 steps. |
| 4 | **Checkpoint selection on a 40-layout dev split** of fresh layouts (every 5k), with the full 220 run only on the chosen one. | ~30 min per checkpoint | Removes the ±13-point checkpoint lottery (finding 5). |
| 5 | Only if 1–3 plateau: **finer wrist features.** A stride-16 layer3 map or a higher-resolution wrist crop, plus random-shift augmentation. | full retrain | 1 mm ≈ a few wrist-view pixels, and stride-32 tokens are coarse. |

Steps 1 and 2 are a single `dagger --policy-type flow` run plus one task collection. Concrete
command file for `tools/slurm/peg_insert_run.sbatch`:

```bash
F=$PERSIST/flow/paired400_711188/run; W=$PERSIST/workspace/independent_708647/run
CK=$F/training/checkpoints/225000.pt
for set in fresh:$PERSIST/corrections/moved_hole_709670/run/fresh_layouts.json exposed:$W/heldout_layouts.json; do
  $PY -m humanoid_mujoco_scenes.peg_insert.eval --checkpoint $CK --device cuda --layouts ${set#*:} \
    --report $RUN/eval_225k_${set%%:*}.json --trace-dir $RUN/traces_${set%%:*}
  $PY -m humanoid_mujoco_scenes.peg_insert.collect_corrections --source traces \
    --eval-report $RUN/eval_225k_${set%%:*}.json --trace-dir $RUN/traces_${set%%:*} --output $RUN/corr_${set%%:*}
  $PY -m humanoid_robot_learning.raw_to_lerobot --raw $RUN/corr_${set%%:*} --root $RUN/corr_${set%%:*}_ds
done
$PY -m humanoid_mujoco_scenes.peg_insert.curate add --plan $F/selection_plan.json \
  --dataset $RUN/corr_fresh_ds --dataset $RUN/corr_exposed_ds --output $RUN/selection.json
$PY -m humanoid_il.flow_transformer.train --root $RUN/training $(python3 -c "import json;print(' '.join('--dataset '+r for r in json.load(open('$RUN/selection.json'))['roots']))") \
  --selection $RUN/selection.json --phase-config $REPO/src/robot_learning/config/peg_insert_phases.json \
  --resume-from $CK --schedule constant --lr 1e-5 --vision-lr 1e-6 --steps 250000 --save-freq 5000
```

Note that the fresh failures become training data in step 1. Fresh results after that round
are *exposed*, so judge it on the interpolation cells and a newly sampled fresh set
(`workspace sample --split any --seed <new>`).

## Flow scripts → current tools

| Old | Now |
|---|---|
| `flow_policy`, `flow_data`, `train_flow`, `continue_flow` | `humanoid_il.flow_transformer.{policy,data,train}` (`--resume-from --schedule constant` for continuations) |
| `eval_flow`, `eval_flow_checkpoint`, `flow_round` | `peg_insert.eval` (any checkpoint; per-group summary in the report) |
| half selection, `spatial_dataset`, `expand_hole_dataset` | `peg_insert.curate fraction / spatial / add` |
| `spatial_flow_round` interpolation layouts | `workspace sample --cells-from <plan> --jitter-mm 1` |
| `paired_hole_round` | `collect_corrections --source tasks --tasks paired --map ... --selection ...` (reproduces the 23 original tasks exactly) |
| `audit_flow_hole`, `benchmark_chunks` | `peg_insert.audit hole-shift / latency` |

The rewritten `FlowPolicy` has the same parameter names and outputs as the old one (max
difference 0.0), and checkpoint payloads keep their keys, so every NAS `.pt` loads and can be
branched from.
