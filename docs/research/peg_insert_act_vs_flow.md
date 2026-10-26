# Peg insertion in MuJoCo: ACT with expert corrections

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

In round 3, the 40-correction cap filled in file order: nominal failures first, then holes moved 0–3
cm. So no correction came from the hardest case, a hole moved 3–7 cm (flow 26 nominal + 14 hole
0–3 cm; ACT 23 + 17).

**Round 4: corrections balanced across groups.** `collect_corrections --max-corrections 40
--group-quota hole_3_7=20` took 20 corrections from holes moved 3–7 cm and spread the other 20
round-robin across nominal, hole 0–3 cm and both peg groups. These came from each policy's own
round-3 failures. Training continued 15k updates as before (flow 265k → 280k, ACT 387k → 402k),
about 2.3 h per job. Both were evaluated on new held-out layouts, identical for the two: nominal
seeds from 19000, and 99 new jitters.

| Held-out (new layouts) | Flow 280k | ACT 402k | p (flow vs ACT) |
|---|---|---|---|
| nominal | **17/20** | 3/20 | |
| fresh: hole moved 0–3 / 3–7 cm | 16/20 · **16/30** | 1/20 · 2/30 | |
| fresh: peg moved 0–3 / 3–7 cm | 14/20 · 16/29 | 7/20 · 18/29 | |
| **fresh total** | **62/99** | 28/99 | < 0.001 |
| all held-out | **79/119 (66%)** | 31/119 (26%) | < 0.001 |
| lifted (all held-out) | 119/119 | 119/119 | |

**Flow** improved on every group from round 3 to round 4:

- fresh 44 → 62/99 (p = 0.015)
- hole moved 3–7 cm 7 → 16/30 (p = 0.03)
- nominal 10 → 17/20 (p = 0.04)

Targeting the hardest group helped more than the round-3 mix, and the other groups held up too.

**ACT** did not improve with the same data budget: fresh 37 → 28/99 and nominal 6 → 3/20 (both
within noise, p = 0.23 and 0.45). Holes moved 3–7 cm stayed at 2/30, and only peg moves far
remain strong (18/29).

In the workcell, flow now inserts in two thirds of held-out episodes, against a quarter for ACT.
That's the clearest flow-over-ACT result in this study.

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
