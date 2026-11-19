# Loop closure

`slam_core/slam_core/loop_closure/` is the loop closure module. ROS-free, numpy-only (OpenCV only inside the PnP
verifier). Status: working and tested on a synthetic world; **not yet validated on real D455 or TUM data**.

## See it work (no camera, no GPU)

```bash
pip install numpy scipy matplotlib
python slam_tools/loop_closure_demo.py                  # writes loop_closure_demo.png
python slam_tools/loop_closure_demo.py --drift heavy --show
cd slam_core && pip install -e '.[test]' && pytest tests -q     # 20 tests; the iSAM2 ones need gtsam
```

The demo walks a robot around a building twice plus two side hallways that are identical copies of each other
(36 m apart). Odometry drifts. The same pose graph optimizer runs after three detectors:

| | detector | what it is |
|---|---|---|
| A | prototype | what `vo.py` did before: old keyframes within 1 m of the drifted position, first that verifies wins |
| B | appearance only | place recognition + verification, no false-loop defenses ("just plug in DBoW2") |
| C | full module | everything below |

Typical result (581 keyframes, seed 0, raw odometry error 5.5 m RMS, moderate drift; heavy drift in brackets):

| detector | true loops | false loops | error after |
|---|---|---|---|
| A prototype | 29 (19) | 0 | 0.27 m (0.72 m) |
| B appearance only | 40 (40) | **4** | **8.98 m** (15.1 m) |
| C full module | 36 (33) | 0 | 0.28 m (0.32 m) |

Read it honestly: in this world the old radius search is already decent (it finds fewer loops and degrades faster
as drift grows). The clear win is B vs C: once loop candidates come from appearance, the look-alike hallways get
accepted and wreck the map unless the defenses are there. The numbers come from a simulator, not real images.

## Pipeline (`LoopClosureDetector.process(keyframe, global_descriptor)`)

1. **Retrieve** by appearance, not by the drifted position (`retrieval.py`): a bag of visual words over the local
   descriptors (XFeat), the DBoW2 idea (tf-idf weighted word histograms) written in numpy with a vocabulary trained on our
   own recordings, brute-force cosine search. VLAD is available as an alternative. Keyframes within `min_hops` odometry
   edges are excluded (graph distance, not a time rule). Why not the DBoW2 library: its stock vocabulary is for binary ORB
   descriptors, XFeat's are 64-d floats, and its Python bindings need a source build.
2. **Verify** every candidate, keep the best by inliers (`verification.py`): features + PnP/RANSAC against the old
   keyframe's depth, `min_inliers`. The verifier is passed in, so the logic is testable without images.
3. **Gate** against odometry (`gate.py`): Mahalanobis distance between the loop and odometry's estimate, with
   covariance propagated along the keyframe tree. Rejects loops that disagree by far more than drift allows.
4. **Confirm** over consecutive keyframes (ORB-SLAM3-style): `confirm_count` keyframes in a row must produce
   agreeing loops.
5. **Pairwise consistency** (`consistency.py`, the idea behind PCM / Kimera-RPGO): loops that disagree with the
   largest mutually consistent group are dropped. A later loop can retract an earlier one.
6. **Output** `LoopUpdate`: loops to add, loop ids to retract. Loop covariance shrinks with the inlier count.

## Using it

- `prototype/vo.py --loop-mode module` (default is still `radius`, the old behaviour). The vocabulary (500 words,
  `--loop-encoder bow`, or `vlad`) is trained on the first `--loop-vocab-kfs` keyframes, or loaded with `--loop-vocab
  loop_vocab.npz`; each run writes `loop_vocab.npz`, so train it once on a varied recording and reuse it.
- `prototype/pose_graph.py` now scales the loop sigmas by `sqrt(100 / inliers)`.
- `slam_core/backend/incremental.py` is the iSAM2 backend (needs gtsam). Feed it each keyframe with `add_keyframe`, and each
  detector `LoopUpdate` with `apply`. Loop factors are remembered by `LoopConstraint.id`, so retracted loops are removed
  (RTAB-Map does the same). After a loop goes in, its residual is checked against its own covariance and a loop the rest of
  the graph cannot live with is taken back out (RTAB-Map's optimization error ratio idea).
  Try it: `sim.run(scenario, detector, backend="isam2")`.

## Parameters (`LoopParams`) and how the defaults were chosen

Defaults: `min_inliers` 30, `confirm_count` 2, `top_k` 3, `min_score` 0.3, `min_hops` 20, `min_path` 2.0 m, `cooldown` 10, gate and pairwise
consistency on. In the simulators (which only see ~25 landmarks) `min_inliers` is 10.

They come from a sweep of the whole detector (gate, confirmation and consistency included) with the real XFeat verifier
on TUM sequences (`slam_tools/eval_detector.py`, scored against ground truth: coverage = fraction of revisit keyframes that
have a correct loop within 5 keyframes; verifications = how many candidate pairs get the expensive matching, the main cost):

| setting | coverage fr1/room, fr2/desk, fr3/office | verifications (same order) |
|---|---|---|
| old: min_inliers 40, confirm 3, top_k 5, min_score 0.2 | 0.78 / 0.95 / 0.88 | 584 / 496 / 538 |
| **new: min_inliers 30, confirm 2, top_k 3, min_score 0.3** | **0.82 / 1.00 / 1.00** | **327 / 271 / 300** (-44%) |
| top_k 2 (even cheaper) | loses a loop on fr3 (coverage 0.72) | |

Final check on all seven TUM sequences, with the corrected scoring (a revisit needs 5 m of real travel and a view within 1 m
and 40 deg of an earlier one; coverage = revisit keyframes with a correct loop within 5 keyframes). fr1/desk, fr1/desk2 and fr1/360
were not used for tuning. old = min_inliers 40, confirm 3, top_k 5, min_score 0.2; new = the defaults:

| sequence | keyframes | revisit kfs | coverage old / new | loops (correct) old / new | verifications old / new |
|---|---|---|---|---|---|
| fr1/room | 148 | 49 | 0.78 / 0.82 | 4 (4) / 4 (4) | 584 / 327 |
| fr2/desk | 129 | 19 | 0.95 / 1.00 | 2 (2) / 2 (2) | 496 / 271 |
| fr3/long_office | 141 | 25 | 0.88 / 1.00 | 3 (3) / 4 (4) | 538 / 300 |
| fr1/desk | 62 | 20 | 0.60 / 0.95 | 2 (2) / 3 (3) | 162 / 82 |
| fr1/desk2 | 79 | 30 | 0.93 / 0.97 | 4 (4) / 4 (4) | 235 / 119 |
| fr1/360 (spins in place) | 100 | 23 | 0.83 / 0.87 | 2 (2) / 3 (2) | 363 / 188 |
| fr2/large_with_loop | 263 | 0 | no scoreable revisit | 1 (1) / 1 (1) | 1153 / 674 |

Over the six scored sequences: mean coverage 0.83 -> 0.94, correct loops 17 -> 19, verifications 2378 -> 1287 (-46%). The one
"incorrect" loop (fr1/360, KF 70 -> KF 0) is not a wrong place: those keyframes are really 25 cm apart, but it had only 33 inliers
and its pose is off by 7.7 cm and 5.1 deg (the cut-off for "correct" is 10 cm and 5 deg). That is the price of `min_inliers`
30 over 40: slightly less accurate weakest loops (their sigmas are inflated for it: sqrt(100 / inliers)). In the simulators
(8 runs, two drift levels) the new setting finds more true loops (309 vs 276) and still no false ones.

fr2/large_with_loop: an earlier version of the scoring counted ~190 "revisits" there that were really the camera standing
still at the start (keyframes kept being created with no motion). The detector now excludes candidates closer than `min_path`
(2 m) along the odometry path, and the scoring needs 5 m of true travel. Views of a far scene with no depth (beyond the
sensor range) cannot be verified with PnP at all; a 2D-2D fallback (essential matrix, up to scale) would be needed for them.

Caveat: TUM has no look-alike places, so these sweeps cannot measure false-loop risk (the gate rejected nothing); that is
tested in the simulators. All of it should be re-swept on D455 recordings with labelled revisits.

## Verified vs not

- Verified on the simulator: the 20 tests (Linux, GTSAM 4.3.0, numpy 2.5) including detector + iSAM2 backend, and the real
  `prototype/pose_graph.py` on simulator output (graph error 5132 -> 10.5, start-to-end gap 18.9 m -> 1.2 m).
- **Verified on real images: three TUM sequences** (real XFeat + LighterGlue features, PnP, real GTSAM, every 2nd frame,
  CPU only, one run each). Reproduce: `slam_tools/run_tum_loop.py DATASET --loop-mode none|radius|module --out DIR`
  and `slam_tools/eval_loops_pr.py DATASET` (needs torch, kornia, tqdm, opencv, gtsam; XFeat via torch.hub).

  Whole system, trajectory error (ATE RMSE):

  | sequence | VO only | old radius search | new module |
  |---|---|---|---|
  | fr1/room | 0.319 m | 0.125 m (2 loops) | 0.112 m (4 loops) |
  | fr2/desk | 0.089 m | 0.089 m (2 loops) | 0.092 m (2 loops) |
  | fr3/long_office | 0.124 m | 0.030 m (4 loops) | 0.030 m (3 loops) |

  Loop closure on its own (`eval_loops_pr.py`: retrieval + verification vs ground truth; a revisit = a keyframe at least
  20 odometry edges after another one within 1 m and 40 deg; correct = within 10 cm and 5 deg of the true relative pose):

  | sequence | keyframes with a revisit | retrieval recall@5 | precision at >=40 inliers | recall at >=40 inliers |
  |---|---|---|---|---|
  | fr1/room | 49 / 148 | 0.80 | 1.00 (33/33) | 0.67 |
  | fr2/desk | 19 / 129 | 0.89 | 1.00 (15/15) | 0.79 |
  | fr3/long_office | 25 / 141 | 0.88 | 1.00 (24/24) | 0.84 |

  Retrieval and verification experiments (same three sequences, `eval_loops_pr.py`, cached VO keyframes; retrieval recall@5 /
  recall of verified loops at 100% precision; fr1/room, fr2/desk, fr3/long_office):

  | variant | retrieval recall@5 | verified recall at 100% precision | decision |
  |---|---|---|---|
  | VLAD (64 clusters), baseline | 0.80 / 0.89 / 0.88 | 0.67 / 0.79 / 0.84 | alternative |
  | **bag of words, 500 words** | **0.84 / 0.95 / 0.96** | 0.69 / 0.79 / 0.84 | **default** |
  | bag of words, 1000 words | 0.86 / 0.95 / 0.88 | 0.69 / 0.79 / 0.88 | same within noise, bigger |
  | group re-ranking (window 3 / 6) | 0.80 / 0.89 / 0.80 (w3) | 0.61 / 0.79 / 0.84 (w3), 0.59 (room, w6) | off: no gain, worse on fr1 |
  | guided matching (second PnP pass) | 0.80 / 0.89 / 0.88 | 0.67 / 0.79 / 0.84 (identical to baseline) | off: no measurable effect |

  Precision was 100% in every variant. Samples are small (19-49 revisit keyframes), so one or two loops are noise; the
  retrieval recall gain of bag of words (+4, +6, +8 points) is the clearest signal. Guided matching (`loop_guided=1`) and group
  re-ranking (`LoopParams.group_window`) stay in the code, off by default, because they did not help on this data.

  Ceiling for any loop closure (`slam_tools/oracle_loops.py`: the same pose graph fed perfect loops built from the ground
  truth, no images): fr1/room 0.319 -> 0.116 m, fr2/desk 0.089 -> 0.089 m, fr3/long_office 0.124 -> 0.036 m. Our detector
  (old defaults) got 0.112 / 0.092 / 0.030 m, i.e. at the ceiling (it can even go slightly under it, since the oracle is not
  a strict bound). So on these sequences better loop detection cannot lower the whole-system error further: what is left is
  the visual odometry's local noise. The detector improvements show up in coverage, cost and false-loop protection instead.

  Reading it honestly:
  - Precision was 100% on all three (all 72 accepted verified loops correct), and the earlier fr1/room check of the accepted
    loops against the ground truth also found no false loops. Recall at 100% precision is 67-84%; retrieval misses 11-20% of
    revisits.
  - End to end, the module is **not measurably better than the old radius search on these sequences** (differences of 1-13 mm,
    within what one run can show; slightly worse on fr2/desk). These sequences only revisit nearby, with little drift, which
    is the case the radius search handles. Loop closure itself helps a lot where drift is large (fr1/room 2.8x, fr3 4.1x)
    and not at all on fr2/desk, whose VO error is already small.
  - Whole-system accuracy is well behind ORB-SLAM2 (published roughly 5 cm / 1 cm / 1 cm on these three, protocols differ):
    the gap is the VO frontend (no local map or bundle adjustment, every 2nd frame), not the loop closure.
  - The precision/recall sweep covers retrieval + verification only, not the gate, confirmation and consistency stages;
    sample sizes are small (19-49 revisit keyframes per sequence); one run each.
- `vo.py --loop-mode module` works end to end on real data with the default `LoopParams`.

## Not done / known limits

- Real-data validation and tuning of thresholds (see above).
- The gate can reject true loops under heavy unmodelled drift (it did reject many candidates in the heavy-drift runs); the
  iSAM2 backend's Huber loss treats a loop that is far from odometry as an outlier, so large true corrections come in
  gradually.
- Identical places that are also geometrically identical and *close in odometry terms* cannot be told apart by
  anything here.
- A learned global descriptor (NetVLAD / DINOv2-based) may beat bag of words; check licences first (SALAD is GPL-3.0, the
  DINOv2 weights' licence is unclear). Gait blur: only let sharp keyframes be loop candidates.
- The arm-masking and keyframe stability gating for the humanoid are separate work.
