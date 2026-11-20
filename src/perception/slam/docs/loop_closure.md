# Loop closure

`slam_core/slam_core/loop_closure/` detects when the robot is back somewhere it has been, so the pose graph can remove the
drift that visual odometry builds up. It is ROS-free and numpy-only (OpenCV only inside `PnPVerifier`), with an incremental
GTSAM backend in `slam_core/slam_core/backend/incremental.py`.

Status: works end to end on real images (7 TUM RGB-D sequences, real XFeat features, real GTSAM) and in two simulators
built to provoke false loops. **Not yet tested on D455 recordings or on a GPU.** In `prototype/vo.py` it is opt-in
(`--loop-mode module`); the default is still the old 1 m radius search.

## Try it

```bash
# no camera, no GPU: a synthetic building with two identical hallways 36 m apart (writes loop_closure_demo.png)
pip install numpy scipy matplotlib
python slam_tools/loop_closure_demo.py              # add --drift heavy, --show

# unit tests (the GTSAM ones are skipped where gtsam is not installed)
cd slam_core && pip install -e '.[test]' && pytest tests -q

# real data: a TUM RGB-D sequence (needs torch, kornia, tqdm, opencv; gtsam for the pose graph)
python slam_tools/run_tum_loop.py DATASET --loop-mode module --out runs/room       # VO + loops + pose graph + error
python slam_tools/eval_loops_pr.py DATASET --cache room.pt                         # retrieval + verification vs truth
python slam_tools/eval_detector.py DATASET --cache room.pt                         # whole detector vs truth, settings sweep
python slam_tools/oracle_loops.py DATASET runs/room --out runs/room_oracle          # ceiling: perfect loops
python slam_tools/bench_matching.py DATASET                                        # matching time, GPU vs CPU
python slam_tools/compare_loops.py DATASET --ours runs/room --rtabmap rtabmap.db    # same scoring for RTAB-Map's loops
```

## How it works

`LoopClosureDetector.process(keyframe, global_descriptor)` runs once per new keyframe and returns a `LoopUpdate`:
loops to add, and ids of earlier loops to take back out.

1. **Retrieve** old keyframes that look like the new one: a bag of visual words over the XFeat descriptors (the DBoW2
   idea, tf-idf weighted, written in numpy with a vocabulary trained on our own data; VLAD is available too). Not the
   DBoW2 library: its stock vocabulary is for binary ORB descriptors (XFeat's are 64-d floats) and its Python bindings
   need a source build. Recent keyframes are excluded: fewer than `min_hops` odometry edges or `min_path` metres travelled.
2. **Verify** every candidate geometrically (feature matching + PnP RANSAC against the old keyframe's depth) and keep
   the best by inlier count. The verifier is passed in, so the detector logic is testable without images.
3. **Gate** against odometry: a loop that disagrees with odometry's estimate by far more than the accumulated drift
   allows (Mahalanobis distance, covariance propagated along the keyframe tree) is rejected.
4. **Confirm**: `confirm_count` consecutive keyframes must find agreeing loops (ORB-SLAM3's local consistency idea).
5. **Pairwise consistency** (the PCM / Kimera-RPGO idea): loops must agree with each other through the odometry
   between them; the largest mutually consistent group is kept, so a later loop can retract an earlier one.
6. **Output** with a covariance that grows as the inlier count falls (sigma_ref * sqrt(100 / inliers)).

The iSAM2 backend adds odometry edges as keyframes arrive and applies each `LoopUpdate`: loop factors are remembered by
id so retracted loops are removed (`removeFactorIndices`, the RTAB-Map approach), loop edges get a Huber loss, and a loop
whose residual the rest of the graph cannot accept after optimizing is taken back out (RTAB-Map's error-ratio check).

The design follows ORB-SLAM3, VINS-Fusion, RTAB-Map, Kimera-RPGO and cuVSLAM (read for the ideas; no code copied).

## Using it

- `prototype/vo.py --loop-mode module`: the bag-of-words vocabulary is trained on the first `--loop-vocab-kfs` keyframes,
  or loaded with `--loop-vocab loop_vocab.npz` (every run writes one, so train it once on a varied recording and reuse it).
  Loops go to `loops.txt` as before.
- `prototype/pose_graph.py` now scales each loop's sigmas by its inlier count.
- A live backend would call `IncrementalBackend.add_keyframe(kf)` for each keyframe and `apply(update)` for each
  `LoopUpdate`; `apply` returns the ids of loops it rejected after optimizing.

## Parameters (`LoopParams`)

| parameter | default | meaning |
|---|---|---|
| `min_inliers` | 30 | inliers a verified loop needs (10 in the simulators, which see ~25 landmarks) |
| `top_k` | 3 | candidates verified per keyframe (each one costs a feature matching) |
| `min_score` | 0.3 | retrieval similarity floor |
| `confirm_count` | 2 | consecutive keyframes that must agree (1 = off) |
| `min_hops` / `min_path` | 20 / 2.0 m | a candidate must be this far back along the odometry (so standing still makes no loops) |
| `cooldown` | 10 | keyframes to wait after a loop |
| `gate`, `gate_inflation` | on, 1.0 | the odometry plausibility gate; raise the inflation if true loops get gated out |
| `pcm` | on | pairwise consistency |

How they were chosen: the whole detector was swept with the real verifier on fr1/room, fr2/desk and fr3/long_office
(`eval_detector.py`), then checked on four sequences not used for tuning and in the simulators. The old values were
`min_inliers` 40, `confirm_count` 3, `top_k` 5, `min_score` 0.2. Lowering `top_k` to 2 started losing loops. These are
TUM values, not D455 values: re-sweep on labelled D455 recordings.

## Results

### Simulators (identical look-alike hallways, drifting odometry)

581 keyframes, raw odometry error 5.5 m (moderate drift) / 12.2 m (heavy). Same pose graph after each detector:

| detector | moderate: true / false loops, error | heavy: true / false loops, error |
|---|---|---|
| A: old prototype (1 m radius search) | 29 / 0, 0.27 m | 19 / 0, 0.72 m |
| B: appearance only, no defenses ("just plug in DBoW2") | 40 / **4**, **9.38 m** | 40 / **4**, **16.14 m** |
| C: this module | 39 / 0, 0.24 m | 37 / 0, 0.38 m |

Without the defenses the look-alike hallways get accepted and wreck the map. Over 8 runs (2 drift levels x 4 seeds) the
new defaults found 309 true loops and no false ones (old defaults: 276 / 0). The 3D simulator (6-DoF, climbing path, drift
in all axes, real iSAM2 backend) gives the same picture; there, the backend's own check also rejects the false loops.

### Real data: the detector on its own (7 TUM sequences, scored against ground truth)

A *revisit* is a keyframe within 1 m and 40 deg of an earlier one after at least 5 m of true travel; *coverage* is the
fraction of revisit keyframes with a correct loop (within 10 cm and 5 deg of the true relative pose) within 5 keyframes;
*verifications* is how many candidate pairs got the expensive matching. fr1/desk, fr1/desk2, fr1/360 and
fr2/large_with_loop were not used for tuning.

| sequence | keyframes | revisit kfs | coverage old / new | loops (correct) old / new | verifications old / new |
|---|---|---|---|---|---|
| fr1/room | 148 | 49 | 0.78 / 0.82 | 4 (4) / 4 (4) | 584 / 327 |
| fr2/desk | 129 | 19 | 0.95 / 1.00 | 2 (2) / 2 (2) | 496 / 271 |
| fr3/long_office | 141 | 25 | 0.88 / 1.00 | 3 (3) / 4 (4) | 538 / 300 |
| fr1/desk | 62 | 20 | 0.60 / 0.95 | 2 (2) / 3 (3) | 162 / 82 |
| fr1/desk2 | 79 | 30 | 0.93 / 0.97 | 4 (4) / 4 (4) | 235 / 119 |
| fr1/360 | 100 | 23 | 0.83 / 0.87 | 2 (2) / 3 (2) | 363 / 188 |
| fr2/large_with_loop | 263 | 0 | no scoreable revisit | 1 (1) / 1 (1) | 1153 / 674 |

Six scored sequences: mean coverage 0.83 -> 0.94, correct loops 17 -> 19, verifications -46%. No loop was at a wrong
place; the one "incorrect" loop is a weak one (see failure cases). Retrieval alone (`eval_loops_pr.py`, fr1/room, fr2/desk,
fr3/long_office) puts a true revisit among its top 5 candidates 84 / 95 / 96% of the time with bag of words (80 / 89 / 88%
with VLAD). Two other ideas were measured and dropped because they did not help: re-ranking candidates by neighbouring
keyframes' scores, and a second guided-matching pass in verification.

### Real data: whole-system trajectory error (ATE RMSE, every 2nd frame, CPU)

| sequence | VO only | old radius search | this module | ceiling: perfect loops |
|---|---|---|---|---|
| fr1/room | 0.319 m | 0.125 m | 0.115 m | 0.116 m |
| fr2/desk | 0.089 m | 0.089 m | 0.088 m | 0.089 m |
| fr3/long_office | 0.124 m | 0.030 m | 0.031 m | 0.036 m |
| fr1/desk | 0.038 m | 0.031 m | 0.026 m | |
| fr1/desk2 | 0.155 m | 0.044 m | 0.040 m | |
| fr1/360 | 0.185 m | 0.185 m (no loops found) | 0.122 m | |

The module is never worse than the old search and clearly better where the old search finds nothing (fr1/360). (The old-search runs on fr1/desk, fr1/desk2 and fr1/360 used 30 inliers instead of its default 40, which can only have helped it.) The
ceiling column (`oracle_loops.py`: the same pose graph fed perfect loops from the ground truth) shows the module is already
at what any loop detector could reach on these sequences: what is left is the visual odometry's local noise, not loop
closure. Every accepted loop on fr1/room was also checked by hand against the ground truth (1.1-4.9 cm, 0.4-2.7 deg).
fr2/large_with_loop was not finished (the run stopped when the laptop slept).

### Compared with RTAB-Map

RTAB-Map 0.23.14 (default settings, `rtabmap-rgbd_dataset`) and this module on the same sequences, scored by
`slam_tools/compare_loops.py` with the same ground-truth rules for both. The systems make keyframes differently (RTAB-Map adds a
map node once per second), so scoring is by time: a loop is correct within 10 cm and 5 deg of the true relative pose; the
true trajectory is sampled every 0.5 s and a sample is a revisit if it is within 1 m and 40 deg of a point at least 5 m of
travel earlier; coverage = revisit samples with a correct loop within 2 s. RTAB-Map's transforms are in its base frame, so one
frame/direction convention is fitted for all its loops per sequence (the one that fits the ground truth best).

| sequence | ours: loops (correct), median error, coverage | RTAB-Map: loops (correct), median error, coverage | trajectory error ours / RTAB-Map |
|---|---|---|---|
| fr1/room | 4 (4), 1.6 cm, **0.78** | 6 (6), 1.9 cm, 0.62 | 0.115 / **0.058** m |
| fr2/desk | 2 (2), 2.8 cm, 0.52 | 16 (16), 2.2 cm, **0.77** | **0.088** / 0.128 m |
| fr3/long_office | 4 (4), 1.9 cm, 0.64 | 26 (26), 1.8 cm, **0.86** | 0.031 / **0.017** m |
| fr1/desk | 3 (3), 2.4 cm, **0.80** | 6 (6), 1.7 cm, 0.73 | 0.026 / 0.028 m |
| fr1/desk2 | 4 (4), 3.4 cm, **1.00** | 6 (6), 1.2 cm, 0.88 | 0.040 / 0.043 m |
| fr1/360 | 3 (2), 3.1 cm, **0.76** | 0 loops, 0.00 | **0.122** / 0.307 m |
| fr2/large_with_loop | 1 (1), 9.1 cm, no revisits to score | 4 (0 within 10 cm / 5 deg; median 16 cm, none >1 m off) | 0.534 / 0.424 m |

- Precision: the same. Neither system put a loop at a wrong place; our only miss is the weak fr1/360 loop.
- Coverage: about even. Ours is higher on 4 sequences, RTAB-Map's on 2; mean 0.75 vs 0.64 (0.75 vs 0.77 without fr1/360,
  where RTAB-Map found no loops).
- RTAB-Map adds many more loops on the longer sequences (mostly by appearance: 14 of 16 on fr2/desk, 20 of 26 on fr3); ours is
  probably held back by `cooldown` (10 keyframes after each loop), which is worth revisiting.
- RTAB-Map's loop poses are slightly more accurate (it refines each loop more).
- The trajectory error mostly reflects odometry (RTAB-Map's frame-to-map odometry with local bundle adjustment is better than
  `prototype/vo.py`), so it is not a loop closure comparison. On fr2/large_with_loop our VO lost tracking on 1030 of 2591
  frames (a large hall with little valid depth), and RTAB-Map's 4 loops all miss the 10 cm / 5 deg cut-off; whether that is
  real inaccuracy or a frame-convention artefact of the scoring was not resolved, so treat that row as inconclusive.
- One run each, RTAB-Map at default settings, small sequences.

### Cost

Feature matching is almost all of it. On a 20-thread laptop CPU (with other jobs running): XFeat detection about 140 ms
per image, LighterGlue matching about 145 ms, one candidate verification about 150-170 ms. The new defaults average about
2 verifications per keyframe, so about 0.3 s of loop closure work per keyframe on CPU, which should run in a background
thread. **GPU speed is not measured yet**: run `slam_tools/bench_matching.py` on a CUDA machine (and again with
`CUDA_VISIBLE_DEVICES=` to force the CPU) to get it.

## Failure cases (cause and fix)

| failure | cause | what happens | fix / status |
|---|---|---|---|
| False loop between look-alike places | appearance and local geometry both match (perceptual aliasing) | without defenses: 4 false loops, map error 5.5 -> 9.4 m (simulator) | odometry gate, confirmation, pairwise consistency, backend error check. Tested in the simulators only: TUM has no look-alike places |
| "Loops" while standing still | VO keeps making keyframes with no motion, so a hop count says the robot travelled | the robot loop-closes to where it already is (found on fr2/large_with_loop) | fixed: `min_path` (2 m along the odometry path); unit-tested |
| A weak loop is slightly off | `min_inliers` 30 accepts loops with few matches | fr1/360: KF 70 -> 0, right place, 33 inliers, off by 7.7 cm / 5.1 deg | accepted trade-off: its sigmas are inflated by the inlier count, and the loop edge has a Huber loss |
| Revisit not retrieved | viewpoint change between the two visits | 4-16% of revisits never reach verification | open: a learned global descriptor (check licences: SALAD is GPL-3.0, DINOv2 weights unclear) |
| View with no depth | far scene beyond the depth sensor's range | the PnP verifier cannot verify it at all | open: needs a 2D-2D (essential matrix) fallback |
| True loop gated out under heavy drift | odometry covariances smaller than the real drift | fewer loops (simulator, heavy drift: many candidates gated) | `gate_inflation`; tune on D455 data |

Tracking loss (the other half of the README's failure-case item) belongs to the VO frontend and is not covered here.

## Not done

- D455 recordings with labelled revisits (re-sweep the parameters there) and a real look-alike environment.
- GPU timing; running loop detection in a background thread; batching a keyframe's candidates in one matching call.
- `--loop-mode module` as the default in `vo.py`; a ROS node and `LoopClosure` message; keyframe database save/load;
  rebuilding the dense map after a loop.
- A 2D-2D verification fallback for views without depth.
- More loops on long sequences (RTAB-Map adds many more; try a shorter `cooldown`), and RTAB-Map at tuned settings / on D455.
