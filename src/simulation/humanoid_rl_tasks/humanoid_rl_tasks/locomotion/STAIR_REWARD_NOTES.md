# Onshape stair courses: reward and termination rationale

Why the straight and spiral course configs
(`config/pioneer_humanoid_v1/onshape_env_cfg.py`, `onshape_spiral_env_cfg.py`)
set what they set. The code carries a one-line summary per term; the
measurements and history live here. Numbers are from earlier runs (model 1999,
model_1350, `2026-10-20_*`, `2026-11-10`) and are context, not guarantees on the
current whole-body walker.

Reward weights are per-second rates (multiplied by `dt = 0.02`), so a term only
matters if its per-step contribution is a visible fraction of the progress
income (~0.03 per step on the straight course). Several terms were inert for
exactly that reason; the weights below were calibrated against it.

## Straight course

### Shove (`push_by_setting_velocity`)
Root velocity is perturbed every 8-12 s. Lateral is the interesting axis (a
biped has to catch itself without stepping wide); fore/aft is smaller so a bump
does not push the robot back down the flight. Training and Play share the
numbers. +/-0.5 m/s sideways is the shipping setting (83.5 % completion, 11.8 %
falls). +/-0.9 m/s was tried and was worse (62.6 % completion, 27.5 % falls,
action std 3.7 -> 4.9).

### Knee clip -1.10
The shared recipe clips the knee target to -0.95 rad while the joint allows
-1.134 rad. Forward kinematics (hip 0.788 m, thigh 0.262 m, shank 0.394 m) gives
a max sagittal foot lift of ~0.155 m at 0.95 rad but ~0.217 m at the hardware
limit. Course risers are 0.100-0.115 m on part 1 rising to ~0.21 m on parts 5-6,
so under -0.95 the tall risers are outside sagittal reach and the walker
cleared them by abducting the hip sideways. The clip is task-local; no knee
reward is added.

### Model-fix era (2026-10-19)
Several terms were written to compensate for asymmetries that were actually
defects in the robot description (right ankle-pitch axis not the mirror of the
left -> ~9.5 deg toe-down landings; right hip-abduction axis not the mirror ->
circumduction). With the axes fixed and the spawn re-centred the source of the
bias is gone. Those terms had only been validated by warm-starting an existing
walker: three from-scratch probes sat at reward ~-12 while the 2026-10-17 set
reached +24 by iteration 210. Hence `joint_bias_mirror`-style compensation terms
are weight 0, kept for a warm-started clean-up pass.

### `feet_impact` -0.5
Landing speed was inert at the earlier weight (~-0.5 per episode against
~+6500 forward-progress income on model 1999), so a foot could slam down for
free. -0.5 is 12x stronger. The "hard landing" report was mostly the right-foot
toe-down URDF defect.

### `foot_flat` -15
Measured on model 1999 (36 touchdowns): right foot 18.8 deg off horizontal on
average (53.9 max) vs left 2.9 (11.7 max). Re-measured on model_1350 (24 s, no
pushes): right 22.2 deg mean (p95 32.4) vs left 12.9 (p95 28.9), and the left
foot slid 0.62 m/s for 0.365 s per contact (max 1.29 s). Measured on the foot
*body*, not the ankle joint (joint angle did not predict landing tilt).

- The 0.15 s landing window is load-bearing: over a whole stance the foot
  legitimately pitches to ~40 deg (up to 67 deg at toe-off), which any
  whole-contact deadband would tax.
- Deadband 0.23 rad is calibrated to the leg that already lands correctly.
- Weight: the term fires on ~7 landing frames per step and is squared, mean
  ~0.0055 rad^2 per step. 0.0055 * 0.02 * weight against 0.032/step progress:
  -0.5 would be 0.2 % of income (inert); -15 puts a foot landing 9 deg outside
  the deadband at ~5 %. The spiral's -0.5 landing term is inert by the same
  arithmetic.
- Check `Episode_Reward/foot_flat`: it should be a few percent of progress
  income, not a rounding error.

### `feet_slide` -0.5
A foot translating 0.62 m/s while touching is dragging. At -0.1 the term logged
-0.0008 against +0.054 progress (inert). -0.5 makes planting cheaper than
skating without dominating the walk.

### Symmetry terms (weight 0)
A raw pairwise L/R joint term fights the anti-phase swing, so these compare slow
averages (see `mdp/rewards.py`): time-averaged mirrored joint angles, and
time-averaged stance duty. Measured bias on model_1350: hip 0.56 rad, knee
0.82 rad, duty 0.81 vs 0.18. Targets if enabled: knee bias < ~0.2 rad, duty
within ~0.15. Candidate weights: `joint_bias_mirror` -0.05 (~5 % of income),
`feet_duty_symmetry` -0.2 (~7 %).

### Goal distance 20.5
The course is a sawtooth ("Stair Wave (Up-Down Progressive)"), so the goal must
sit past the final descent to keep the summit drop in distribution. At 22.6 an
earlier setup ended the episode on the plateau before the drop, leaving every
term (including `feet_impact`) blind to the landing the visual eval exposed.
Part 6 ends at y = 19.3; 20.5 is just past it so completed episodes reset
promptly instead of walking ~6 m of flat.

### `hip_abduction_splay` -6, deadband 0.30
The right leg cleared risers by abducting ~0.9 rad with a near-straight knee
and dragged the foot ~0.5 m sideways while the left tucked properly. After the
knee-tuck fix the right hip came down to +0.39 rad but the term went inert:
deadband 0.45 sat above the residual. The deadband is calibrated to the leg the
policy already swings correctly (left hip mean +0.26 rad), so a correct gait is
free and one-sided splay pays.

### Knee tuck (`knee_swing_flexion` 0.28 / -4.0)
The inherited 0.35 had been relaxed to 0.22 while fixing a rear "butt-kick", but
with the tuck relaxed the right leg kept the knee straight and swung out
sideways. The rear lift is now bounded separately by `foot_swing_clearance`'s
forward phase gate and `foot_backswing_cost`. 0.40 rad plus the clearance peak
produced a high-knee march; 0.28 binds on the flat without marching, and risers
get their tuck from the clearance reward (paid up to band 0.20 m).

### `feet_air_time` 1.0 / 0.2 s
The inherited weight 4.0 made the visual eval report "jumping". Cutting weight,
not re-thresholding: the term pays `max(air - threshold, 0)`, so a higher
threshold makes the policy swing longer. Tried 2.5/0.28 (`2026-10-20_03-52-36`):
air-time income fell only 20 %, hip flexion rose and landing tilt went from
4.6/2.0 to 15.0/7.2 deg. Do not zero it or gate it on speed (that produced a
feet-glued stander).

### `termination_penalty` -800
-800 is a *warm-start* setting. A fresh random policy sees a -16 spike (the whole
iteration-0 return) and its action std runs away into flail-and-fall, which is
why from-scratch lineages used -200. Every run that climbed this course was
warm-started with -800, so a policy that stalls at a riser loses as much as one
that falls.

### Other
- `flat_orientation_l2` -3.0: continuous torso-tilt price, strong enough for a
  recovery gradient, slight next to the walk.
- `dof_acc_l2` -5e-7 on leg joints: smooth swings.
- `track_ang_vel_z_exp` 1.0, `action_rate_l2` -0.002: trimmed two taxes that
  dominated the landscape.
- Height-scan noise (`LidarNoiseCfg`): Gaussian + per-beam bias + dropout, ramped
  over ~500 iterations so a warm-started checkpoint adapts instead of meeting
  full corruption on its first rollout.

## Arm swing (`StairsStraightArmSwing`)

- **Action scale 0.5 on arms.** The inherited per-joint scale dict lists only
  leg joints, so arm actions fell back to 1.0, 4-16x any leg channel. With
  exploration std drifting to ~3 rad every arm action hit its clip and the
  shoulders ended pinned behind the torso.
- **Clip envelope** shoulder (-0.9, 0.45), elbow (-1.4, 0): a walking swing
  (51 deg forward / 26 deg back, no elbow hyperextension), not the joint's full
  range. +1.5 rad shoulder (86 deg behind) and +0.5 rad elbow were reachable and
  looked like flung arms.
- **`joint_deviation_arm` -0.03.** Nothing priced arm posture, so the arm action
  means random-walked under Adam until they saturated. L1 deviation from the
  at-the-sides pose gives a restoring force; weight is below `joint_deviation_hip`
  (-0.1) because it spans 4 joints at ~0.5 rad (~-0.06/step).
- **`arm_swing_coupling` -0.1.** The anchor alone makes moving the arms a pure
  cost (0.02 rad total arm deviation after 500 iterations, still 0.02 after
  another 230 with a *position*-coupled version). The velocity form tracks
  `v_shoulder ~ 0.5 * v_hip` (same side); hip velocity is zero-mean, unlike hip
  position. Untracked error on model_225 was 2.78 rad/s ~ 0.28/step vs ~0.3/step
  of leg income: loud enough to act on without dictating the gait.
- **`entropy_coef` 0.002.** Course reward is ~0.06/step while the inherited 0.006
  entropy bonus was ~0.24/step, so entropy dominated: action std grew 0.97 ->
  3.33 rad, past the clip, leaving the action means no usable gradient. Baked in
  from iteration 0 because late corrections crashed.

## Spiral course

The straight course's world-axis terms are zeroed and replaced with path-relative
ones (`mdp/spiral_course.py`); weights otherwise carry over, so the balance
against the fall penalty is unchanged.

- **Spawn.** On the marker from the geometry file, facing world +Y. The first
  tread's tangent is also +Y, so the robot faces the right way once on the
  staircase; the heading command turns it onto the tangent during the approach.
  Offsets exist for a nudge without a re-export.
- **Command speed 0.22-0.30 m/s.** Made the base walk stably (falls 74 % -> 6 %,
  episode 54 s, reward +13.9). 0.45-0.55 m/s (to match the 0.61 m tread spacing)
  was clearly worse (reward -81, progress negative, robot stopped).
- **`base_height_l2` off.** The scan always sees the staircase rising around the
  robot (mean ray 1.6 m above the tread), charging a systematic -13/episode.
- **`stance_height_l2` -5.0** (root height above the stance foot, target 0.84 m).
  Max clearable riser is 0.182 m at 0.837 m hip height but 0.143 m at 0.780 m,
  so a crouched robot cannot clear 0.12 m risers, and without the scanner-based
  term nothing held the stance up.
- **`hip_yaw_splay` off.** It prices the two hip-roll joints differing, and a
  humanoid turns by rolling them differentially. With it active the robot turned
  the wrong way, right hip roll pinned at -0.524, circling at r = 1.45 m instead
  of the 1.16 m walking line. Absolute roll is still priced by
  `hip_abduction_splay` and the ankle terms.
- **`hip_abduction_splay` -1.0** (straight: -6.0). It forbids the hip abduction a
  0.12 m step-up needs. With splay priced the policy cleared risers (bonus
  0.0146), but 83 % of episodes ended in base_tilt because the abducted step-up
  is unstable on the curve, PPO removed it, and the bonus went back to 0. Easing
  the price keeps some abduction. The warm start is the old splayed walker, the
  only policy that could already clear a 0.12 m riser; a staged curriculum
  (learn turn+climb, then price splay out) is the next step once one climbs.
- **Fall penalty and riser bonus work as a pair.** Current: penalty -800, riser
  bonus weight 2000 (+40 per riser in per-step units). The bonus must outweigh
  the immediate cost of a failed attempt (a fall costs 16/step), or PPO never
  tries; at +3 (weight 150) the policy kept its distance and the gait collapsed.
  At +40 attempting pays even at a 29 % success rate. At a penalty of -200 the
  policy stopped caring about falling and flailed (96 % of episodes ended in
  base_tilt within a second). Earlier experiments with -200 let lifting be worth
  the risk but a cheap fall invites flailing, so the two settings only work
  together.
- **`spiral_path_progress` 2.0** (not 5.0). At 5.0 (0.026/step) it was worth twice
  the clearance shaping (0.014/step) and the policy traded the gait for raw
  speed: clearance income collapsed and the robot shuffled. It is the dense term
  that pays for the ~0.9 m flat approach and the curve; it is farmable by
  circling, which `spiral_no_progress` (height gain) and the riser bonus guard
  against.
- **`spiral_riser_bonus`.** The height-gain rate is dense in time but not in
  outcome: a step-up's payoff is spread over the filter time constant, so an
  attempt looks like -16 (fall) or +0.9 and PPO cannot tell success from failure.
  A per-riser bonus is immediate and cannot be farmed by walking.
- **`spiral_face_tangent` -0.25, deadband 0.25.** The heading command already
  steers; at the straight course's -0.5 / 0.15 rad it cost 0.06/step (> 2x the
  progress income) and the policy ignored the climb.
- **`spiral_radial_error` -5.0, deadband 0.10.** 0.2 m off the line costs
  0.02/step and 0.4 m costs 0.18 (7x income). At -25 it cost 40x income and the
  policy never climbed; at -2.0 the robot circled at whatever radius it liked
  (60 % of episodes out of bounds). Target is the command's radius, a hair inside
  the marker, since treads extend 0.14 m past the line but 0.5 m before the
  middle.
- **`foot_swing_clearance` (scalar band, weight 40).** Band is the real 0.14 m
  riser + 0.03 m term margin, as a scalar rather than the straight course's
  per-position table: on the 0.84 m flat approach the stance-foot height is 0, so
  a table would pay only 0.06 m of clearance exactly where the policy must
  prepare to step up. Excess coefficient 30 keeps the lift at the band. A high
  weight is needed because without it the policy tracked velocity perfectly by
  shuffling with straight legs (0.074/step tracking vs 0.0006 clearance; swing
  knees collapsed 0.97 -> 0.13 rad), and a lift on the staircase risks a fall.
- **`foot_flat` zeroed; `foot_flat_landing` -0.5.** The spiral uses the
  landing-window term instead of the straight whole-stance one (deadband 0.23 rad
  calibrated to the good foot's 11.7 deg worst case).
- **`course_out_of_bounds` is wide.** A guidance error is not a fall. The treads
  are an annulus barely a metre wide and a light shove slides the robot off the
  line; terminating that with the -800 penalty made 68 % of episodes end out of
  bounds. The radial reward steers; base-tilt catches real falls.
- **`spiral_no_progress` 0.25 m in 60 s.** Only has to catch a feet-glued
  stander. 0.3 m in 15 s fired on 60 % of episodes, the -16 penalties dominated
  the return, the value scale blew up and the policy diverged (raw actions at
  +/-50). The 0.84 m flat approach eats the first seconds, so a stricter bar
  would truncate every episode before a sustained climb.
- **Episode length 240 s.** The staircase is ~7 m tall and the climb takes ~50 s.
