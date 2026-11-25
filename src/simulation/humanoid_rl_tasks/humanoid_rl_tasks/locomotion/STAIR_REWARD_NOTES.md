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

## Foothold tasks (`StairsStraightFoothold`, `StairsSpiralFoothold`)

### Straight foothold
- **Extra patch row behind the heel.** A 7x7 patch at 5 cm reaches 15 cm behind the
  predicted foothold, but the ankle sits 8.0 cm from the heel of a 24.3 cm sole, so the
  step-off edge a descending heel can clip is at the patch's rear boundary. One more row
  extends the reach to 20 cm; the extra heights are flattened first so the original 49
  keep their order. The critic sees the same layout.
- **Spawn at the foot of any part.** Starting 1.5 m before the first riser made the
  14.5-16 cm parts a small slice of each 46 s episode. The y slots are the centres of the
  0.4 m ground-level gaps (`straight_stairs_centreline.npz`), where the whole 24 cm sole
  sits on flat ground; jitters are small for the same reason.
- **Soft landings -5.0.** Touchdown frames summed 91.7 (m/s)^2 over a 75 s episode on
  model_3549, worth 0.012 at -0.5 (inert). -5.0 is ~0.12, about 9 % of the 1.34
  forward-progress income.
- **Push ramp 0.35 -> 1.0** over 6000 env steps (~250 iterations): `push_robot` kicks up to
  0.8 m/s per axis every 5-10 s, so starting at ~1/3 lets the gait settle first; the
  endpoint is unchanged.
- **Torso stability.** model_2999 (28 s, no pushes): pitch -0.9 +/- 4.6 deg, roll
  +0.3 +/- 3.2 deg, |wx| 0.36 / |wy| 0.44 rad/s. Mean tilt ~0, so the defect is oscillation;
  the rate term goes from ~7 % to ~14 % of velocity-tracking income.

### Spiral foothold (the measurements behind each `SPIRAL_*` knob)
- **Defaults** reproduce `model_249_slow`: dense height-rate climb income capped at the
  commanded path speed (weight 80 at 0.10 m/s), riser bonus 20 (makes a step-up worth the
  fall), flight penalty 25, velocity tracking 5, torso damped (-2 roll / -10 tilt).
- **`feet_impact_contact`.** The inherited `feet_impact` reads sensors that filter the
  terrain mesh, but here the feet touch the staircase helicoid, so it logged -0.0011 per
  episode (0.1 % of income). The replacement reads the `contact_forces` sensor.
- **`feet_slide` -4.** Plant speed while loaded was 0.19 m/s here vs 0.09 on the straight
  course (the foot is still travelling when it lands). The inherited -0.5 was 2 % of income;
  -4 is ~7 % and is what `model_299_plant` used. It is linear in plant speed, so it prices a
  skid where the squared vertical-impact term cannot.
- **Back-kick -1.5, allow 0.10.** `kick_probe.py` over 381 swings aligned at lift-off: the
  foot leaves the tread 0.24 m behind the root and moves 3 cm further backwards over 0.26 s
  with the knee at -0.75 rad. 43 % of swings carry the foot backwards in world frame for the
  first 0.2 s; averaging hides it. Raw term averages 0.055 (m/s)^2/step (p90 0.117) against
  ~0.18 reward/step: -0.5 (~15 %) changed nothing, -1.5 charges ~46 %. It uses
  `body_lin_vel_w` projected on the heading, gated on airborne, so a planted foot is free.
- **Knee fold: off.** Knee goes -0.23 -> -0.73 rad within 0.24 s of lift-off. Pricing it at
  -1.0 (34 % of income) changed nothing: folding is geometrically required to clear a riser
  from a foot that leaves 0.24 m behind the body.
- **Shank rotation -0.02.** Shank pitch -22.9 -> -39.5 deg over 0.28 s while the thigh comes
  forward only 12.5 deg. Pricing joint angle or foot position both failed; pricing the
  rotation forces the hip to lead. Raw 1.54 (rad/s)^2/step, so -0.02 is ~17 % of income.
- **Action-rate -0.02, not -0.002.** The action changed 2.9 sigma per step (straight: 0.34)
  and legs accelerated at 99 rad/s^2 (straight: 43). At -0.002 the term was 3 % of income and
  the policy buzzed at the control Nyquist rate (the knee target, clipped to (-1.1, -0.05),
  was a two-state relay). At -0.02 the buzz costs 35 % and the smooth regime 1.5 %.
- **Slew cap on leg targets: off.** On `2026-11-18_19-08-38` the mean target step is tiny
  (knee 0.03 rad) but tails are huge (Ankle_P_L and Hip_F_L p99 0.74 rad, max 1.37 rad in
  one step). A 12 rad/s cap clips only those, yet trained in it cost 20 % completion
  (0.951 -> 0.749) and left the leg-acceleration tail unchanged (p99 270 -> 246): the spikes
  are joints hitting position stops, not command slew. It was briefly default-on, which put
  earlier probes and videos under the cap. The mechanism stays in `mdp/joint_rate_limit.py`
  behind `SPIRAL_LEG_RATE` / `SPIRAL_KNEE_RATE`.
- **Knee slew cap: tried, off.** Knee target step median 0.000 rad, p99 0.89, max 1.05 rad
  (the whole range in one 20 ms step: a relay). A 16 rad/s cap cost nothing but left velocity
  reversals unchanged (23.96 -> 23.93 /s, 24.16 retrained) and taps per riser went 3.6 -> 5.1:
  a slew cap bounds step size, not reversal rate.
- **Knee action scale 0.05.** The inherited 0.30 maps the 1.05 rad clip range onto a in
  [-2.0, +1.5] while the knee output is |a| ~ 8-13, so the command sits on a rail ~85 % of
  steps, with ~5x too much small-signal gain and ~24 reversals/s (straight: 12-16). 0.05
  spans a in [-12, +9].
- **`action_l2` -2e-3.** On the shipped model mean |a| was 10.8-13.9 (hips) and 9.9-11.3
  (knees) vs 0.9-1.3 on the straight deliverable; commands were outside the joint range on
  50-84 % of steps and motors at effort limit on 26-59 %. The drift came with action std
  growing 0.84 -> 2.01 rad (`entropy_coef` 0 leaves it no restoring force). `action_rate_l2`
  prices only changes, so a policy sitting in saturation pays nothing. The reward manager
  charges `weight * sum(a^2) * dt` and logs it divided by episode length (which is why -2e-4
  looked 39x too small); -2e-3 is ~30 % of gross reward at |a| ~ 10 and ~1 % at |a| ~ 1.
- **`knee_target_overshoot`.** Charges the unclipped target beyond the window (smooth-L1,
  linear tail). The inherited -0.025 was ~1 % of reward and the relay survived three runs;
  -0.5 is ~22 % but the policy paid it instead of leaving the rail (completion
  0.939 -> 0.903), so it stays at the inherited value until the target signal is measured.
- **Swing target vs band bonus.** The straight deliverable prices swing height with a target
  (`feet_swing_height_relative`, 0.10 m, weight -40); the inherited band bonus pays for
  extreme flexion, and the knee sat on a clip value 85 % of steps. `dof_vel` and
  `contact_no_vel` are the two other smoothness terms this task lacked.
- **Fly-in: off.** At -2 over 300 iterations it cut peak touchdown force 680 -> 670 N but
  completion fell 0.894 -> 0.797 and out-of-bounds doubled (0.039 -> 0.086): pricing the low
  forward carry makes the policy stub risers. The horizontal arrival is a symptom of the
  short stride.
- **Command gate threshold.** Swing clearance, air time and knee tuck gate on a strict `>`
  against the command norm, and this task commands exactly 0.10 m/s, equal to the shared 0.1
  threshold, so all three read 0.0000 and the robot shuffled up on straight knees. The
  threshold now sits below the command; the shared default is untouched.
- **Scanner at the base, not lifted.** The spiral recipe lifts the scanner 1.5 m, but from
  there ~75 % of rays hit the turn above (heights ~-1.5 m, clipped to -1), which is exactly
  the information the foothold patches need.
- **Observation clips.** The base-motion, joint-velocity and last-action terms were
  unbounded, and `last_action` feeds back. Both spiral runs died at ~450 iterations:
  `action_rate_l2` spiked 350,000x in one iteration, value loss reached 1e12, then std went
  NaN, with actor weights flat (max |w| 0.44) -- the blow-up came from the input.
- **`invalid_height_scan` off.** A ray leaving the staircase from the base-level scanner is
  genuinely no ground and is already encoded as -1; terminating on it charged -800 for the
  outer columns and ended 25-67 % of transfer-run episodes.
- **Spawn spread.** Starting every episode on the flat 0.84 m approach (standing is free
  there) was the attractor the straight-to-spiral transfer sat in for 455 iterations without
  climbing once; half the batch now starts on treads.
- **Riser bonus capped in rate.** Uncapped, it pays more per second the faster the robot
  moves and turned the gait into a sprint (2.5-3.5 risers/s vs a 0.20 m/s command). The cap
  pays at most one riser per `SPIRAL_BONUS_MIN_STEPS`; dense progress terms are capped at
  the commanded path speed.
- **Action window scaling.** On the old scales the low edge was reached at |a| of 2.3-3.5
  while the policy sat at 8-10; outside the window the reward is flat in the action, so no
  gradient. Scaling so |a| = `_ACT_RANGE` reaches the far edge keeps every relevant command
  inside the gradient region.
- **Keep a foot loaded.** A staircase is a support-transfer task. The straight policy keeps
  a foot loaded 66 % of the time; the spiral was at 27 % (4.3 footfalls/s, 44 deg forward
  lean). The term is zero in every supported step, so it does not prefer standing. With it:
  54 % loaded, completion 0.939 -> 0.994, falls 0.052 -> 0.005.
