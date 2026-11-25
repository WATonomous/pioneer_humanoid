# Onshape stair courses: reward and termination rationale

Why the course configs (`config/pioneer_humanoid_v1/onshape_*.py`) set what they
set. The code carries a one-line summary per term; this file records the reason,
and the measurement where one exists. Weights are per-second rates (multiplied by
`dt = 0.02`), so a term only matters if its per-step share is visible against the
~0.03/step forward-progress income; several inherited terms were inert for
exactly that reason. Numbers are from earlier runs and are context, not
guarantees on the current walker.

## Straight course (`onshape_env_cfg.py`)

- **Shove** every 8-12 s: +/-0.5 m/s sideways is the shipping setting (83.5 %
  completion). +/-0.9 was worse (62.6 %, action std 3.7 -> 4.9): the lateral
  catch, not the fore/aft bump, is the axis that matters.
- **Knee clip -1.10.** The shared recipe clips to -0.95 rad, which lifts the foot
  only ~0.155 m; the tall risers (~0.21 m) are then outside sagittal reach and
  the walker clears them by abducting sideways. Task-local clip, no knee reward.
- **Bias-compensation terms are weight 0.** They were written for two mirror-axis
  defects in the robot description (right ankle pitch, right hip abduction). With
  the axes fixed the bias source is gone; they are kept at 0 for a warm-started
  clean-up pass.
- **`feet_impact` -0.5** is 12x the inherited weight, which was inert (~-0.5 per
  episode against ~+6500 progress income).
- **`foot_flat` -15**, deadband 0.23 rad, 0.15 s landing window. Measured on the
  foot *body*, not the ankle joint (the joint angle did not predict landing
  tilt). The short window is load-bearing: a whole stance legitimately pitches
  to 40-67 deg. -15 puts a foot 9 deg outside the deadband at ~5 % of income;
  -0.5 is inert by the same arithmetic.
- **`feet_slide` -0.5.** A loaded foot moving 0.62 m/s is dragging; at -0.1 the
  term logged 0.0008 against 0.054 progress (inert).
- **Symmetry terms are weight 0.** A pairwise L/R term fights the anti-phase
  swing, so they compare time-averaged joint angles and stance duty instead
  (`mdp/rewards.py`).
- **Goal distance 20.5.** The course is a sawtooth, so the goal sits past the
  final descent; ending on the plateau left every term blind to the landing.
- **`hip_abduction_splay` -6**, deadband 0.30, calibrated to the leg that already
  swings correctly, so a correct gait is free and one-sided splay pays.
- **Knee tuck 0.28 / -4.0.** 0.40 marched; risers get their tuck from the
  clearance reward, which pays up to a 0.20 m band.
- **`feet_air_time` 1.0 / 0.2 s.** Cut the weight, not the threshold: the term
  pays `max(air - threshold, 0)`, so raising the threshold lengthens the swing
  instead. Zeroing it or gating it on speed produces a feet-glued stander.
- **`termination_penalty` -800** is a warm-start setting. From scratch the
  iteration-0 return is the penalty spike and the action std runs away; -200 was
  used for from-scratch lineages.
- **Height-scan noise** (`LidarNoiseCfg`): Gaussian + per-beam bias + dropout,
  ramped over ~500 iterations so a warm-started checkpoint adapts.

## Arm swing (`onshape_env_cfg.py`, arm-swing classes)

- **Arm action scale 0.5.** The inherited scale dict lists leg joints only, so
  arm actions fell back to 1.0, 4-16x any leg channel; the shoulders ended pinned
  behind the torso.
- **Clip envelope** shoulder (-0.9, 0.45), elbow (-1.4, 0): a walking swing, not
  the joint's full range. The full range looked like flung arms.
- **`joint_deviation_arm` -0.03** gives arm posture a restoring force, below
  `joint_deviation_hip` because it spans four joints.
- **`arm_swing_coupling` -0.1.** The anchor alone makes moving the arms a pure
  cost; the velocity form tracks `v_shoulder ~ 0.5 * v_hip` (same side), which is
  zero-mean unlike hip position.
- **`entropy_coef` 0.002.** The inherited 0.006 was ~0.24/step against ~0.06/step
  of course reward, so entropy dominated and the action std grew past the clip.

## Spiral course (`onshape_spiral_env_cfg.py`)

World-axis terms are replaced with path-relative ones (`mdp/spiral_course.py`);
weights otherwise carry over, so the balance against the fall penalty is
unchanged.

- **Spawn** on the marker from the geometry file, facing +Y; the first tread's
  tangent is +Y, so the heading command turns the robot onto it.
- **Command 0.22-0.30 m/s** walks stably (falls 74 % -> 6 %); 0.45-0.55 m/s (the
  0.61 m tread spacing) was clearly worse.
- **`base_height_l2` off.** The scan always sees the staircase rising around the
  robot, charging a systematic -13/episode.
- **`stance_height_l2` -5.0**, target 0.84 m. A crouched robot cannot clear
  0.12 m risers (0.182 m max at 0.837 m hip height vs 0.143 m at 0.780 m).
- **`hip_yaw_splay` off.** A humanoid turns by rolling the hip-roll joints
  differentially; with it active the robot circled the wrong way. Absolute roll
  is still priced by `hip_abduction_splay` and the ankle terms.
- **`hip_abduction_splay` -1.0** (straight: -6.0). A 0.12 m step-up needs some
  abduction; pricing it out removed the only clearance the policy had.
- **Fall penalty and riser bonus are a pair.** -800 / +40 per riser: the bonus
  must outweigh the immediate cost of a failed attempt (a fall costs 16/step) or
  PPO never tries. At -200 flailing paid.
- **`spiral_path_progress` 2.0.** At 5.0 it was worth twice the clearance shaping
  and the policy traded the gait for speed. `spiral_no_progress` and the riser
  bonus guard against farming it by circling.
- **`spiral_riser_bonus`.** Height gain is dense in time but not in outcome, so
  an attempt looks like -16 or +0.9 and PPO cannot tell success from failure.
- **`spiral_face_tangent` -0.25** (straight: -0.5). The heading command already
  steers; the stronger price cost 2x the progress income.
- **`spiral_radial_error` -5.0**, deadband 0.10. 0.2 m off the line costs 0.02
  and 0.4 m costs 0.18 (7x income); -25 never climbed, -2.0 circled.
- **`foot_swing_clearance`** is a scalar band (0.14 m riser + 0.03 m) rather than
  the straight course's per-position table: on the flat approach a table would
  pay only 0.06 m exactly where the step-up is prepared.
- **`foot_flat_landing` -0.5** replaces the straight whole-stance term.
- **`course_out_of_bounds` is wide.** A guidance error is not a fall; the treads
  are a ~1 m annulus and a light shove slides the robot off, so terminating with
  -800 ended 68 % of episodes. The radial reward steers; base-tilt catches falls.
- **`spiral_no_progress` 0.25 m in 60 s.** Only has to catch a stander; a
  stricter bar fired during the flat approach and the -16 penalties diverged PPO.
- **Episode 240 s:** the staircase is ~7 m tall and the climb takes ~50 s.

## Foothold tasks

### Straight (`onshape_foothold_env_cfg.py`)

- **One extra patch row behind the heel.** A 7x7 patch at 5 cm reaches 15 cm
  behind the foothold, but the heel sits 8.0 cm behind the centre of a 24.3 cm
  sole, so the step-off edge was at the patch boundary. The extra row is
  flattened first, so the original 49 keep their order.
- **Spawn at the foot of any part.** The y slots are the centres of the 0.4 m
  ground-level gaps, where the whole sole sits on flat ground; starting 1.5 m
  before the first riser made the tall parts a small slice of each episode.
- **Soft landings -5.0.** At -0.5 the term was 0.012 against 1.34 income;
  -5.0 is ~9 %.
- **Push ramp 0.35 -> 1.0** over ~250 iterations, so the gait settles before
  full shoves; the endpoint is unchanged.

### Spiral (`onshape_spiral_foothold_env_cfg.py`, the `SPIRAL_*` knobs)

- **Defaults** reproduce `model_249_slow`.
- **`feet_impact_contact`** replaces the inherited `feet_impact`, which reads
  filtered terrain sensors and logged 0.1 % of income on the helicoid.
- **`feet_slide` -4** (straight: -0.5): plants are still moving at 0.19 m/s here
  vs 0.09 on the straight course.
- **Back-kick -1.5**, allow 0.10 m: 43 % of swings carry the foot backwards for
  the first 0.2 s, and averaging hides it. -0.5 changed nothing.
- **Knee fold off.** Folding is geometrically required to clear a riser from a
  foot that leaves 0.24 m behind the body.
- **Shank rotation -0.02** makes the hip lead the swing instead of the shank.
- **Action rate -0.02, not -0.002.** The action changed 2.9 sigma/step and legs
  accelerated at 99 rad/s^2 (straight: 43); at -0.002 the policy buzzed at the
  control Nyquist rate.
- **Knee action scale 0.05** (straight: 0.30). At 0.30 the command sat on a rail
  ~85 % of steps with ~24 reversals/s.
- **`action_l2` -2e-3.** Mean |a| was 10.8-13.9, with commands outside the joint
  range on 50-84 % of steps; `action_rate_l2` prices only changes, so saturation
  was free.
- **Swing target instead of a band bonus**, plus `dof_vel` and `contact_no_vel`.
- **Fly-in off.** Pricing the low forward carry made the policy stub risers.
- **Command gate threshold** sits below the 0.10 m/s command; at equality the
  strict `>` made clearance, air time and knee tuck all read 0.0000.
- **Scanner at the base, not lifted.** Lifted 1.5 m, ~75 % of rays hit the turn
  above, which is exactly the information the foothold patches need.
- **Observation clips.** Unbounded base-motion, joint-velocity and last-action
  terms let `action_rate_l2` spike 350,000x and the run go NaN.
- **`invalid_height_scan` off.** A ray leaving the staircase is genuine no-ground
  and already encoded as -1; terminating charged -800 for the outer columns.
- **Spawn spread.** Starting every episode on the flat approach was the
  attractor the transfer run sat in without climbing; half the batch starts on
  treads.
- **Riser bonus capped in rate**, or it pays for sprinting (2.5-3.5 risers/s).
- **Action window scaling.** Outside the window the reward is flat in the action,
  so scaling to `_ACT_RANGE` keeps every relevant command inside the gradient.
- **Keep a foot loaded.** A staircase is a support-transfer task; the straight
  policy keeps a foot loaded 66 % of the time vs 27 % here, and the term is zero
  in every supported step.
