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
