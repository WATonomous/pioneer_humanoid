# Onshape stair course: reward and termination rationale

Why the course config (`config/pioneer_humanoid_v1/onshape_foothold_env_cfg.py`) sets what it
sets. The code carries a one-line summary per term; this file records the reason,
and the measurement where one exists. Weights are per-second rates (multiplied by
`dt = 0.02`), so a term only matters if its per-step share is visible against the
~0.03/step forward-progress income; several inherited terms were inert for
exactly that reason. Numbers are from earlier runs and are context, not
guarantees on the current walker.

## Foothold tasks

### Straight (`onshape_foothold_env_cfg.py`)

- **One extra patch row behind the heel.** A 7x7 patch at 5 cm reaches 15 cm
  behind the foothold, but the heel sits 8.0 cm behind the centre of a 24.3 cm
  sole, so the step-off edge was at the patch boundary. The extra row is
  flattened first, so the original 49 keep their order.
- **Spawn at the foot of any part.** The y slots are the centres of the 0.4 m
  ground-level gaps, where the whole sole sits on flat ground; starting 1.5 m
  before the first riser made the tall parts a small slice of each episode.
- **Lidar noise on the policy's rays** (`LidarNoiseCfg`: 2 cm range noise, 1 cm per-beam
  bias, 2 % dropout + 2 % dead beams per episode), applied to the height map before the foothold patches
  are cut, on top of the existing +/-0.1 output noise. Ramped in over ~500 iterations so a warm start
  adapts. The critic's scan stays clean. `model_599` was trained without it.
- **Soft landings -5.0.** At -0.5 the term was 0.012 against 1.34 income;
  -5.0 is ~9 %.
- **Push ramp 0.35 -> 1.0** over ~250 iterations, so the gait settles before
  full shoves; the endpoint is unchanged.
