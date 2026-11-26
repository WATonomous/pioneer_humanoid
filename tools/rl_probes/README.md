# RL probes

Throwaway diagnostics written while tuning the stair policies. Each is run inside the Isaac container:

    isaaclab.sh -p tools/rl_probes/<script>.py

They locate the repo relative to this folder. Several still default to specific local run folders
(`outputs/rl/<task>/<date>/model_N.pt`); point them at a checkpoint with the env vars the script reads
(`CKPT_OVERRIDE`, `TASK_OVERRIDE`, ...). Not maintained as a library: read a script before trusting its numbers.
The per-swing sole-tilt probe lives in `tools/onshape_course/step_probe.py`.

| Script | What it does |
|---|---|
| `backkick_check.py` | Smoke test + calibration for the swing-leg backward-kick penalty. |
| `buzz_probe.py` | Is the leg buzz in the contact, or in the joint? |
| `cadence.py` | Step cadence and sway on a course, with the arms swinging or frozen. |
| `check_gait_terms.py` | Verify the 3 s clock, the foot-land tread bonus and the mismatch term. |
| `check_polish_cfg.py` | Verify the polish reward set builds and the new terms are registered. |
| `check_recipe.py` | Verify the simplified spiral recipe. |
| `check_spawn.py` | Sanity-check the multi-slot course spawn and the push ramp on the TRAIN task. |
| `check_spiral_command.py` | Verify the spiral command is now the tangent-following one. |
| `diff_cfg_probe.py` | (no docstring) |
| `distill_policy.py` | Distill a saturated policy into a re-parameterised (well-conditioned) action space. |
| `dump_joint.py` | One joint, in detail: raw action, scale, target, position, velocity, torque. |
| `dump_knee.py` | Dump the left knee's velocity and commanded target for two seconds. |
| `eval_marker_spawn.py` | Marker-spawn evaluation: same start point for every checkpoint. |
| `eval_push.py` | Marker-spawn evaluation: same start point for every checkpoint. |
| `eval_spiral_final.py` | Completion rate of the final spiral foothold model, training disturbance, many envs. |
| `eval_straight_final.py` | Completion of a straight-stairs foothold policy on the (deterministic) Play course: terminations per term. |
| `gait_quality.py` | Gait quality of a spiral checkpoint: speed vs command, sway, yaw flail, foot slip. |
| `gait_true.py` | Debounced gait measurement: footfalls, step length, treads per step, path speed. |
| `joint_info.py` | Joint ranges, default pose, action scale/offset/clip and effort limits for the leg action term. |
| `kick_probe.py` | What does the swing leg do right after it leaves the ground? |
| `knee_rate_stats.py` | How fast does the knee's target actually need to move? |
| `l2_calib.py` | What does the reward manager actually charge for action_l2 per step? |
| `landing_probe.py` | What does the foot actually do at the instant it lands? |
| `leg_quality.py` | Measure leg smoothness and landing hardness on the spiral. |
| `obs_jitter.py` | Frame-to-frame jitter of each policy observation term. |
| `obs_layout.py` | (no docstring) |
| `probe_contact.py` | Is the terrain-contact sensor alive on this course? |
| `probe_swing.py` | Do the terrain-gated swing rewards actually fire while walking? |
| `rate_limit_check.py` | Smoke test for the in-term knee rate limit (mdp/joint_rate_limit.py). |
| `reset_probe.py` | Minimal check: does this env survive being reset twice in one process? |
| `reward_breakdown.py` | Per-step reward breakdown of the spiral foothold policy (PLAY). |
| `sat_probe.py` | Is the shipped policy's action saturation real, or an artefact of a fall? |
| `slew_check.py` | Does capping the knee target's slew rate remove the swing chatter? |
| `slew_probe.py` | Does a slew-rate limit on the commanded joint target fix the leg chatter? |
| `smooth_probe.py` | Where does the jerk and the hard landing actually come from? |
| `target_probe.py` | Where does the leg jitter actually live: the command, or the joint? |
| `trace_spiral.py` | Per-step trace of command vs motion on the spiral (model_900, PLAY). |
| `train_act_probe.py` | Does the policy behave differently under training-time observation noise? |
| `wobble.py` | Quantify the gait wobble: yaw-rate command vs tracking, lateral sway, speed steadiness. |
