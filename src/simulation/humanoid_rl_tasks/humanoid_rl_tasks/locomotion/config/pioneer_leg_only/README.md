# Pioneer humanoid: flat-ground walking

Velocity-commanded walking on flat ground for two robots, with the same policy observations and actions:

- leg-only humanoid (`assets/pioneer_leg_only_humanoid`): 12 leg joints, 32.2 kg;
- whole-body humanoid (`assets/pioneer_whole_body_humanoid`): arms, hands and head, 44.7 kg; only the 12 leg joints
  move, everything above the hips is held rigid.

Masses are the URDFs' estimates. Robot configs: `pioneer_humanoid/leg_only.py` (each URDF re-rooted at the hips with
+X forward; the assets' root is at the soles, -Y forward).

| Task | Robot |
|---|---|
| `Isaac-Locomotion-Flat-PioneerLegOnly-v0` | leg-only humanoid |
| `Isaac-Locomotion-Flat-PioneerWholeBody-v0` | whole-body humanoid, everything above the hips held rigid |

Each has a `-Play-v0` variant. In the `simulation_isaac` container, from `$HUMANOID_ROOT`:

```bash
rl-train --task=Isaac-Locomotion-Flat-PioneerLegOnly-v0 --headless          # ~1 h for 3000 iterations
rl-play --task=Isaac-Locomotion-Flat-PioneerLegOnly-Play-v0 --num_envs 1 --video --checkpoint <run>/model_<n>.pt
# PLAY_COMMAND="vx,vy,wz" (default "0.5,0,0") or PLAY_COMMAND=random (a new command every 3 s)
```

A whole-body run can start from a leg-only checkpoint (same observations and actions): `--resume True --load_run <run> --checkpoint <ckpt>`.

## Rewards

The reward is the weighted sum of these terms, every 20 ms step (`flat_env_cfg.py`; gait terms in `gait.py`).

| Category | Term | Weight | What |
|---|---|---|---|
| Task | `track_lin_vel_xy_exp` | 1.0 | exp(-error² / 0.25) of the commanded x-y velocity (yaw frame) |
| | `track_ang_vel_z_exp` | 0.5 | exp(-error² / 0.25) of the commanded yaw rate |
| | `alive` | 0.15 | per step not fallen |
| Gait | `contact` | 0.18 | feet (0-2) whose ground contact matches the clock: stance for the first 55% of each leg's 1 s cycle, legs half a cycle apart |
| | `feet_swing_height` | -20 | squared error of each airborne sole from 10 cm |
| | `contact_no_vel` | -0.2 | squared velocity of planted feet (slip) |
| | `stance_knee` | -1.0 | squared distance of a stance knee from -0.15 rad (near-straight supporting leg) |
| | `feet_flat` | -2.0 | squared tilt of planted feet (land on the sole, not its edge) |
| | `feet_too_close` | -40 | squared amount the feet come closer than 0.2 m sideways or cross |
| Posture | `base_height` | -10 | squared error of base height from 0.835 m |
| | `flat_orientation_l2` | -1.0 | squared base tilt |
| | `hip_pos` | -1.0 | squared hip roll and yaw angles |
| Smoothness | `lin_vel_z_l2` | -2.0 | squared base vertical velocity |
| | `ang_vel_xy_l2` | -0.05 | squared base roll and pitch rates |
| | `dof_vel` | -1e-3 | squared joint velocities |
| | `dof_acc_l2` | -2.5e-7 | squared hip and knee accelerations |
| | `dof_torques_l2` | -2e-6 | squared hip and knee torques |
| | `action_rate_l2` | -0.02 | squared change of the action |
| Limits | `dof_pos_limits` | -5.0 | joint angles beyond 90% of their range |
| | `termination_penalty` | -200 | episode ends by a fall (base touches the ground) |

Randomised for transfer: friction 0.3-1.2, base mass -1 to +3 kg, base centre of mass ±3 cm, motor stiffness and damping
×0.7-1.3, joint armature ×0.5-2, pushes up to 0.8 m/s every 5-10 s, start pose ±0.1 rad.
PPO trains every batch also on its left-right mirror image (`symmetry.py`), for an even gait.

## Policy interface (for deployment)

50 Hz. Observation (50): base linear velocity of the base's centre of mass and angular velocity, both in the base frame
(3 + 3), projected gravity (3), command vx, vy, wz (3), joint positions minus default (12), joint velocities (12), last
action (12), gait clock sin, cos of 2π·(t mod 1 s) (2). Action (12): joint target = default + 0.25 · action. Joint order:
`Hip_F_L Hip_F_R Hip_A_L Hip_A_R Hip_R_L Hip_R_R Knee_L Knee_R Ankle_P_L Ankle_P_R Ankle_R_L Ankle_R_R`. Default pose:
Hip_F 0.25, Knee -0.5, Ankle_P -0.25 rad, the rest 0. PD gains: Hip_F/Knee 200, Hip_A/Hip_R 150, ankles 40; damping 5
(ankles 2).

## Notes

- Sideways walking is slow (~0.2 m/s for 0.3): the hip can swing a leg only 10° inward.
- `gait.py` zeroes non-finite rewards (a rare PhysX glitch otherwise turned training into NaNs at a random iteration) and
  `physics_blowup` resets such robots.
