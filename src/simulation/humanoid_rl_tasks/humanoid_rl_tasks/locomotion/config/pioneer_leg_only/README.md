# Pioneer leg-only humanoid: walking

Velocity-commanded walking for `assets/pioneer_leg_only_humanoid` (12 leg joints, 32.2 kg), and the same policy on the
whole-body humanoid (`assets/pioneer_whole_body_humanoid`, arms and hands, upper body held). Robot config:
`pioneer_humanoid/leg_only.py` (the URDF re-rooted at the hips with +X forward; the asset's root is at the soles, -Y forward).

| Task | What |
|---|---|
| `Isaac-Locomotion-Flat-PioneerLegOnly-v0` | Isaac Lab's G1 flat recipe, Pioneer names |
| `Isaac-Locomotion-Gait-PioneerLegOnly-v0` | + Unitree rl_gym gait clock (1 s cycle), swing height, no-slip, straight stance knee, flat-foot landing, randomised motors/friction/mass/pushes |
| `Isaac-Locomotion-GaitSym-PioneerLegOnly-v0` | + left-right symmetry augmentation (`symmetry.py`). **The recipe to use.** |
| `Isaac-Locomotion-GaitWholeBody-PioneerLegOnly-v0` | GaitSym on the whole-body humanoid, everything above the hips fixed |
| `Isaac-Locomotion-GaitArmSwing-PioneerLegOnly-v0` | the same with shoulders and elbows swung on the gait clock (`arm_swing.py`), fists closed |

Each has a `-Play-v0` variant. In the `simulation_isaac` container, from `$HUMANOID_ROOT`:

```bash
rl-train --task=Isaac-Locomotion-GaitSym-PioneerLegOnly-v0 --headless          # ~1 h for 3000 iterations
rl-play --task=Isaac-Locomotion-GaitSym-PioneerLegOnly-Play-v0 --num_envs 1 --video --checkpoint <run>/model_<n>.pt
# PLAY_COMMAND="vx,vy,wz" (default "0.5,0,0") or PLAY_COMMAND=random (a new command every 3 s)
```

A whole-body run can start from a leg-only checkpoint (same observations and actions): `--resume True --load_run <run> --checkpoint <ckpt>`.

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
