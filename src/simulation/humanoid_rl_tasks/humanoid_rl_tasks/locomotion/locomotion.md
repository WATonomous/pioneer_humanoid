# Locomotion: Pioneer humanoid V1 velocity tracking

Velocity-commanded bipedal locomotion for **Pioneer humanoid Simulation Model V1** in Isaac Lab. The agent receives base velocity commands $(v_x, v_y, \omega_z)$ and is rewarded for tracking them while staying upright.

Asset: `assets/old_leg_only_humanoid/` (SolidWorks URDF export).
Training uses `old_leg_only_humanoid.urdf`, which adds a Z-up `base` link and a fixed joint so the CAD Y-long frame stands in Isaac without a spawn rotation.

**Environments**

| Task ID | Terrain | Mode |
| :--- | :--- | :--- |
| `Isaac-Locomotion-Flat-PioneerHumanoid-v0` | Plane | Train |
| `Isaac-Locomotion-Flat-PioneerHumanoid-Play-v0` | Plane | Play |
| `Isaac-Locomotion-Rough-PioneerHumanoid-v0` | Procedural rough | Train |
| `Isaac-Locomotion-Rough-PioneerHumanoid-Play-v0` | Procedural rough | Play |

Legacy aliases `Isaac-Velocity-*` register the same configs.

## Train & play

Run inside the **`simulation_isaac`** container (Isaac Lab 2.3.2 / Sim 5.1). Host setup: [`docker/simulation/isaac_lab/README.md`](../../../../../docker/simulation/isaac_lab/README.md).

```bash
# Host
cd ~/Desktop/humanoid && ./watod up -d && ./watod -t simulation_isaac

# Inside container — from $HUMANOID_ROOT
cd $HUMANOID_ROOT

# Train — flat terrain
rl-train --task=Isaac-Locomotion-Flat-PioneerHumanoid-v0 --headless

# Play — loads latest under outputs/rl/pioneer_humanoid_flat/
rl-play --task=Isaac-Locomotion-Flat-PioneerHumanoid-Play-v0 --num_envs=1

# Play — specific checkpoint
rl-play --task=Isaac-Locomotion-Flat-PioneerHumanoid-Play-v0 --num_envs=1 \
  --checkpoint outputs/rl/pioneer_humanoid_flat/<run>/model_<iter>.pt
```

Rough-terrain variants: replace `Flat` with `Rough` and use `outputs/rl/pioneer_humanoid_rough/`.

**Spawn / joint smoke checks**

```bash
$ISAACLAB/isaaclab.sh -p $RL_RUNNERS/diagnose_spawn.py \
  --task=Isaac-Locomotion-Flat-PioneerHumanoid-v0 --headless --num_envs=1 --steps=10

$ISAACLAB/isaaclab.sh -p $RL_RUNNERS/diagnose_joints.py \
  --task=Isaac-Locomotion-Flat-PioneerHumanoid-v0 --headless --num_envs=1
```

## Joints & bodies

| Role | Names |
| :--- | :--- |
| Root body | `base` (merged upright root from isaac URDF) |
| Feet (contacts) | `Foot_L`, `Foot_R` |
| Hip flexion | `Hip_F_L`, `Hip_F_R` |
| Hip abduction | `Hip_A_L`, `Hip_A_R` |
| Hip rotation | `Hip_R_L`, `Hip_R_R` |
| Knee | `Knee_L`, `Knee_R` |
| Ankle pitch / roll | `Ankle_P_*`, `Ankle_R_*` |

Config: `pioneer_humanoid/whole_body.py`, tasks under `config/pioneer_humanoid_v1/`.
