# Light switch manipulation task (MuJoCo)

Wall light-switch scenes for the Pioneer 8dof arms, plus scripted demo plans.

| Scene | Arm | What | Scripted plan |
|---|---|---|---|
| `toggle_switch` | left | lever on a wall plate, flip up = on | yes (`toggle_plan`) |
| `rocker_switch` | left | rocker paddle, press the top half in = on | yes (`rocker_plan`) |
| `toggle_switch_right`, `rocker_switch_right` | right | mirrored wall in front of the right arm (the leader-arm unit drives the right arm) | no |
| `lever_switch` | left | older table-top lever, kept as an easy fallback | no |

- Shared wall / plate / hinge / detent code: `humanoid_mujoco_scenes/_wall_switch.py`. The detent snaps the switch to its nearer end like a real one.
- Every reset shifts the plate on the wall by up to +-3 cm (seeded), so a policy has to look for the switch.
- The wall is 9 cm beyond the arm's home-pose fingertips (no contact at start) and well inside its reach.
- `success` = the switch is past its "on" threshold (toggle 0.35 rad of +-0.5, rocker 0.1 rad of +-0.15).
- Scripted operator on the 8dof arm: toggle_switch 10/10 and rocker_switch 10/10 (seeds 0-9). Run it with
  `python src/teleop/pioneer_leader_arm_teleop/synthetic/synthetic_teleop.py --scene toggle_switch --record --num_episodes 100`.
- Human teleop: `--scene toggle_switch` works in the Quest and leader-arm teleops. Close the gripper first: with it open
  the lever passes between the two fingers.
- Tests: `src/simulation/mujoco_scenes/tests/test_light_switch_scenes.py`.

## ACT baseline on the 8dof arm (new gripper)

100 scripted demos of `toggle_switch` (left arm, 640x480 `ego` + `wrist_left`, 25 fps, 14,528 frames) recorded on a WATcloud GPU
node (`MUJOCO_GL=egl`, 6 workers, about 40 min). ACT: chunk_size 20, n_action_steps 20, 8000 steps, batch 8, one RTX 4090
(about 11 min). Evaluated on 20 unseen layouts (seeds 1000-1019, `humanoid_robot_learning.sim.evaluate`):

| Scene | Success | 95% CI |
|---|---|---|
| toggle_switch | 20/20 | 84-100% |

Limits: demos are scripted (not human teleop), layout jitter is only +-3 cm, 20 episodes is a small sample, sim only.
`evaluate.py` needs `--task "flip the light switch up to turn the light on"` when the training dataset is not on disk,
and the `transformers` package (the policy registry imports the DiT policy).
