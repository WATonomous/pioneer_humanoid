# Light switch manipulation task (MuJoCo)

Two wall light-switch scenes for the Pioneer left arm, plus scripted demo plans and an ACT baseline.

- `toggle_switch`: lever on a wall plate, flip up = on.
- `rocker_switch`: rocker paddle, press the top half in = on (gripper tilted ~25 deg down, since the fingers are taller than the paddle).
- Shared wall/hinge/detent code: `humanoid_mujoco_scenes/_wall_switch.py`.
- Demo plans (`toggle_plan`, `rocker_plan`) in `synthetic/synthetic_tasks.py`. Scripted operator: 30/30 successes per scene.

## ACT baseline

50 scripted demos per scene (320x240, `ego` + `wrist_left`, 25 fps). ACT, chunk_size 20, 8000 steps, batch 8, trained on an M1 MacBook (MPS), about 2.5 h per scene.
Evaluated on 20 unseen layouts (seeds 1000-1019, `humanoid_robot_learning.sim.evaluate`):

| Scene | Success | 95% CI |
|---|---|---|
| toggle_switch | 20/20 | 84-100% |
| rocker_switch | 20/20 | 84-100% |

Limits: demos are scripted (not human teleop), layout jitter is only +-3 cm, 20 episodes is a small sample, sim only.

Run: `synthetic_teleop.py --scene toggle_switch --record --num_episodes 50 --image_size 320x240`, then `lerobot-train ... --policy.type=act`, then `humanoid_robot_learning.sim.evaluate --scene toggle_switch --policy <ckpt> --episodes 20 --seed 1000`.
