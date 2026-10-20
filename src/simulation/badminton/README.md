# Badminton receive — stationary bimanual arm, RL

MuJoCo simulation of the WATonomous bimanual arm receiving badminton serves
with a racket in each hand, and the RL pipeline that trains the receive
policy (mjlab / MuJoCo Warp, rsl_rl PPO).

One single-arm policy plays both arms (the shared-arm tasks,
`humanoid_badminton/shared.py`). It drives whichever arm is active, in that
arm's frame; the left arm sees the mirrored world, so the two sides are
symmetric by construction. Both arms start each serve hanging at the rest
pose (`control.shared_start_pose`). A chooser compares the critic's value
for "right arm takes it" and "left arm takes it" on the robot's own
observations, and locks the arm once one leads by `--margin` or at
`control.shared_go_max_s` (0.2 s). The other arm stays at rest, and both
return to rest after the hit. Racket/forearm contact between the arms is
penalised and ends the episode.

## Setup

```bash
uv sync --extra train          # Linux + NVIDIA GPU (CPU: --extra train-cpu, smoke tests only)
uv run scripts/build_scene.py  # regenerate scene/badminton.xml after editing scene/params.yaml
uv run python -c "import launcher; launcher.build_workspace()"   # both arms' reach -> scene/workspace_W.npz
```

The arm is `assets/pioneer_bimanual_arm/urdf/pioneer_bimanual_arm.urdf` (repo
root), the same URDF the Isaac Lab and hardware stacks use; its `<limit>` tags
cap the joint ranges. `scene/params.yaml` is the single source of truth for
everything else (shuttle aerodynamics, arm joint ranges within the URDF limits,
torque and speed limits, control gains, launcher bank, perception noise). Note
the speed cap: `control.target_velocity_max` is in the hardware `joint_command`
units, where the effective steady-state speed is
`(1 - low_pass_alpha) * velocity_max` (15%).

## Train

```bash
# 1. teacher: PPO on privileged state (true shuttle state + intercept point);
#    the active arm is drawn 50/50 each episode
uv run scripts/train_rl.py Mjlab-Badminton-Shared-Teacher --env.scene.num-envs 1024

# 2. student: distil the teacher into a policy on realistic perception
#    (EKF-tracked shuttle + trajectory prior). Symlink the teacher run dir
#    into logs/rsl_rl/badminton_shared_student/ (any name containing
#    "badminton_shared_teacher").
uv run scripts/train_rl.py Mjlab-Badminton-Shared-Student --env.scene.num-envs 1024

# 3. fine-tune the student with PPO on its own reward; the -Priv task's
#    critic sees privileged state
uv run scripts/student_to_ppo.py --task Mjlab-Badminton-Shared-Student-PPO-Priv \
    --student logs/rsl_rl/badminton_shared_student/<run>/model_1499.pt \
    --out logs/rsl_rl/badminton_shared_student_ppo_priv/init/model_0.pt
uv run scripts/train_rl.py Mjlab-Badminton-Shared-Student-PPO-Priv \
    --agent.resume True --agent.load-run init --agent.load-checkpoint model_0.pt

# 4. chooser: a privileged critic cannot run on the robot, so the arm is
#    picked by the critic of a Mjlab-Badminton-Shared-Student-PPO run
#    (critic on student observations), trained the same way as step 3
```

Runs log to Weights & Biases (project `mjlab`); checkpoints land in
`logs/rsl_rl/<experiment>/<timestamp>/`. rsl_rl opens a new timestamped
directory on every launch, including resumes.

## Evaluate / watch

```bash
M=../../../models
uv run scripts/eval_rl.py --task Mjlab-Badminton-Shared-Student-PPO-Priv \
    --checkpoint-file $M/badminton_shared_student_ppo/model_11996.pt \
    --critic-task Mjlab-Badminton-Shared-Student-PPO \
    --critic-checkpoint $M/badminton_shared_chooser/model_11997.pt \
    --episodes 8192        # hit rate, per-arm split, return quality, joint feasibility
MUJOCO_GL=egl uv run scripts/demo_quad.py <same task/checkpoint/critic args> \
    --record runs/demo.mp4 --no-web --supersample 2   # four-camera video
uv run scripts/plot_eval.py --label "<model>"         # figures from runs/eval_pstar_hits.npy:
uv run scripts/plot_eval_story.py --label "<model>"   # picks, hit chance, returns, face map,
                                                      # scorecard, strike zone, landing map,
                                                      # net view, rally -> runs/figs/
```

Baselines are checked in under `models/` (repo root):
`badminton_shared_student_ppo/model_11996.pt` with the chooser
`badminton_shared_chooser/model_11997.pt` scores 98.4% bank hits at margin
1.0, hits split 49/51 right/left, every lateral bin >= 97.1%, 89% of
returns clear the net. `--margin` defaults to 1.0.
The wrist (joint6, GL40) sits at its 0.73 Nm clamp in about half the
episodes and above its rated 0.25 Nm for ~81% of the swing. The scripted
baseline in `baseline/` drives the right arm only.

## Layout

```
aero.py, launcher.py, perception*.py, predictor.py, mjsim.py   physics, serve bank, EKF prior, plain-MuJoCo sim
baseline/        scripted receive (IK + min-jerk swing) used as the reference before RL
humanoid_badminton/ the mjlab task: env config, rewards, observations, action moderation, perception command
scripts/         build_scene, mesh_prep, train_rl, student_to_ppo, eval_rl, play_rl, plot_eval, plot_eval_story
scene/           params.yaml, generated badminton.xml, meshes
```

Policy I/O (shared student): 61 inputs at 50 Hz, all in the active arm's
frame (6 joint pos/vel, racket face pose, last action, EKF shuttle state +
8-point trajectory prior, EKF uncertainty) → 6 joint-position targets for
the active arm (mirrored for the left), rate-limited and low-passed like the
hardware path. The idle arm holds its rest pose, and after the hit both arms
return to rest.
