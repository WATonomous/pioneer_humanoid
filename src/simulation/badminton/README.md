# Badminton receive — stationary bimanual arm, RL

MuJoCo simulation of the WATonomous bimanual arm receiving badminton serves
with a racket in each hand, and the RL pipeline that trains the receive
policy (mjlab / MuJoCo Warp, rsl_rl PPO).

Each serve is assigned to one arm: the arm on the side (x relative to the
stand centre) where the predicted flight crosses the strike plane
`control.assign_strike_y`. The teacher uses the true flight; the student
uses its EKF estimate, which is fixed after `control.assign_latch_s`. The
other arm holds its ready pose, so both rackets never go for the same
shuttle. Racket/forearm contact between the arms is penalised and ends the
episode.

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
# 1. teacher: PPO on privileged state (true shuttle state + intercept point)
uv run scripts/train_rl.py Mjlab-Badminton-Receive-Teacher --env.scene.num-envs 1024

# 2. student: distil the teacher into a policy on realistic perception
#    (EKF-tracked shuttle + trajectory prior). Symlink the teacher run dir
#    into logs/rsl_rl/badminton_student/ (any name containing "badminton_teacher").
uv run scripts/train_rl.py Mjlab-Badminton-Receive-Student

# 3. fine-tune the student with PPO on its own reward (recovers what
#    distillation loses to perception noise; critic sees privileged state)
uv run scripts/student_to_ppo.py --student logs/rsl_rl/badminton_student/<run>/model_1499.pt \
    --out logs/rsl_rl/badminton_student_ppo/init/model_0.pt
uv run scripts/train_rl.py Mjlab-Badminton-Receive-Student-PPO \
    --agent.resume True --agent.load-run init --agent.load-checkpoint model_0.pt
```

Warm start from single-arm checkpoints: `scripts/widen_checkpoint.py` turns a
right-arm PPO checkpoint into a bimanual one (the actor becomes two
block-diagonal copies, right arm and mirrored left arm; see its docstring),
then resume from it:

```bash
uv run scripts/widen_checkpoint.py --group teacher \
    --old ../../../models/badminton_teacher/model_5996.pt \
    --out logs/rsl_rl/badminton_teacher/init_bimanual/model_0.pt
uv run scripts/train_rl.py Mjlab-Badminton-Receive-Teacher \
    --agent.resume True --agent.load-run init_bimanual --agent.load-checkpoint model_0.pt
# student PPO: --group student, the student_ppo model, Mjlab-Badminton-Receive-Student-PPO
```

Runs log to Weights & Biases (project `mjlab`); checkpoints land in
`logs/rsl_rl/<experiment>/<timestamp>/`. rsl_rl opens a new timestamped
directory on every launch, including resumes.

## Evaluate / watch

```bash
uv run scripts/eval_rl.py --task Mjlab-Badminton-Receive-Student-PPO \
    --checkpoint-file <model.pt> --episodes 8192      # hit rate, return quality, joint feasibility
uv run scripts/play_rl.py Mjlab-Badminton-Receive-Student-PPO --viewer viser \
    --checkpoint-file <model.pt>                      # http://localhost:8080
```

Bimanual baselines are checked in under `models/` (repo root):
`models/badminton_bimanual_student_ppo/model_2999.pt` (97.8% bank hits, 66%
net clearance, hits split 49/51% right/left) and its teacher
`models/badminton_bimanual_teacher/model_2999.pt` (96.7%). Both were
warm-started from the single-arm baselines with `scripts/widen_checkpoint.py`
and fine-tuned for 3000 iterations.

Single-arm baselines are also checked in under `models/` (repo root):
`models/badminton_student_ppo/model_4997.pt` (98.8% bank hits, 78% net
clearance) and its teacher `models/badminton_teacher/model_5996.pt`. They
run on the bimanual env only after `scripts/widen_checkpoint.py`. The
scripted baseline in `baseline/` drives the right arm only.

## Layout

```
aero.py, launcher.py, perception*.py, predictor.py, mjsim.py   physics, serve bank, EKF prior, plain-MuJoCo sim
baseline/        scripted receive (IK + min-jerk swing) used as the reference before RL
humanoid_badminton/ the mjlab task: env config, rewards, observations, action moderation, perception command
scripts/         build_scene, mesh_prep, train_rl, student_to_ppo, eval_rl, play_rl
scene/           params.yaml, generated badminton.xml, meshes
```

Policy I/O: 87 inputs at 50 Hz (12 joint pos/vel, both racket face poses,
last action, arm-assignment one-hot, EKF shuttle state + 8-point trajectory
prior, EKF uncertainty) → 12 joint-position targets (right arm first),
rate-limited and low-passed like the hardware path; the unassigned arm's
targets are replaced by its ready pose, and after the hit both arms return
to their rest poses.
