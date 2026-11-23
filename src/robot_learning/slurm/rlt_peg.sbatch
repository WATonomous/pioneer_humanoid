#!/bin/bash
# RLT (RL token + TD3) on a frozen Pioneer-1, peg_insert critical phase (humanoid_robot_learning/rlt). From the repo
# root:  sbatch src/robot_learning/slurm/rlt_peg.sbatch   (BASE= the Pioneer-1 training output dir, OUT= results)
# Hole: the scene default (4 mm, the task the demos show); PEG_CLEARANCE=0.008 widens it. Settings vs the first runs
# (Brev, 8 mm): TD target clipped to [0, 1] (actor_critic.py), 2 updates per transition instead of 5, exploration
# 0.02 instead of 0.01 (the actor moved 5-15x further than it explored), ~150 eval start states instead of 55. To
# reuse another run's start states and RL token (same eval set, comparable numbers), copy its starts.pkl and
# rl_token.pt into OUT first.
#SBATCH --job-name=rlt_peg
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --gres=shard:20480,tmpdisk:10240
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j-%x.out
set -euo pipefail
source ~/venvs/pioneer/bin/activate
export MUJOCO_GL=egl
cd src/robot_learning
BASE=${BASE:-$HOME/outputs/pioneer1_peg_100}
OUT=${OUT:-$HOME/outputs/rlt_peg_100}
python -m humanoid_robot_learning.rlt.train --policy "$BASE/checkpoints/last/pretrained_model" --out "$OUT" \
  --envs 8 --train_starts 240 --eval_starts 130 --prefix 2 --baselines ${BASELINES:-5:2,16:6} \
  --utd ${UTD:-2} --noise ${NOISE:-0.02} --beta ${BETA:-10} \
  --minutes ${MINUTES:-150} --eval_every ${EVAL_EVERY:-10000}
