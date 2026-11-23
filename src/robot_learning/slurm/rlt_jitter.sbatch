#!/bin/bash
# How jerky the commanded motion is (rlt/jitter.py), per replanning setting and with inference-time smoothing.
# From the repo root:  sbatch src/robot_learning/slurm/rlt_jitter.sbatch   (BASE=, RLT=, VARIANTS=, REPORT= optional)
#SBATCH --job-name=rlt_jitter
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --gres=shard:20480,tmpdisk:10240
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j-%x.out
set -euo pipefail
source ~/venvs/pioneer/bin/activate
export MUJOCO_GL=egl
cd src/robot_learning
BASE=${BASE:-$HOME/outputs/pioneer1_peg_100}
RLT=${RLT:-$HOME/outputs/rlt_peg_100}
python -m humanoid_robot_learning.rlt.jitter --policy "$BASE/checkpoints/last/pretrained_model" --rlt "$RLT" \
  ${VARIANTS:+--variants $VARIANTS} ${REPORT:+--report $REPORT}
