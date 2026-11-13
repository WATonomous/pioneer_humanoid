#!/bin/bash
# One-time setup on a WATcloud login node (run from the repo root):  bash src/robot_learning/slurm/setup_watcloud.sh
# Makes ~/venvs/pioneer with the DiT trainer and the MuJoCo eval. Then `hf auth login` (the dataset is private).
set -e
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  module load StdEnv/2023 python/3.11
fi
python3 -m venv ~/venvs/pioneer
source ~/venvs/pioneer/bin/activate
pip install -U pip
pip install -e "src/robot_learning[dit,sim]" -e src/simulation/mujoco_scenes mujoco scipy
mkdir -p logs
echo "Done. Next: source ~/venvs/pioneer/bin/activate && hf auth login"
