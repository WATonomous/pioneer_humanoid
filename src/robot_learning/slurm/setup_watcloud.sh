#!/bin/bash
# One-time setup on a WATcloud login node (run from the repo root):  bash src/robot_learning/slurm/setup_watcloud.sh
# Makes ~/venvs/pioneer with the Pioneer-1 trainer and the MuJoCo eval. Then `hf auth login` (private dataset, gated DINOv3).
set -e
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  module load StdEnv/2023 python/3.11
fi
python3 -m venv ~/venvs/pioneer
source ~/venvs/pioneer/bin/activate
pip install -U pip
pip install -e "src/robot_learning[pioneer1,sim]" -e src/simulation/mujoco_scenes -e src/pioneer_humanoid mujoco scipy
mkdir -p logs
echo "Done. Next: source ~/venvs/pioneer/bin/activate && hf auth login"
