#!/bin/bash
# Run a python script in the garment-fold Isaac container (same mounts/caches as watod's simulation_isaac_garment).
#   isaac_run.sh NAME script.py [args...]      (script in this folder; image: docker build -f docker/simulation/isaac_lab/isaac_lab_garment.Dockerfile --build-arg TAG=main -t humanoid_isaac_garment:local docker/simulation/isaac_lab)
# Stop gracefully with: docker stop -t 120 NAME   (SIGTERM first; never kill -9 Isaac)
set -e
NAME=$1; shift
SCRIPT=$1; shift
C=${ISAAC_SIM_CACHE_DIR:-$HOME/docker/isaac-sim}
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
LEHOME=${LEHOME_ROOT:-$HOME/Desktop/lehome_ref}   # fetch.py downloads LeHome data/assets here
docker run --rm --name "$NAME" --gpus all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e FABRIC=${FABRIC:-0} -e CLOSE_FRAMES=${CLOSE_FRAMES:-10} -e CLOTH_CFG=${CLOTH_CFG:-} \
  -e HUMANOID_ROOT=/workspace/humanoid -e HOST_UID=1000 -e HOST_GID=1000 -e OWNERSHIP_FIX_INTERVAL=100000 \
  -v "$REPO":/workspace/humanoid:rw \
  -v "$LEHOME":/lehome:rw \
  -v "$REPO"/docker/simulation/isaac_lab/entrypoint.sh:/entrypoint.sh:ro \
  -v $C/cache/kit:/isaac-sim/kit/cache:rw -v $C/cache/ov:/root/.cache/ov:rw -v $C/cache/pip:/root/.cache/pip:rw \
  -v $C/cache/glcache:/root/.cache/nvidia/GLCache:rw -v $C/cache/computecache:/root/.nv/ComputeCache:rw \
  -v $C/logs:/root/.nvidia-omniverse/logs:rw -v $C/data:/root/.local/share/ov/data:rw \
  --entrypoint /entrypoint.sh humanoid_isaac_garment:local \
  bash -c "cd /workspace/humanoid && /workspace/isaaclab/_isaac_sim/python.sh -m pip install -q -e src/pioneer_humanoid -e src/simulation/garment_fold_task -e src/robot_learning --no-deps --no-build-isolation >/dev/null 2>&1; \
           cd /workspace/humanoid/src/simulation/garment_fold_task/lehome_retarget && /workspace/isaaclab/isaaclab.sh -p $SCRIPT $*; st=\$?; chown -R 1000:1000 /lehome /workspace/humanoid/src; exit \$st"
