#!/bin/bash
# Evaluate one saved peg_insert policy on every scenario shift (scenarios.py): 50 held-out layouts (7000+) each,
# m = 0.6 as in bc.py. Writes results_8dof/bc_exec/<tag>__ensemble_m0.6_<scenario>.json.
#   ./scenario_eval.sh pioneer_v5_delta
TAG=$1; N=${2:-50}
for S in fog glare lens shadow clutter background; do
  f=results_8dof/bc_exec/${TAG}__ensemble_m0.6_${S}.json
  [ -e "$f" ] && { echo "skip $TAG $S (done)"; continue; }
  BENCH_ROBOT=8dof MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv/bin/python eval_policy.py --tag "$TAG" --ens_m 0.6 \
    --variant "$S" --episodes "$N" --eval_batch 10 2>&1 | grep -E "^\[|Traceback|Error" || true
done
miss=0
for S in fog glare lens shadow clutter background; do
  [ -e results_8dof/bc_exec/${TAG}__ensemble_m0.6_${S}.json ] || { echo "MISSING $TAG $S"; miss=1; }
done
exit $miss
