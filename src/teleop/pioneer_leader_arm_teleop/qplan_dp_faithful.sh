#!/bin/bash
# Q-Planning on a frozen Diffusion Policy, following the paper's protocol (arXiv 2608.21204, Sec. 3 and Alg. 1) as far as
# this code allows: Q initialised on the BC demonstrations with the chunked Bellman target (EMA bootstrap from the next
# chunk), an "offline" evaluation before any rollout, then ITERS rounds of M planner rollouts + S Q-only gradient steps.
#
#   POLICY=<dp ckpt>/pretrained_model DEMOS=<demo dataset root (the one the policy was trained on)> OUT=<dir> ./qplan_dp_faithful.sh
#
# Still not the paper: Q reads the frozen policy's observation encoding (an MLP) instead of its own DINOv2 + T5 encoders.
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
N=${N:-32}; M=${M:-100}; S=${S:-200}; ITERS=${ITERS:-10}; INIT_STEPS=${INIT_STEPS:-5000}
PY="${PYTHON:-python} qplan_dp.py"
mkdir -p "$OUT"
res() { grep 'QDP\] peg' | sed 's/.*success/success/' | cut -c1-30; }
ev() {  # ev <label> <collect args...>: 200 layouts (1000-1199) in two processes
  for s in 1000 1100; do echo "$(date +%T) $1, layouts $s-$((s + 99)): $($PY collect --policy "$POLICY" --out "$OUT/eval_${1// /_}_$s.pt" --seed $s --episodes 100 "${@:2}" 2>&1 | res)" | tee -a "$OUT/status"; done
}
$PY demos --policy "$POLICY" --dataset_root "$DEMOS" --out "$OUT/demos.pt" ${VIDEO_BACKEND:+--video_backend $VIDEO_BACKEND} | tail -1
$PY train --data "$OUT/demos.pt" --out "$OUT/q0.pt" --td --steps $INIT_STEPS | tail -1
ev "policy alone" --select sample --n 1
ev "offline Q" --select qavg --q "$OUT/q0.pt" --n $N
for i in $(seq 1 $ITERS); do
  $PY collect --policy "$POLICY" --out "$OUT/r$i.pt" --seed $((10000 + 100 * i)) --episodes $M --select qavg --q "$OUT/q$((i - 1)).pt" --n $N > "$OUT/r$i.log" 2>&1
  echo "$(date +%T) round $i planner rollouts (layouts $((10000 + 100 * i))+): $(res < "$OUT/r$i.log")" | tee -a "$OUT/status"
  $PY train --data "$OUT/demos.pt" "$OUT/r*.pt" --out "$OUT/q$i.pt" --init "$OUT/q$((i - 1)).pt" --td --steps $S --lr 1e-4 | tail -1
  if [ $((i % 2)) -eq 0 ] || [ $i -eq $ITERS ]; then ev "round $i Q" --select qavg --q "$OUT/q$i.pt" --n $N; fi
done
