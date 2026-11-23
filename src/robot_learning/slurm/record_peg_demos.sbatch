#!/bin/bash
# Extend the synthetic peg_insert demos (zabsy/pioneer_8dof_peg_insert_synthetic, 40 takes) to TARGET takes, on a GPU
# node (MuJoCo EGL). From the repo root:  sbatch src/robot_learning/slurm/record_peg_demos.sbatch
# Needs `hf auth login` once (the source dataset is private). Rerun-safe: the probe log is reused and seeds already
# in the dataset are skipped, so a job that stops early picks up where it left off.
#
# 1. a local copy of the 40-take dataset (ROOT, in $HOME: it survives the job)
# 2. probe seeds FIRST_SEED.. without cameras (parallel), keep successful takes <= MAX_TAKE_S long. The first 40 used
#    30 s; 45 s also keeps takes where the operator misses the hole and corrects, the recovery the policy lacks.
# 3. record the kept seeds one at a time with cameras, appending to ROOT
#SBATCH --job-name=peg_demos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:4096,tmpdisk:10240
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j-%x.out
set -euo pipefail
source ~/venvs/pioneer/bin/activate
export MUJOCO_GL=egl
unset PEG_CLEARANCE                      # the demos use the scene default: 4 mm hole (2 mm a side)

SRC_REPO=${SRC_REPO:-zabsy/pioneer_8dof_peg_insert_synthetic}
ROOT=${ROOT:-$HOME/datasets/pioneer_8dof_peg_insert_synthetic_100}
TARGET=${TARGET:-100}
MAX_TAKE_S=${MAX_TAKE_S:-45}
FIRST_SEED=${FIRST_SEED:-72}             # the first 40 came from seeds 0..71
TRIES=${TRIES:-200}
SYN=src/teleop/pioneer_leader_arm_teleop/synthetic/synthetic_teleop.py
PROBE_LOG=$ROOT.probe.log

# MuJoCo must render on the GPU: on a CPU fallback (no NVIDIA GL libraries) each take takes ~15 min instead of seconds.
python - <<'EOF'
import mujoco
from OpenGL import GL
ctx = mujoco.GLContext(64, 64)
ctx.make_current()
renderer = GL.glGetString(GL.GL_RENDERER).decode()
print("GL_RENDERER:", renderer, flush=True)
ctx.free()
if any(s in renderer.lower() for s in ("llvmpipe", "softpipe", "swrast")):
    raise SystemExit("MuJoCo EGL renders on the CPU on this node: NVIDIA's GL libraries are missing")
EOF

if [ ! -f "$ROOT/meta/info.json" ]; then
  python -c "from huggingface_hub import snapshot_download as d; d('$SRC_REPO', repo_type='dataset', local_dir='$ROOT')"
fi
echo "[demos] $(grep -c . "$ROOT/meta/synthetic_takes.jsonl") takes in $ROOT"

if [ ! -s "$PROBE_LOG" ]; then
  # no --record: probe only. It tries all TRIES seeds and exits 1 when fewer than TRIES succeed, which is expected.
  python $SYN --scene peg_insert --seed "$FIRST_SEED" --num_episodes "$TRIES" --max_tries "$TRIES" \
    --workers "$SLURM_CPUS_PER_TASK" > "$PROBE_LOG.tmp" 2>&1 || true
  grep -q "^\[SYNTH\] seed" "$PROBE_LOG.tmp" || { cat "$PROBE_LOG.tmp"; echo "[demos] probe produced no takes"; exit 1; }
  mv "$PROBE_LOG.tmp" "$PROBE_LOG"
fi

SEEDS=$(python - "$PROBE_LOG" "$ROOT/meta/synthetic_takes.jsonl" "$TARGET" "$MAX_TAKE_S" <<'EOF'
import json, re, sys
log, takes, target, max_s = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
done = {json.loads(l)["seed"] for l in open(takes) if l.strip()}
ok = [(int(m[1]), float(m[2])) for m in re.finditer(r"^\[SYNTH\] seed (\d+): ok\s+([\d.]+) s take", open(log).read(), re.M)]
pick = [s for s, t in ok if t <= max_s and s not in done][:max(0, target - len(done))]
print(f"[demos] probe: {len(ok)} ok, {sum(t <= max_s for _, t in ok)} <= {max_s} s; recording {len(pick)}", file=sys.stderr)
if len(done) + len(pick) < target:
    sys.exit(f"[demos] only {len(done) + len(pick)} of {target} takes available: raise TRIES or MAX_TAKE_S")
print(" ".join(map(str, pick)))
EOF
)

for s in $SEEDS; do
  n=$(( $(grep -c . "$ROOT/meta/synthetic_takes.jsonl") + 1 ))
  python $SYN --scene peg_insert --record --seed "$s" --max_tries 1 --num_episodes "$n" --dataset_root "$ROOT"
done
echo "[demos] done: $(grep -c . "$ROOT/meta/synthetic_takes.jsonl") takes in $ROOT"
