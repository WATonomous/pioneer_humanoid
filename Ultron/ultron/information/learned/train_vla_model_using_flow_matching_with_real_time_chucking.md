---
key: train_vla_model_using_flow_matching_with_real_time_chucking
task: Train VLA model using flow matching with real-time chucking (RTC) on arm manipulation task for 5 minutes, then upload final checkpoint and config to HuggingFace model repo as TestVLA, also run the VLA model one time and then tell me on how to run it as well
successes: 1
failures_since: 1
last_verified: 2026-10-04
mode: new
node: trpro-slurm1
---
# Train VLA model using flow matching with real-time chucking (RTC) on arm manipulation task for 5 minutes, then upload final checkpoint and config to HuggingFace model repo as TestVLA, also run the VLA model one time and then tell me on how to run it as well

## Run details
- Mode: new on node trpro-slurm1
- Resources: time=10:00:00, gpus=2, cpus=8, mem=64G, tmpdisk_mb=102400, partition=compute, gpu_type=None, shard=None, node=None
- Duration: 20 min
- Working directory: /home/rijul_chaddha/IsaacLab/FallRepo/Ultron

## Working commands
The sequence that succeeded, in order (failed attempts are left out).

1. **Start dockerd**
```bash
export DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/mnt/wato-drive2/rijul_chaddha/docker_data}"; export DOCKER_HOST=unix:///tmp/run/docker.sock; slurm-start-dockerd.sh > /tmp/ultron_dockerd.log 2>&1 & for i in $(seq 1 36); do docker info > /dev/null 2>&1 && break; sleep 5; done; tail -n 15 /tmp/ultron_dockerd.log; docker info --format 'dockerd up, root={{.DockerRootDir}}'
```

2. **Check host GPU**
```bash
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

3. **Build/start isaac-lab-ros2 container**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; cd /home/rijul_chaddha/IsaacLab && ./docker/container.py start ros2 > /tmp/ultron_container_start.log 2>&1; RC=$?; tail -n 25 /tmp/ultron_container_start.log; exit $RC
```

4. **Verify GPU inside container**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; docker exec isaac-lab-ros2 nvidia-smi -L
```

5. **Ensure humanoid + lerobot repos in container**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; C=isaac-lab-ros2; if ! docker exec $C test -e /workspace/humanoid; then if docker exec $C test -d /workspace/isaaclab/FallRepo/humanoid; then docker exec $C ln -s /workspace/isaaclab/FallRepo/humanoid /workspace/humanoid; else docker cp /home/rijul_chaddha/IsaacLab/FallRepo/humanoid $C:/workspace/humanoid; fi; fi; if ! docker exec $C test -d /workspace/isaaclab/lerobot/src/lerobot; then docker exec $C mkdir -p /workspace/isaaclab; docker cp /home/rijul_chaddha/IsaacLab/lerobot $C:/workspace/isaaclab/lerobot; fi; docker exec $C ls -d /workspace/humanoid/src/il /workspace/humanoid/src/simulation/so101_vial_task/scripts /workspace/isaaclab/lerobot/src/lerobot /workspace/isaaclab/lerobot/examples/rtc/eval_dataset.py
```

6. **Discover RTC eval flags and task layout**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; docker exec isaac-lab-ros2 bash -c 'cd /workspace/isaaclab/lerobot; grep -nE "^version|requires-python|^(smolvla|dataset|av-dep) *=" pyproject.toml; echo "--- eval_dataset.py config fields:"; grep -nE "^class |^    [a-z_]+: .*=|output_dir|policy_path|draccus|parser.wrap" examples/rtc/eval_dataset.py | head -n 60; echo "--- so101 task scripts:"; ls /workspace/humanoid/src/simulation/so101_vial_task /workspace/humanoid/src/simulation/so101_vial_task/scripts; echo "--- humanoid_so101_vial_task package location:"; find /workspace/humanoid/src/simulation/so101_vial_task/ -maxdepth 4 -type d -name humanoid_so101_vial_task; true'
```

7. **Create py3.12 venv, install lerobot[smolvla,dataset]**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; docker exec isaac-lab-ros2 bash -c 'set -eo pipefail; export PATH=/root/.local/bin:$HOME/.local/bin:$PATH; if ! command -v uv > /dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi; uv --version; if [ ! -x /opt/vla_env/bin/python ]; then uv venv --python 3.12 /opt/vla_env; fi; uv pip install --python /opt/vla_env/bin/python --link-mode=copy "/workspace/isaaclab/lerobot[smolvla,dataset]" "av>=15.0.0,<16.0.0" "datasets>=4.0.0,<5.0.0" matplotlib huggingface_hub 2>&1 | tail -n 25; uv pip list --python /opt/vla_env/bin/python 2>/dev/null | grep -iE "^(lerobot|torch|torchvision|transformers|datasets|av|accelerate|num2words|numpy) "'
```

8. **Validate env, CUDA and SmolVLA/RTC config**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; docker exec -w /tmp -e PYTHONPATH=/workspace/isaaclab/lerobot/src:/workspace/humanoid/src/il isaac-lab-ros2 /opt/vla_env/bin/python -c "import torch, lerobot, datasets, av; from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig; from lerobot.policies.rtc.configuration_rtc import RTCConfig; c = SmolVLAConfig(); print('lerobot', lerobot.__version__, lerobot.__file__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'datasets', datasets.__version__, 'av', av.__version__); print('gpu', torch.cuda.get_device_name(0)); print('chunk_size', c.chunk_size, 'n_action_steps', c.n_action_steps, 'load_vlm_weights', c.load_vlm_weights, 'rtc_config', c.rtc_config); print('RTCConfig', RTCConfig()); assert torch.cuda.is_available()"
```

9. **Check HF auth and dataset access**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; docker exec -w /tmp -e HF_TOKEN isaac-lab-ros2 /opt/vla_env/bin/python -c "from huggingface_hub import HfApi; a = HfApi(); print('HF user:', a.whoami()['name']); i = a.dataset_info('CursedRock17/so101_teleop_vials_sim_and_real'); print('dataset ok:', i.id, 'files:', len(i.siblings))"
```

10. **Write 5-minute training script**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; docker exec -i isaac-lab-ros2 bash -c 'cat > /tmp/vla_train.sh' << 'EOF'
#!/bin/bash
# Train SmolVLA (flow matching) for 5 minutes of wall-clock time, then stop at the next checkpoint.
OUT=/workspace/humanoid/outputs/train/TestVLA
LOG=/tmp/vla_train.log
export PYTHONPATH=/workspace/isaaclab/lerobot/src:/workspace/humanoid/src/il
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
if [ -e "$OUT" ]; then mv "$OUT" "${OUT}_prev_$(date +%s)"; fi
mkdir -p "$(dirname "$OUT")"
cd /workspace/isaaclab/lerobot
/opt/vla_env/bin/python -m lerobot.scripts.lerobot_train --dataset.repo_id=CursedRock17/so101_teleop_vials_sim_and_real --policy.type=smolvla --policy.load_vlm_weights=true --policy.push_to_hub=false --policy.device=cuda --output_dir="$OUT" --steps=200000 --batch_size=8 --num_workers=4 --save_freq=100 --log_freq=10 --wandb.enable=false --job_name=so101_smolvla_rtc > "$LOG" 2>&1 &
PID=$!
# wait (up to 60 min: first run downloads the dataset and the VLM weights) for the first logged step
for i in $(seq 1 720); do
  grep -q "step:" "$LOG" && break
  kill -0 $PID 2>/dev/null || break
  if [ $((i % 12)) -eq 0 ]; then echo "[wait ${i}x5s] $(tail -c 200 "$LOG" | tr '\r\n' '  ' | tail -c 160)"; fi
  sleep 5
done
if ! grep -q "step:" "$LOG"; then
  echo "Training did not reach the first logged step. Log tail:"
  tail -n 60 "$LOG"
  kill -TERM $PID 2>/dev/null
  exit 1
fi
echo "First training step logged at $(date -u +%H:%M:%S); training for 300 s"
sleep 300
BEFORE=$(readlink "$OUT/checkpoints/last" 2>/dev/null)
for i in $(seq 1 120); do
  NOW=$(readlink "$OUT/checkpoints/last" 2>/dev/null)
  if [ -n "$NOW" ] && [ "$NOW" != "$BEFORE" ]; then break; fi
  kill -0 $PID 2>/dev/null || break
  sleep 5
done
sleep 5
kill -TERM $PID 2>/dev/null
sleep 10
pkill -KILL -f lerobot.scripts.lerobot_train 2>/dev/null
echo "Training stopped at $(date -u +%H:%M:%S). Last logged steps:"
grep "step:" "$LOG" | tail -n 5
ls -la "$OUT/checkpoints/"
ls -la "$OUT/checkpoints/last/pretrained_model/"
if ! test -f "$OUT/checkpoints/last/pretrained_model/model.safetensors"; then echo "No checkpoint was written. Log tail:"; tail -n 40 "$LOG"; exit 1; fi
test -f "$OUT/checkpoints/last/pretrained_model/config.json"
EOF
```

11. **Train SmolVLA flow-matching policy (5 min)**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; docker exec -e HF_TOKEN isaac-lab-ros2 bash /tmp/vla_train.sh
```

12. **Run the VLA once with RTC (dataset inference)**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; docker exec -e HF_TOKEN -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/workspace/isaaclab/lerobot/src:/workspace/humanoid/src/il isaac-lab-ros2 bash -c 'set -o pipefail; mkdir -p /workspace/humanoid/outputs/train/TestVLA_rtc_run && cd /workspace/humanoid/outputs/train/TestVLA_rtc_run && /opt/vla_env/bin/python /workspace/isaaclab/lerobot/examples/rtc/eval_dataset.py --policy.path=/workspace/humanoid/outputs/train/TestVLA/checkpoints/last/pretrained_model --dataset.repo_id=CursedRock17/so101_teleop_vials_sim_and_real --rtc.execution_horizon=8 --rtc.max_guidance_weight=10.0 --rtc.prefix_attention_schedule=EXP --device=cuda --seed=10 2>&1 | tee /tmp/vla_rtc_run.log | tail -n 40; echo "--- files produced:"; find . -type f | head -n 30'
```

13. **Sim RTC rollout, 1 episode (best-effort)**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; docker exec -e HF_TOKEN -w /workspace/humanoid/src/simulation/so101_vial_task isaac-lab-ros2 bash -c 'T=/workspace/humanoid/src/simulation/so101_vial_task; PKG=$(find $T/ -maxdepth 4 -type d -name humanoid_so101_vial_task | head -n 1); export PYTHONPATH=/workspace/humanoid/src/il:$T:${PKG:+$(dirname $PKG)}; timeout 1200 /workspace/isaaclab/isaaclab.sh -p scripts/lerobot_eval_rtc.py --task Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval --policy_path /workspace/humanoid/outputs/train/TestVLA/checkpoints/last/pretrained_model --num_episodes 1 --execution_horizon 10 --headless > /tmp/vla_sim_eval.log 2>&1; RC=$?; echo "exit code $RC" > /tmp/vla_sim_eval.status; grep -q "most recent call last" /tmp/vla_sim_eval.log && echo "exit code $RC, Python exception: $(grep -E "^[A-Za-z]+Error" /tmp/vla_sim_eval.log | tail -n 1 | cut -c1-150)" > /tmp/vla_sim_eval.status; echo "Best-effort sim rollout result: $(cat /tmp/vla_sim_eval.status) (a failure is expected if lerobot 0.6.2 cannot be imported under Isaac Python 3.10; full log in /tmp/vla_sim_eval.log)"; true'
```

14. **Stage checkpoint, config, logs and README**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; docker exec -i -w /tmp -e HF_TOKEN -e SLURM_JOB_ID isaac-lab-ros2 bash -s << 'EOF'
set -e
OUT=/workspace/humanoid/outputs/train/TestVLA
STAGE=/tmp/TestVLA_upload
mkdir -p $STAGE/logs
cp -rL $OUT/checkpoints/last/pretrained_model/. $STAGE/
cp /tmp/vla_train.log $STAGE/logs/train.log
if [ -f /tmp/vla_rtc_run.log ]; then cp /tmp/vla_rtc_run.log $STAGE/logs/rtc_inference.log; fi
if [ -f /tmp/vla_sim_eval.log ]; then cp /tmp/vla_sim_eval.log $STAGE/logs/sim_eval.log; fi
if [ -d /workspace/humanoid/outputs/train/TestVLA_rtc_run ]; then mkdir -p $STAGE/rtc_run; cp -r /workspace/humanoid/outputs/train/TestVLA_rtc_run/. $STAGE/rtc_run/; fi
USERNAME=$(/opt/vla_env/bin/python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])")
CKPT=$(basename "$(readlink -f $OUT/checkpoints/last)")
LASTSTEP=$(grep -o "step:[0-9A-Za-z.]*" /tmp/vla_train.log | tail -n 1 | cut -d: -f2)
SIM=$(cat /tmp/vla_sim_eval.status 2>/dev/null || echo "not run")
if [ -s /tmp/vla_rtc_run.log ] && ! grep -q Traceback /tmp/vla_rtc_run.log; then RTC="completed without errors"; else RTC="did not complete cleanly"; fi
cat > $STAGE/README.md << 'MD'
---
library_name: lerobot
tags:
- robotics
- smolvla
- flow-matching
- real-time-chunking
- so101
---
# TestVLA: SmolVLA (flow matching) with Real-Time Chunking on the SO101 vial task

Short smoke-training run produced by an Ultron SLURM job. It is a pipeline test, not a converged policy.

| Item | Value |
|---|---|
| Policy | LeRobot SmolVLA (VLM backbone + flow-matching action expert), lerobot 0.6.2 |
| Dataset | CursedRock17/so101_teleop_vials_sim_and_real |
| Training budget | 5 minutes wall-clock, batch size 8, one GPU |
| Uploaded checkpoint | step __CKPT__ (last logged step: __LASTSTEP__) |
| SLURM job | __JOB__ |
| Date (UTC) | __DATE__ |
| RTC run on dataset samples | __RTC__, see logs/rtc_inference.log and rtc_run/ |
| Isaac Lab sim RTC rollout (best-effort, Isaac Python 3.10) | __SIM__, see logs/sim_eval.log |

Real-time chunking (RTC) is an inference-time technique for flow-matching policies: while the robot is still executing the previous action chunk, the next chunk is generated and guided to stay consistent with the actions that are already committed. The weights are trained normally and RTC is switched on when the policy is run.

## Files
- model.safetensors, config.json, train_config.json and processor files: the LeRobot pretrained_model directory
- logs/train.log: full training log
- logs/rtc_inference.log, rtc_run/: output of the RTC inference run
- logs/sim_eval.log: output of the simulation rollout attempt

## How to run it

lerobot 0.6.2 needs Python 3.12 or newer, so use a Python 3.12 environment and not the Isaac Sim Python 3.10. Inside the isaac-lab-ros2 container the job built one like this:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=/root/.local/bin:$PATH
uv venv --python 3.12 /opt/vla_env
uv pip install --python /opt/vla_env/bin/python "/workspace/isaaclab/lerobot[smolvla,dataset]" "av>=15.0.0,<16.0.0" matplotlib huggingface_hub
```

### 1. Download
```bash
hf download __REPO__ --local-dir ./TestVLA
```
(The repo is private, so HF_TOKEN must be set. On older huggingface_hub versions the command is `huggingface-cli download`.)

### 2. Run once with RTC on dataset samples (this is what the job ran)
```bash
export PYTHONPATH=/workspace/isaaclab/lerobot/src:/workspace/humanoid/src/il
/opt/vla_env/bin/python /workspace/isaaclab/lerobot/examples/rtc/eval_dataset.py \
  --policy.path=./TestVLA \
  --dataset.repo_id=CursedRock17/so101_teleop_vials_sim_and_real \
  --rtc.execution_horizon=8 --rtc.max_guidance_weight=10.0 \
  --rtc.prefix_attention_schedule=EXP --device=cuda --seed=10
```
It takes two dataset samples, uses the actions of the first as the previous chunk, and generates actions for the second with and without RTC.

### 3. Use RTC from Python
```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig

policy = SmolVLAPolicy.from_pretrained("__REPO__").to("cuda").eval()
policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=8, max_guidance_weight=10.0)
policy.init_rtc_processor()
# batch = preprocessed observation (state, camera images, task string)
# prev_chunk = the not-yet-executed actions of the previous chunk, inference_delay = control steps spent on inference
actions = policy.predict_action_chunk(batch, inference_delay=4, prev_chunk_left_over=prev_chunk)
```
This snippet is a sketch that was not executed by the job; examples/rtc/eval_dataset.py in the lerobot repo is the authoritative reference for the API.

### 4. Roll out in Isaac Lab simulation with RTC
```bash
cd /workspace/humanoid/src/simulation/so101_vial_task
PYTHONPATH=/workspace/humanoid/src/il:$(pwd):$PYTHONPATH \
/workspace/isaaclab/isaaclab.sh -p scripts/lerobot_eval_rtc.py \
  --task Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval \
  --policy_path /path/to/TestVLA \
  --num_episodes 1 --execution_horizon 10 --headless
```
The result of this command in the training job is the sim rollout row in the table above. If that row shows an error, the simulation path needs a lerobot build that runs under Isaac Sim's Python 3.10 (and the humanoid_so101_vial_task package on PYTHONPATH) before it will work.

### 5. Reproduce the training
```bash
cd /workspace/isaaclab/lerobot
PYTHONPATH=/workspace/isaaclab/lerobot/src:/workspace/humanoid/src/il \
/opt/vla_env/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=CursedRock17/so101_teleop_vials_sim_and_real \
  --policy.type=smolvla --policy.load_vlm_weights=true \
  --policy.push_to_hub=false --policy.device=cuda \
  --output_dir=/workspace/humanoid/outputs/train/TestVLA \
  --steps=200000 --batch_size=8 --num_workers=4 --save_freq=100 --log_freq=10 \
  --wandb.enable=false --job_name=so101_smolvla_rtc
```
The job stopped this command after 5 minutes of training, at the next checkpoint save.
MD
sed -i -e "s|__REPO__|$USERNAME/TestVLA|g" -e "s|__CKPT__|$CKPT|g" -e "s|__LASTSTEP__|$LASTSTEP|g" -e "s|__JOB__|${SLURM_JOB_ID:-unknown}|g" -e "s|__DATE__|$(date -u +%Y-%m-%d)|g" -e "s|__RTC__|$RTC|g" -e "s|__SIM__|$SIM|g" $STAGE/README.md
ls -la $STAGE $STAGE/logs
EOF
```

15. **Upload checkpoint + config to HF TestVLA**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; docker exec -i -w /tmp -e HF_TOKEN isaac-lab-ros2 /opt/vla_env/bin/python - << 'EOF'
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()  # reads HF_TOKEN from the environment
user = api.whoami()["name"]
repo_id = f"{user}/TestVLA"
stage = Path("/tmp/TestVLA_upload")
assert (stage / "model.safetensors").is_file() and (stage / "config.json").is_file(), "staging dir incomplete"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
api.upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="model", commit_message="Upload SmolVLA (flow matching + RTC) checkpoint, config, logs and README from Ultron job")
files = api.list_repo_files(repo_id, repo_type="model")
print("\n".join(sorted(files)))
assert "model.safetensors" in files and "config.json" in files, "upload verification failed"
print(f"Uploaded: https://huggingface.co/{repo_id}")
EOF
```

16. **Save and print run instructions**
```bash
export DOCKER_HOST=unix:///tmp/run/docker.sock; mkdir -p outputs && docker cp isaac-lab-ros2:/tmp/TestVLA_upload/README.md outputs/TestVLA_README.md && cat outputs/TestVLA_README.md
```
