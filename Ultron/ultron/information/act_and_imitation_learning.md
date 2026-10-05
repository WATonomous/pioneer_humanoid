# ACT & Imitation Learning Training Guide

This guide details the exact procedures, configuration files, and commands for training Action Chunking with Transformers (ACT) and LeRobot / GR00T policies in IsaacLab.

---

## 1. SO101 & LeRobot ACT Workflow (Primary)

### Codebase & Task Locations:
- Task Directory: `/workspace/humanoid/src/simulation/so101_vial_task`
- Core Modules: `/workspace/humanoid/src/il` (`humanoid_il`), `/workspace/isaaclab/lerobot`
- Python Environment: IsaacLab Python `/workspace/isaaclab/isaaclab.sh -p`

### Training Command (LeRobot ACT):
Training uses LeRobot's train runner (`il-train` or `python -m lerobot.scripts.lerobot_train`) with remote Hugging Face dataset IDs (e.g. `CursedRock17/so101_teleop_vials_sim_and_real`):
```bash
PYTHONPATH=/workspace/isaaclab/lerobot/src:/workspace/humanoid/src/il:$PYTHONPATH \
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=CursedRock17/so101_teleop_vials_sim_and_real \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=/workspace/humanoid/outputs/train/TestVLAModel \
  --policy.device=cuda \
  --steps=10000 \
  --batch_size=8 \
  --save_freq=5000 \
  --job_name=so101_act_train
```
*Note: Checkpoints are saved to `{output_dir}/checkpoints/last/pretrained_model`.*

### Evaluation Command (In Simulation):
To evaluate policy checkpoints in IsaacLab simulation:
```bash
cd /workspace/humanoid/src/simulation/so101_vial_task
PYTHONPATH=/workspace/humanoid/src/il:$(pwd):$PYTHONPATH \
/workspace/isaaclab/isaaclab.sh -p scripts/lerobot_eval.py \
  --task Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval \
  --policy_type lerobot \
  --policy_path /workspace/humanoid/outputs/train/TestVLAModel/checkpoints/last/pretrained_model \
  --num_episodes 2 \
  --headless
```
*Note: Always include `/workspace/humanoid/src/il` in PYTHONPATH so `humanoid_il` imports successfully.*

---

## 2. Legacy / Isaac-GR00T Working Directory
- Host: `/home/rijul_chaddha/IsaacLab/ImitationLearning`
- Container: `/workspace/isaaclab/ImitationLearning`
- Checkpoints: `/workspace/isaaclab/ImitationLearning/checkpoints`

---

## 2. Dependencies Installation Inside Container

Before training, ensure required Python packages are installed in the container:
```bash
/isaac-sim/kit/python/bin/python3 -m pip install \
    tyro filelock packaging pydantic mpmath psutil python-dateutil pytz Pillow wheel huggingface_hub
```

For GR00T:
```bash
cd /workspace/Isaac-GR00T && /isaac-sim/kit/python/bin/python3 -m pip install -e .
```

---

## 3. Training Commands (In Order)

### Standard Imitation Learning / Transformer Policy:
```bash
cd /workspace/Isaac-GR00T

export DS_BUILD_OPS=0
export DEEPSPEED_SKIP_CUDA_CHECK=1
export CUDA_HOME=/isaac-sim/kit/python/lib/python3.10/site-packages/nvidia/cuda_runtime
export DS_SKIP_CUDA_CHECK=1

/isaac-sim/kit/python/bin/python3 gr00t/experiment/launch_finetune.py \
    --experiment-dir "/workspace/isaaclab/ImitationLearning/checkpoints" \
    --dataset-path "/workspace/isaaclab/ImitationLearning/demonstrations/robomimic_dataset.hdf5" \
    --dataset-val-path "/workspace/isaaclab/ImitationLearning/demonstrations/robomimic_dataset.hdf5" \
    --model-type "transformer_policy" \
    --training-batch-size 64 \
    --training-learning-rate 1e-4 \
    --training-num-epochs 200
```

### Multi-GPU Fine-tuning (2x RTX 3090):
```bash
cd /workspace/Isaac-GR00T

export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

/isaac-sim/kit/python/bin/python3 -m torch.distributed.run --nproc_per_node=2 gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /workspace/isaaclab/ImitationLearning/demonstrations/lerobot_dataset \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path /workspace/Isaac-GR00T/gr00t/configs/data/custom_embodiment.py \
    --output-dir /workspace/isaaclab/ImitationLearning/checkpoints \
    --global-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 1e-4 \
    --max-steps 5000 \
    --dataloader-num-workers 4 \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --num-gpus 2
```

---

## 4. Checkpoints & Hugging Face Synchronization

### Pull Existing Weights:
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Ultrox9504/arm-model-weights',
    local_dir='/workspace/isaaclab/ImitationLearning/checkpoints',
    token=os.environ.get('HF_TOKEN')
)
```

### Push Updated Weights:
```python
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.upload_folder(
    folder_path='/workspace/isaaclab/ImitationLearning/checkpoints',
    repo_id='Ultrox9504/arm-model-weights',
    repo_type='model'
)
```
