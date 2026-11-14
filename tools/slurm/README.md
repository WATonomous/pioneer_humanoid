# WATcloud jobs

One parameterised script per job type. Each job unpacks a code snapshot on the node, builds a venv there
([`node_env.sh`](node_env.sh): Python 3.12, torch 2.7.1 cu126, LeRobot 0.7 with pi0.5, MuJoCo, the repo packages; wheels
cached on NAS) and writes its outputs, the snapshot and the exact command to `$PERSIST/runs/<name>_<jobid>/`.

| Script | Job |
|---|---|
| [`train.sbatch`](train.sbatch) | Train a LeRobot policy (`python -m humanoid_robot_learning.train`), optionally on one arm of a bimanual dataset |
| [`eval.sbatch`](eval.sbatch) | Evaluate a training run's checkpoints in sim (`sim.evaluate`), recording the rollouts with their RECAP sidecar |
| [`rollouts.sbatch`](rollouts.sbatch) | RECAP rollouts as a job array: each task records one seed range with `sim.evaluate --record` (policy on its GPU) |

| Path (defaults; override with `PERSIST`, `LOGIN`) | Holds |
|---|---|
| `/home/$USER/humanoid/` (login node) | `artifacts/code.tgz`, Slurm logs in `logs/` |
| `/mnt/wato-drive/$USER/humanoid/datasets/` | recorded datasets, read in place |
| `/mnt/wato-drive/$USER/humanoid/runs/<name>_<jobid>/` | one run: `code.tgz`, `command.txt`, `train/`, `logs/pipeline.log`, `logs/exit_status` |
| `/mnt/wato-drive/$USER/humanoid/cache/` | uv wheels and Python, Hugging Face models (pi05_base) |

## Submit

```bash
# once: the log folder Slurm writes to
ssh wato1 'mkdir -p ~/humanoid/logs ~/humanoid/artifacts'
# snapshot the code (from the repo root) and copy it, with the dataset, to WATcloud
git archive --format=tar.gz -o /tmp/code.tgz HEAD
scp /tmp/code.tgz wato1:humanoid/artifacts/code.tgz
rsync -a datasets/drawer_stow_train_v1/sim/ wato1:/mnt/wato-drive/$USER/humanoid/datasets/drawer_stow_train_v1/
```

pi0.5 (action expert only, the VLM frozen, relative actions) on a dataset whose action stats were recomputed as
relative (`lerobot-edit-dataset --operation.type recompute_stats --operation.relative_action true`):

```bash
ssh wato1 'mkdir -p ~/humanoid/jobs' && scp tools/slurm/train.sbatch wato1:humanoid/jobs/
ssh wato1 '/opt/slurm/bin/sbatch --job-name=pi05-drawer --nodelist=trpro-slurm2 --mem=48G --gres=gpu:rtx_4090:1,tmpdisk:65536 \
  ~/humanoid/jobs/train.sbatch pi05_drawer /mnt/wato-drive/$USER/humanoid/datasets/drawer_stow_train_v1 \
  --policy.path=lerobot/pi05_base --policy.train_expert_only=true --policy.use_relative_actions=true \
  --policy.gradient_checkpointing=true --policy.push_to_hub=false --policy.device=cuda \
  --steps=10000 --batch_size=4 --save_freq=2000 --trackio.enable=false'
ssh wato1 '/opt/slurm/bin/squeue -u $USER'
```

pi0.5's text tokenizer (`google/paligemma-3b-pt-224`) is gated on Hugging Face. Without a token on the cluster, copy
the tokenizer files (no weights) from a machine that has them and add
`--policy.text_tokenizer_name=/mnt/wato-drive/$USER/humanoid/models/paligemma-3b-pt-224-tokenizer`.
Check capacity first (`ssh wato1 /opt/slurm/bin/sinfo`); pick a GPU with `--nodelist` and `--gres`. pi0.5 in fp32
with the VLM frozen is close to a 24 GB 3090's limit: if it runs out of memory, lower `--batch_size`.

RECAP rollouts, 8 tasks x 10 episodes (seeds 1000..1079), then merge the shards (CPU only, no GPU needed):

```bash
scp tools/slurm/rollouts.sbatch wato1:humanoid/jobs/
ssh wato1 '/opt/slurm/bin/sbatch --array=0-7 --job-name=rollouts-drawer --gres=gpu:1,tmpdisk:65536 \
  ~/humanoid/jobs/rollouts.sbatch drawer_r1 10 1000 1 --scene drawer_stow \
  --policy /mnt/wato-drive/$USER/humanoid/runs/<train run>/train/checkpoints/last/pretrained_model'
# when every task wrote <run>/logs/exit_<i> = "exit 0": rsync the shard_* folders and merge them
humanoid-merge-takes <run>/shard_* --output datasets/recap/drawer_r1_rollouts
```

Each task writes `<run>/shard_<i>` (a dataset with its RECAP sidecar), `logs/task_<i>.log` and `logs/exit_<i>`; the
run folder is named with the array's job id. Value function and labels: [robot_learning README](../../src/robot_learning/README.md#recap-value-function-and-advantage-labels).

Evaluate chosen checkpoints of a run once it has trained (one GPU job, checkpoints in turn; `STEPS` picks them, default
all), each recorded as `<run>/rollouts_<step>`; `summary.txt` collects the success lines:

```bash
scp tools/slurm/eval.sbatch wato1:humanoid/jobs/
ssh wato1 'STEPS="006000 010000" /opt/slurm/bin/sbatch --job-name=eval-pi05 --dependency=afterok:<train job> \
  ~/humanoid/jobs/eval.sbatch eval_pi05 /mnt/wato-drive/$USER/humanoid/runs/<train run>/train/checkpoints \
  --scene drawer_stow --episodes 40 --seed 1000 --max_s 90'
```

More on `humanoid_robot_learning.sim.evaluate`: [robot_learning README](../../src/robot_learning/README.md).
