# WATcloud jobs for the peg-insertion pipeline

One parameterised job script runs any sequence of the pipeline's CLI tools on a GPU node.
`stage_input.sbatch` is the one-off checksum-verified upload of a large input archive to NAS.

| Path | Holds |
|---|---|
| `/home/law_zou/humanoid-peg-insert/` | login-side `artifacts/code.tgz`, command files, Slurm logs |
| `/mnt/wato-drive/law_zou/humanoid-peg-insert/datasets/` | converted LeRobot datasets (read in place, never copied) |
| `/mnt/wato-drive/law_zou/humanoid-peg-insert/runs/<name>_<job>/` | one run: `code.tgz`, `commands.sh`, outputs, `logs/` |

Runs before this layout (`outputs/`, `corrections/`, `workspace/`, `flow/`) stay where they are;
[`docs/research/peg_insert_act_vs_flow.md`](../../docs/research/peg_insert_act_vs_flow.md) maps each one to its job.

## Submit

```bash
# 1. snapshot the code (from the repo root) and copy it with a command file to the login node
git archive --format=tar.gz -o /tmp/code.tgz HEAD
scp /tmp/code.tgz wato1:/home/law_zou/humanoid-peg-insert/artifacts/code.tgz
cat > /tmp/eval_heldout.sh <<'EOF'
D=$PERSIST/datasets
$PY -m humanoid_mujoco_scenes.peg_insert.eval --checkpoint $PERSIST/corrections/moved_hole_709670/run/training/checkpoints/347314/pretrained_model \
  --device cuda --layouts $PERSIST/workspace/independent_708647/run/heldout_layouts.json --n-action-steps 50 \
  --report $RUN/heldout.json --trace-dir $RUN/traces
EOF
scp /tmp/eval_heldout.sh wato1:/home/law_zou/humanoid-peg-insert/jobs/
# 2. submit (pick the GPU here) and follow the log
ssh wato1 '/opt/slurm/bin/sbatch --job-name=peg-eval /home/law_zou/humanoid-peg-insert/jobs/peg_insert_run.sbatch \
  eval_heldout /home/law_zou/humanoid-peg-insert/jobs/eval_heldout.sh'
ssh wato1 '/opt/slurm/bin/squeue -u law_zou'
ssh wato1 'tail -f /home/law_zou/humanoid-peg-insert/logs/peg-eval-JOBID.out'
```

Use `--nodelist=trpro-slurm2 --gres=gpu:rtx_4090:1,tmpdisk:65536` for the RTX 4090. The job
installs torch 2.7.1 (cu126), LeRobot `e670ac5daf9b76`, MuJoCo 3.15.0 and the three repo
packages into a venv on the node's `/tmp`, sets `MUJOCO_GL=egl`, and runs the command file with
`$PY`, `$RUN` (output directory), `$PERSIST` and `$REPO` defined. A command file is a few
lines per round; the tools resume from what already exists (`expert --output`, `dagger`,
trainer checkpoints), so a resubmitted job continues where it stopped.

Typical command files:

```bash
# continue ACT with corrections, then evaluate
$PY -m humanoid_robot_learning.lerobot_finetune --config_path=$CKPT/pretrained_model/train_config.json --resume=true \
  --dataset.root=$PERSIST/datasets/pioneer_peg_insert --extra-dataset=$PERSIST/runs/corr_r1_1/dataset \
  --phase-config=$REPO/src/robot_learning/config/peg_insert_phases.json --phase-metrics \
  --steps=174000 --batch_size=8 --save_freq=4000 --policy.device=cuda --output_dir=$RUN/training
# DAgger rounds until 20/20 on held-out seeds
$PY -m humanoid_mujoco_scenes.peg_insert.dagger --root $RUN --checkpoint $CKPT --base-dataset $PERSIST/datasets/pioneer_peg_insert \
  --trigger stall --corrections 10 --phase-config $REPO/src/robot_learning/config/peg_insert_phases.json --eval-seed 5000
```
