# SLURM Cluster Reference Guide

This reference provides cluster parameters, partition configurations, and resource options for jobs submitted by Ultron.

---

## 1. Partitions & Hardware

| Partition | Time Limit | State | Default Nodes | Recommended For |
|---|---|---|---|---|
| `compute` | 1-00:00:00 (24h) | Mixed | `delta-slurm1`, `tr-slurm2`, `trpro-slurm[1-2]` | General training, short debug runs (<24h) |
| `compute_dense` | 7-00:00:00 (7d) | Mixed | `delta-slurm1`, `tr-slurm2`, `trpro-slurm[1-2]` | Long multi-day training runs |

---

## 2. Resource Request Specifications

- **Default User Account:** `uw-reality-lab`
- **Default GPU Allocation:**
  - Standard single GPU: `#SBATCH --gres=gpu:1`
  - Explicit RTX 3090: `#SBATCH --gres=gpu:rtx_3090:1`
  - Dual RTX 3090 with scratch tempdisk: `#SBATCH --gres=gpu:rtx_3090:2,tmpdisk:102400`
- **CPUs per Task:**
  - Typical: `#SBATCH --cpus-per-task=4` or `--cpus-per-task=8`
- **Memory:**
  - Typical: `#SBATCH --mem=32G` or `--mem=64G`

---

## 3. Monitoring & Job Control

- Check status of all your jobs:
  ```bash
  squeue -u $USER
  ```
- Cancel a job:
  ```bash
  scancel <job_id>
  ```
- Detailed job resource accounting:
  ```bash
  sacct -j <job_id> --format=JobID,JobName,Partition,AllocCPUS,Elapsed,State,ExitCode
  ```
- View node status:
  ```bash
  sinfo -p compute
  ```
