# Ultron Information & Reference Library

This folder contains operational references and instructions that the Ultron AI agent loads dynamically to guide planning, execution, and troubleshooting.

## Available Guides (hand-written, highest authority):
- **`isaaclab_startup.md`**: Launching IsaacLab, Docker daemon startup on compute nodes, running commands inside `isaac-lab-ros2`, Isaac Sim paths, and environment flags.
- **`act_and_imitation_learning.md`**: Step-by-step ACT and Isaac-GR00T training instructions, dataset formatting, multi-GPU runs, and Hugging Face checkpointing.
- **`huggingface_storage.md`**: Creating repositories and uploading checkpoints, datasets and artifacts to Hugging Face.
- **`troubleshooting_and_fail_steps.md`**: Ordered recovery instructions for dockerd, container crashes, CUDA OOM, NCCL errors, missing nvcc, and Ollama connection failures.
- **`slurm_reference.md`**: Cluster partitions (`compute`, `compute_dense`), GPU naming, resource limits, and monitoring commands.

Any additional `.md` file placed into this directory is automatically included into the Ultron agent's prompt context during plan generation and self-healing.

## `learned/` (written automatically)
After a job is **verified successful**, Ultron writes what worked to `learned/<task>.md`: the working command
sequence, each error it hit with the fix that worked, and the lessons. Secrets are redacted before writing.
A later successful run of the same kind of task updates the entry; a later failed run flags it as possibly stale.
`learned/INDEX.md` lists every entry. The planner reads the index and opens the relevant entries; healing prompts
get the entries that match the failing command. Learned entries are lower authority than the guides above —
edit or delete any entry that is wrong.
