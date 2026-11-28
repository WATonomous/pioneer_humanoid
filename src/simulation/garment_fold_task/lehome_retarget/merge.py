"""Merge the per-garment LeRobot datasets of the retarget into one training dataset.

  python merge.py OUT_ROOT KIND_DIR [KIND_DIR ...]     (e.g. ~/Desktop/lehome_ref/out/top_short)

Datasets recorded before each garment type had its own instruction say "fold the garment"; those are relabeled to
their type's instruction (run_retarget.TASK)."""
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK = {"top_short": "fold the short-sleeve top", "top_long": "fold the long-sleeve top", "pant_short": "fold the shorts",
        "pant_long": "fold the trousers"}
out = Path(sys.argv[1]).expanduser()
tmp = Path(tempfile.mkdtemp())
parts = []
for kind_dir in map(Path, sys.argv[2:]):
    for d in sorted((kind_dir / "dataset").iterdir()):
        if not (d / "meta" / "episodes").is_dir():
            print("skip (no saved episodes):", d)
            continue
        c = tmp / f"{kind_dir.name}_{d.name}"
        shutil.copytree(d, c)
        tasks = pd.read_parquet(c / "meta" / "tasks.parquet")
        if list(tasks.index) == ["fold the garment"]:
            tasks.index = [TASK[kind_dir.name]]
            tasks.index.name = "task"
            tasks.to_parquet(c / "meta" / "tasks.parquet")
        parts.append(LeRobotDataset(repo_id=f"humanoid/{c.name}", root=c))
        print(c.name, parts[-1].num_episodes, "episodes", list(tasks.index))
merged = merge_datasets(parts, output_repo_id="humanoid/pioneer_8dof_lehome_fold", output_dir=out)
print("merged:", merged.num_episodes, "episodes,", merged.num_frames, "frames ->", out)
shutil.rmtree(tmp)
