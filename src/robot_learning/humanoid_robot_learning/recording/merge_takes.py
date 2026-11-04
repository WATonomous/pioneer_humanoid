"""Combine humanoid-record takes into one LeRobot dataset, optionally uploading it.

humanoid-record saves every take as its own small dataset (<session>/take_NNN/). Training and
the Hugging Face Hub want one dataset, so merge first:

    humanoid-merge-takes datasets/pioneer_v1_left_arm/real/001 datasets/pioneer_v1_left_arm/real/002 \\
        --output datasets/pioneer_v1_left_arm/real_merged
    # add --push_to_hub <hf-user>/<name> to also upload it (needs `huggingface-cli login`)

Inputs can be session folders (all their takes, in order) or single take folders. The takes are
only read; the merged copy is written to --output, which must not exist yet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from humanoid_robot_learning.recording.recorder import take_dirs

logger = logging.getLogger(__name__)


def _is_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def collect_takes(paths: list[Path]) -> list[Path]:
    """Expand session folders into their takes; keep take folders as given."""
    takes: list[Path] = []
    for path in paths:
        if _is_dataset(path):
            takes.append(path)
            continue
        found = take_dirs(path)
        if not found:
            raise SystemExit(f"{path}: not a take folder and has no take_NNN/ folders")
        for take in found:
            if _is_dataset(take):
                takes.append(take)
            else:
                logger.warning("Skipping %s: no meta/info.json (unfinished take?)", take)
    return takes


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="session folders and/or take folders")
    parser.add_argument("--output", type=Path, required=True, help="merged dataset folder (must not exist)")
    parser.add_argument(
        "--push_to_hub",
        metavar="REPO_ID",
        default=None,
        help="also upload the merged dataset to this Hugging Face repo, e.g. <user>/pioneer_v1_left_arm",
    )
    parser.add_argument("--private", action="store_true", help="make the uploaded repo private")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.output.exists():
        raise SystemExit(f"--output {args.output} already exists; pick a new folder")
    takes = collect_takes(args.paths)
    if not takes:
        raise SystemExit("No saved takes found.")

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets = [LeRobotDataset(repo_id=f"local/{t.parent.name}_{t.name}", root=t) for t in takes]
    repo_id = args.push_to_hub or f"local/{args.output.name}"
    logger.info("Merging %d takes into %s", len(takes), args.output)
    merged = merge_datasets(datasets, output_repo_id=repo_id, output_dir=args.output)
    print(f"Merged {merged.meta.total_episodes} episodes ({merged.meta.total_frames} frames) into {args.output}")

    if args.push_to_hub:
        logger.info("Uploading to https://huggingface.co/datasets/%s", repo_id)
        merged.push_to_hub(private=args.private)
        print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
