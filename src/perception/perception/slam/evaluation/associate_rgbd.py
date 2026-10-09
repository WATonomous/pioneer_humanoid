#!/usr/bin/env python3
"""Associate timestamped RGB and depth frames for RTAB-Map datasets."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

# This functions converts this '1548266469.85281 rgb/1548266469.85281.png' into a proper tuple in a list
def read_index(index_path: Path) -> list[tuple[float, Path]]:
    entries: list[tuple[float, Path]] = []
    with index_path.open(encoding="utf-8") as index_file:
        for line_number, line in enumerate(index_file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"{index_path}:{line_number}: expected timestamp and path")

            try:
                timestamp = float(fields[0])
            except ValueError as error:
                raise ValueError(
                    f"{index_path}:{line_number}: invalid timestamp {fields[0]!r}"
                ) from error

            entries.append((timestamp, Path(fields[1])))

    if not entries:
        raise ValueError(f"No image entries found in {index_path}")
    return sorted(entries)


def associate(
    rgb_entries: list[tuple[float, Path]],
    depth_entries: list[tuple[float, Path]],
    max_difference: float,
) -> list[tuple[tuple[float, Path], tuple[float, Path]]]:
    candidates: list[tuple[float, int, int]] = []
    depth_start = 0

    for rgb_index, (rgb_time, _) in enumerate(rgb_entries):
        while (
            depth_start < len(depth_entries)
            and depth_entries[depth_start][0] < rgb_time - max_difference
        ):
            depth_start += 1

        depth_index = depth_start
        while (
            depth_index < len(depth_entries)
            and depth_entries[depth_index][0] <= rgb_time + max_difference
        ):
            difference = abs(rgb_time - depth_entries[depth_index][0])
            candidates.append((difference, rgb_index, depth_index))
            depth_index += 1

    used_rgb: set[int] = set()
    used_depth: set[int] = set()
    matches: list[tuple[tuple[float, Path], tuple[float, Path]]] = []

    for _, rgb_index, depth_index in sorted(candidates):
        if rgb_index in used_rgb or depth_index in used_depth:
            continue
        used_rgb.add(rgb_index)
        used_depth.add(depth_index)
        matches.append((rgb_entries[rgb_index], depth_entries[depth_index]))

    return sorted(matches, key=lambda match: match[0][0])


def populate_links(
    dataset_path: Path,
    output_path: Path,
    entries: list[tuple[float, Path]],
) -> None:
    for timestamp, relative_source in entries:
        source = (dataset_path / relative_source).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Image listed for {timestamp} does not exist: {source}")

        destination = output_path / relative_source.name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Duplicate output filename: {destination.name}")

        destination.symlink_to(os.path.relpath(source, output_path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match TUM/Bonn RGB and depth frames by timestamp."
    )
    parser.add_argument("dataset", type=Path, help="dataset containing rgb.txt and depth.txt")
    parser.add_argument(
        "--max-difference",
        type=float,
        default=0.02,
        help="maximum RGB-depth timestamp difference in seconds (default: 0.02)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing rgb_sync and depth_sync directories",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_dir():
        parser.error(f"dataset directory does not exist: {dataset_path}")
    if args.max_difference <= 0:
        parser.error("--max-difference must be positive")

    rgb_index = dataset_path / "rgb.txt"
    depth_index = dataset_path / "depth.txt"
    if not rgb_index.is_file() or not depth_index.is_file():
        parser.error("dataset must contain rgb.txt and depth.txt")

    rgb_output = dataset_path / "rgb_sync"
    depth_output = dataset_path / "depth_sync"
    if rgb_output.exists() or depth_output.exists():
        if not args.force:
            parser.error("rgb_sync or depth_sync already exists; use --force to replace them")
        shutil.rmtree(rgb_output, ignore_errors=True)
        shutil.rmtree(depth_output, ignore_errors=True)

    rgb_entries = read_index(rgb_index)
    depth_entries = read_index(depth_index)
    matches = associate(rgb_entries, depth_entries, args.max_difference)
    if not matches:
        raise RuntimeError(
            f"No RGB-depth pairs found within {args.max_difference:.6f} seconds"
        )

    rgb_temp = Path(tempfile.mkdtemp(prefix=".rgb_sync_", dir=dataset_path))
    depth_temp = Path(tempfile.mkdtemp(prefix=".depth_sync_", dir=dataset_path))
    try:
        populate_links(dataset_path, rgb_temp, [match[0] for match in matches])
        populate_links(dataset_path, depth_temp, [match[1] for match in matches])
        rgb_temp.rename(rgb_output)
        depth_temp.rename(depth_output)
    except Exception:
        shutil.rmtree(rgb_temp, ignore_errors=True)
        shutil.rmtree(depth_temp, ignore_errors=True)
        raise

    unmatched_rgb = len(rgb_entries) - len(matches)
    unmatched_depth = len(depth_entries) - len(matches)
    print(
        f"Associated {len(matches)} RGB-depth pairs "
        f"({unmatched_rgb} RGB and {unmatched_depth} depth frames unmatched)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
