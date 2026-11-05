"""Build a video-disjoint YOLO dataset from a hand-labeled JSONL manifest.

The manifest must contain image, video, label, and bbox fields. Positive rows
use bbox=[left, top, right, bottom] in original image pixels. Negative rows get
an empty YOLO label file. Pass validation videos explicitly; every other video
in the manifest is assigned to training.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import yaml



def load_rows(manifest_path: Path) -> list[dict]:
    with manifest_path.open(encoding="utf-8") as manifest:
        rows = [json.loads(line) for line in manifest if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def yolo_label(row: dict, image_width: int, image_height: int) -> str:
    if row["label"] == "negative":
        return ""
    if row["label"] != "positive" or not row.get("bbox"):
        raise ValueError(f"Positive rows require bbox coordinates: {row}")

    left, top, right, bottom = map(float, row["bbox"])
    left = max(0.0, min(left, image_width))
    right = max(0.0, min(right, image_width))
    top = max(0.0, min(top, image_height))
    bottom = max(0.0, min(bottom, image_height))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid or empty bbox: {row['bbox']}")

    center_x = ((left + right) / 2) / image_width
    center_y = ((top + bottom) / 2) / image_height
    box_width = (right - left) / image_width
    box_height = (bottom - top) / image_height
    return f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n"


def build_dataset(manifest_path: Path, output_dir: Path, validation_videos: list[str], clean: bool) -> None:
    rows = load_rows(manifest_path.resolve())
    output_dir = output_dir.resolve()
    all_videos = {row.get("video") for row in rows}
    if None in all_videos:
        raise ValueError("Every manifest row must include a source video")
    unknown_validation = set(validation_videos) - all_videos
    if unknown_validation:
        raise ValueError(f"Validation video(s) are not in the manifest: {sorted(unknown_validation)}")
    training_videos = all_videos - set(validation_videos)
    if not training_videos:
        raise ValueError("Validation selection consumes every source video")

    if clean:
        for split in ("train", "val"):
            shutil.rmtree(output_dir / "images" / split, ignore_errors=True)
            shutil.rmtree(output_dir / "labels" / split, ignore_errors=True)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "train_negative": 0, "val_negative": 0}
    split_rows = []
    for row in rows:
        split = "val" if row["video"] in validation_videos else "train"
        source = Path(row["image"])
        image = cv2.imread(str(source))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        image_height, image_width = image.shape[:2]
        image_name = source.name
        label_name = source.with_suffix(".txt").name
        shutil.copy2(source, output_dir / "images" / split / image_name)
        (output_dir / "labels" / split / label_name).write_text(
            yolo_label(row, image_width, image_height), encoding="utf-8"
        )
        counts[split] += 1
        if row["label"] == "negative":
            counts[f"{split}_negative"] += 1
        split_rows.append({"image": image_name, "video": row["video"], "split": split})

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "shuttlecock"},
    }
    (output_dir / "shuttlecock.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )
    with (output_dir / "split_manifest.jsonl").open("w", encoding="utf-8") as split_manifest:
        for split_row in split_rows:
            split_manifest.write(json.dumps(split_row) + "\n")

    print(f"Training videos: {sorted(training_videos)}")
    print(f"Validation videos: {sorted(validation_videos)}")
    print(f"Train: {counts['train']} ({counts['train_negative']} negative)")
    print(f"Val: {counts['val']} ({counts['val_negative']} negative)")
    print(f"Dataset YAML: {output_dir / 'shuttlecock.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--val_video",
        action="append",
        required=True,
        help="Source video reserved for validation; repeat for multiple videos.",
    )
    parser.add_argument("--clean", action="store_true", help="Clear existing train/val folders first.")
    args = parser.parse_args()
    build_dataset(args.manifest, args.output_dir, args.val_video, args.clean)
