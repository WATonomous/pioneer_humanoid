import argparse
import random
import shutil
from pathlib import Path

import yaml


def split_dataset(labeled_dir, frames_dir, out_dir, val_frac, seed=42):
    labels_dir = Path(labeled_dir) / "labels"
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)

    label_files = sorted(labels_dir.glob("*.txt"))
    random.seed(seed)
    random.shuffle(label_files)

    n_val = int(len(label_files) * val_frac)
    val_set = set(f.stem for f in label_files[:n_val])

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train, n_val_actual, n_empty = 0, 0, 0
    for label_file in label_files:
        stem = label_file.stem
        frame_file = frames_dir / f"{stem}.jpg"
        if not frame_file.exists():
            continue

        split = "val" if stem in val_set else "train"
        shutil.copy(frame_file, out_dir / "images" / split / frame_file.name)
        shutil.copy(label_file, out_dir / "labels" / split / label_file.name)

        if label_file.stat().st_size == 0:
            n_empty += 1
        if split == "train":
            n_train += 1
        else:
            n_val_actual += 1

    yaml_content = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "shuttlecock"},
    }
    yaml_path = out_dir / "shuttlecock.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    print(
        f"Train: {n_train}, Val: {n_val_actual}, Empty-label (hard negatives): {n_empty}"
    )
    print(f"Dataset yaml written to {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_dir", default="data/labeled")
    parser.add_argument("--frames_dir", default="data/frames")
    parser.add_argument("--out_dir", default="data/dataset")
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    split_dataset(args.labeled_dir, args.frames_dir,
                  args.out_dir, args.val_frac)
