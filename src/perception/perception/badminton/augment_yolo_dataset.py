"""Create mild offline augmentations for a YOLO training split.

Only the training split is augmented. Validation and the separate Ticket 3
manifest must remain untouched for an honest comparison.

Ultralytics already applies random flips and HSV jitter during training, so
these copies add no new information; they mostly double the epoch time. They
are kept for reproducing yolo26n_v1. Use --remove to delete them.
"""

import argparse
import random
import re
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def read_labels(path: Path) -> list[list[float]]:
    labels = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Expected YOLO row in {path}: {line}")
            labels.append([float(value) for value in values])
    return labels


def write_labels(path: Path, labels: list[list[float]]) -> None:
    path.write_text(
        "".join(" ".join(f"{value:.6f}" for value in label) + "\n" for label in labels),
        encoding="utf-8",
    )


def augment(image, labels, rng: random.Random):
    augmented = image
    transformed_labels = [label[:] for label in labels]

    if rng.random() < 0.5:
        augmented = cv2.flip(augmented, 1)
        for label in transformed_labels:
            label[1] = 1.0 - label[1]

    hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[:, :, 1] = (hsv[:, :, 1] * rng.uniform(0.85, 1.15)).clip(0, 255)
    hsv[:, :, 2] = (hsv[:, :, 2] * rng.uniform(0.85, 1.15)).clip(0, 255)
    augmented = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return augmented, transformed_labels


def augment_training_split(dataset_dir: Path, copies: int, seed: int) -> None:
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS and "_aug" not in path.stem
    )
    if not image_paths:
        raise FileNotFoundError(f"No base training images found in {images_dir}")

    rng = random.Random(seed)
    generated = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        labels = read_labels(labels_dir / f"{image_path.stem}.txt")
        for copy_index in range(1, copies + 1):
            augmented, augmented_labels = augment(image, labels, rng)
            output_stem = f"{image_path.stem}_aug{copy_index:02d}"
            output_image = images_dir / f"{output_stem}{image_path.suffix.lower()}"
            output_label = labels_dir / f"{output_stem}.txt"
            cv2.imwrite(str(output_image), augmented, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_labels(output_label, augmented_labels)
            generated += 1

    print(f"Base training images: {len(image_paths)}")
    print(f"Generated augmented images: {generated}")
    print("Validation and Ticket 3 evaluation data were not modified.")


def remove_augmented(dataset_dir: Path) -> None:
    """Delete *_augNN images/labels and the Ultralytics label cache."""
    removed = 0
    for folder in (dataset_dir / "images" / "train", dataset_dir / "labels" / "train"):
        for path in folder.iterdir():
            if re.search(r"_aug\d+$", path.stem):
                path.unlink()
                removed += 1
    cache = dataset_dir / "labels" / "train.cache"
    if cache.exists():
        cache.unlink()
    print(f"Removed {removed} augmented files from {dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--remove", action="store_true", help="Delete previously generated copies instead.")
    args = parser.parse_args()
    if args.remove:
        remove_augmented(args.dataset_dir.resolve())
    else:
        if args.copies < 1:
            parser.error("--copies must be at least 1")
        augment_training_split(args.dataset_dir.resolve(), args.copies, args.seed)
