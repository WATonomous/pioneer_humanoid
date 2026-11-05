"""Prepare a manually annotated shuttlecock evaluation set.

This script intentionally keeps images and annotations outside the repository.
It samples frames from selected videos, writes JPEGs to an external output
folder, and creates a JSONL manifest for annotate_eval.py.
"""

import argparse
import json
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def sample_video(video_path: Path, output_dir: Path, every_seconds: float) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * every_seconds))
    rows = []

    for frame_index in range(0, frame_count, step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue

        image_name = f"{video_path.stem}_f{frame_index:06d}.jpg"
        image_path = output_dir / image_name
        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(
            {
                "image": str(image_path.resolve()),
                "video": video_path.name,
                "frame_index": frame_index,
                "label": None,
                "center_x": None,
                "center_y": None,
                "tags": [],
            }
        )

    capture.release()
    return rows


def prepare_eval(input_dir: Path, output_dir: Path, videos: list[str], every_seconds: float) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if videos:
        video_paths = [input_dir / name for name in videos]
    else:
        video_paths = sorted(
            path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS
        )

    missing = [path for path in video_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing video(s): " + ", ".join(map(str, missing)))
    if not video_paths:
        raise FileNotFoundError(f"No videos found in {input_dir}")

    rows = []
    for video_path in video_paths:
        video_rows = sample_video(video_path, images_dir, every_seconds)
        rows.extend(video_rows)
        print(f"{video_path.name}: sampled {len(video_rows)} frames")

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in rows:
            manifest.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} frames to {images_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="Video filename to sample; repeat for multiple videos. Defaults to all videos.",
    )
    parser.add_argument(
        "--every_seconds",
        type=float,
        default=1.0,
        help="Time between sampled frames. Use a smaller value for a larger eval set.",
    )
    args = parser.parse_args()
    if args.every_seconds <= 0:
        parser.error("--every_seconds must be positive")

    prepare_eval(args.input_dir, args.output_dir, args.video, args.every_seconds)
