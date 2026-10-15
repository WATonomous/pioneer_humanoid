"""Copy a LeRobot dataset with its camera videos scaled down (faster training; ACT is image-bound at 640x480).

    python downscale_dataset.py <src_root> <dst_root> --height 240 --width 320

Frames, timestamps and everything but the videos are copied as they are; sim_policy_eval.py scales its renders
the same way (area filter) when the policy's image size is smaller than the cameras'.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--codec", choices=["libsvtav1", "libx264"], default="libsvtav1",
                        help="libx264 for an ffmpeg built without SVT-AV1 (e.g. the one imageio-ffmpeg ships)")
    args = parser.parse_args()
    # An existing copy is updated: metadata and frames again, only the new videos (one file per episode) are scaled.
    shutil.copytree(args.src, args.dst, ignore=shutil.ignore_patterns("videos", "images"), dirs_exist_ok=True)
    videos = [v for v in sorted((args.src / "videos").rglob("*.mp4")) if not (args.dst / v.relative_to(args.src)).exists()]

    def scale(path: Path) -> None:
        out = args.dst / path.relative_to(args.src)
        out.parent.mkdir(parents=True, exist_ok=True)
        encode = (["-c:v", "libsvtav1", "-preset", "10", "-crf", "24", "-g", "2", "-pix_fmt", "yuv420p", "-svtav1-params", "lp=1"]
                  if args.codec == "libsvtav1" else
                  ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-g", "2", "-pix_fmt", "yuv420p", "-threads", "1"])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vf", f"scale={args.width}:{args.height}:flags=area",
                        *encode, str(out)], check=True)

    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(scale, videos))
    info_path = args.dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for ft in info["features"].values():
        if ft.get("dtype") == "video":
            ft["shape"] = [args.height, args.width, 3]
            ft["info"]["video.height"], ft["info"]["video.width"] = args.height, args.width
            if args.codec == "libx264":
                ft["info"]["video.codec"] = "h264"
    info_path.write_text(json.dumps(info, indent=4))
    print(f"{len(videos)} videos -> {args.dst} at {args.width}x{args.height}")


if __name__ == "__main__":
    main()
