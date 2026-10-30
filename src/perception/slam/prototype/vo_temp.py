"""Run the existing VO frontend on an image/depth folder.

The folder must contain images/, depth/, timestamps.txt, and
calibration/camera.yaml. Images and depth files must have matching names.
Depth PNG values are converted to metres using depth_scale from camera.yaml.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from slam_core.types import Frame

from vo import Params, VO, draw, load_xfeat


def folder_frames(folder):
    folder = Path(folder)
    with (folder / "calibration" / "camera.yaml").open() as fh:
        calibration = yaml.safe_load(fh) or {}

    K = np.asarray(calibration["K"], dtype=np.float64)
    dist = np.asarray(calibration.get("distortion", []), dtype=np.float64)
    depth_scale = float(calibration.get("depth_scale", 0.001))
    timestamps = folder / "timestamps.txt"

    entries = []
    if timestamps.exists():
        for line in timestamps.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            _, stamp, image_name, depth_name = line.split()
            entries.append((float(stamp), image_name, depth_name))
    else:
        image_paths = sorted((folder / "images").glob("*.png"))
        entries = [
            (index / 30.0, f"images/{path.name}", f"depth/{path.name}")
            for index, path in enumerate(image_paths)
        ]

    for seq, (stamp, image_name, depth_name) in enumerate(entries):
        image = cv2.imread(str(folder / image_name), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(folder / depth_name), cv2.IMREAD_UNCHANGED)
        if image is None or depth is None:
            raise FileNotFoundError(f"missing image/depth pair: {image_name}, {depth_name}")
        depth = depth.astype(np.float32) * depth_scale
        yield Frame(stamp=stamp, image=image, depth=depth, K=K, dist=dist, seq=seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="folder containing images, depth, and calibration")
    parser.add_argument("--out", default="vo_folder_output")
    parser.add_argument("--no-viz", action="store_true")
    for name, value in vars(Params()).items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=type(value),
            default=value,
        )
    args = parser.parse_args()
    params = Params(**{name: getattr(args, name) for name in vars(Params())})
    vo = VO(load_xfeat(params.top_k), params)
    frame_count = 0

    try:
        for frame in folder_frames(args.input):
            result = vo.process(frame)
            frame_count += 1
            if result.state == "lost" and vo.lost == 1:
                print(f"[{frame.stamp:.2f}] lost: {result.reason}")
            elif result.state in ("recovered", "reinit"):
                print(f"[{frame.stamp:.2f}] {result.state}")
            if result.loop_to >= 0:
                print(f"[{frame.stamp:.2f}] loop: KF {result.new_kf.id} -> KF {result.loop_to}")
            if not args.no_viz:
                cv2.imshow("vo", draw(frame, result, len(vo.refs), len(vo.loops)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()
        if vo.refs:
            vo.write(args.out)

    print(f"processed {frame_count} frames, created {len(vo.refs)} keyframes")
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
