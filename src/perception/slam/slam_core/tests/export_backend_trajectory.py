"""Expand optimized keyframe poses into a corrected full-frame trajectory."""

import argparse
from pathlib import Path

from slam_core.types.pose import from_tum, to_tum


def read_poses(path):
    poses = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = line.split()
        poses.append(from_tum(values[1:8]))
    return poses


def export_trajectory(optimized_keyframes, frames, output):
    keyframe_poses = read_poses(optimized_keyframes)
    output_rows = []

    for line in Path(frames).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = line.split()
        stamp = float(values[0])
        keyframe_id = int(values[1])
        T_kf_cur = from_tum(values[2:9])
        T_map_cur = keyframe_poses[keyframe_id] @ T_kf_cur
        output_rows.append((stamp, to_tum(T_map_cur)))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        file.write("# timestamp tx ty tz qx qy qz qw\n")
        for stamp, pose in output_rows:
            file.write(
                f"{stamp:.9f} "
                + " ".join(f"{value:.9f}" for value in pose)
                + "\n"
            )
    print(f"Wrote {len(output_rows)} poses to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("optimized_keyframes")
    parser.add_argument("frames")
    parser.add_argument("output")
    args = parser.parse_args()
    export_trajectory(args.optimized_keyframes, args.frames, args.output)


if __name__ == "__main__":
    main()
