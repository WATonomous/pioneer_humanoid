"""Replay VO keyframes through the incremental GTSAM pose-graph backend."""

import argparse
from pathlib import Path

from slam_core.backend import PoseGraphBackend
from slam_core.types.pose import to_tum

from test_vo_keyframes import load_vo_keyframes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keyframes", type=Path)
    parser.add_argument("kf_vo", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    keyframes = load_vo_keyframes(args.keyframes, args.kf_vo)
    if not keyframes:
        raise ValueError("no keyframes were loaded")

    backend = PoseGraphBackend()
    for keyframe in keyframes:
        backend.add_keyframe(keyframe)

    estimates = backend.optimized_poses()
    print(f"Loaded keyframes: {len(keyframes)}")
    print(f"Optimized poses: {len(estimates)}")
    print(f"First keyframe: X{keyframes[0].id}")
    print(f"Last keyframe: X{keyframes[-1].id}")

    if args.output:
        with args.output.open("w") as output:
            output.write("# timestamp tx ty tz qx qy qz qw\n")
            for keyframe in keyframes:
                tum_pose = to_tum(estimates[keyframe.id].matrix())
                output.write(
                    f"{keyframe.stamp:.9f} "
                    + " ".join(f"{value:.9f}" for value in tum_pose)
                    + "\n"
                )
        print(f"Wrote optimized trajectory: {args.output}")


if __name__ == "__main__":
    main()
