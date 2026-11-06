"""Replay VO keyframes through the incremental GTSAM pose-graph backend."""

import argparse
from pathlib import Path

from slam_core.backend import PoseGraphBackend
from slam_core.types.pose import to_tum

from load_vo_data import load_vo_keyframes, load_vo_loops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keyframes", type=Path)
    parser.add_argument("kf_vo", type=Path)
    parser.add_argument("--loops", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    keyframes = load_vo_keyframes(args.keyframes, args.kf_vo)
    if not keyframes:
        raise ValueError("no keyframes were loaded")
    loops = load_vo_loops(args.loops) if args.loops else []
    loops_by_new_id = {}
    for loop in loops:
        loops_by_new_id.setdefault(loop.new_id, []).append(loop)

    backend = PoseGraphBackend()
    for keyframe in keyframes:
        backend.add_keyframe(keyframe)
        for loop in loops_by_new_id.get(keyframe.id, []):
            backend.add_loop_constraint(loop)

    estimates = backend.optimized_poses()
    print(f"Loaded keyframes: {len(keyframes)}")
    print(f"Optimized poses: {len(estimates)}")
    print(f"First keyframe: X{keyframes[0].id}")
    print(f"Last keyframe: X{keyframes[-1].id}")

    if args.loops:
        print(f"Loaded loop closures: {len(loops)}")
        for loop in loops:
            print(f"  X{loop.old_id} -> X{loop.new_id} ({loop.n_inliers} inliers)")

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
