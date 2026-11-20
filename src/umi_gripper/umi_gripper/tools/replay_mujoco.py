"""Replay a take on the 8dof arm in MuJoCo: what the arm would do if it followed the demo through TCP IK.

    python -m umi_gripper.tools.replay_mujoco datasets/umi_raw/<session>/take_0003            # viewer (mjpython on macOS)
    python -m umi_gripper.tools.replay_mujoco <take> --video replay.mp4                         # headless, for remote work

Kinematic only (joint positions are set, nothing is simulated): it checks reach and wrist motion, not grasps. The
demo starts at the arm's home TCP pose, as in postprocess.reachability. Red status = a step the arm couldn't reach.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from umi_gripper import config as umi_config
from umi_gripper._paths import PIONEER_HUMANOID, add
from umi_gripper.postprocess.quality import split_episode
from umi_gripper.postprocess.reachability import _arm_joints, check_reach, relative_tcp
from umi_gripper.postprocess.sync import sync_take
from umi_gripper.take import load_take

add(PIONEER_HUMANOID)


def build_model():
    import mujoco
    from pioneer_humanoid.mujoco_bimanual_arm_8dof import arm_spec, set_home
    model = arm_spec().compile()
    data = mujoco.MjData(model)
    set_home(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def apply(model, data, side: str, joints: np.ndarray, closure: float) -> None:
    import mujoco
    from pioneer_humanoid.arm_8dof_params import GRIPPER_STROKE
    for name, value in zip(_arm_joints(side), joints):
        data.qpos[model.joint(name).qposadr[0]] = value
    for name in (f"{side}_gripper", f"{side}_gripper_mimic"):
        data.qpos[model.joint(name).qposadr[0]] = float(np.clip(closure, 0, 1)) * GRIPPER_STROKE
    mujoco.mj_forward(model, data)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("take", type=Path)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--episode", type=int, default=0, help="which episode of the take (it may split at gaps)")
    parser.add_argument("--joint-speed", type=float, default=2.0)
    parser.add_argument("--video", type=Path, default=None, help="render to this .mp4 instead of opening a viewer")
    parser.add_argument("--loop", action="store_true", help="viewer: replay until closed")
    args = parser.parse_args(argv)
    cfg = umi_config.load_config()
    fps = float(cfg["postprocess"]["fps"])
    split = split_episode(sync_take(load_take(args.take), cfg), cfg["postprocess"])
    for reason in split.reasons:
        print(f"note: {reason}")
    if not split.ok:
        print("nothing to replay")
        return 1
    if not 0 <= args.episode < len(split.episodes):
        print(f"--episode must be in [0, {len(split.episodes)}): the take splits into {len(split.episodes)} episodes")
        return 1
    episode = split.episodes[args.episode]
    reach = check_reach(relative_tcp(episode), fps, args.arm, args.joint_speed)
    summary = reach.summary()
    print(f"{summary['reachable_fraction']:.0%} of steps reachable; max error "
          f"{summary['max_position_error_m'] * 100:.1f} cm / {summary['max_orientation_error_deg']:.0f} deg")
    model, data = build_model()

    if args.video is not None:
        import cv2
        import mujoco
        renderer = mujoco.Renderer(model, 480, 640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.35, 0.0, -0.15]
        camera.distance, camera.azimuth, camera.elevation = 1.4, 150.0, -20.0
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
        for step in range(len(episode)):
            apply(model, data, args.arm, reach.joints[step], episode.closure[step])
            renderer.update_scene(data, camera)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            if not reach.reachable[step]:
                cv2.putText(frame, "UNREACHABLE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            writer.write(frame)
        writer.release()
        print(f"wrote {args.video}")
        return 0

    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for step in range(len(episode)):
                if not viewer.is_running():
                    break
                started = time.monotonic()
                apply(model, data, args.arm, reach.joints[step], episode.closure[step])
                viewer.sync()
                time.sleep(max(0.0, 1 / fps - (time.monotonic() - started)))
            if not args.loop:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
