"""Quest Touch controller teleoperation in MuJoCo: both arms, any humanoid_mujoco_scenes scene.

The headset only tracks (index.html streams its poses): open http://localhost:8444/ in Meta Browser after
`adb reverse tcp:8444 tcp:8444` (start_quest.sh does it all).
Left Y (or 6 with the window focused) resets the arms and the scene. --record: right B starts / saves a take;
--record_arms left|right records one arm (8 values) for single-arm tasks.

This machine shows one camera window (ego on top, both wrist cameras below; plain ``python``);
``--headless`` shows nothing. Runs on a Mac without a GPU.
"""
from __future__ import annotations

import argparse

from session import MujocoSession, add_session_args, open_rig, serve   # first: puts the repo's packages on sys.path
from scene_driver import check_controller_args  # noqa: E402
from humanoid_robot_learning.recording.schema import ARM_SETS  # noqa: E402
from humanoid_robot_learning.sim.sim_teleop_record import add_record_args, load_record_schema  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Quest Touch controller teleoperation in MuJoCo (both arms).")
    add_session_args(parser)
    add_record_args(parser, task_description="Quest VR teleop demonstration")
    parser.add_argument("--record_arms", choices=tuple(ARM_SETS), default="bimanual",
                        help="arms to record (both are always driven): left or right records that arm's 8 values "
                             "and its wrist camera into datasets/pioneer_8dof_<arm>_arm, for single-arm tasks")
    args = parser.parse_args()
    check_controller_args(parser, args)
    sides = ARM_SETS[args.record_arms]
    if args.record and args.cameras is None:
        args.cameras = ",".join(["ego", *(f"wrist_{side}" for side in sides)])   # the recorded arms' wrists
    return args, load_record_schema(parser, args, arms=list(sides), robot="8dof")


def run() -> None:
    args, record = parse_args()
    sim = MujocoSession(args.scene)
    sim.reset()
    rig = open_rig(args, record, sim, ARM_SETS[args.record_arms])
    print(f"[Quest] MuJoCo scene {args.scene}: open http://localhost:{args.controller_port}/ in Meta Browser "
          f"(after adb reverse tcp:{args.controller_port} tcp:{args.controller_port}). Left Y or 6 in the window resets.", flush=True)
    if rig.recording is not None:
        rig.window.on_saved = lambda take: (rig.recording.saved(), rig.reset_key.set())

    def start_episode(driver) -> str:
        sim.reset()
        driver.reset(sim.joint_positions())
        return "Scene reset; arms back to home."

    serve(args, rig, sim, start_episode)


if __name__ == "__main__":
    run()
