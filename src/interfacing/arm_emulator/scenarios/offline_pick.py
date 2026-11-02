"""Can the arm pick up the cube at its real gains? The synthetic operator, the offline stand-in, no ROS.

    python scenarios/offline_pick.py --episodes 5 [--force-scale 3]

A fast answer to the grasp question before collecting demos: fails for the same physical reasons a demo
would. The end-to-end version through the real joint_command is synthetic_demos.py.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from arm_emulator.physics import EmulatedArm
from arm_emulator import config  # noqa: F401 -- sys.path for the scenes

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "teleop" / "pioneer_leader_arm_teleop" / "synthetic"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_command import CONTROL_DT, OfflineCommand  # noqa: E402

HOME_DEG = [0.0, 0.0, 0.0, -90.0, 0.0, 0.0]


def run_episode(seed: int, force_scale: float | None, friction: float | None) -> tuple[bool, str, EmulatedArm]:
    from human_operator import HumanOperator
    from pioneer_humanoid.arm_params import LEFT_ARM_JOINTS
    from synthetic_tasks import TASKS, TaskFailed

    arm = EmulatedArm(seed=seed, start_pose_deg=HOME_DEG, supported=True)
    if force_scale is not None:
        arm.grip_force_scale = force_scale
    if friction is not None:
        arm.model.geom_friction[arm.model.geom("rubiks_cube").id][0] = friction
    command = OfflineCommand(arm)
    command.seed()
    substeps = round(CONTROL_DT / arm.dt)

    def tick(q_rad, grip: float) -> None:
        command.tick(np.degrees(q_rad), grip)
        for _ in range(substeps):
            arm.step()

    rng = np.random.default_rng([seed, 1])
    op = HumanOperator(arm.model, lambda: arm.data, tick, LEFT_ARM_JOINTS, np.radians(HOME_DEG), CONTROL_DT, rng)
    notes: list[str] = []
    try:
        TASKS["rubiks_cube"].plan(op, arm.model, arm.data, notes)
        return arm.success(), "; ".join(notes), arm
    except TaskFailed as e:
        return False, f"{e}; " + "; ".join(notes), arm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--force-scale", type=float, default=None, help="override motor_models.yaml gripper.force_scale")
    p.add_argument("--friction", type=float, default=None, help="override the cube's friction (scene: 1.0)")
    args = p.parse_args()
    wins = 0
    for seed in range(args.episodes):
        ok, notes, arm = run_episode(seed, args.force_scale, args.friction)
        wins += ok
        worst = {s.name: round(float(arm.peak_torque[s.motor_id]), 3) for s in arm.specs}
        print(f"seed {seed}: {'lifted' if ok else 'FAILED'}  closure {arm.closure():.2f}  sim {arm.data.time:.1f} s  "
              f"contact {arm.peak_contact:.0f} N  breaches {arm.breaches or 'none'}  {notes}\n  peak torque {worst}", flush=True)
    print(f"{wins}/{args.episodes} lifted")


if __name__ == "__main__":
    main()
