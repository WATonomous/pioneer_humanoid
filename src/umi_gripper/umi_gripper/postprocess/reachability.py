"""Can the Pioneer 8dof arm follow a demo? TCP IK on the real kinematics, at the arm's speed limits.

The demo is placed so its first TCP pose is the arm's home TCP pose (as UMI checks its demos), then every 25 fps
step is tracked by one velocity-capped mink QP (quest_controller_teleop/mink_control.py's solver, with the task
moved from the wrist pivot to the TCP). A step the arm can't reach, or can't reach that fast, shows up as tracking
error. TcpIK is also what a deployed tool-frame policy would use to turn its actions into joint targets.

Kinematics only, CPU: pip install mujoco mink daqp.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from umi_gripper import geometry
from umi_gripper._paths import PIONEER_HUMANOID, QUEST_TELEOP, add

add(QUEST_TELEOP, PIONEER_HUMANOID)


def _arm_joints(side: str) -> list[str]:
    from pioneer_humanoid.arm_8dof_params import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
    return list(LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS)


def make_tcp_ik(side: str = "left", joint_speed: float = 2.0, orientation_weight: float = 0.5,
                posture_weight: float = 0.01):
    """TcpIK for one 8dof arm (built lazily: mink, mujoco and the URDF load only here)."""
    from mink_control import PioneerMinkIK
    from pioneer_humanoid.arm_8dof_params import TCP_LOCAL

    class TcpIK(PioneerMinkIK):
        """PioneerMinkIK with one frame task on the TCP site (TCP_LOCAL on the last link) instead of pivot + tool."""

        def _mjcf(self):
            x, y, z = TCP_LOCAL
            return super()._mjcf().replace('<site name="tool" pos="0 0 0"/>',
                                           f'<site name="tool" pos="0 0 0"/><site name="tcp" pos="{x} {y} {z}"/>')

        def setup_tcp(self, orientation_cost: float) -> None:
            self.tcp_task = self.mink.FrameTask("tcp", "site", position_cost=1.0, orientation_cost=orientation_cost,
                                                lm_damping=1e-6)

        def tcp_pose(self, qpos) -> np.ndarray:
            self.cfg.update(np.asarray(qpos, dtype=float))
            return self.cfg.get_transform_frame_to_world("tcp", "site").as_matrix()

        def solve_tcp(self, target: np.ndarray, dt: float) -> np.ndarray:
            """One velocity-capped step toward the TCP pose `target` (4x4, base_link); returns the joint command."""
            mink = self.mink
            seed = self.command.copy()
            self.cfg.update(seed)
            goal = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(target[:3, :3]), target[:3, 3])
            self.tcp_task.set_target(goal)
            try:
                v = mink.solve_ik(self.cfg, [self.tcp_task, self.posture], dt, "daqp", damping=1e-4,
                                  limits=[self.box])
            except mink.exceptions.NoSolutionFound:
                self.infeasible += 1
                v = np.zeros(len(seed))
            step = self.velocity * dt
            lo = np.minimum(np.maximum(self.limits[:, 0], seed - step), seed + step)
            hi = np.maximum(np.minimum(self.limits[:, 1], seed + step), seed - step)
            self.command = np.clip(seed + v * dt, lo, hi)
            return self.command.copy()

    ik = TcpIK(_arm_joints(side), velocity=joint_speed, orientation_weight=0.0, posture_weight=posture_weight)
    ik.setup_tcp(orientation_weight)
    return ik


@dataclass
class Reach:
    joints: np.ndarray            # (n, 7) joint commands
    position_error: np.ndarray    # (n,) m
    orientation_error: np.ndarray  # (n,) rad
    reachable: np.ndarray         # (n,) bool

    def summary(self) -> dict:
        return {"reachable_fraction": float(self.reachable.mean()) if len(self.reachable) else 0.0,
                "max_position_error_m": float(self.position_error.max(initial=0.0)),
                "max_orientation_error_deg": math.degrees(float(self.orientation_error.max(initial=0.0)))}


def anchored_targets(relative_tcp: np.ndarray, home_tcp: np.ndarray) -> np.ndarray:
    """Start-relative TCP poses (n,4,4) -> base_link targets, the first one at the arm's home TCP."""
    return np.einsum("ij,njk->nik", home_tcp, relative_tcp)


def check_reach(relative_tcp: np.ndarray, fps: float, side: str = "left", joint_speed: float = 2.0,
                substeps: int = 4, max_position_error: float = 0.01,
                max_orientation_error_deg: float = 10.0, ik=None) -> Reach:
    ik = ik or make_tcp_ik(side, joint_speed)
    home = ik.tcp_pose(ik.rest)
    targets = anchored_targets(relative_tcp, home)
    ik.command = ik.rest.copy()
    dt = 1.0 / fps / substeps
    joints, pos_err, ori_err = [], [], []
    for target in targets:
        for _ in range(substeps):
            q = ik.solve_tcp(target, dt)
        reached = ik.tcp_pose(q)
        joints.append(q)
        pos_err.append(np.linalg.norm(reached[:3, 3] - target[:3, 3]))
        ori_err.append(np.linalg.norm(geometry.rotation_vector(reached[:3, :3].T @ target[:3, :3])))
    pos_err, ori_err = np.array(pos_err), np.array(ori_err)
    ok = (pos_err <= max_position_error) & (ori_err <= math.radians(max_orientation_error_deg))
    return Reach(np.array(joints), pos_err, ori_err, ok)


def relative_tcp(episode) -> np.ndarray:
    origin = geometry.invert(episode.tcp[0])
    return np.einsum("ij,njk->nik", origin, episode.tcp)


def main(argv=None) -> int:
    from umi_gripper import config as umi_config
    from umi_gripper.postprocess.quality import check_episode
    from umi_gripper.postprocess.sync import sync_take
    from umi_gripper.take import find_takes, load_take

    parser = argparse.ArgumentParser(description="Check that the 8dof arm can follow each take (TCP IK)")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--joint-speed", type=float, default=2.0, help="rad/s cap per joint")
    args = parser.parse_args(argv)
    cfg = umi_config.load_config()
    fps = float(cfg["postprocess"]["fps"])
    for path in [t for root in args.inputs for t in find_takes(root)]:
        verdict = check_episode(sync_take(load_take(path), cfg), cfg["postprocess"])
        if not len(verdict.episode):
            print(f"{path}: no valid steps")
            continue
        reach = check_reach(relative_tcp(verdict.episode), fps, args.arm, args.joint_speed)
        s = reach.summary()
        print(f"{path}: {s['reachable_fraction']:.0%} reachable, max error {s['max_position_error_m'] * 100:.1f} cm "
              f"/ {s['max_orientation_error_deg']:.0f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
