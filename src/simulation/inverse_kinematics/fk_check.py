"""Check MuJoCo's forward kinematics against the repo's analytic FK for the left arm.

For random joint angles inside the limits, compare
  - the wrist pivot position (MuJoCo joint anchor vs PioneerKinematics.kinematics()[0]), and
  - the last link's orientation (MuJoCo xmat vs PioneerKinematics.kinematics()[1]),
then check the tool point is where it should be (link7 origin + rotation @ offset).

    mjpython is not needed (no viewer):   python fk_check.py
"""
import sys
from pathlib import Path

import numpy as np

from arm_model import LEFT_ARM_JOINTS, SRC, TCP_OFFSET, LeftArm

sys.path.append(str(SRC / "teleop" / "quest_controller_teleop"))
from control import PioneerKinematics  # noqa: E402

N = 500
POS_TOL = 1e-3     # m
ROT_TOL = 1e-3     # rad


def rotation_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def main():
    arm = LeftArm()
    kin = PioneerKinematics(LEFT_ARM_JOINTS)
    rng = np.random.default_rng(0)

    pivot_err, rot_err, tcp_err = [], [], []
    for _ in range(N):
        q = arm.random_q(rng)
        pivot_ref, rot_ref, _, _ = kin.kinematics(q)
        pivot_mj = arm.wrist_pivot(q)
        rot_mj = arm.link7_rotation(q)
        tcp_pos, _ = arm.tcp_pose(q)
        link7_pos = arm.data.xpos[arm.link7_id]
        pivot_err.append(np.linalg.norm(pivot_mj - pivot_ref))
        rot_err.append(rotation_angle(rot_ref.T @ rot_mj))
        tcp_err.append(np.linalg.norm(tcp_pos - (link7_pos + rot_mj @ np.array(TCP_OFFSET))))

    print(f"{N} random joint configurations inside the limits")
    print(f"wrist pivot position error: max {max(pivot_err) * 1000:.4f} mm, mean {np.mean(pivot_err) * 1000:.4f} mm")
    print(f"last link rotation error:   max {np.degrees(max(rot_err)):.4f} deg, mean {np.degrees(np.mean(rot_err)):.4f} deg")
    print(f"tool point vs link7 + offset: max {max(tcp_err) * 1000:.4f} mm")

    q_home = arm.rest
    pos, rot = arm.tcp_pose(q_home)
    print(f"\ntool point at the home pose (base frame, m): {np.round(pos, 4)}")

    ok = max(pivot_err) < POS_TOL and max(rot_err) < ROT_TOL and max(tcp_err) < POS_TOL
    print("\nPASS" if ok else "\nFAIL: the two forward kinematics disagree")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
