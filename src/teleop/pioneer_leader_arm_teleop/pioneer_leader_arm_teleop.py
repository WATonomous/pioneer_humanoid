"""Leader-arm teleop of the Pioneer left arm (no IK), in Isaac Sim (default), plain MuJoCo or on the real arm.

    isaaclab.sh -p pioneer_leader_arm_teleop.py --scene push [--record]
    python pioneer_leader_arm_teleop.py --target mujoco --scene peg_insert
    python3 pioneer_leader_arm_teleop.py --target real [--live] # dry run; --live drives the real arm

--target picks the backend before anything simulator-specific is imported, so --target mujoco
and --target real run without Isaac. See isaac_sim.py / mujoco_sim.py / real_arm.py and README.md.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--target", choices=("isaac", "mujoco", "real"), default="isaac")
_target = _pre.parse_known_args()[0].target
if _target == "mujoco":
    import mujoco_sim as backend
elif _target == "real":
    import real_arm as backend
else:
    import isaac_sim as backend

if __name__ == "__main__":
    backend.run()
