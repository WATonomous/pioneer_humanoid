"""Make the repo's sibling, non-installed modules importable (the same sys.path pattern the teleops use).

The Quest teleop (transport.py, index.html), the leader arm's STS3215 reader and the wrist-camera rectifier are plain
script folders, not packages; pioneer_humanoid and robot_learning may or may not be pip-installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]          # <repo>/src
REPO_ROOT = SRC.parent
QUEST_TELEOP = SRC / "teleop" / "quest_controller_teleop"
LEADER_TELEOP = SRC / "teleop" / "pioneer_leader_arm_teleop"
CAMERA_CALIBRATION = SRC / "perception" / "camera_calibration"
PIONEER_HUMANOID = SRC / "pioneer_humanoid"
ROBOT_LEARNING = SRC / "robot_learning"


def add(*paths: Path) -> None:
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
