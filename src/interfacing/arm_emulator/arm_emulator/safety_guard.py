"""Refuse to run if this process could be mistaken for, or reach, the real arm.

The teleop and deploy commands are the same ones that drive the real arm, so what keeps them away from it
is the ROS graph they join. An emulator on the real graph would also feed joint_command fake feedback.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

FORBIDDEN_NODES = {"can_node"}
ARPHRD_CAN = "280"  # /sys/class/net/<if>/type of a CAN interface


def environment_failures(env: dict[str, str] | None = None, net_dir: Path = Path("/sys/class/net")) -> list[str]:
    """Why this environment isn't safely apart from the real arm; empty when it is."""
    env = os.environ if env is None else env
    out = []
    domain = env.get("ROS_DOMAIN_ID", "0")
    if domain in ("", "0"):
        out.append("ROS_DOMAIN_ID is 0, the real arm's domain: use the emulator's compose file (it sets 42)")
    if env.get("ROS_LOCALHOST_ONLY") != "1":
        out.append("ROS_LOCALHOST_ONLY is not 1: the nodes could reach other machines on this network")
    if net_dir.is_dir():
        for interface in net_dir.iterdir():
            kind = interface / "type"
            if kind.is_file() and kind.read_text().strip() == ARPHRD_CAN:
                out.append(f"CAN interface {interface.name} is visible: the emulator must have no path to a real bus")
    return out


def graph_failures(node_names: Iterable[str], feedback_publishers: Iterable[str], own_name: str) -> list[str]:
    """Why the ROS graph looks like it has real hardware on it."""
    out = [f"node {name} is on this ROS graph: is the real interfacing stack running?"
           for name in node_names if name in FORBIDDEN_NODES]
    out += [f"{name} also publishes /interfacing/motorFeedback" for name in feedback_publishers if name != own_name]
    return out
