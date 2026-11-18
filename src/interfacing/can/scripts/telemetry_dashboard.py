#!/usr/bin/env python3
"""Live telemetry dashboard for the humanoid: one web page with every joint, the CAN bus, the
command pipeline, the host and the event log.

READ-ONLY. It only subscribes; it has no publisher and cannot command a motor. Its warnings
mirror joint_command's limits (arm_actuators.yaml) for display: joint_command's watchdog and the
hardware E-stop stay the only things that stop the arm.

    /interfacing/motorFeedback  MotorFeedback, motor frame   -> position, torque, current, temp
    /interfacing/motorCMD       MotorCmd, motor frame        -> setpoint really sent, gains, mode
    /arm/<side>/joint_targets   ArmPose, command frame       -> what teleop / the policy asked for
    /rosout                     warnings and errors          -> event log

Everything is shown in the command frame (degrees, URDF zero), like telemetry_record.py.
Standard library + PyYAML only (the interfacing container has no numpy), no JS dependencies
(the robot PC may be offline).

Examples (interfacing or joint_command container, ROS sourced)::

  python3 telemetry_dashboard.py                    # http://127.0.0.1:8765
  python3 telemetry_dashboard.py --host 0.0.0.0     # view from another machine on the LAN
  python3 telemetry_dashboard.py --camera ego=/ego_camera/color/image_raw/compressed
  python3 telemetry_dashboard.py --demo             # simulated data, no ROS needed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_config import (ARM_POSE_JOINTS, DEFAULT_CALIBRATIONS, MIT_PROFILES,  # noqa: E402
                          actuator_configs, arm_pose_topic)

STATIC_DIR = Path(__file__).resolve().parent / "dashboard"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}

# MotorCmd.control_type (common_msgs/msg/MotorCmd.msg).
MIT_CONTROL, POSITION_LOOP, POSITION_VELOCITY = 0, 4, 6
MIT_ENTER, MIT_EXIT, MIT_SET_ZERO, MIT_CLEAR_ERRORS = 9, 10, 11, 12
CONTROL_NAMES = {0: "MIT", 1: "current", 2: "brake", 3: "velocity", 4: "position", 5: "set origin",
                 6: "pos-vel", 7: "duty", 8: "disable"}

# Drive status / error words (can/include/mit_protocol.hpp). The families disagree on code 1.
GL2_STATUS = {0: "Disabled", 1: "Enabled", 8: "Over-voltage", 9: "Under-voltage",
              10: "Over-current", 11: "MOSFET over-temp", 12: "Winding over-temp",
              13: "Comms loss", 14: "Overload"}
AK_ERROR = {0: "No fault", 1: "Motor over-temp", 2: "Over-current", 3: "Over-voltage",
            4: "Under-voltage", 5: "Encoder fault", 6: "MOSFET over-temp", 7: "Motor lock-up"}

# Display thresholds, as fractions of the configured limit. The limits themselves are config.
WARN_TEMP_MARGIN_C = 8.0      # warn this far below mit_max_temperature
WARN_TORQUE_FRACTION = 0.8    # of mit_max_torque
WARN_TRACK_FRACTION = 0.7     # of mit_max_track_err
LIMIT_SLACK_DEG = 1.0         # measured this far outside a position limit -> warn
DRIVEN_TIMEOUT_S = 1.0        # no command for this long -> the joint counts as not driven
OFFLINE_AFTER_S = 2.0         # feedback older than this -> shown as lost, not just late

HISTORY_PERIOD_S = 0.2
HISTORY_SECONDS = 900
STREAM_PERIOD_S = 0.1

# Fallback kinematic chain for the 3D view when the URDF is not reachable (the interfacing
# container mounts only src/interfacing). Joint origins and axes of pioneer_bimanual_arm.urdf
# (yaw wrist), in ArmPose slot order; every rpy in that URDF is 0.
FALLBACK_CHAIN = {
    "right": [
        ([0.0216, -0.1633, 0.0605], [0, 1, 0]), ([-0.0428, -0.1160, -0.0002], [1, 0, 0]),
        ([0.0360, 0.0, -0.1163], [0, 0, -1]), ([0.0, -0.0252, -0.1846], [0, -1, 0]),
        ([0.0, 0.0230, -0.1075], [0, 0, 1]), ([0.0, 0.0, -0.1280], [-1, 0, 0]),
    ],
    "left": [
        ([0.0216, 0.1633, 0.0605], [0, -1, 0]), ([-0.0088, 0.1160, -0.0002], [1, 0, 0]),
        ([0.0020, 0.0, -0.1173], [0, 0, -1]), ([0.0, 0.0252, -0.1836], [0, 1, 0]),
        ([0.0, -0.0230, -0.1692], [0, 0, 1]), ([0.0, 0.0, -0.0663], [-1, 0, 0]),
    ],
}
TOOL_OFFSET = [0.0, 0.0, -0.178]  # link6 -> fingertips
# The repo root when run from the source tree (src/interfacing/can/scripts); harmless elsewhere.
_PARENTS = Path(__file__).resolve().parents
REPO_ROOT = _PARENTS[4] if len(_PARENTS) > 4 else Path("/")
URDF_CANDIDATES = [
    str(REPO_ROOT / "assets" / "pioneer_bimanual_arm" / "urdf" / "pioneer_bimanual_arm.urdf"),
    "/assets/pioneer_bimanual_arm/urdf/pioneer_bimanual_arm.urdf",
]


# ----------------------------------------------------------------------------- config loading

def _first_existing(candidates: List[Optional[str]]) -> Optional[Path]:
    for path in candidates:
        if path and Path(path).exists():
            return Path(path).resolve()
    return None


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             cwd=Path(__file__).resolve().parent, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_limb_joints(path: Path, kind: str) -> List[dict]:
    """Every block with a can_id in a calibration file -> one joint dict, in file order.

    Unlike joint_config.load_joint_map this accepts a null direction / zero_offset (the
    uncalibrated leg file): the joint is listed and flagged, and shown in the motor frame.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    joints: List[dict] = []

    def walk(node: Any, side: str, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        if "can_id" in node:
            direction, zero = node.get("direction"), node.get("zero_offset")
            limited = bool(node.get("limit_range", False))
            joints.append({
                "id": int(node["can_id"]),
                "side": side,
                "kind": kind,
                "limb": f"{side} {kind}",
                "name": prefix,
                "direction": float(direction) if direction else 1.0,
                "zero_offset": float(zero) if zero is not None else 0.0,
                "calibrated": direction is not None and zero is not None,
                "lower": float(node["lower_limit"]) if limited and "lower_limit" in node else None,
                "upper": float(node["upper_limit"]) if limited and "upper_limit" in node else None,
                "slot": None,
            })
            return
        for key, child in node.items():
            walk(child, side, f"{prefix}.{key}" if prefix else str(key))

    for side, block in data.items():
        walk(block, str(side), "")
    if kind == "arm":
        slots = {f"{group}.{joint}": i for i, (group, joint) in enumerate(ARM_POSE_JOINTS)}
        for joint in joints:
            joint["slot"] = slots.get(joint["name"])
    return joints


def load_safety(path: Optional[Path]) -> Dict[Tuple[str, str], dict]:
    """{(side, "group.joint"): the joint's block merged over safety.global}; ("", "") = global."""
    if path is None:
        return {}
    cfg = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("safety", {}) or {}
    base = cfg.get("global", {}) or {}
    out: Dict[Tuple[str, str], dict] = {("", ""): dict(base)}
    for side, groups in cfg.items():
        if side in ("global", "units") or not isinstance(groups, dict):
            continue
        for group, members in groups.items():
            for joint, node in (members or {}).items():
                out[(str(side), f"{group}.{joint}")] = {**base, **(node or {})}
    return out


def load_drive_models(path: Optional[Path]) -> Dict[int, dict]:
    if path is None:
        return {}
    motors = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("motors", {}) or {}
    return {int(k): v for k, v in motors.items() if isinstance(v, dict)}


def _num(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def build_joint_table(calibrations: List[Tuple[Path, str]], actuators: Optional[Path],
                      profiles: Optional[Path]) -> List[dict]:
    safety = load_safety(actuators)
    models = load_drive_models(profiles)
    joints: List[dict] = []
    seen: set = set()
    for path, kind in calibrations:
        for joint in load_limb_joints(path, kind):
            if joint["id"] in seen:  # ids must be unique: feedback is matched by id alone
                joint["duplicate_id"] = True
                continue
            seen.add(joint["id"])
            # Only an arm joint is under arm_actuators.yaml; a leg has no safety block yet.
            block = safety.get((joint["side"], joint["name"])) if kind == "arm" else None
            block = block if block is not None else (safety.get(("", ""), {}) if kind == "arm" else {})
            model = models.get(joint["id"], {})
            control_type = block.get("control_type")
            joint.update({
                "model": model.get("model"),
                # The drive's dialect decides what its status word means (GL II 1 = Enabled, AK 1 =
                # over-temp). A drive missing from mit_profiles.yaml falls back to mit_family.
                "family": (block.get("mit_family") if control_type == 0 else model.get("family"))
                or block.get("mit_family"),
                "active": bool(block.get("active", True)) if block else None,
                "control_type": control_type,
                "max_torque": _num(block.get("mit_max_torque")),
                "peak_torque": _num(block.get("mit_peak_torque")),
                "max_temp": _num(block.get("mit_max_temperature")),
                "max_track_err": _num(block.get("mit_max_track_err")),
                "fb_timeout": _num(block.get("mit_feedback_timeout")) or 0.2,
                "velocity_max": _num(block.get("velocity_max")),
                "cfg_kp": _num(block.get("mit_kp")),
                "cfg_kd": _num(block.get("mit_kd")),
            })
            joints.append(joint)
    return joints


def load_chain(urdf: Optional[Path]) -> Tuple[Dict[str, list], str]:
    """Arm kinematic chains for the 3D view: {side: [{"origin", "axis"} x6]}, and their source."""
    chain = {side: [{"origin": o, "axis": a} for o, a in links]
             for side, links in FALLBACK_CHAIN.items()}
    if urdf is None:
        return chain, "built-in copy"
    try:
        parsed: Dict[str, Dict[int, dict]] = {"left": {}, "right": {}}
        for joint in ET.parse(urdf).getroot().findall("joint"):
            match = re.fullmatch(r"joint([1-6])([lL]?)", joint.get("name", ""))
            origin, axis = joint.find("origin"), joint.find("axis")
            if not match or origin is None or axis is None:
                continue
            side = "left" if match.group(2) else "right"
            parsed[side][int(match.group(1)) - 1] = {
                "origin": [float(v) for v in origin.get("xyz", "0 0 0").split()],
                "axis": [float(v) for v in axis.get("xyz", "0 0 1").split()],
            }
        for side, links in parsed.items():
            if len(links) == 6:
                chain[side] = [links[i] for i in range(6)]
        return chain, str(urdf)
    except (ET.ParseError, OSError, ValueError):
        return chain, "built-in copy"


# ----------------------------------------------------------------------------- telemetry store

def _blank_runtime() -> dict:
    return {
        "pos_motor": None, "pos": None, "vel": None, "tau": None, "cur": None, "temp": None,
        "code": 0, "fb_time": None, "fb_count": 0, "fb_hz": 0.0, "seen": False,
        "sp": None, "req": None, "kp": None, "kd": None, "ff": None, "ctrl": None,
        "cmd_time": None, "cmd_count": 0, "cmd_hz": 0.0, "driven": False, "mit_enabled": None,
        "peak_tau": 0.0, "peak_temp": None, "peak_err": 0.0,
        "state": "offline", "why": "",
    }


class TelemetryStore:
    """Thread-safe latest-value store + health rules. No ROS imports, so it is unit-testable.

    Ingest (on_feedback / on_cmd / on_pose / on_log) from any thread, tick() at a few Hz, then
    snapshot() for the page.
    """

    def __init__(self, joints: List[dict], demo: bool = False, clock=time.monotonic):
        self.lock = threading.RLock()
        self.clock = clock
        self.demo = demo
        self.joints = joints
        self.by_id: Dict[int, dict] = {j["id"]: j for j in joints}
        self.rt: Dict[int, dict] = {j["id"]: _blank_runtime() for j in joints}
        self.unknown_ids: Dict[int, int] = {}  # feedback from ids that are in no config file
        self.streams: Dict[str, dict] = {}
        self.events: deque = deque(maxlen=400)
        self.event_seq = 0
        self.history: deque = deque(maxlen=int(HISTORY_SECONDS / HISTORY_PERIOD_S))
        self.system: Dict[str, Any] = {}
        self.ros: Dict[str, Any] = {"connected": False, "nodes": {}, "topics": {}}
        self.start = clock()
        self._rate_mark = clock()
        self._hist_mark = 0.0
        self._overall = ("offline", "")

    # -- ingest ---------------------------------------------------------------------------

    def on_feedback(self, motor_id: int, position: float, velocity: float, current: float,
                    torque: float, temperature: float, error_code: int) -> None:
        now = self.clock()
        with self.lock:
            info, rt = self.by_id.get(motor_id), self.rt.get(motor_id)
            if info is None or rt is None:
                self.unknown_ids[motor_id] = self.unknown_ids.get(motor_id, 0) + 1
                return
            pos = info["zero_offset"] + float(position) / info["direction"]
            if rt["pos"] is not None and rt["fb_time"] is not None:
                dt = now - rt["fb_time"]
                if 1e-4 < dt < 0.5:
                    raw = (pos - rt["pos"]) / dt
                    rt["vel"] = raw if rt["vel"] is None else 0.7 * rt["vel"] + 0.3 * raw
            rt.update(pos_motor=float(position), pos=pos, tau=float(torque), cur=float(current),
                      temp=float(temperature), code=int(error_code), fb_time=now, seen=True)
            rt["fb_count"] += 1
            rt["peak_tau"] = max(rt["peak_tau"], abs(float(torque)))
            if temperature:
                rt["peak_temp"] = max(rt["peak_temp"] or float(temperature), float(temperature))
            if rt["driven"] and rt["sp"] is not None:
                rt["peak_err"] = max(rt["peak_err"], abs(pos - rt["sp"]))

    def on_cmd(self, motor_id: int, control_type: int, position: float, kp: float, kd: float,
               torque: float) -> None:
        now = self.clock()
        with self.lock:
            info, rt = self.by_id.get(motor_id), self.rt.get(motor_id)
            if info is None or rt is None:
                return
            name = f"{info['limb']} {info['name']}"
            if control_type in (MIT_ENTER, MIT_EXIT, MIT_SET_ZERO, MIT_CLEAR_ERRORS):
                label = {MIT_ENTER: "MIT enter (enabled)", MIT_EXIT: "MIT exit (limp)",
                         MIT_SET_ZERO: "zero set at the current position",
                         MIT_CLEAR_ERRORS: "drive errors cleared"}[control_type]
                if control_type in (MIT_ENTER, MIT_EXIT):
                    changed = rt["mit_enabled"] != (control_type == MIT_ENTER)
                    rt["mit_enabled"] = control_type == MIT_ENTER
                    if control_type == MIT_EXIT:
                        rt["driven"] = False
                    if not changed:
                        return
                self._event("info", name, label)
                return
            rt["cmd_time"] = now
            rt["cmd_count"] += 1
            rt["ctrl"] = int(control_type)
            if control_type == MIT_CONTROL:
                rt.update(kp=float(kp), kd=float(kd), ff=float(torque))
                if kp == 0.0:  # limp / damp / seed frame: not a setpoint
                    rt["driven"] = False
                    return
                rt["sp"] = info["zero_offset"] + math.degrees(position) / info["direction"]
                rt["driven"] = True
            elif control_type in (POSITION_LOOP, POSITION_VELOCITY):
                rt.update(kp=None, kd=None, ff=None)
                rt["sp"] = info["zero_offset"] + float(position) / info["direction"]
                rt["driven"] = True
            else:
                rt.update(kp=None, kd=None, ff=None, driven=control_type != 8)

    def on_pose(self, side: str, slots: List[float], include_gripper: bool = False,
                gripper_position: float = 0.0) -> None:
        now = self.clock()
        with self.lock:
            stream = self.streams.setdefault(side, {"count": 0, "hz": 0.0, "time": None,
                                                    "live": False, "gripper": None})
            stream["count"] += 1
            stream["time"] = now
            stream["gripper"] = float(gripper_position) if include_gripper else None
            for info in self.joints:
                if info["side"] != side or info["kind"] != "arm":
                    continue
                if info["slot"] is not None and info["slot"] < len(slots):
                    self.rt[info["id"]]["req"] = float(slots[info["slot"]])
                elif info["name"].startswith("gripper") and include_gripper and info["upper"]:
                    self.rt[info["id"]]["req"] = float(gripper_position) * info["upper"]

    def on_log(self, level: int, source: str, text: str) -> None:
        severity = "fault" if level >= 40 else "warn" if level >= 30 else "info"
        with self.lock:
            # Throttled ROS logs repeat every second or two: keep one line per burst.
            for event in list(self.events)[-8:]:
                if event["src"] == source and event["msg"] == text and \
                        self.clock() - event["mono"] < 10.0:
                    event["n"] += 1
                    return
            self._event(severity, source, text)

    def set_ros(self, connected: bool, nodes: Dict[str, bool], topics: Dict[str, dict]) -> None:
        with self.lock:
            self.ros = {"connected": connected, "nodes": nodes, "topics": topics}

    def set_system(self, system: Dict[str, Any]) -> None:
        with self.lock:
            self.system = system

    # -- health ---------------------------------------------------------------------------

    def _event(self, severity: str, source: str, text: str) -> None:
        self.event_seq += 1
        self.events.append({"seq": self.event_seq, "t": time.time(), "mono": self.clock(),
                            "sev": severity, "src": source, "msg": text, "n": 1})

    @staticmethod
    def code_name(info: dict, code: int) -> str:
        table = AK_ERROR if info.get("family") == "ak" else GL2_STATUS \
            if info.get("family") == "gl2" else {0: "OK"}
        return table.get(code, f"code {code}")

    @staticmethod
    def code_is_fault(info: dict, code: int) -> bool:
        if info.get("family") == "gl2":
            return code >= 8  # 0 Disabled / 1 Enabled are healthy
        return code != 0

    def _joint_state(self, info: dict, rt: dict, now: float) -> Tuple[str, str]:
        if not rt["seen"]:
            return "offline", "no feedback received"
        age = now - rt["fb_time"]
        if age > OFFLINE_AFTER_S:
            return "stale", f"feedback lost {age:.0f} s ago"
        if age > info["fb_timeout"]:
            return "stale", f"feedback {age * 1000:.0f} ms old (limit {info['fb_timeout'] * 1000:.0f})"
        if self.code_is_fault(info, rt["code"]):
            return "fault", f"drive fault: {self.code_name(info, rt['code'])}"
        temp, tau = rt["temp"] or 0.0, abs(rt["tau"] or 0.0)
        if info["max_temp"] and temp >= info["max_temp"]:
            return "fault", f"{temp:.0f} C at or above the {info['max_temp']:.0f} C limit"
        hard_torque = info["peak_torque"] or info["max_torque"]
        if hard_torque and tau > hard_torque:
            return "fault", f"torque {tau:.1f} N.m above the {hard_torque:.1f} N.m cap"
        err = abs(rt["pos"] - rt["sp"]) if rt["driven"] and rt["sp"] is not None else None
        if err is not None and info["max_track_err"] and err > info["max_track_err"]:
            return "fault", f"tracking error {err:.1f} deg above {info['max_track_err']:.1f}"
        if info["max_temp"] and temp >= info["max_temp"] - WARN_TEMP_MARGIN_C:
            return "warn", f"{temp:.0f} C, limit {info['max_temp']:.0f} C"
        if info["max_torque"] and tau >= WARN_TORQUE_FRACTION * info["max_torque"]:
            return "warn", f"torque {tau:.1f} of {info['max_torque']:.1f} N.m rated"
        if err is not None and info["max_track_err"] and \
                err >= WARN_TRACK_FRACTION * info["max_track_err"]:
            return "warn", f"tracking error {err:.1f} of {info['max_track_err']:.1f} deg"
        if info["lower"] is not None and rt["pos"] < info["lower"] - LIMIT_SLACK_DEG:
            return "warn", f"{rt['pos']:.1f} deg is below the lower limit {info['lower']:.1f}"
        if info["upper"] is not None and rt["pos"] > info["upper"] + LIMIT_SLACK_DEG:
            return "warn", f"{rt['pos']:.1f} deg is above the upper limit {info['upper']:.1f}"
        return ("ok", "") if rt["driven"] else ("idle", "feedback healthy, not driven")

    def tick(self) -> None:
        """Rates, health transitions -> events, history. Call at >= 5 Hz."""
        now = self.clock()
        with self.lock:
            window = now - self._rate_mark
            if window >= 1.0:
                for rt in self.rt.values():
                    rt["fb_hz"], rt["cmd_hz"] = rt["fb_count"] / window, rt["cmd_count"] / window
                    rt["fb_count"] = rt["cmd_count"] = 0
                for side, stream in self.streams.items():
                    stream["hz"] = stream["count"] / window
                    stream["count"] = 0
                    live = stream["time"] is not None and now - stream["time"] < 1.0
                    if live != stream["live"]:
                        self._event("info", f"{side} arm targets",
                                    "command stream started" if live else "command stream stopped")
                    stream["live"] = live
                self._rate_mark = now

            rank = {"fault": 4, "stale": 3, "warn": 2}
            for info in self.joints:
                rt = self.rt[info["id"]]
                if rt["driven"] and (rt["cmd_time"] is None or
                                     now - rt["cmd_time"] > DRIVEN_TIMEOUT_S):
                    rt["driven"] = False
                state, why = self._joint_state(info, rt, now)
                if state != rt["state"]:
                    name = f"{info['limb']} {info['name']}"
                    if state in rank:
                        self._event("fault" if state in ("fault", "stale") else "warn", name, why)
                    elif rt["state"] in rank:
                        self._event("info", name, f"recovered ({rt['state']} cleared)")
                    elif rt["state"] == "offline":
                        self._event("info", name, f"online (motor {info['id']})")
                rt["state"], rt["why"] = state, why

            if now - self._hist_mark >= HISTORY_PERIOD_S:
                self._hist_mark = now
                row = {}
                for info in self.joints:
                    rt = self.rt[info["id"]]
                    if rt["seen"] and now - rt["fb_time"] <= OFFLINE_AFTER_S:
                        row[info["id"]] = [_r(rt["pos"]), _r(rt["sp"] if rt["driven"] else None),
                                           _r(rt["req"]), _r(rt["tau"], 3), _r(rt["ff"], 3),
                                           _r(rt["temp"], 0)]
                if row:
                    self.history.append([round(time.time(), 2), row])

    def overall(self) -> Tuple[str, str]:
        states = [(self.by_id[i], rt) for i, rt in self.rt.items()]
        worst = [(info, rt) for info, rt in states if rt["state"] == "fault"]
        if worst:
            info, rt = worst[0]
            more = f" (+{len(worst) - 1} more)" if len(worst) > 1 else ""
            return "fault", f"{info['limb']} {info['name']}: {rt['why']}{more}"
        stale = [(info, rt) for info, rt in states if rt["state"] == "stale"]
        if stale:
            info, rt = stale[0]
            return "degraded", f"{len(stale)} joint(s) lost feedback, first {info['limb']} " \
                               f"{info['name']}: {rt['why']}"
        warn = [(info, rt) for info, rt in states if rt["state"] == "warn"]
        if warn:
            info, rt = warn[0]
            more = f" (+{len(warn) - 1} more)" if len(warn) > 1 else ""
            return "warning", f"{info['limb']} {info['name']}: {rt['why']}{more}"
        if not any(rt["state"] in ("ok", "idle") for _, rt in states):
            return "offline", "no motor feedback: is can_node running and the arm powered?"
        driven = sum(1 for _, rt in states if rt["state"] == "ok")
        online = sum(1 for _, rt in states if rt["state"] in ("ok", "idle"))
        return "nominal", f"{online} joints online, {driven} driven"

    # -- output ---------------------------------------------------------------------------

    def snapshot(self) -> dict:
        now = self.clock()
        with self.lock:
            joints = []
            for info in self.joints:
                rt = self.rt[info["id"]]
                fresh = rt["seen"] and now - rt["fb_time"] <= OFFLINE_AFTER_S
                err = rt["pos"] - rt["sp"] if fresh and rt["driven"] and rt["sp"] is not None \
                    else None
                if rt["ctrl"] == MIT_CONTROL:
                    mode = "MIT" if rt["driven"] else "MIT damp" if (rt["kd"] or 0) > 0 \
                        else "MIT limp"
                elif rt["ctrl"] is not None:
                    mode = CONTROL_NAMES.get(rt["ctrl"], str(rt["ctrl"]))
                else:
                    mode = None
                if rt["cmd_time"] is None or now - rt["cmd_time"] > DRIVEN_TIMEOUT_S:
                    mode = None
                joints.append({
                    "id": info["id"], "st": rt["state"], "why": rt["why"],
                    "pos": _r(rt["pos"]) if fresh else None,
                    "sp": _r(rt["sp"]) if rt["driven"] else None,
                    "req": _r(rt["req"]), "err": _r(err),
                    "vel": _r(rt["vel"], 1) if fresh else None,
                    "tau": _r(rt["tau"], 3) if fresh else None,
                    "cur": _r(rt["cur"], 2) if fresh else None,
                    "temp": _r(rt["temp"], 0) if fresh and rt["temp"] else None,
                    "code": rt["code"] if fresh else None,
                    "codeName": self.code_name(info, rt["code"]) if fresh else None,
                    "hz": _r(rt["fb_hz"], 1), "cmdHz": _r(rt["cmd_hz"], 1),
                    "age": _r(now - rt["fb_time"], 3) if rt["seen"] else None,
                    "kp": _r(rt["kp"], 2), "kd": _r(rt["kd"], 2), "ff": _r(rt["ff"], 3),
                    "mode": mode, "enabled": rt["mit_enabled"],
                    "pkTau": _r(rt["peak_tau"], 2), "pkTemp": _r(rt["peak_temp"], 0),
                    "pkErr": _r(rt["peak_err"], 2),
                })
            state, text = self.overall()
            streams = {side: {"hz": _r(s["hz"], 1), "live": s["live"], "gripper": _r(s["gripper"]),
                              "age": _r(now - s["time"], 2) if s["time"] is not None else None}
                       for side, s in self.streams.items()}
            return {
                "t": round(time.time(), 3), "demo": self.demo, "uptime": round(now - self.start, 1),
                "overall": {"state": state, "text": text}, "joints": joints, "streams": streams,
                "system": self.system, "ros": self.ros,
                "unknownIds": dict(self.unknown_ids), "eventSeq": self.event_seq,
            }

    def events_after(self, seq: int) -> List[dict]:
        with self.lock:
            return [{k: v for k, v in e.items() if k != "mono"}
                    for e in self.events if e["seq"] > seq]

    def history_since(self, seconds: float) -> list:
        cutoff = time.time() - seconds
        with self.lock:
            return [row for row in self.history if row[0] >= cutoff]


def _r(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), digits)


# ----------------------------------------------------------------------------- host + CAN stats

class SystemMonitor:
    """Host load and CAN interface counters, from /proc and /sys (Linux)."""

    def __init__(self, can_interface: str, outputs_dir: Path):
        self.can_interface = can_interface
        self.outputs_dir = outputs_dir
        self._cpu_prev: Optional[Tuple[int, int]] = None
        self._can_prev: Optional[Tuple[float, int, int]] = None
        self._gpu: Optional[dict] = None
        self._gpu_time = 0.0
        self._can_detail: dict = {}
        self._can_detail_time = 0.0

    @staticmethod
    def _read(path: str) -> Optional[str]:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return None

    def _cpu(self) -> Optional[float]:
        line = (self._read("/proc/stat") or "").split("\n", 1)[0].split()
        if len(line) < 5:
            return None
        fields = [int(v) for v in line[1:]]
        idle, total = fields[3] + (fields[4] if len(fields) > 4 else 0), sum(fields)
        prev, self._cpu_prev = self._cpu_prev, (idle, total)
        if prev is None or total == prev[1]:
            return None
        return round(100.0 * (1.0 - (idle - prev[0]) / (total - prev[1])), 1)

    def _memory(self) -> Optional[dict]:
        info = {}
        for line in (self._read("/proc/meminfo") or "").splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                info[key] = int(rest.split()[0]) / 1048576.0
        if "MemTotal" not in info or "MemAvailable" not in info:
            return None
        return {"used": round(info["MemTotal"] - info["MemAvailable"], 1),
                "total": round(info["MemTotal"], 1)}

    def _cpu_temp(self) -> Optional[float]:
        temps = []
        for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            raw = self._read(str(zone))
            if raw and raw.lstrip("-").isdigit() and 0 < int(raw) < 150000:
                temps.append(int(raw) / 1000.0)
        return round(max(temps), 0) if temps else None

    def _gpu_stats(self, now: float) -> Optional[dict]:
        if now - self._gpu_time < 5.0:
            return self._gpu
        self._gpu_time = now
        self._gpu = None
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                     "temperature.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3)
                parts = [p.strip() for p in out.stdout.strip().split("\n")[0].split(",")]
                if out.returncode == 0 and len(parts) == 5:
                    self._gpu = {"name": parts[0], "util": float(parts[1]),
                                 "memUsed": round(float(parts[2]) / 1024, 1),
                                 "memTotal": round(float(parts[3]) / 1024, 1),
                                 "temp": float(parts[4])}
            except (OSError, ValueError, subprocess.SubprocessError):
                self._gpu = None
        return self._gpu

    def _can(self, now: float) -> dict:
        base = f"/sys/class/net/{self.can_interface}"
        if not Path(base).exists():
            self._can_prev = None
            return {"iface": self.can_interface, "present": False}
        stat = {k: int(self._read(f"{base}/statistics/{k}") or 0)
                for k in ("rx_packets", "tx_packets", "rx_errors", "tx_errors", "rx_dropped")}
        out = {"iface": self.can_interface, "present": True,
               "operstate": self._read(f"{base}/operstate"), **stat}
        prev, self._can_prev = self._can_prev, (now, stat["rx_packets"], stat["tx_packets"])
        if prev is not None and now > prev[0]:
            out["rxHz"] = round(max(0, stat["rx_packets"] - prev[1]) / (now - prev[0]), 0)
            out["txHz"] = round(max(0, stat["tx_packets"] - prev[2]) / (now - prev[0]), 0)
        if now - self._can_detail_time >= 2.0 and shutil.which("ip"):
            self._can_detail_time = now
            self._can_detail = {}
            try:
                res = subprocess.run(["ip", "-details", "-statistics", "-json", "link", "show",
                                      self.can_interface], capture_output=True, text=True,
                                     timeout=2)
                link = (json.loads(res.stdout) or [{}])[0] if res.returncode == 0 else {}
                data = (link.get("linkinfo") or {}).get("info_data") or {}
                xstats = (link.get("linkinfo") or {}).get("info_xstats") or {}
                self._can_detail = {
                    "state": data.get("state"),
                    "bitrate": (data.get("bittiming") or {}).get("bitrate"),
                    "berrTx": (data.get("berr_counter") or {}).get("tx"),
                    "berrRx": (data.get("berr_counter") or {}).get("rx"),
                    "busOff": xstats.get("bus_off"), "restarts": xstats.get("restarts"),
                }
            except (OSError, ValueError, subprocess.SubprocessError):
                self._can_detail = {}
        out.update({k: v for k, v in self._can_detail.items() if v is not None})
        return out

    def poll(self) -> dict:
        now = time.monotonic()
        try:
            load = [round(v, 2) for v in os.getloadavg()]
        except OSError:
            load = None
        disk = None
        try:
            target = self.outputs_dir if self.outputs_dir.exists() else Path("/")
            usage = shutil.disk_usage(target)
            disk = {"free": round(usage.free / 2**30, 1), "total": round(usage.total / 2**30, 1),
                    "path": str(target)}
        except OSError:
            pass
        return {
            "host": {"name": os.uname().nodename, "cpu": self._cpu(), "cores": os.cpu_count(),
                     "load": load, "mem": self._memory(), "temp": self._cpu_temp(),
                     "gpu": self._gpu_stats(now), "disk": disk},
            "can": self._can(now),
        }


# ----------------------------------------------------------------------------- data sources

class CameraHub:
    """Latest JPEG per camera, relayed as MJPEG. CompressedImage topics only (no decoder here)."""

    def __init__(self, names: List[str]):
        self.cond = threading.Condition()
        self.frames: Dict[str, Tuple[int, bytes]] = {name: (0, b"") for name in names}

    def push(self, name: str, data: bytes) -> None:
        with self.cond:
            self.frames[name] = (self.frames[name][0] + 1, data)
            self.cond.notify_all()

    def wait(self, name: str, last: int, timeout: float = 2.0) -> Tuple[int, bytes]:
        with self.cond:
            self.cond.wait_for(lambda: self.frames[name][0] != last, timeout)
            return self.frames[name]


def run_ros(store: TelemetryStore, sides: List[str], cameras: Dict[str, str], hub: CameraHub,
            stop: threading.Event) -> None:
    """Subscribe-only ROS node. Imported lazily so --demo works without ROS."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rcl_interfaces.msg import Log
    from sensor_msgs.msg import CompressedImage

    from common_msgs.msg import ArmPose, MotorCmd, MotorFeedback

    class DashboardNode(Node):
        def __init__(self):
            super().__init__("telemetry_dashboard")
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=100)
            self.counts: Dict[str, int] = {}
            self.mark = time.monotonic()
            self.watch = {"/interfacing/motorFeedback": "can_node",
                          "/interfacing/motorCMD": "joint_command_node"}
            self.create_subscription(MotorFeedback, "/interfacing/motorFeedback",
                                     self.on_feedback, qos)
            self.create_subscription(MotorCmd, "/interfacing/motorCMD", self.on_cmd, qos)
            for side in sides:
                topic = arm_pose_topic(side)
                self.watch[topic] = "teleop / policy"
                self.create_subscription(ArmPose, topic,
                                         lambda msg, side=side, topic=topic:
                                         self.on_pose(msg, side, topic), qos)
            self.create_subscription(Log, "/rosout", self.on_log, 100)
            for name, topic in cameras.items():
                self.watch[topic] = f"camera {name}"
                self.create_subscription(CompressedImage, topic,
                                         lambda msg, name=name, topic=topic:
                                         self.on_image(msg, name, topic), qos_profile_sensor_data)
            self.create_timer(1.0, self.on_graph)

        def bump(self, topic: str) -> None:
            self.counts[topic] = self.counts.get(topic, 0) + 1

        def on_feedback(self, msg) -> None:
            self.bump("/interfacing/motorFeedback")
            store.on_feedback(int(msg.motor_id), float(msg.position), float(msg.velocity),
                              float(msg.current), float(msg.torque), float(msg.temperature),
                              int(msg.error_code))

        def on_cmd(self, msg) -> None:
            self.bump("/interfacing/motorCMD")
            store.on_cmd(int(msg.motor_id), int(msg.control_type), float(msg.position),
                         float(msg.kp), float(msg.kd), float(msg.torque))

        def on_pose(self, msg, side: str, topic: str) -> None:
            self.bump(topic)
            slots = list(msg.shoulder.position[:3]) + list(msg.elbow.position[:2]) + \
                list(msg.wrist.position[:1])
            store.on_pose(side, [float(v) for v in slots], bool(msg.include_gripper),
                          float(msg.gripper_position))

        def on_log(self, msg) -> None:
            if msg.level >= 30 and msg.name != self.get_name():
                store.on_log(int(msg.level), msg.name, msg.msg)

        def on_image(self, msg, name: str, topic: str) -> None:
            self.bump(topic)
            if "jpeg" in msg.format.lower() or "jpg" in msg.format.lower():
                hub.push(name, bytes(msg.data))

        def on_graph(self) -> None:
            now = time.monotonic()
            window, self.mark = now - self.mark, now
            names = set(self.get_node_names())
            nodes = {name: name in names for name in ("can_node", "joint_command_node")}
            topics = {}
            for topic, source in self.watch.items():
                topics[topic] = {"hz": round(self.counts.get(topic, 0) / max(window, 1e-3), 1),
                                 "pubs": self.count_publishers(topic), "source": source}
            self.counts = {}
            store.set_ros(True, nodes, topics)

    rclpy.init()
    node = DashboardNode()
    try:
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def run_demo(store: TelemetryStore, stop: threading.Event) -> None:
    """Simulated robot, fed through the same ingest path as ROS. Clearly labelled on the page."""
    t0 = time.monotonic()
    arms = [j for j in store.joints if j["kind"] == "arm"]
    dead = {j["id"] for j in arms if j["side"] == "right" and j["name"].startswith("wrist")}

    def span(info: dict) -> Tuple[float, float]:
        lower = info["lower"] if info["lower"] is not None else -60.0
        upper = info["upper"] if info["upper"] is not None else 60.0
        return (lower + upper) / 2.0, min(25.0, (upper - lower) * 0.3)

    state = {j["id"]: {"pos": span(j)[0] + (span(j)[1] * math.sin(i) if j["side"] == "left" else 0.0),
                        "temp": 30.0 + (j["id"] % 5)} for i, j in enumerate(arms)}
    step = 0
    store.on_log(30, "demo", "Simulated telemetry: these numbers do not come from the robot")
    while not stop.is_set():
        t = time.monotonic() - t0
        step += 1
        requested: Dict[str, List[float]] = {}
        for index, info in enumerate(arms):
            if info["id"] in dead:
                continue
            mid, amp = span(info)
            sim = state[info["id"]]
            driven = info["side"] == "left" and info.get("active") is not False
            cap = info["max_torque"] or 5.0
            torque, heat = 0.0, 0.0
            if driven:
                target = mid + amp * math.sin(0.5 * t + index)
                sim["pos"] += (target - sim["pos"]) * 0.12
                torque = cap * (0.3 * math.sin(math.radians(sim["pos"]) + index)
                                + 0.03 * (target - sim["pos"]))
                # The elbow runs warm on purpose, to show a temperature warning.
                heat = 0.01 * abs(torque) / cap + (0.03 if info["name"] == "elbow.pitch" else 0.0)
                motor_deg = info["direction"] * (target - info["zero_offset"])
                store.on_cmd(info["id"], MIT_CONTROL, math.radians(motor_deg),
                             info.get("cfg_kp") or 40.0, info.get("cfg_kd") or 0.6, 0.0)
                if info["slot"] is not None:
                    requested.setdefault(info["side"], [0.0] * 6)[info["slot"]] = \
                        mid + amp * math.sin(0.5 * (t + 0.12) + index)
            sim["temp"] += heat - 0.0015 * (sim["temp"] - 30.0)
            motor = info["direction"] * (sim["pos"] - info["zero_offset"])
            healthy = 1 if info.get("family") == "gl2" else 0
            store.on_feedback(info["id"], motor + 0.05 * math.sin(40 * t + index), 0.0,
                              abs(torque) / 1.3, torque, round(sim["temp"]), healthy)
        if step % 2 == 0:
            for side, slots in requested.items():
                store.on_pose(side, slots, True, 0.5 + 0.5 * math.sin(0.3 * t))
        stop.wait(0.02)


# ----------------------------------------------------------------------------- HTTP

class Dashboard:
    def __init__(self, store: TelemetryStore, config: dict, hub: CameraHub):
        self.store, self.config, self.hub = store, config, hub
        self.stop = threading.Event()


def make_handler(app: Dashboard):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet: one line per request floods at 10 Hz
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any) -> None:
            self._send(json.dumps(payload, separators=(",", ":")).encode(), "application/json")

        def do_GET(self):  # noqa: N802 - http.server's name
            url = urlparse(self.path)
            try:
                if url.path in STATIC_FILES:
                    name, content_type = STATIC_FILES[url.path]
                    self._send((STATIC_DIR / name).read_bytes(), content_type)
                elif url.path == "/api/config":
                    self._json(app.config)
                elif url.path == "/api/snapshot":
                    self._json(app.store.snapshot())
                elif url.path == "/api/history":
                    seconds = float(parse_qs(url.query).get("s", ["600"])[0])
                    self._json(app.store.history_since(min(max(seconds, 1.0), HISTORY_SECONDS)))
                elif url.path == "/api/stream":
                    self._stream()
                elif url.path.startswith("/camera/") and url.path[8:] in app.hub.frames:
                    self._camera(url.path[8:])
                else:
                    self._send(b"not found", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError, ValueError):
                pass

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            seq = max(0, app.store.event_seq - 200)
            while not app.stop.is_set():
                snap = app.store.snapshot()
                events = app.store.events_after(seq)
                if events:
                    seq = events[-1]["seq"]
                    snap["events"] = events
                self.wfile.write(b"data: " + json.dumps(snap, separators=(",", ":")).encode()
                                 + b"\n\n")
                self.wfile.flush()
                time.sleep(STREAM_PERIOD_S)

        def _camera(self, name: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            last = 0
            while not app.stop.is_set():
                count, frame = app.hub.wait(name, last)
                if count == last or not frame:
                    continue
                last = count
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                self.wfile.flush()

    return Handler


def background(store: TelemetryStore, monitor: SystemMonitor, stop: threading.Event) -> None:
    last_system = 0.0
    while not stop.is_set():
        store.tick()
        now = time.monotonic()
        if now - last_system >= 1.0:
            last_system = now
            store.set_system(monitor.poll())
        stop.wait(0.1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to serve on (default 127.0.0.1; 0.0.0.0 = the whole LAN, no login)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--calibration", help="path to arm_calibration.yaml")
    ap.add_argument("--leg-calibration", help="path to leg_calibration.yaml (default: next to the "
                    "arm file, if present)")
    ap.add_argument("--actuators", help="path to arm_actuators.yaml")
    ap.add_argument("--urdf", help="URDF for the 3D view (default: the repo's pioneer_bimanual_arm)")
    ap.add_argument("--can-interface", default="can0")
    ap.add_argument("--camera", action="append", default=[], metavar="NAME=TOPIC",
                    help="show a sensor_msgs/CompressedImage (jpeg) topic; repeatable")
    ap.add_argument("--demo", action="store_true", help="simulated data, no ROS (UI check only)")
    args = ap.parse_args(argv)

    arm_file = _first_existing([args.calibration] if args.calibration else DEFAULT_CALIBRATIONS)
    if arm_file is None:
        raise SystemExit("could not find arm_calibration.yaml; pass --calibration PATH")
    leg_file = _first_existing([args.leg_calibration] if args.leg_calibration
                               else [str(arm_file.parent / "leg_calibration.yaml")])
    actuators = _first_existing([args.actuators] if args.actuators else actuator_configs("arm"))
    profiles = _first_existing(MIT_PROFILES)
    calibrations = [(arm_file, "arm")] + ([(leg_file, "leg")] if leg_file else [])
    joints = build_joint_table(calibrations, actuators, profiles)
    chain, chain_source = load_chain(_first_existing([args.urdf] if args.urdf else URDF_CANDIDATES))

    cameras: Dict[str, str] = {}
    for item in args.camera:
        name, sep, topic = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not topic.startswith("/"):
            raise SystemExit(f"--camera wants NAME=/topic, got '{item}'")
        cameras[name] = topic

    store = TelemetryStore(joints, demo=args.demo)
    hub = CameraHub(list(cameras))
    sides = sorted({j["side"] for j in joints if j["kind"] == "arm"})
    repo_outputs = REPO_ROOT / "outputs"
    outputs = next((p for p in (Path("/outputs"), repo_outputs) if p.is_dir()), Path("/"))
    config = {
        "robot": "Pioneer humanoid", "demo": args.demo, "git": _git_sha(),
        "joints": joints, "chain": chain, "tool": TOOL_OFFSET, "chainSource": chain_source,
        "cameras": cameras, "canInterface": args.can_interface,
        "files": {"calibration": str(arm_file), "legCalibration": str(leg_file) if leg_file else None,
                  "actuators": str(actuators) if actuators else None,
                  "mitProfiles": str(profiles) if profiles else None},
        "thresholds": {"warnTempMarginC": WARN_TEMP_MARGIN_C,
                       "warnTorqueFraction": WARN_TORQUE_FRACTION,
                       "warnTrackFraction": WARN_TRACK_FRACTION},
        "historySeconds": HISTORY_SECONDS,
    }
    app = Dashboard(store, config, hub)

    threads = [threading.Thread(target=background, daemon=True,
                                args=(store, SystemMonitor(args.can_interface, outputs), app.stop))]
    if args.demo:
        threads.append(threading.Thread(target=run_demo, args=(store, app.stop), daemon=True))
    else:
        threads.append(threading.Thread(target=run_ros, daemon=True,
                                        args=(store, sides, cameras, hub, app.stop)))
    for thread in threads:
        thread.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True
    print(f"calibration: {arm_file}")
    print(f"actuators:   {actuators or 'NOT FOUND (no limits shown)'}")
    print(f"joints:      {len(joints)} ({', '.join(sorted({j['limb'] for j in joints}))})")
    print(f"source:      {'DEMO (simulated)' if args.demo else 'ROS (subscribe-only)'}")
    print(f"dashboard:   http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
