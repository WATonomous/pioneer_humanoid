#!/usr/bin/env python3
"""Run folders + CSV telemetry, shared by the recorders and tools/gl40_telemetry_plot.py.

Standard library only: it runs in the interfacing container, which has no numpy/matplotlib.

One run = one folder::

    outputs/gl40_bench/20260921-174455_wrist-40deg/
        telemetry.csv   one row per tick, per motor
        run.json        gains, limits, rates, motor model, git sha, abort reason
        *.png           written later by the host-side plotter

CSV columns:

    t_s        seconds since the run started
    source     "script" (raw SocketCAN bench tool) or "ros" (joint_command pipeline)
    motor_id   CAN node id
    joint      human name from arm_calibration.yaml, e.g. shoulder.pitch
    phase      hold | ramp | settle | monitor | step | stream | dwell | return | rest
    sp_deg     setpoint actually SENT to the motor, degrees (after clamp/rate limiting)
    sp_raw_deg the angle that was REQUESTED before moderation (ros source only)
    pos_deg    measured position, degrees
    vel_dps    measured velocity, degrees/second (blank if the drive's scale is unverified)
    tau_nm     measured torque, N.m (MIT drives only)
    current_a  measured current, A (servo-mode feedback only)
    drive_c    drive temperature, degC
    motor_c    motor temperature, degC
    status     drive status/error word
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

CSV_FIELDS = [
    "t_s",
    "source",
    "motor_id",
    "joint",
    "phase",
    "sp_deg",
    "sp_raw_deg",
    "pos_deg",
    "vel_dps",
    "tau_nm",
    "current_a",
    "drive_c",
    "motor_c",
    "status",
    "ff_nm",
    "kp",
]

#: Container mount point for the repo's gitignored outputs/ directory (see
#: modules/docker-compose.interfacing.yaml). Falls back to ./outputs when running on the host.
_CONTAINER_OUTPUTS = Path("/outputs")


def default_run_root() -> Path:
    """Where run folders go, honouring $HUMANOID_TELEMETRY_DIR."""
    env = os.environ.get("HUMANOID_TELEMETRY_DIR")
    if env:
        return Path(env)
    if _CONTAINER_OUTPUTS.is_dir():
        return _CONTAINER_OUTPUTS / "gl40_bench"
    return Path.cwd() / "outputs" / "gl40_bench"


def _git_sha() -> str:
    for cwd in (Path(__file__).resolve().parent, Path.cwd()):
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=cwd, capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
    return "unknown"


def _chown_like_parent(path: Path) -> None:
    """Give new files the mount point's owner, so root-run tools leave no root-owned files."""
    if os.geteuid() != 0:
        return
    try:
        target = path.parent
        while not target.exists() and target != target.parent:
            target = target.parent
        stat = target.stat()
        if stat.st_uid == 0:
            return
        for root, dirs, files in os.walk(path):
            for name in dirs + files:
                os.chown(os.path.join(root, name), stat.st_uid, stat.st_gid)
        os.chown(path, stat.st_uid, stat.st_gid)
    except OSError:
        pass  # best effort -- never fail a run over file ownership


class RunFolder:
    """Creates the folder, streams rows into telemetry.csv, writes run.json on close."""

    def __init__(self, label: str, source: str, meta: Optional[Dict[str, Any]] = None,
                 root: Optional[Path] = None, enabled: bool = True):
        self.enabled = enabled
        self.source = source
        self.meta: Dict[str, Any] = dict(meta or {})
        self.path: Optional[Path] = None
        self._csv_file = None
        self._writer: Optional[csv.DictWriter] = None
        self._t0 = time.monotonic()
        self._rows = 0

        if not enabled:
            return

        safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in label).strip("-")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = root or default_run_root()
        self.path = base / f"{stamp}_{safe}" if safe else base / stamp
        self.path.mkdir(parents=True, exist_ok=True)

        self._csv_file = (self.path / "telemetry.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        # Chown now, not in close(): a killed run would otherwise leave root-owned files.
        _chown_like_parent(self.path)

        self.meta.setdefault("label", label)
        self.meta.setdefault("source", source)
        self.meta.setdefault("started_utc", datetime.now(timezone.utc).isoformat())
        self.meta.setdefault("git_sha", _git_sha())

    # -- writing ---------------------------------------------------------
    def row(self, motor_id: int, phase: str, joint: str = "", sp_deg=None, pos_deg=None,
            vel_dps=None, tau_nm=None, current_a=None, drive_c=None, motor_c=None,
            status=None, t_s: Optional[float] = None, sp_raw_deg=None, ff_nm=None,
            kp=None) -> None:
        if not self.enabled or self._writer is None:
            return

        def num(value, digits=4):
            return "" if value is None else f"{float(value):.{digits}f}"

        self._writer.writerow({
            "t_s": f"{(time.monotonic() - self._t0) if t_s is None else t_s:.4f}",
            "source": self.source,
            "motor_id": motor_id,
            "joint": joint,
            "phase": phase,
            "sp_deg": num(sp_deg),
            "sp_raw_deg": num(sp_raw_deg),
            "pos_deg": num(pos_deg),
            "vel_dps": num(vel_dps),
            "tau_nm": num(tau_nm, 5),
            "current_a": num(current_a),
            "drive_c": "" if drive_c is None else int(drive_c),
            "motor_c": "" if motor_c is None else int(motor_c),
            "status": "" if status is None else status,
            "ff_nm": num(ff_nm, 5),
            "kp": num(kp),
        })
        self._rows += 1
        if self._rows % 50 == 0:
            self._csv_file.flush()

    def note(self, **kwargs: Any) -> None:
        """Add / overwrite run.json fields (e.g. the abort reason as it happens)."""
        self.meta.update(kwargs)

    # -- lifecycle -------------------------------------------------------
    def close(self, outcome: str = "completed", **extra: Any) -> Optional[Path]:
        if not self.enabled or self.path is None:
            return None
        self.meta.setdefault("outcome", outcome)
        self.meta["outcome"] = outcome
        self.meta.update(extra)
        self.meta["rows"] = self._rows
        self.meta["duration_s"] = round(time.monotonic() - self._t0, 3)
        self.meta["ended_utc"] = datetime.now(timezone.utc).isoformat()
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        (self.path / "run.json").write_text(json.dumps(self.meta, indent=2, sort_keys=True),
                                            encoding="utf-8")
        _chown_like_parent(self.path)
        return self.path

    def __enter__(self) -> "RunFolder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close("aborted" if exc_type is not None else self.meta.get("outcome", "completed"))

    def describe(self) -> str:
        if not self.enabled or self.path is None:
            return "telemetry disabled (--no-log)"
        return f"telemetry -> {self.path}"


def load_run(path: Path):
    """Read a run folder back: (meta dict, list of row dicts with floats parsed)."""
    path = Path(path)
    meta_file = path / "run.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    rows = []
    with (path / "telemetry.csv").open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            for key in ("t_s", "sp_deg", "sp_raw_deg", "pos_deg", "vel_dps", "tau_nm",
                        "current_a", "ff_nm", "kp"):
                row[key] = float(raw[key]) if raw.get(key) not in (None, "") else None
            for key in ("motor_id", "drive_c", "motor_c"):
                row[key] = int(raw[key]) if raw.get(key) not in (None, "") else None
            rows.append(row)
    return meta, rows


# ---------------------------------------------------------------------------
# Analysis (stdlib only -- shared by the recorders and the host plotter)
# ---------------------------------------------------------------------------

def split_by_motor(rows) -> Dict[int, list]:
    """Group rows by motor id, preserving order."""
    out: Dict[int, list] = {}
    for row in rows:
        out.setdefault(row["motor_id"], []).append(row)
    return out


def differentiate(t_s, pos_deg, window: int = 3):
    """Central-difference velocity (deg/s), `window` samples each side; None where undefined.

    Used instead of the drives' reported velocity, whose scale is unverified.
    """
    n = len(t_s)
    vel = [None] * n
    for i in range(n):
        lo, hi = max(0, i - window), min(n - 1, i + window)
        dt = t_s[hi] - t_s[lo]
        if hi == lo or dt <= 0 or pos_deg[lo] is None or pos_deg[hi] is None:
            continue
        vel[i] = (pos_deg[hi] - pos_deg[lo]) / dt
    return vel


def _finite(values):
    return [v for v in values if v is not None]


def sustained_velocity(t_s, pos_deg, window_s: float = 0.5) -> float:
    """Fastest average speed (deg/s) over any `window_s`. The per-sample derivative reads ~2x
    when the sampling beats against the feedback rate."""
    best = 0.0
    n = len(t_s)
    j = 0
    for i in range(n):
        if pos_deg[i] is None:
            continue
        while j < n and t_s[j] - t_s[i] < window_s:
            j += 1
        if j >= n or pos_deg[j] is None:
            continue
        dt = t_s[j] - t_s[i]
        if dt > 0:
            best = max(best, abs(pos_deg[j] - pos_deg[i]) / dt)
    return best


# Phases that bring a joint back to where the run started (see motor_metrics).
RETURN_PHASES = ("return", "rest")


def motor_metrics(rows, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Benchmark numbers for one motor's rows. Keys are None when not measurable."""
    meta = meta or {}
    t = [r["t_s"] for r in rows]
    pos = [r["pos_deg"] for r in rows]
    sp = [r["sp_deg"] for r in rows]
    tau = _finite([r["tau_nm"] for r in rows])
    temps = _finite([r["drive_c"] for r in rows])

    meas_vel = _finite(differentiate(t, pos))
    # A step test jumps the setpoint on purpose: differentiate commanded piecewise, cut at steps.
    cuts = [0] + [i for i in range(1, len(rows))
                  if rows[i].get("phase") == "step" and rows[i - 1].get("phase") != "step"]
    cmd_vel = []
    for seg_lo, seg_hi in zip(cuts, cuts[1:] + [len(rows)]):
        cmd_vel += _finite(differentiate(t[seg_lo:seg_hi], sp[seg_lo:seg_hi]))
    # Measured velocity is judged on the non-step stretches only.
    steady_segments, seg_start = [], None
    for i, r in enumerate(rows + [{"phase": "step"}]):
        if r.get("phase") != "step" and seg_start is None:
            seg_start = i
        elif r.get("phase") == "step" and seg_start is not None:
            steady_segments.append((seg_start, i))
            seg_start = None
    limits = meta.get("limits", {}) or {}

    m: Dict[str, Any] = {
        "samples": len(rows),
        "duration_s": round(t[-1] - t[0], 3) if len(t) > 1 else 0.0,
        "peak_measured_vel_dps": round(max((abs(v) for v in meas_vel), default=0.0), 2),
        "sustained_measured_vel_dps": round(max(
            (sustained_velocity(t[lo:hi], pos[lo:hi]) for lo, hi in steady_segments),
            default=0.0), 2),
        "peak_commanded_vel_dps": round(max((abs(v) for v in cmd_vel), default=0.0), 2),
        "peak_requested_vel_dps": round(max(
            (abs(v) for v in _finite(differentiate(t, [r.get("sp_raw_deg") for r in rows]))),
            default=0.0), 2) or None,
        "peak_torque_nm": round(max((abs(v) for v in tau), default=0.0), 4),
        "peak_drive_temp_c": max(temps, default=None),
        "min_pos_deg": round(min(_finite(pos)), 3) if _finite(pos) else None,
        "max_pos_deg": round(max(_finite(pos)), 3) if _finite(pos) else None,
    }

    # Tracking error, where both traces exist.
    errs = [abs(p - s) for p, s in zip(pos, sp) if p is not None and s is not None]
    m["peak_track_err_deg"] = round(max(errs), 3) if errs else None
    m["final_track_err_deg"] = round(errs[-1], 3) if errs else None

    # Round trips: measure the outbound part (up to the first return phase); the way back is
    # scored as the error against the start.
    back_i = next((i for i, r in enumerate(rows) if r.get("phase") in RETURN_PHASES), None)
    # Clamp detection on the outbound part too (a round trip ends by requesting the origin).
    out_rows = rows if back_i is None else rows[:back_i]
    # A clamp is commanded short of requested, not measured short (PD sag).
    raw = _finite([r.get("sp_raw_deg") for r in out_rows])
    out_sp = _finite([r["sp_deg"] for r in out_rows])
    if raw:
        m["requested_deg"] = round(raw[-1], 3)
        # A run stopped mid-ramp is still lagging its request (low-pass), which is not a clamp.
        still_ramping = bool(out_rows) and out_rows[-1].get("phase") == "ramp" and back_i is not None
        if out_sp and abs(raw[-1] - out_sp[-1]) > 0.5 and not still_ramping:
            m["clamped_to_deg"] = round(out_sp[-1], 3)
    if back_i is not None:
        start_sp = next((v for v in sp if v is not None), None)
        end_pos = next((v for v in reversed(pos) if v is not None), None)
        if start_sp is not None and end_pos is not None:
            m["return_err_deg"] = round(end_pos - start_sp, 3)
        t, pos, sp = t[:back_i], pos[:back_i], sp[:back_i]
        if _finite(sp):
            m["benchmark_deg"] = round(_finite(sp)[-1], 3)

    # Step-response shape, measured from the first commanded setpoint change.
    start_i = next((i for i in range(1, len(sp))
                    if sp[i] is not None and sp[0] is not None and abs(sp[i] - sp[0]) > 1e-6), None)
    if start_i is not None and pos[start_i] is not None:
        target = _finite(sp)[-1]
        start_pos = pos[start_i]
        travel = target - start_pos
        if abs(travel) > 1e-6:
            def frac(p):
                return (p - start_pos) / travel

            def cross(level):
                for i in range(start_i, len(pos)):
                    if pos[i] is not None and frac(pos[i]) >= level:
                        return t[i] - t[start_i]
                return None

            t10, t90 = cross(0.1), cross(0.9)
            m["rise_time_s"] = round(t90 - t10, 3) if (t10 is not None and t90 is not None) else None
            peak = max(_finite(pos), key=frac) if _finite(pos) else None
            m["overshoot_pct"] = round(max(0.0, (frac(peak) - 1.0) * 100.0), 2) if peak else None
            # Settling: last time the trace was more than 0.5 deg from target.
            last_out = None
            for i in range(start_i, len(pos)):
                if pos[i] is not None and abs(pos[i] - target) > 0.5:
                    last_out = t[i]
            m["settle_time_s"] = round(last_out - t[start_i], 3) if last_out is not None else 0.0
            m["steady_state_err_deg"] = round(target - pos[-1], 3) if pos[-1] is not None else None

    # Pass/fail against whatever limits the run recorded.
    checks = []
    # This joint's own ceiling; the scalar (min over all MIT joints) is for pre-per-joint runs.
    joint = next((r.get("joint") for r in rows if r.get("joint")), None)
    max_torque = (limits.get("max_torque_nm_by_joint") or {}).get(joint, limits.get("max_torque_nm"))
    if max_torque is not None:
        checks.append(("peak torque <= max_torque",
                       m["peak_torque_nm"] <= max_torque + 1e-9,
                       f"{m['peak_torque_nm']:.3f} / {max_torque} N.m"))
    max_track = limits.get("max_track_err_deg")
    if max_track is not None and m["peak_track_err_deg"] is not None:
        checks.append(("peak tracking error <= max_track_err",
                       m["peak_track_err_deg"] <= max_track + 1e-6,
                       f"{m['peak_track_err_deg']:.2f} / {max_track} deg"))
    soft = limits.get("soft_limits_deg")
    if soft and m["min_pos_deg"] is not None:
        margin = limits.get("soft_limit_margin_deg", 0.0) or 0.0
        lo, hi = soft
        ok = ((lo is None or m["min_pos_deg"] >= lo - margin) and
              (hi is None or m["max_pos_deg"] <= hi + margin))
        checks.append(("position stayed inside soft limits", ok,
                       f"[{m['min_pos_deg']:.1f}, {m['max_pos_deg']:.1f}] vs {soft} deg"))
    # ROS runs: commanded must sit exactly inside arm_calibration.yaml's range; measured gets
    # 2 deg for sag. Joints joint_command never commanded are not judged.
    joint = next((r.get("joint") for r in rows if r.get("joint")), "")
    per_joint = (limits.get("per_joint_deg") or {}).get(joint)
    commanded = _finite([r["sp_deg"] for r in rows])  # the whole run, not the outbound cut
    if not soft and per_joint and commanded:
        lo, hi = per_joint
        checks.append(("commanded stayed inside joint limits",
                       min(commanded) >= lo - 0.05 and max(commanded) <= hi + 0.05,
                       f"[{min(commanded):.2f}, {max(commanded):.2f}] vs [{lo:g}, {hi:g}] deg"))
        if m["min_pos_deg"] is not None:
            checks.append(("measured stayed inside joint limits (+-2 deg sag)",
                           m["min_pos_deg"] >= lo - 2.0 and m["max_pos_deg"] <= hi + 2.0,
                           f"[{m['min_pos_deg']:.2f}, {m['max_pos_deg']:.2f}] vs "
                           f"[{lo:g}, {hi:g}] deg"))
    vel_limit = ((limits.get("velocity_max_dps_per_joint") or {}).get(joint)
                 or limits.get("velocity_max_dps") or limits.get("max_setpoint_vel_dps"))
    if vel_limit is None and limits.get("max_setpoint_vel_rad_s") is not None:
        vel_limit = limits["max_setpoint_vel_rad_s"] * 180.0 / 3.141592653589793
    if vel_limit is not None:
        # Commanded velocity is judged strictly (5% for differentiation); measured gets a wider
        # band (drive dynamics, sampling beat) as a sanity check.
        if m["peak_commanded_vel_dps"]:
            checks.append(("commanded velocity <= velocity_max",
                           m["peak_commanded_vel_dps"] <= vel_limit * 1.05,
                           f"{m['peak_commanded_vel_dps']:.2f} / {vel_limit:.1f} deg/s"))
        checks.append(("sustained measured velocity <= velocity_max",
                       m["sustained_measured_vel_dps"] <= vel_limit * 1.1,
                       f"{m['sustained_measured_vel_dps']:.2f} / {vel_limit:.1f} deg/s "
                       f"(per-sample peak {m['peak_measured_vel_dps']:.2f} includes "
                       f"sampling jitter)"))
    if vel_limit is not None and (m["peak_requested_vel_dps"] or 0) > vel_limit * 1.05:
        # Not a check: notes the run asked for more than the limit, so the limiter was exercised.
        m["velocity_clamp_exercised"] = True
    m["checks"] = [{"name": n, "pass": bool(ok), "detail": d} for n, ok, d in checks]
    m["passed"] = all(c["pass"] for c in m["checks"]) if m["checks"] else None
    return m


def run_metrics(meta: Dict[str, Any], rows) -> Dict[int, Dict[str, Any]]:
    """motor_id -> metrics for a whole run."""
    return {mid: motor_metrics(mrows, meta) for mid, mrows in split_by_motor(rows).items()}


def format_metrics(meta: Dict[str, Any], per_motor: Dict[int, Dict[str, Any]]) -> str:
    """Human-readable report (also the body of summary.md)."""
    lines = []
    label = meta.get("label", "run")
    lines.append(f"# {label}")
    lines.append("")
    bits = [f"source: {meta.get('source', '?')}", f"outcome: {meta.get('outcome', '?')}"]
    if meta.get("git_sha"):
        bits.append(f"git: {meta['git_sha']}")
    lines.append(" | ".join(bits))
    if meta.get("abort_reason"):
        lines.append("")
        lines.append(f"**Aborted:** {meta['abort_reason']}")
    gains = meta.get("gains")
    if gains:
        lines.append("")
        lines.append(f"Gains applied: kp={gains.get('kp_applied')} (raw {gains.get('kp_raw')}), "
                     f"kd={gains.get('kd_applied')} (raw {gains.get('kd_raw')})")

    for motor_id, m in sorted(per_motor.items()):
        joint = next((r for r in [meta.get("joint")] if r), "")
        lines.append("")
        lines.append(f"## motor {motor_id}{f' ({joint})' if joint else ''}")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for key in ("samples", "duration_s", "requested_deg", "clamped_to_deg",
                    "sustained_measured_vel_dps", "peak_measured_vel_dps",
                    "peak_requested_vel_dps", "peak_commanded_vel_dps", "velocity_clamp_exercised",
                    "peak_torque_nm", "peak_track_err_deg", "final_track_err_deg",
                    "benchmark_deg", "rise_time_s", "overshoot_pct", "settle_time_s",
                    "steady_state_err_deg", "return_err_deg", "min_pos_deg", "max_pos_deg", "peak_drive_temp_c"):
            if m.get(key) is not None:
                lines.append(f"| {key} | {m[key]} |")
        if m.get("checks"):
            lines.append("")
            for c in m["checks"]:
                lines.append(f"- {'PASS' if c['pass'] else 'FAIL'} &mdash; {c['name']} "
                             f"({c['detail']})")
    return "\n".join(lines) + "\n"
