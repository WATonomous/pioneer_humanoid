"""500 Hz loop timing (Ticker), run recording (Recorder) and the run folder format (.npz + .json).

Run folder: outputs/actuator_sysid/<motor>_<id>_<profile>_<n>/ with n = 1, 2, ... (next unused).
  log.npz    arrays (see Recorder.arrays); raw frames are kept so old logs can be re-decoded
             if a unit turns out wrong on the bench (e.g. 0x29 position, Task 10).
  meta.json  motor, firmware, fixture, gains, voltage, temps, git commit, limits, profile, events.
"""
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import can
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "outputs" / "actuator_sysid"
LATE_THRESHOLD = 0.001  # s; ticks later than this are counted in the metadata


class Ticker:
    """Fixed-rate loop clock: sleep most of each period, spin the last `spin` seconds.

    Absolute schedule (no drift). If the loop falls more than one period behind, the missed ticks
    are skipped and counted instead of being sent in a burst.
    """

    def __init__(self, period: float = 0.002, spin: float = 0.0002, clock=time.perf_counter):
        self.period, self.spin, self._clock = period, spin, clock
        self.start = clock()
        self._next = self.start
        self.ticks = self.missed = self.late = 0
        self.max_late = 0.0

    @property
    def elapsed(self) -> float:
        """Scheduled time of the current tick, from the start (s)."""
        return self._next - self.start - self.period

    def wait(self) -> float:
        """Block until the next tick. Returns how late it woke (s)."""
        target = self._next
        while True:
            remaining = target - self._clock()
            if remaining <= 0:
                break
            if remaining > self.spin:
                time.sleep(remaining - self.spin)
        lateness = self._clock() - target
        self.ticks += 1
        self.max_late = max(self.max_late, lateness)
        self.late += lateness > LATE_THRESHOLD
        self._next = target + self.period
        if lateness > self.period:
            skip = int(lateness // self.period)
            self.missed += skip
            self._next += skip * self.period
        return lateness

    def stats(self) -> dict:
        return {"period_s": self.period, "ticks": self.ticks, "late_over_1ms": self.late,
                "missed": self.missed, "max_late_s": self.max_late}


class Recorder:
    """Safety listener that keeps every frame, command and event, plus loop ticks."""

    def __init__(self):
        self.frames = []  # (timestamp, id, extended, echo, dlc, data8)
        self.feedback = []  # (timestamp, position, velocity, torque, current, temperature, error)
        self.commands = []  # (wall time before send, p, v, kp, kd, t, kind)
        self.ticks = []  # (scheduled t from start, lateness)
        self.events = []  # (wall time, text)

    # Safety listener interface
    def on_rx(self, rx, fb) -> None:
        f = rx.frame
        self.frames.append((rx.timestamp, f.arbitration_id, f.is_extended_id, rx.is_echo, len(f.data),
                            bytes(f.data).ljust(8, b"\0")))
        if fb is not None:
            self.feedback.append((rx.timestamp, fb.position, fb.velocity, fb.torque,
                                  np.nan if fb.current is None else fb.current, fb.temperature, fb.error))

    def on_command(self, t, cmd, kind) -> None:
        self.commands.append((t, cmd.p, cmd.v, cmd.kp, cmd.kd, cmd.t, kind))

    def on_event(self, t, text) -> None:
        self.events.append((t, text))

    def on_tick(self, t, lateness) -> None:
        self.ticks.append((t, lateness))

    # Output
    def arrays(self) -> dict[str, np.ndarray]:
        def col(rows, i, dtype):
            return np.array([r[i] for r in rows], dtype=dtype)
        fr, fb, cm, tk = self.frames, self.feedback, self.commands, self.ticks
        return {
            "frame_t": col(fr, 0, np.float64), "frame_id": col(fr, 1, np.uint32),
            "frame_ext": col(fr, 2, bool), "frame_echo": col(fr, 3, bool), "frame_dlc": col(fr, 4, np.uint8),
            "frame_data": np.frombuffer(b"".join(r[5] for r in fr), dtype=np.uint8).reshape(-1, 8),
            "fb_t": col(fb, 0, np.float64), "fb_position": col(fb, 1, np.float64),
            "fb_velocity": col(fb, 2, np.float64), "fb_torque": col(fb, 3, np.float64),
            "fb_current": col(fb, 4, np.float64), "fb_temperature": col(fb, 5, np.int16),
            "fb_error": col(fb, 6, np.uint8),
            "cmd_t": col(cm, 0, np.float64), "cmd_p": col(cm, 1, np.float64), "cmd_v": col(cm, 2, np.float64),
            "cmd_kp": col(cm, 3, np.float64), "cmd_kd": col(cm, 4, np.float64), "cmd_t_ff": col(cm, 5, np.float64),
            "cmd_kind": col(cm, 6, "U5"),
            "tick_t": col(tk, 0, np.float64), "tick_late": col(tk, 1, np.float64),
        }

    def temperatures(self) -> dict:
        temps = [r[5] for r in self.feedback]
        if not temps:
            return {"start": None, "end": None, "max": None}
        return {"start": int(temps[0]), "end": int(temps[-1]), "max": int(max(temps))}


def next_run_dir(out: Path, motor: str, motor_id: int, profile: str) -> Path:
    """Create and return out/<motor>_<id>_<profile>_<n>, n = 1 + the largest existing n."""
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"{motor}_{motor_id}_{profile}_"
    pattern = re.compile(re.escape(prefix) + r"(\d+)$")
    taken = [int(m.group(1)) for p in out.iterdir() if (m := pattern.match(p.name))]
    n = max(taken, default=0) + 1
    while True:
        path = out / f"{prefix}{n}"
        try:
            path.mkdir()  # fails if another run took this number meanwhile
            return path
        except FileExistsError:
            n += 1


def git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD") or "unknown",
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


def save_run(run_dir: Path, recorder: Recorder, meta: dict) -> None:
    """Write log.npz and meta.json. meta gets temps, git, host and counts added."""
    np.savez_compressed(run_dir / "log.npz", **recorder.arrays())
    full = dict(meta)
    full.update({
        "temperatures_c": recorder.temperatures(),
        "git": git_state(),
        "host": platform.node(),
        "python_can": can.__version__,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "counts": {"frames": len(recorder.frames), "feedback": len(recorder.feedback),
                   "commands": len(recorder.commands), "ticks": len(recorder.ticks)},
        "events": [{"t": t, "text": text} for t, text in recorder.events],
    })
    tmp = run_dir / "meta.json.tmp"
    tmp.write_text(json.dumps(full, indent=2, default=_json_default) + "\n")
    os.replace(tmp, run_dir / "meta.json")


def load_run(run_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(Path(run_dir) / "log.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads((Path(run_dir) / "meta.json").read_text())


def _json_default(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON serializable: {type(x)}")
