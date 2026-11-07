"""Plot a bench run folder: position, velocity, torque, tracking error and loop timing.

    conda activate sysid
    python -m actuator_sysid.plot_log outputs/actuator_sysid/AK60-39_40_step_1 [more runs] [--show]

Writes plot.png into each run folder. Time axis: seconds from the first command (or first feedback).
"Model torque" = kp*(p_des - p) + kd*(v_des - v) + t_ff from the commands and the feedback just
before each command, i.e. what the motor should have been producing.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

from .logger import load_run


def _t0(a: dict) -> float:
    starts = [a[k][0] for k in ("cmd_t", "fb_t") if len(a[k])]
    return min(starts) if starts else 0.0


def model_torque(a: dict) -> np.ndarray:
    """MIT torque the motor should produce for each command, using the latest earlier feedback."""
    if not len(a["cmd_t"]) or not len(a["fb_t"]):
        return np.full(len(a["cmd_t"]), np.nan)
    i = np.clip(np.searchsorted(a["fb_t"], a["cmd_t"], side="right") - 1, 0, len(a["fb_t"]) - 1)
    p, v = a["fb_position"][i], a["fb_velocity"][i]
    return a["cmd_kp"] * (a["cmd_p"] - p) + a["cmd_kd"] * (a["cmd_v"] - v) + a["cmd_t_ff"]


def plot_run(run_dir: Path, show: bool = False) -> Path:
    import matplotlib.pyplot as plt
    a, meta = load_run(run_dir)
    t0 = _t0(a)
    tc, tf = a["cmd_t"] - t0, a["fb_t"] - t0
    is_cmd = a["cmd_kind"] == "cmd"
    lim = meta.get("limits", {})
    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)

    ax[0].plot(tf, a["fb_position"], label="measured", lw=1)
    ax[0].plot(tc[is_cmd], a["cmd_p"][is_cmd], label="p_des", lw=1, ls="--")
    start = meta.get("start_position")
    if start is not None and lim:
        for edge in (start - lim["position_window"], start + lim["position_window"]):
            ax[0].axhline(edge, color="r", lw=0.6, ls=":")
    ax[0].set_ylabel("position (rad)")

    ax[1].plot(tf, a["fb_velocity"], label="measured", lw=1)
    ax[1].plot(tc[is_cmd], a["cmd_v"][is_cmd], label="v_des", lw=1, ls="--")
    if lim:
        for s in (-1, 1):
            ax[1].axhline(s * lim["max_speed"], color="r", lw=0.6, ls=":")
    ax[1].set_ylabel("velocity (rad/s)")

    ax[2].plot(tf, a["fb_torque"], label="measured", lw=1)
    ax[2].plot(tc, model_torque(a), label="model (from command)", lw=1, alpha=0.8)
    ax[2].plot(tc, a["cmd_t_ff"], label="t_ff", lw=1, ls="--")
    if lim:
        for s in (-1, 1):
            ax[2].axhline(s * lim["max_torque"], color="r", lw=0.6, ls=":")
    ax[2].set_ylabel("torque (Nm)")

    if len(a["fb_t"]) and is_cmd.any():
        held = is_cmd & (a["cmd_kp"] > 0)
        i = np.clip(np.searchsorted(a["fb_t"], a["cmd_t"], side="right") - 1, 0, len(a["fb_t"]) - 1)
        err = np.where(held, a["cmd_p"] - a["fb_position"][i], np.nan)
        ax[3].plot(tc, err, lw=1)
    if lim:
        for s in (-1, 1):
            ax[3].axhline(s * lim["max_track_error"], color="r", lw=0.6, ls=":")
    ax[3].set_ylabel("p_des - p (rad)\n(kp > 0)")

    if len(a["tick_t"]):
        # tick_t counts from the profile start; shift it onto the command time axis.
        first_cmd = tc[is_cmd][0] if is_cmd.any() else 0.0
        ax[4].plot(a["tick_t"] + first_cmd, a["tick_late"] * 1e3, lw=0.6)
    ax[4].set_ylabel("loop lateness (ms)")
    ax[4].set_xlabel("time (s)")

    for e in meta.get("events", []):
        if e["text"].startswith("FAULT") or e["text"].endswith(", stopping"):  # fault / abort_distance stop
            for axis in ax:
                axis.axvline(e["t"] - t0, color="r", lw=1)
    for axis in ax[:3]:
        axis.legend(loc="upper right", fontsize=8)
    for axis in ax:
        axis.grid(alpha=0.3)
    m, res = meta.get("motor", {}), meta.get("result", {})
    prof = meta.get("profile", {})
    fig.suptitle(f"{Path(run_dir).name}: {m.get('name')} id {m.get('id')}, {prof.get('name', '')} "
                 f"(kp {prof.get('kp')}, kd {prof.get('kd')}), ended {res.get('ended')}, fault {res.get('fault')}\n"
                 f"fixture: {meta.get('fixture')}; bus {meta.get('bus', {}).get('kind')}, "
                 f"timestamps {meta.get('bus', {}).get('timestamp_source')}", fontsize=10)
    fig.tight_layout()
    out = Path(run_dir) / "plot.png"
    fig.savefig(out, dpi=110)
    if show:
        plt.show()
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--show", action="store_true", help="also open a window")
    args = ap.parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")
    for run_dir in args.runs:
        print(f"wrote {plot_run(run_dir, args.show)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
