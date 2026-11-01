#!/usr/bin/env python3
"""Plot a telemetry run folder: joint angle, velocity, tracking error and torque over time.

Runs on the HOST (the robot images have no numpy/matplotlib)::

    uv run --with matplotlib --with numpy tools/gl40_telemetry_plot.py outputs/gl40_bench/<run>

Produces, in the run folder itself:

    angle.png     commanded vs measured angle per motor, with limits and phase bands
    velocity.png  measured joint velocity vs the configured ceiling
    tracking.png  |commanded - measured| vs the fault threshold
    torque.png    torque (MIT) or current (servo) vs the ceiling
    temperature.png  drive temperature vs each joint's mit_max_temperature
    summary.md    the same numbers as text

`--sweep RUN [RUN ...]` overlays several runs instead, for gain comparisons.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Reuse telemetry.py's loader and metrics.
_SCRIPTS = Path(__file__).resolve().parent.parent / "src" / "interfacing" / "can" / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import telemetry  # noqa: E402

# --- design tokens (light surface) ------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e4e3df"
# Validated categorical order -- slot per motor id, assigned in sorted order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
CRITICAL = "#e34948"   # status colour: reserved for limits/faults, never a series
COMMANDED = "#52514e"  # setpoints are reference lines, not an identity

LINE_W = 2.0
PHASE_BANDS = {"hold": "#f2f1ed", "ramp": "#e8eef8", "settle": "#eef6f2", "step": "#fbeee8",
               "dwell": "#eef6f2", "return": "#f3ecf7", "rest": "#f2f1ed"}


def style_axes(ax, title: str, ylabel: str, xlabel: str = "time (s)") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


#: Header height in inches, so the title fits whatever the panel count.
_HEADER_IN = 0.85
_PANEL_IN = 2.9


def new_figure(rows: int, title: str, subtitle: str = ""):
    height = _PANEL_IN * rows + _HEADER_IN
    fig, axes = plt.subplots(rows, 1, figsize=(11, height), squeeze=False, facecolor=SURFACE)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left",
                 y=1 - 0.30 / height, va="top")
    if subtitle:
        fig.text(0.01, 1 - 0.62 / height, subtitle, color=INK_SECONDARY, fontsize=9,
                 ha="left", va="top")
    fig._header_frac = _HEADER_IN / height  # consumed by save()
    return fig, [ax[0] for ax in axes]


def layout(fig) -> None:
    # Right margin leaves room for the direct labels, which sit outside the axes.
    fig.tight_layout(rect=(0, 0, 0.88, 1 - getattr(fig, "_header_frac", 0.15)))


def save(fig, path: Path) -> Path:
    layout(fig)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def label_end(ax, xs, ys, text: str, color: str, dy: int = 0) -> None:
    """Direct label at the end of a trace; `dy` staggers labels where traces converge."""
    for x, y in zip(reversed(xs), reversed(ys)):
        if y is not None:
            ax.annotate(text, xy=(x, y), xytext=(6, dy), textcoords="offset points",
                        color=color, fontsize=8, va="center", ha="left",
                        annotation_clip=False)
            return


def trailing_max(t, ys, window_s: float = 1.0):
    """Highest reading over the trailing `window_s`. A whole-degree sensor dithers between two
    values; this drops the dips but keeps every reading that reaches a limit."""
    out, j = [], 0
    for i in range(len(t)):
        while t[i] - t[j] > window_s:
            j += 1
        out.append(max(ys[j:i + 1]))
    return out


def shade_phases(ax, rows) -> None:
    """Background bands showing hold / ramp / settle, so the shape of a run is readable."""
    if not rows:
        return
    start = rows[0]["t_s"]
    current = rows[0]["phase"]
    seen = set()
    for row in rows:
        if row["phase"] != current:
            if current in PHASE_BANDS:
                ax.axvspan(start, row["t_s"], color=PHASE_BANDS[current], zorder=0)
                seen.add(current)
            start, current = row["t_s"], row["phase"]
    if current in PHASE_BANDS:
        ax.axvspan(start, rows[-1]["t_s"], color=PHASE_BANDS[current], zorder=0)
        seen.add(current)
    for phase in seen:
        mid = [r["t_s"] for r in rows if r["phase"] == phase]
        if mid:
            # Along the bottom: the top of the axes is where limit rules and their labels live.
            ax.annotate(phase, xy=(sum(mid) / len(mid), 0.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 4), textcoords="offset points", color=INK_MUTED,
                        fontsize=7, ha="center", va="bottom")


def mark_abort(ax, meta: Dict, rows) -> None:
    if meta.get("outcome") not in ("aborted", "error") or not rows:
        return
    t_end = rows[-1]["t_s"]
    ax.axvline(t_end, color=CRITICAL, linewidth=1.6, linestyle="-", zorder=5)
    reason = str(meta.get("abort_reason", "aborted"))
    ax.annotate(reason[:60], xy=(t_end, ax.get_ylim()[1]), xytext=(-6, -6),
                textcoords="offset points", color=CRITICAL, fontsize=8, ha="right", va="top")


def limit_lines(ax, lo: Optional[float], hi: Optional[float], label: str = "limit") -> bool:
    drawn = False
    for value, side in ((lo, "lower"), (hi, "upper")):
        if value is None:
            continue
        ax.axhline(value, color=CRITICAL, linewidth=1.2, linestyle=(0, (6, 4)), zorder=4)
        ax.annotate(f"{label} {value:.4g}", xy=(ax.get_xlim()[0], value),
                    xytext=(4, 3 if side == "upper" else -11), textcoords="offset points",
                    color=CRITICAL, fontsize=8, va="bottom", ha="left")
        drawn = True
    return drawn


def joint_limits(meta: Dict, motor_id: int, joint: str):
    """(lo, hi) in the frame the trace is plotted in, or (None, None)."""
    limits = meta.get("limits", {}) or {}
    soft = limits.get("soft_limits_deg")
    if soft:
        return soft[0], soft[1]
    per_joint = limits.get("per_joint_deg", {}) or {}
    if joint in per_joint:
        return per_joint[joint][0], per_joint[joint][1]
    return None, None


def velocity_ceiling(meta: Dict) -> Optional[float]:
    limits = meta.get("limits", {}) or {}
    if limits.get("velocity_max_dps") is not None:
        return limits["velocity_max_dps"]
    if limits.get("max_setpoint_vel_rad_s") is not None:
        return limits["max_setpoint_vel_rad_s"] * 180.0 / 3.141592653589793
    return None


def run_subtitle(meta: Dict) -> str:
    bits = [f"source {meta.get('source', '?')}", f"outcome {meta.get('outcome', '?')}"]
    gains = meta.get("gains")
    if gains:
        bits.append(f"kp {gains.get('kp_applied'):.4g} (raw {gains.get('kp_raw')})")
        bits.append(f"kd {gains.get('kd_applied'):.4g} (raw {gains.get('kd_raw')})")
    if meta.get("git_sha"):
        bits.append(f"git {meta['git_sha']}")
    return "  |  ".join(str(b) for b in bits)


def plot_run(run_dir: Path) -> List[Path]:
    meta, rows = telemetry.load_run(run_dir)
    if not rows:
        # Header-only CSV: the run died before reaching the bus. Nothing to plot.
        print(f"skipping {run_dir.name}: telemetry.csv has no data rows "
              f"({'no run.json either -- the run never completed' if not meta else ''})")
        return []
    by_motor = telemetry.split_by_motor(rows)
    metrics = telemetry.run_metrics(meta, rows)
    motors = sorted(by_motor)
    label = meta.get("label", run_dir.name)
    written: List[Path] = []

    # --- angle: one panel per motor (small multiples -- never 7 motors on one axis) ---
    fig, axes = new_figure(len(motors), f"Joint angle over time - {label}", run_subtitle(meta))
    for ax, motor_id in zip(axes, motors):
        mrows = by_motor[motor_id]
        joint = mrows[0]["joint"] or f"motor {motor_id}"
        t = [r["t_s"] for r in mrows]
        pos = [r["pos_deg"] for r in mrows]
        sp = [r["sp_deg"] for r in mrows]
        colour = SERIES[motors.index(motor_id) % len(SERIES)]

        style_axes(ax, f"{joint}  (motor {motor_id})", "angle (deg)")
        if any(v is not None for v in sp):
            ax.plot(t, sp, color=COMMANDED, linewidth=1.6, linestyle=(0, (5, 3)),
                    label="commanded (after limits)", zorder=3)
            label_end(ax, t, sp, "commanded", COMMANDED, dy=7)
        ax.plot(t, pos, color=colour, linewidth=LINE_W, label="measured", zorder=4)
        label_end(ax, t, pos, "measured", colour, dy=-7)

        lo, hi = joint_limits(meta, motor_id, joint)
        has_limits = limit_lines(ax, lo, hi)
        # The ArmPose request, before the clamp. y-range frozen first, so an over-limit request
        # runs off the panel while commanded stops at the limit.
        raw = [r.get("sp_raw_deg") for r in mrows]
        has_request = any(v is not None for v in raw)
        if has_request:
            y_lo, y_hi = ax.get_ylim()
            ax.plot(t, raw, color=INK_MUTED, linewidth=1.0, linestyle=(0, (1, 2)),
                    label="requested", zorder=2)
            ax.set_ylim(y_lo, y_hi)
        shade_phases(ax, mrows)
        mark_abort(ax, meta, mrows)

        handles = [Line2D([], [], color=COMMANDED, linewidth=1.6, linestyle=(0, (5, 3)),
                          label="commanded"),
                   Line2D([], [], color=colour, linewidth=LINE_W, label="measured")]
        if has_request:
            handles.insert(0, Line2D([], [], color=INK_MUTED, linewidth=1.0,
                                     linestyle=(0, (1, 2)), label="requested"))
        if has_limits:
            handles.append(Line2D([], [], color=CRITICAL, linewidth=1.2, linestyle=(0, (6, 4)),
                                  label="joint limit"))
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8,
                  labelcolor=INK_SECONDARY)
        m = metrics.get(motor_id, {})
        if m.get("requested_deg") is not None and m.get("clamped_to_deg") is not None:
            # Usually off-scale, so annotated rather than plotted.
            ax.annotate(f"requested {m['requested_deg']:+.1f} deg -> clamped to "
                        f"{m['clamped_to_deg']:+.1f}",
                        xy=(0.985, 0.93), xycoords="axes fraction", color=CRITICAL,
                        fontsize=8, ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                                  edgecolor="none", alpha=0.85))
        if m.get("steady_state_err_deg") is not None:
            ax.annotate(f"steady-state error {m['steady_state_err_deg']:+.2f} deg",
                        xy=(0.015, 0.13), xycoords="axes fraction", color=INK_SECONDARY,
                        fontsize=8, ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                                  edgecolor="none", alpha=0.85))
    written.append(save(fig, run_dir / "angle.png"))

    # --- velocity: differentiated position (the trustworthy measure) ---
    ceiling = velocity_ceiling(meta)
    fig, axes = new_figure(1, f"Joint velocity over time - {label}",
                           "differentiated from measured angle; the drives' reported velocity "
                           "depends on an unverified parameter-page scale")
    ax = axes[0]
    style_axes(ax, "measured joint velocity", "velocity (deg/s)")
    for motor_id in motors:
        mrows = by_motor[motor_id]
        joint = mrows[0]["joint"] or f"motor {motor_id}"
        t = [r["t_s"] for r in mrows]
        vel = telemetry.differentiate(t, [r["pos_deg"] for r in mrows])
        colour = SERIES[motors.index(motor_id) % len(SERIES)]
        ax.plot(t, vel, color=colour, linewidth=LINE_W, label=joint, zorder=4)
        label_end(ax, t, vel, joint, colour)
    if ceiling is not None:
        limit_lines(ax, -ceiling, ceiling, "velocity limit")
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY,
              ncol=min(3, len(motors)))
    written.append(save(fig, run_dir / "velocity.png"))

    # --- tracking error ---
    if any(r["sp_deg"] is not None and r["pos_deg"] is not None for r in rows):
        fig, axes = new_figure(1, f"Tracking error over time - {label}", run_subtitle(meta))
        ax = axes[0]
        style_axes(ax, "|commanded - measured|", "error (deg)")
        for motor_id in motors:
            mrows = by_motor[motor_id]
            joint = mrows[0]["joint"] or f"motor {motor_id}"
            pts = [(r["t_s"], abs(r["pos_deg"] - r["sp_deg"]))
                   for r in mrows if r["pos_deg"] is not None and r["sp_deg"] is not None]
            if not pts:
                continue
            colour = SERIES[motors.index(motor_id) % len(SERIES)]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=colour, linewidth=LINE_W, label=joint, zorder=4)
            label_end(ax, list(xs), list(ys), joint, colour)
        threshold = (meta.get("limits", {}) or {}).get("max_track_err_deg")
        if threshold is not None:
            limit_lines(ax, None, threshold, "fault at")
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY,
                  ncol=min(3, len(motors)))
        written.append(save(fig, run_dir / "tracking.png"))

    # --- torque / current ---
    has_torque = any(r["tau_nm"] is not None for r in rows)
    has_current = any(r["current_a"] is not None for r in rows)
    if has_torque or has_current:
        panels = int(has_torque) + int(has_current)
        fig, axes = new_figure(panels, f"Motor effort over time - {label}", run_subtitle(meta))
        idx = 0
        if has_torque:
            ax = axes[idx]
            idx += 1
            style_axes(ax, "torque (MIT drives)", "torque (N.m)")
            for motor_id in motors:
                mrows = [r for r in by_motor[motor_id] if r["tau_nm"] is not None]
                if not mrows:
                    continue
                joint = mrows[0]["joint"] or f"motor {motor_id}"
                colour = SERIES[motors.index(motor_id) % len(SERIES)]
                xs = [r["t_s"] for r in mrows]
                ys = [r["tau_nm"] for r in mrows]
                ax.plot(xs, ys, color=colour, linewidth=LINE_W, label=joint, zorder=4)
                label_end(ax, xs, ys, joint, colour)
            limits = meta.get("limits", {}) or {}
            by_joint = limits.get("max_torque_nm_by_joint") or {}
            plotted = {r["joint"] for r in rows if r["tau_nm"] is not None and r["joint"]}
            joints = [j for j in (meta.get("moving") or sorted(plotted)) if j in by_joint]
            if joints:
                # Each joint against its own ceiling; one line per distinct value.
                ceilings: Dict[float, List[str]] = {}
                for j in joints:
                    ceilings.setdefault(float(by_joint[j]), []).append(j)
                for value, names in sorted(ceilings.items()):
                    limit_lines(ax, -value, value, f"{', '.join(names)} ceiling")
            elif limits.get("max_torque_nm") is not None:
                ceiling_nm = limits["max_torque_nm"]
                limit_lines(ax, -ceiling_nm, ceiling_nm, "torque ceiling")
            ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY,
                      ncol=min(3, len(motors)))
        if has_current:
            ax = axes[idx]
            style_axes(ax, "current (servo drives)", "current (A)")
            for motor_id in motors:
                mrows = [r for r in by_motor[motor_id] if r["current_a"] is not None]
                if not mrows:
                    continue
                joint = mrows[0]["joint"] or f"motor {motor_id}"
                colour = SERIES[motors.index(motor_id) % len(SERIES)]
                xs = [r["t_s"] for r in mrows]
                ys = [r["current_a"] for r in mrows]
                ax.plot(xs, ys, color=colour, linewidth=LINE_W, label=joint, zorder=4)
                label_end(ax, xs, ys, joint, colour)
            ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_SECONDARY,
                      ncol=min(3, len(motors)))
        written.append(save(fig, run_dir / "torque.png"))

    # --- drive temperature (0 = no reading, as joint_command treats it) ---
    if any(r["drive_c"] for r in rows):
        fig, axes = new_figure(1, f"Drive temperature over time - {label}", run_subtitle(meta))
        ax = axes[0]
        style_axes(ax, "drive temperature, highest reading in the last 1 s", "temperature (C)")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plotted, ends = [], []
        for motor_id in motors:
            mrows = [r for r in by_motor[motor_id] if r["drive_c"]]
            if not mrows:
                continue
            joint = mrows[0]["joint"] or f"motor {motor_id}"
            plotted.append(joint)
            colour = SERIES[motors.index(motor_id) % len(SERIES)]
            xs = [r["t_s"] for r in mrows]
            ys = trailing_max(xs, [r["drive_c"] for r in mrows])
            ax.plot(xs, ys, color=colour, linewidth=LINE_W, drawstyle="steps-post", label=joint,
                    zorder=4)
            ends.append((xs, ys, joint, colour))
        mit = (meta.get("limits", {}) or {}).get("mit") or {}
        ceilings: Dict[float, List[str]] = {}
        for j in plotted:
            value = (mit.get(j) or {}).get("max_temperature_c")
            if value is not None:
                ceilings.setdefault(float(value), []).append(j)
        for value, names in sorted(ceilings.items()):
            # Name the joints only when the line does not cover every trace.
            limit_lines(ax, None, value, "fault at" if names == plotted
                        else f"{', '.join(names)} fault at")
        # Whole degrees, so traces end on the same or adjacent values: stack the labels upward
        # at least 10 pt apart. Measured after the limit lines (they can stretch the y-range) and
        # the layout (it shrinks the panel).
        layout(fig)
        lo, hi = ax.get_ylim()
        pt_per_c = ax.get_window_extent().height * 72 / fig.dpi / (hi - lo)
        floor = -float("inf")
        for xs, ys, joint, colour in sorted(ends, key=lambda e: e[1][-1]):
            at = max(ys[-1] * pt_per_c, floor + 10)
            label_end(ax, xs, ys, joint, colour, dy=at - ys[-1] * pt_per_c)
            floor = at
        # Above the panel: the traces span the full width and height.
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8,
                  labelcolor=INK_SECONDARY, ncol=min(3, len(plotted)))
        written.append(save(fig, run_dir / "temperature.png"))

    summary = telemetry.format_metrics(meta, metrics)
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    written.append(run_dir / "summary.md")
    return written


def plot_sweep(run_dirs: List[Path], out: Optional[Path]) -> List[Path]:
    """Overlay several runs -- one line per run -- for gain comparison."""
    out = out or run_dirs[0].parent / "sweep.png"
    loaded = [(d, *telemetry.load_run(d)) for d in run_dirs]

    fig, axes = new_figure(2, "Gain sweep", f"{len(loaded)} runs overlaid")
    ax = axes[0]
    style_axes(ax, "measured angle, aligned to the start of each run", "angle (deg)")
    points = []
    for i, (d, meta, rows) in enumerate(loaded):
        colour = SERIES[i % len(SERIES)]
        gains = meta.get("gains") or {}
        name = f"kp {gains.get('kp_applied', '?'):.3g}" if gains.get("kp_applied") \
            else meta.get("label", d.name)
        t = [r["t_s"] for r in rows]
        base = next((r["pos_deg"] for r in rows if r["pos_deg"] is not None), 0.0)
        pos = [None if r["pos_deg"] is None else r["pos_deg"] - base for r in rows]
        ax.plot(t, pos, color=colour, linewidth=LINE_W, label=name, zorder=4)
        label_end(ax, t, pos, name, colour)
        m = telemetry.run_metrics(meta, rows)
        first = m[sorted(m)[0]] if m else {}
        points.append((name, gains.get("kp_applied"), first, colour))
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY,
              ncol=min(3, len(loaded)))

    ax = axes[1]
    style_axes(ax, "steady-state error and overshoot vs kp", "value", "kp (N.m/rad)")
    kps = [p[1] for p in points if p[1] is not None]
    if kps:
        ax.plot(kps, [abs(p[2].get("steady_state_err_deg") or 0) for p in points if p[1]],
                color=SERIES[0], linewidth=LINE_W, marker="o", markersize=8,
                label="|steady-state error| (deg)")
        ax.plot(kps, [p[2].get("overshoot_pct") or 0 for p in points if p[1]],
                color=SERIES[1], linewidth=LINE_W, marker="s", markersize=8,
                label="overshoot (%)")
        ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return [save(fig, out)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("run", nargs="+", type=Path, help="run folder(s)")
    ap.add_argument("--sweep", action="store_true",
                    help="overlay the given runs on one figure instead of plotting each")
    ap.add_argument("--out", type=Path, help="--sweep: where to write sweep.png")
    args = ap.parse_args(argv)

    for d in args.run:
        if not (d / "telemetry.csv").exists():
            sys.exit(f"{d} has no telemetry.csv -- is it a run folder?")

    written = plot_sweep(args.run, args.out) if args.sweep else \
        [p for d in args.run for p in plot_run(d)]
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
