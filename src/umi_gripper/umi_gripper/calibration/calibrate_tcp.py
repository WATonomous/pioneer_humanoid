"""Controller -> TCP offset by pivot calibration (mounts.<headset>.grip_to_tcp.xyz in umi_calibration.json).

Hold the TCP point (between the jaws, 25 mm up from the tips: clamp the calibration pin there) still in a divot,
press the mounted controller's thumbstick, then tilt and twist the gripper about that point for --seconds. Every
sample satisfies  p_i + R_i t = c  (p_i, R_i the grip pose, t the offset in the grip frame, c the fixed point),
solved for t and c together by least squares, the same model as the Quest teleop's wrist pivot (control.fit_pivot).
The TCP axes (rpy_deg) are not observable this way: they come from your CAD and stay in umi_gripper.yaml.

    python -m umi_gripper.calibration.calibrate_tcp              # writes on success
    python -m umi_gripper.calibration.calibrate_tcp --dry-run
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from umi_gripper import config as umi_config
from umi_gripper import geometry, quest_tracker

MAX_RESIDUAL_M = 0.003
MIN_SAMPLES = 60


def fit_pivot(positions: np.ndarray, rotations: np.ndarray, max_offset: float = 0.5,
              max_residual: float = MAX_RESIDUAL_M) -> tuple[np.ndarray, np.ndarray, float]:
    """(offset in grip frame, fixed point in world, RMS residual m). Raises ValueError on a bad recording."""
    positions = np.asarray(positions, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    n = len(positions)
    if n < MIN_SAMPLES:
        raise ValueError(f"only {n} poses; pivot for longer (need {MIN_SAMPLES})")
    a = np.concatenate([rotations, -np.tile(np.eye(3), (n, 1, 1))], axis=2).reshape(-1, 6)
    b = -positions.reshape(-1)
    solution, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    if rank < 6 or singular[-1] / singular[0] < 1e-3:
        raise ValueError("not enough rotation: tilt about at least two different axes, not only twist")
    offset, point = solution[:3], solution[3:]
    residual = float(np.sqrt(np.mean(np.sum((positions + rotations @ offset - point) ** 2, axis=1))))
    if np.linalg.norm(offset) > max_offset:
        raise ValueError(f"offset {np.linalg.norm(offset):.2f} m is implausible; was the TCP held still?")
    if residual > max_residual:
        raise ValueError(f"residual {residual * 1000:.1f} mm > {max_residual * 1000:.0f} mm: the TCP slid in the "
                         "divot, or the controller lost tracking; retry")
    return offset, point, residual


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--headset", choices=umi_config.HEADSETS, default=None)
    parser.add_argument("--controller", choices=umi_config.SIDES, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in (("headset", args.headset), ("controller", args.controller)) if v}
    cfg = umi_config.load_config(overrides=overrides)
    tracker = quest_tracker.from_config(cfg).start()
    try:
        tracker.set_status("TCP calibration: TCP in the divot, click the thumbstick, then pivot")
        print("TCP in the divot, click the mounted controller's thumbstick, then tilt/twist about the TCP ...")
        while not any(e.name == "stick" for e in tracker.poll_events()):
            tracker.drain()
            time.sleep(0.02)
        tracker.drain()
        end = time.monotonic() + args.seconds
        samples = []
        while time.monotonic() < end:
            samples += [s for s in tracker.drain() if s.tracked]
            tracker.set_status(f"Pivoting ... {end - time.monotonic():4.1f} s")
            time.sleep(0.02)
    finally:
        tracker.stop()
    positions = np.array([s.position for s in samples]).reshape(-1, 3)
    rotations = np.array([geometry.quat_to_matrix(s.quat_xyzw) for s in samples]).reshape(-1, 3, 3)
    try:
        offset, _, residual = fit_pivot(positions, rotations)
    except ValueError as exc:
        print(f"FAILED: {exc}")
        return 1
    cad = umi_config.grip_to_tcp(cfg)[:3, 3]
    print(f"grip -> TCP offset {np.round(offset, 4)} m, residual {residual * 1000:.2f} mm "
          f"(CAD {np.round(cad, 4)}, differs by {np.linalg.norm(offset - cad) * 1000:.1f} mm)")
    if not args.dry_run:
        umi_config.update_calibration({"mounts": {cfg["headset"]: {"grip_to_tcp": {"xyz": offset.round(5).tolist()}}}})
        print(f"saved to {umi_config.DEFAULT_CALIBRATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
