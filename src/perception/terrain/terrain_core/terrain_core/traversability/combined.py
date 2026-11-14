"""Combine slope, step and roughness into one traversability score.

Each metric is turned into a cost in [0, 1] by dividing by the robot's limit
and clipping. The cell's score is 1 - max(costs), so the worst metric decides.
A cell is `traversable` only if every metric is within its limit.

Unknown cells (NaN elevation, or variance above `max_variance` when given) get
NaN score and traversable == False: planners must never step on terrain the
map has no estimate for.

Known trade-off: slope is skipped on non-planar cells (residual > planar_tol_m). A noisy, steep ramp whose
residual sits between planar_tol_m and max_roughness_m therefore escapes the slope check; set
planar_tol_m=None to enforce slope everywhere (at the cost of rejecting small steps as "steep slopes").

The limit values come from the locomotion team (ticket TR-6); the defaults here
are placeholders so the code runs, not real robot numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .slope import plane_fit
from .step import compute_step


@dataclass(frozen=True)
class RobotLimits:
    max_slope_deg: float = 7.0       # TR-6, locomotion 2026-11-10
    max_step_m: float = 0.08         # TR-6, locomotion 2026-11-10
    max_roughness_m: float = 0.03    # PLACEHOLDER - tune on course bags
    max_variance: float | None = None  # m^2; None disables the confidence gate
    planar_tol_m: float | None = 0.01  # plane-fit residual below which a cell counts as a planar surface.


def compute_traversability(
    elevation: np.ndarray,
    resolution: float,
    limits: RobotLimits = RobotLimits(),
    variance: np.ndarray | None = None,
    plane_radius: int = 1,
    step_radius: int = 1,
) -> dict[str, np.ndarray]:
    """Return a dict of layers, each shaped like `elevation`.

    Keys: slope (rad), step (m), roughness (m), traversability (0..1, NaN if
    unknown), traversable (bool).
    """
    elevation = np.asarray(elevation, dtype=float)
    slope, rough = plane_fit(elevation, resolution, plane_radius)
    planar_tol = np.inf if limits.planar_tol_m is None else limits.planar_tol_m
    step = compute_step(elevation, resolution, step_radius, slope, rough, planar_tol)

    known = np.isfinite(elevation) & np.isfinite(slope)
    if variance is not None and limits.max_variance is not None:
        known &= np.asarray(variance, dtype=float) <= limits.max_variance

    # Near a step edge or box wall the plane fit reports a steep slope that is really the step.
    # The step and roughness limits cover those cells, so slope is only enforced on planar ones.
    with np.errstate(invalid="ignore"):
        planar = rough <= planar_tol
    slope_checked = np.where(planar, slope, 0.0)

    max_slope = np.deg2rad(limits.max_slope_deg)
    step0 = np.nan_to_num(step, nan=0.0)
    with np.errstate(invalid="ignore"):
        c_slope = np.clip(slope_checked / max_slope, 0.0, 1.0)
        c_step = np.clip(step0 / limits.max_step_m, 0.0, 1.0)
        c_rough = np.clip(rough / limits.max_roughness_m, 0.0, 1.0)
    cost = np.maximum(np.maximum(c_slope, c_step), c_rough)

    score = 1.0 - cost
    score[~known] = np.nan

    with np.errstate(invalid="ignore"):
        traversable = known & (slope_checked <= max_slope) & (step0 <= limits.max_step_m) \
            & (rough <= limits.max_roughness_m)

    return {
        "slope": slope,
        "step": step,
        "roughness": rough,
        "traversability": score,
        "traversable": traversable,
    }
