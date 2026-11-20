"""Combine slope, step and roughness into one traversability score.

Each metric is turned into a cost in [0, 1] by dividing by the robot's limit and clipping. The cell's score is
1 - max(costs), so the worst metric decides. A cell is `traversable` only if every metric is within its limit.

Unknown cells (NaN elevation, or variance above `max_variance` when given) get NaN score and traversable == False:
planners must never step on terrain the map has no estimate for.

Two scales, set in metres so they work at any grid resolution:
  - slope and roughness come from a plane fit over `plane_window_m` (default 0.40 m), large enough that 1-2 cm of
    elevation noise does not tilt a flat floor or a 5 degree ramp past the 7 degree limit (0.30 m is enough up to
    about 1 cm of noise, 0.20 m is not);
  - step height uses `step_radius_m` (default 5 cm: the immediate neighbours), because a step is a local event.

Slope is enforced only on planar cells (residual RMS <= planar tolerance, see slope.planar_tolerance), because near a
step edge the plane fit's steep tilt is really the step, which the step limit already covers. The tolerance is set
from the map's own noise level, either given (`noise_sigma_m`) or estimated from the median roughness of the map, so
it does not break when the noise changes.

Known trade-offs:
  * A step smaller than a few times the map noise cannot be told from tilt: it is treated as planar, so the plane
    fit's steep slope blocks its edge cells (conservative). Noise-free or 1 cm-noise maps still pass legal steps.
  * The noise estimate is global. On a map that is mostly rough it overestimates the noise, which stops steep
    slopes being checked as strictly; pass `noise_sigma_m` explicitly if you know the sensor noise.

The limit values come from the locomotion team (ticket TR-6, sim results 2026-11-10).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._window import window_radius
from .slope import estimate_noise_sigma, plane_fit, planar_tolerance
from .step import compute_step


@dataclass(frozen=True)
class RobotLimits:
    max_slope_deg: float = 7.0       # TR-6, locomotion 2026-11-10 (non-ideal, 6 m ramp up)
    max_step_m: float = 0.08         # TR-6, locomotion 2026-11-10 (non-ideal, step up/down)
    max_roughness_m: float = 0.03    # PLACEHOLDER - tune on course bags
    max_variance: float | None = None  # m^2; None disables the confidence gate


@dataclass(frozen=True)
class TraversabilityParams:
    plane_window_m: float = 0.40         # plane-fit window width for slope and roughness
    step_radius_m: float = 0.05          # neighbour reach for step height
    noise_sigma_m: float | None = None   # per-cell elevation noise; None = estimate from the map
    planar_floor_m: float = 0.01         # planar tolerance never goes below this
    planar_noise_factor: float = 1.5     # planar tolerance = max(floor, factor * noise sigma)


def compute_traversability(
    elevation: np.ndarray,
    resolution: float,
    limits: RobotLimits = RobotLimits(),
    variance: np.ndarray | None = None,
    params: TraversabilityParams = TraversabilityParams(),
) -> dict[str, np.ndarray]:
    """Return a dict of layers, each shaped like `elevation`.

    Keys: slope (rad), step (m), roughness (m), traversability (0..1, NaN if unknown), traversable (bool).
    """
    elevation = np.asarray(elevation, dtype=float)
    slope, rough = plane_fit(elevation, resolution, params.plane_window_m)

    if params.noise_sigma_m is None:
        cells = (2 * window_radius(params.plane_window_m, resolution) + 1) ** 2
        sigma = estimate_noise_sigma(rough, cells)
    else:
        sigma = params.noise_sigma_m
    planar_tol = planar_tolerance(sigma, params.planar_floor_m, params.planar_noise_factor)
    with np.errstate(invalid="ignore"):
        planar = rough <= planar_tol

    step = compute_step(elevation, resolution, params.step_radius_m, slope, planar)

    known = np.isfinite(elevation) & np.isfinite(slope)
    if variance is not None and limits.max_variance is not None:
        known &= np.asarray(variance, dtype=float) <= limits.max_variance

    # Slope is only enforced on planar cells; elsewhere the step and roughness limits are responsible.
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
