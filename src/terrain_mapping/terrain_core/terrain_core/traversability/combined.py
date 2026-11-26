"""Combine slope, step and roughness into one traversability score.

A cell is traversable only if every metric is within its limit; unknown cells never are. Slope and roughness use a
plane fit over `plane_window_m`, step height a small `step_radius_m` window. Slope is only enforced on planar cells,
since near a step edge the steep plane fit is really the step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._window import window_radius
from .slope import estimate_noise_sigma, plane_fit, planar_tolerance
from .step import compute_step


@dataclass(frozen=True)
class RobotLimits:
    max_slope_deg: float = 7.0       # TR-6, locomotion 2026-11-10 (non-ideal, 6 m ramp up)
    max_step_m: float = 0.08         # TR-6, locomotion 2026-11-10 (non-ideal, step up/down)
    max_roughness_m: float = 0.03    # placeholder, tune on course bags
    max_variance: float | None = None  # m^2; None disables the confidence gate


@dataclass(frozen=True)
class TraversabilityParams:
    plane_window_m: float = 0.40         # plane-fit window for slope and roughness
    step_radius_m: float = 0.05          # neighbour reach for step height
    noise_sigma_m: float | None = None   # per-cell elevation noise; None = estimate from the map
    planar_floor_m: float = 0.01         # planar tolerance = max(floor, factor * noise)
    planar_noise_factor: float = 1.5


@dataclass(frozen=True)
class TraversabilityConfig:
    """TR-6 limits and TR-7 algorithm settings loaded from ``terrain.yaml``.

    Keeping this separate from :func:`compute_traversability` lets the offline
    evaluator and a future ROS node use exactly the same, version-controlled
    values. Missing YAML keys keep the safe defaults above.
    """

    limits: RobotLimits = field(default_factory=RobotLimits)
    params: TraversabilityParams = field(default_factory=TraversabilityParams)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TraversabilityConfig":
        """Read the top-level ``traversability:`` section of a terrain YAML file."""
        import yaml

        with open(path) as fh:
            settings = yaml.safe_load(fh) or {}
        section = settings.get("traversability") or {}
        return cls(
            limits=RobotLimits(**(section.get("robot_limits") or {})),
            params=TraversabilityParams(**(section.get("params") or {})),
        )


def compute_traversability(
    elevation: np.ndarray,
    resolution: float,
    limits: RobotLimits = RobotLimits(),
    variance: np.ndarray | None = None,
    params: TraversabilityParams = TraversabilityParams(),
) -> dict[str, np.ndarray]:
    """Return layers shaped like `elevation`: slope (rad), step (m), roughness (m), traversability, traversable."""
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

    known = np.isfinite(slope) & np.isfinite(step) & np.isfinite(rough)
    if variance is not None and limits.max_variance is not None:
        known &= np.asarray(variance, dtype=float) <= limits.max_variance

    slope_checked = np.where(planar, slope, 0.0)          # off planar cells the step limit applies instead

    max_slope = np.deg2rad(limits.max_slope_deg)
    with np.errstate(invalid="ignore"):
        c_slope = np.clip(slope_checked / max_slope, 0.0, 1.0)
        c_step = np.clip(step / limits.max_step_m, 0.0, 1.0)
        c_rough = np.clip(rough / limits.max_roughness_m, 0.0, 1.0)
    cost = np.maximum(np.maximum(c_slope, c_step), c_rough)

    score = 1.0 - cost
    score[~known] = np.nan

    with np.errstate(invalid="ignore"):
        traversable = known & (slope_checked <= max_slope) & (step <= limits.max_step_m) \
            & (rough <= limits.max_roughness_m)

    return {
        "slope": slope,
        "step": step,
        "roughness": rough,
        "traversability": score,
        "traversable": traversable,
    }
