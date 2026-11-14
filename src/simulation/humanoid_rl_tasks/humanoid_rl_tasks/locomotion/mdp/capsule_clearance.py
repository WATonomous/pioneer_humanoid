"""Batched capsule geometry for the leg-clearance reward and its diagnostics."""

from __future__ import annotations

import torch


def segment_segment_distance(
    first_start: torch.Tensor,
    first_end: torch.Tensor,
    second_start: torch.Tensor,
    second_end: torch.Tensor,
) -> torch.Tensor:
    """Return exact distances between closed 3D segments with shape ``(..., 3)``.

    The closest pair is either an interior line-line solution or an endpoint
    projected onto the other segment. Evaluating both cases also handles
    parallel segments and zero-length segments without unstable division.
    """
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    offset = first_start - second_start
    first_length_sq = first_direction.square().sum(dim=-1)
    second_length_sq = second_direction.square().sum(dim=-1)
    directions_dot = (first_direction * second_direction).sum(dim=-1)
    first_offset_dot = (first_direction * offset).sum(dim=-1)
    second_offset_dot = (second_direction * offset).sum(dim=-1)
    determinant = first_length_sq * second_length_sq - directions_dot.square()
    safe_determinant = determinant.clamp_min(1.0e-12)
    first_fraction = (
        directions_dot * second_offset_dot - second_length_sq * first_offset_dot
    ) / safe_determinant
    second_fraction = (
        first_length_sq * second_offset_dot - directions_dot * first_offset_dot
    ) / safe_determinant
    interior_valid = (
        (determinant > 1.0e-6 * first_length_sq * second_length_sq)
        & (first_fraction >= 0.0)
        & (first_fraction <= 1.0)
        & (second_fraction >= 0.0)
        & (second_fraction <= 1.0)
    )
    interior_offset = (
        offset
        + first_fraction.unsqueeze(-1) * first_direction
        - second_fraction.unsqueeze(-1) * second_direction
    )
    distance_sq = torch.where(interior_valid, interior_offset.square().sum(dim=-1), torch.inf)

    def point_segment_distance_sq(point, start, direction, length_sq):
        fraction = ((point - start) * direction).sum(dim=-1) / length_sq.clamp_min(1.0e-12)
        closest = start + fraction.clamp(0.0, 1.0).unsqueeze(-1) * direction
        return (point - closest).square().sum(dim=-1)

    for point in (first_start, first_end):
        distance_sq = torch.minimum(
            distance_sq,
            point_segment_distance_sq(point, second_start, second_direction, second_length_sq),
        )
    for point in (second_start, second_end):
        distance_sq = torch.minimum(
            distance_sq,
            point_segment_distance_sq(point, first_start, first_direction, first_length_sq),
        )
    return distance_sq.clamp_min(0.0).sqrt()
