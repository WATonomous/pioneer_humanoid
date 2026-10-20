"""Shared observation snapshot for real-robot and sim recorders."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObservationSnapshot:
    """One timestep: proprio state, commanded action, optional camera frames."""

    state: np.ndarray | None
    action: np.ndarray | None
    images: dict[str, np.ndarray] = field(default_factory=dict)
