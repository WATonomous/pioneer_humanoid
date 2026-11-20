"""Small helpers ported from lehome/utils/record.py (experiment paths, rate limiting)."""

from __future__ import annotations

import time
from pathlib import Path


class RateLimiter:
    """Sleep to enforce a fixed loop rate in Hz."""

    def __init__(self, hz: float) -> None:
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.period = 1.0 / hz
        self._next = time.monotonic()

    def sleep(self) -> None:
        self._next += self.period
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            while self._next < time.monotonic():
                self._next += self.period


def get_next_experiment_path_with_gap(base_path: Path) -> Path:
    """Return `<base>/001`, `<base>/002`, … using the lowest unused index."""
    base_path.mkdir(parents=True, exist_ok=True)
    indices: set[int] = set()
    for folder in base_path.iterdir():
        if folder.is_dir():
            try:
                indices.add(int(folder.name))
            except ValueError:
                continue
    index = 1
    while index in indices:
        index += 1
    return base_path / f"{index:03d}"


def resolve_config_path(path: str | Path, *, anchor: Path) -> Path:
    """Resolve config relative to anchor when not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (anchor / p).resolve()


# Repo root (src/robot_learning/humanoid_robot_learning/recording/record_utils.py -> repo). Datasets live
# under <repo>/datasets/ wherever a recorder is launched from.
REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_dataset_root(
    cfg: dict, override: str | Path | None = None, *, subdir: str | None = None, default: str = "datasets/record"
) -> Path:
    """Output directory: ``override`` as given, else the schema's ``record.root`` under REPO_ROOT (+ subdir)."""
    if override:
        return Path(override)
    root = Path((cfg.get("record") or {}).get("root", default))
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / subdir if subdir else root
