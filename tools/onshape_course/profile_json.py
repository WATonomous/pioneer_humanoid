"""Write the Onshape course profile tables without exploding every number.

``json.dump(..., indent=1)`` puts each number on its own line: the straight
course profile was 4,330 lines for 540 samples. A single compact line is worse
to review. This writer keeps the scalar metadata inline and emits each item of
the long arrays (profile samples, tread angles, ...) on one line, so the file
stays readable and diffable:

    {
      "usd": "straight_stairs_terrain.usd",
      "y0": -3.975,
      "bin": 0.05,
      "samples": [
        [0.0,0.06],
        [0.0,0.06],
        ...
      ]
    }

The straight profile stores only ``[height, band]`` per sample. The y
coordinate is implicit: ``y = y0 + index * bin``. :func:`assert_uniform_grid`
guards the conversion from an explicit y column.
"""

from __future__ import annotations

import json
from pathlib import Path


def assert_uniform_grid(positions: list[float], bin_size: float, tol: float = 1e-6) -> None:
    """Raise ``ValueError`` unless ``positions`` is a uniform ``bin_size`` grid."""
    for index in range(1, len(positions)):
        delta = float(positions[index]) - float(positions[index - 1])
        if abs(delta - float(bin_size)) > tol:
            raise ValueError(
                f"profile positions are not on a uniform {bin_size} m grid: "
                f"index {index - 1}->{index} is {delta:.9f} m"
            )


def write_profile(
    path: str,
    data: dict,
    per_line: tuple[str, ...] = (),
    per_physical_line: int = 1,
) -> None:
    """Write ``data`` as JSON with ``per_physical_line`` items per line for ``per_line`` keys.

    Keys not in ``per_line`` are written inline, so short lists such as a
    centre or marker stay on the metadata line.
    """
    selected = set(per_line)
    lines = ["{"]
    entries = list(data.items())
    for entry_index, (key, value) in enumerate(entries):
        comma = "," if entry_index + 1 < len(entries) else ""
        if key in selected and isinstance(value, list):
            lines.append(f"  {json.dumps(key)}: [")
            # Group items into lines of `per_physical_line` elements each.
            for group_start in range(0, len(value), per_physical_line):
                group = value[group_start : group_start + per_physical_line]
                is_last_group = group_start + per_physical_line >= len(value)
                items_str = ", ".join(
                    json.dumps(item, separators=(',', ':')) for item in group
                )
                group_comma = "," if not is_last_group else ""
                lines.append(f"    {items_str}{group_comma}")
            lines.append(f"  ]{comma}")
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value, separators=(',', ':'))}{comma}")
    lines.append("}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
