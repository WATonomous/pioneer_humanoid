"""Write the Onshape course profile tables without exploding every number.

``json.dump(..., indent=1)`` puts each number on its own line: the straight
course profile was 4,330 lines for 540 samples. A single compact line is worse
to review. This writer keeps the scalar metadata inline and emits each item of
the long arrays (``heights``/``band`` pairs, tread angles, ...) on one line, so
the file stays readable and diffable:

    {
      "usd": "straight_stairs_terrain.usd",
      "bin": 0.05,
      "heights": [
        [-3.975,0.0],
        [-3.925,0.0]
      ],
      "band": [
        [-3.975,0.06]
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path


def write_profile(path: str, data: dict, per_line: tuple[str, ...] = ()) -> None:
    """Write ``data`` as JSON with one item per line for ``per_line`` keys.

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
            for item_index, item in enumerate(value):
                item_comma = "," if item_index + 1 < len(value) else ""
                lines.append(f"    {json.dumps(item, separators=(',', ':'))}{item_comma}")
            lines.append(f"  ]{comma}")
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value, separators=(',', ':'))}{comma}")
    lines.append("}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
