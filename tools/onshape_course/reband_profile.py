"""Recompute the clearance band table from an existing course profile.

The course profile JSON holds the terrain height at every y bin, which is all
the band table needs -- so the band can be re-derived (different lookahead,
different clamp) without booting Isaac Sim again.

    python3 tools/onshape_course/reband_profile.py \\
        --profile assets/staircase_onshape/course_easy_profile.json \\
        --lookahead 0.30 --band-min 0.06 --band-max 0.16
"""

from __future__ import annotations

import argparse
import json

parser = argparse.ArgumentParser(description="Recompute the required-clearance band table.")
parser.add_argument("--profile", required=True, help="Course profile JSON (from course_profile.py).")
parser.add_argument("--lookahead", type=float, default=0.30, help="How far ahead a riser must be paid for (m).")
parser.add_argument("--margin", type=float, default=0.03, help="Foot-to-foot margin the term already subtracts.")
parser.add_argument("--band-min", type=float, default=0.06, help="Floor for the band (a natural flat step).")
parser.add_argument("--band-max", type=float, default=0.16, help="Ceiling for the band (the tallest riser + margin).")
args_cli = parser.parse_args()

with open(args_cli.profile) as handle:
    profile = json.load(handle)

heights = [(float(y), float(z)) for y, z in profile["heights"]]
bin_size = float(profile.get("bin", 0.05))

band = []
for index, (y, z) in enumerate(heights):
    limit = y + args_cli.lookahead
    ahead = [zz for yy, zz in heights[index + 1 :] if yy <= limit + 1e-9]
    needed = (max(ahead) if ahead else z) - z
    band.append([round(y, 4), round(min(max(needed + args_cli.margin, args_cli.band_min), args_cli.band_max), 4)])

profile["band"] = band
profile["lookahead"] = args_cli.lookahead
profile["margin"] = args_cli.margin

with open(args_cli.profile, "w") as handle:
    json.dump(profile, handle, indent=1)

values = [b for _, b in band]
print(f"[REBAND] {args_cli.profile}: lookahead={args_cli.lookahead} margin={args_cli.margin} "
      f"band_min={args_cli.band_min} band_max={args_cli.band_max}")
print(f"[REBAND] band range {min(values):.3f}..{max(values):.3f}  bin={bin_size}")
step = 10
for index in range(0, len(band), step):
    y, b = band[index]
    print(f"[REBAND]   y={y:7.2f} band={b:.3f}")
