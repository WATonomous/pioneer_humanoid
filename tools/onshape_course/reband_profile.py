"""Recompute the clearance band table from an existing course profile.

The course profile JSON holds the terrain height at every y bin, which is all
the band table needs -- so the band can be re-derived (different lookahead,
different clamp) without booting Isaac Sim again.

    python3 tools/onshape_course/reband_profile.py \\
        --profile assets/staircase_onshape/straight_stairs_profile.json \\
        --lookahead 0.30 --band-min 0.06 --band-max 0.16
"""

from __future__ import annotations

import argparse
import json

from profile_json import assert_uniform_grid, write_profile

parser = argparse.ArgumentParser(description="Recompute the required-clearance band table.")
parser.add_argument("--profile", required=True, help="Course profile JSON (from course_profile.py).")
parser.add_argument("--lookahead", type=float, default=0.30, help="How far ahead a riser must be paid for (m).")
parser.add_argument("--margin", type=float, default=0.03, help="Foot-to-foot margin the term already subtracts.")
parser.add_argument("--band-min", type=float, default=0.06, help="Floor for the band (a natural flat step).")
parser.add_argument("--band-max", type=float, default=0.16, help="Ceiling for the band (the tallest riser + margin).")
args_cli = parser.parse_args()

with open(args_cli.profile) as handle:
    profile = json.load(handle)

bin_size = float(profile.get("bin", 0.05))
if "samples" in profile:
    y0 = float(profile["y0"])
    positions = [y0 + index * bin_size for index in range(len(profile["samples"]))]
    heights = [float(row[0]) for row in profile["samples"]]
else:
    # Legacy layout: explicit [y, height] rows.
    positions = [float(y) for y, _ in profile["heights"]]
    heights = [float(z) for _, z in profile["heights"]]

assert_uniform_grid(positions, bin_size)

band = []
for index, (y, z) in enumerate(zip(positions, heights)):
    limit = y + args_cli.lookahead
    ahead = [hz for py, hz in zip(positions[index + 1 :], heights[index + 1 :]) if py <= limit + 1e-9]
    needed = (max(ahead) if ahead else z) - z
    band.append(round(min(max(needed + args_cli.margin, args_cli.band_min), args_cli.band_max), 4))

samples = [[round(z, 4), b] for z, b in zip(heights, band)]

out = {}
for key, value in profile.items():
    if key in {"heights", "band", "samples", "y0"}:
        continue
    if key == "bin":
        out["y0"] = positions[0]
    out[key] = value
if "y0" not in out:
    out["y0"] = positions[0]
if "bin" not in out:
    out["bin"] = bin_size
out["lookahead"] = args_cli.lookahead
out["margin"] = args_cli.margin
out["samples"] = samples

write_profile(args_cli.profile, out, per_line=("samples",))

values = [row[1] for row in samples]
print(f"[REBAND] {args_cli.profile}: lookahead={args_cli.lookahead} margin={args_cli.margin} "
      f"band_min={args_cli.band_min} band_max={args_cli.band_max}")
print(f"[REBAND] band range {min(values):.3f}..{max(values):.3f}  bin={bin_size}")
step = 10
for index in range(0, len(samples), step):
    print(f"[REBAND]   y={positions[index]:7.2f} band={samples[index][1]:.3f}")
