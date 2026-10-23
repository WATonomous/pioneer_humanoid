"""Summarise an onshape probe log: gait symmetry, knee height, foot clearance.

    python3 tools/onshape_course/probe_report.py /tmp/probe599.log

The probe (play_onshape.py) prints one [PLAYARC] line per swing, per foot, at
touchdown, with the joints sampled at peak lift. This turns those into the
numbers that matter for the gait review:

  * knee_z  -- the knee's height above the root at peak lift (the "raises the
               knee too high" complaint; ~0 m is a straight stance leg, ~0.25 m
               is a chest-high march on this robot)
  * dz_max  -- the swing foot's peak height above the root
  * HipF    -- the thigh's pitch at peak lift (the high-knee march signature)
  * dx_max  -- the foot's lateral excursion (the splay check)
  * the L/R split of each (the symmetry check)
"""

from __future__ import annotations

import math
import re
import statistics as st
import sys


def parse(path: str) -> dict:
    out = {"arc": {"L": [], "R": []}, "plants": {"L": 0, "R": 0}, "foot": {"L": [], "R": []},
           "stance": {"L": [], "R": []}, "lead": {}, "episodes": []}
    for line in open(path, errors="ignore"):
        m = re.match(r"\[PLAYARC\] ([LR]) (.*)", line)
        if m:
            out["arc"][m.group(1)].append(
                {k: float(v) for k, v in re.findall(r"(\w+)=([+\-0-9.]+)", m.group(2))})
        m = re.match(r"\[PLAYLEAD\] env=(\d+) plant=([LR])", line)
        if m:
            out["plants"][m.group(2)] += 1
        m = re.match(r"\[PLAYLEAD\] env=(\d+) LEAD=([LR])", line)
        if m:
            out["lead"][m.group(1)] = m.group(2)
        m = re.match(r"\[PLAYLEAD\] env=(\d+) episode_end .*max_foot_z=([+\-0-9.]+)", line)
        if m:
            out["episodes"].append((int(m.group(1)), float(m.group(2))))
        m = re.match(r"\[PLAYARCFOOT\] ([LR]) tilt=\s*([\-0-9.]+)", line)
        if m:
            out["foot"][m.group(1)].append(float(m.group(2)))
        m = re.match(r"\[PLAYARCSTANCE\] ([LR]) td=\s*([\-0-9.]+)", line)
        if m:
            out["stance"][m.group(1)].append(float(m.group(2)))
    return out


def mean(values):
    values = [v for v in values if not math.isnan(v)]
    return st.mean(values) if values else float("nan")


def report(path: str) -> None:
    data = parse(path)
    print(f"=== {path} ===")
    print(f"plants L/R: {data['plants']['L']} / {data['plants']['R']}   "
          f"strides L/R: {len(data['arc']['L'])} / {len(data['arc']['R'])}")
    if data["episodes"]:
        climbed = [z for _, z in data["episodes"] if z > 0.5]
        print(f"episodes: {len(data['episodes'])}  climbed to the summit: {len(climbed)}  "
              f"max foot z: {max(z for _, z in data['episodes']):.2f} m")
    print(f"lead foot: {sorted(set(data['lead'].values()))}")
    header = f"{'metric':30s}{'L':>12s}{'R':>12s}{'|L-R|':>10s}"
    print(header)

    def row(name, values_l, values_r, fmt="{:.3f}"):
        l, r = mean(values_l), mean(values_r)
        print(f"{name:30s}{fmt.format(l):>12s}{fmt.format(r):>12s}{fmt.format(abs(l - r)):>10s}")

    for label, key in (("knee z above root", "knee_z"), ("foot peak dz", "dz_max"),
                       ("hip pitch", "HipF"), ("knee joint", "Knee"),
                       ("foot lateral dx_max", "dx_max")):
        if not data["arc"]["L"]:
            continue
        if key not in data["arc"]["L"][0]:
            continue
        # knee_z / dz_max are signed (negative = below the pelvis): a *less*
        # negative value means a higher knee. Joint angles are magnitudes.
        signed = key in ("knee_z", "dz_max")
        row(label, [x.get(key, float("nan")) if signed else abs(x.get(key, float("nan")))
                    for x in data["arc"]["L"]],
            [x.get(key, float("nan")) if signed else abs(x.get(key, float("nan")))
             for x in data["arc"]["R"]])

    # Flat approach vs course: the knee height on the flat is what the visual
    # eval complains about; on the stairs a lift is required.
    if data["arc"]["L"] and "root_y" in data["arc"]["L"][0]:
        for label, test in (("flat (root_y<1)", lambda r: r < 1.0),
                            ("course (root_y>=1)", lambda r: r >= 1.0)):
            for key in ("knee_z", "dz_max", "HipF"):
                signed = key in ("knee_z", "dz_max")
                row(f"  {label} {key}",
                    [x.get(key, float("nan")) if signed else abs(x.get(key, float("nan")))
                     for x in data["arc"]["L"] if test(x.get("root_y", 0))],
                    [x.get(key, float("nan")) if signed else abs(x.get(key, float("nan")))
                     for x in data["arc"]["R"] if test(x.get("root_y", 0))])
    row("landing tilt (deg)", data["foot"]["L"], data["foot"]["R"], "{:.1f}")
    row("touchdown tilt (deg)", data["stance"]["L"], data["stance"]["R"], "{:.1f}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        report(arg)
        print()
