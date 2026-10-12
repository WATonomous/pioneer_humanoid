#!/usr/bin/env bash

set -euo pipefail

# checks if number of args is not 1
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 ESTIMATED_TRAJECTORY.txt" >&2
  exit 1
fi

# Sets file
ESTIMATED_TRAJECTORY="$1"

# Checks if file exist/non-zero
if [[ ! -s "$ESTIMATED_TRAJECTORY" ]]; then
    echo "Error: trajectory file not found or empty: $ESTIMATED_TRAJECTORY" >&2
    exit 1
fi

python3 - "$ESTIMATED_TRAJECTORY" <<'PY'
import math
import sys

poses = []
with open(sys.argv[1]) as file:
    for line in file:
        fields = line.split()
        if len(fields) >= 8 and not line.lstrip().startswith('#'):
            poses.append([float(value) for value in fields[:8]])

if len(poses) < 2:
    raise SystemExit("Error: trajectory must contain at least two poses.")

start, end = poses[0], poses[-1]
translation_error = math.sqrt(sum((end[i] - start[i]) ** 2 for i in range(1, 4)))

def multiply(a, b):
    return [
        a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
        a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
        a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
        a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2],
    ]

q_start = start[4:8]
q_end = end[4:8]
q_relative = multiply(q_end, [-q_start[0], -q_start[1], -q_start[2], q_start[3]])
norm = math.sqrt(sum(value * value for value in q_relative))
rotation_error = math.degrees(2 * math.atan2(math.sqrt(sum(value * value for value in q_relative[:3])) / norm, abs(q_relative[3]) / norm))

print(f"Loop translation error: {translation_error:.6f} m")
print(f"Loop rotation error: {rotation_error:.6f} deg")
PY
