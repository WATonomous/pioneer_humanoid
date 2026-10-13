#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 input.bag output_ros2_directory" >&2
  exit 1
fi

INPUT_BAG="$(realpath "$1")"
OUTPUT_DIR="$2"
CONVERTER=/opt/evo-venv/bin/rosbags-convert

if [[ ! -f "$INPUT_BAG" || "$INPUT_BAG" != *.bag ]]; then
  echo "Error: input must be an existing ROS 1 .bag file." >&2
  exit 1
fi
if [[ ! -x "$CONVERTER" ]]; then
  echo "Error: $CONVERTER is unavailable. Install the Python 'rosbags' package." >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Error: output already exists: $OUTPUT_DIR" >&2
  exit 1
fi

echo "Converting ROS 1 bag to ROS 2 MCAP..." >&2
"$CONVERTER" \
  --src "$INPUT_BAG" \
  --dst "$OUTPUT_DIR" \
  --dst-storage mcap \
  --dst-typestore ros2_jazzy >&2

echo "ROS 2 bag created: $(realpath "$OUTPUT_DIR")"
