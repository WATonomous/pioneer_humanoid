#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 input.bag output_ros2_directory" >&2
  exit 1
fi

INPUT_BAG="$(realpath "$1")"
OUTPUT_DIR="$2"
RAW_DIR="${OUTPUT_DIR}.raw"
CONVERTER=/opt/evo-venv/bin/rosbags-convert
PYTHON=/opt/evo-venv/bin/python

if [[ ! -f "$INPUT_BAG" || "$INPUT_BAG" != *.bag ]]; then
  echo "Error: input must be an existing ROS 1 .bag file." >&2
  exit 1
fi
if [[ ! -x "$CONVERTER" ]]; then
  echo "Error: $CONVERTER is unavailable. Install the Python 'rosbags' package." >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" || -e "$RAW_DIR" ]]; then
  echo "Error: output or temporary directory already exists." >&2
  exit 1
fi

cleanup() { rm -rf "$RAW_DIR"; }
trap cleanup EXIT

echo "Converting ROS 1 bag to ROS 2 MCAP..." >&2
"$CONVERTER" \
  --src "$INPUT_BAG" \
  --dst "$RAW_DIR" \
  --dst-storage mcap \
  --dst-typestore ros2_jazzy >&2

echo "Normalizing legacy frame IDs..." >&2
"$PYTHON" "$(dirname "$0")/normalize_bag_frames.py" "$RAW_DIR" "$OUTPUT_DIR"

if [[ ! -f "$OUTPUT_DIR/metadata.yaml" ]]; then
  echo "Error: ROS 2 bag conversion did not produce metadata.yaml." >&2
  exit 1
fi

echo "ROS 2 bag created: $(realpath "$OUTPUT_DIR")"
