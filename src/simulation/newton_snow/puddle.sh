#!/bin/bash
cd "$(dirname "$0")"
mkdir -p out
PXR_USDC_EMIT_DEPRECATION_WARNINGS=0 .venv/bin/python puddle_scene.py out/puddle.usda 4.8 -1.0 5.8 -1.2 1.2 2>&1 | grep -v "Warn\|###"
.venv/bin/python puddle_walk.py ../../../models/pioneer_whole_body_arm_swing/model_2299.pt out/puddle_walk.mp4 --bg out/puddle.usda --seconds 12 --size 1280 720 --spp 16 > out/puddle.log 2>&1
grep -E "particles|triangles|wrote|Traceback|Error|memory|fell" out/puddle.log
