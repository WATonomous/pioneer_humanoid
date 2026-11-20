#!/bin/bash
cd "$(dirname "$0")"
mkdir -p out
DOME=600 SUN=1400 PXR_USDC_EMIT_DEPRECATION_WARNINGS=0 .venv/bin/python beach_scene.py out/beach.usda 4.8 -1.0 7.3 -3.0 3.5 0.08 2>&1 | grep -v Warn
.venv/bin/python snow_walk.py ../../../models/pioneer_whole_body_arm_swing/model_2299.pt out/sand_beach.mp4 --whole_body --view tour --depth 0.08 --ppc 2.0 --flakes 0 \
  --seconds 12 --size 1280 720 --rtx --bg out/beach.usda --spp 16 --slab_y -3.0 3.5 --material sand --slab_x1 7.3 > out/beach.log 2>&1
grep -E "SLAB|snow particles|wrote|Traceback|Error|memory|moved|sunk" out/beach.log
