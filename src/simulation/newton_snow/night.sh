#!/bin/bash
# Night snow walk. Physics and path tracing don't fit an 8 GB GPU together, so record the physics, then render it.
cd "$(dirname "$0")"
mkdir -p out
DOME=300 MOON=300 LAMP=200000 NTREES=34 PXR_USDC_EMIT_DEPRECATION_WARNINGS=0 .venv/bin/python night_scene.py out/night.usda 4.8 -1.0 6.3 -3.0 3.5 0.08
A="../../../models/pioneer_whole_body_arm_swing/model_2299.pt out/snow_night.mp4 --whole_body --view tour --depth 0.08 --ppc 2.0 --flakes 7000
  --seconds 12 --size 1280 720 --rtx --bg out/night.usda --spp 16 --slab_y -3.0 3.5"
.venv/bin/python snow_walk.py $A --record out/night_rec > out/night_record.log 2>&1
.venv/bin/python snow_walk.py $A --replay out/night_rec > out/night.log 2>&1
grep -E "SLAB|snow particles|wrote|Traceback|Error|memory" out/night.log
