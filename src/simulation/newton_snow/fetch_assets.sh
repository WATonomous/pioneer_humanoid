#!/bin/bash
# Skies, trees and palms from NVIDIA's public Omniverse content bucket (1.2 GB) into assets/.
set -e
cd "$(dirname "$0")"
PY=${PY:-$PWD/.venv/bin/python}
B=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets
mkdir -p assets/sky assets/trees assets/tropical
cd assets
S=$B/Skies/2022_1/Skies
curl -sf -o sky/moonlit_golf.hdr $S/Night/moonlit_golf.hdr
curl -sf -o sky/evening_road_01.hdr $S/Evening/evening_road_01.hdr
curl -sf -o sky/signal_hill_sunrise.hdr $S/Clear/signal_hill_sunrise.hdr
curl -sf -o sky/noon_grass.hdr $S/Clear/noon_grass.hdr
curl -sf -o sky/syferfontein_18d_clear.hdr $S/Clear/syferfontein_18d_clear.hdr
curl -sf -o sky/exhibition_hall.hdr $S/Indoor/ZetoCGcom_ExhibitionHall_Interior1.hdr
T="Norway_Spruce Japanese_Cherry Japanese_Maple_Fall Scarlet_Oak_fall Red_Ash_Fall Black_Oak_Fall"
for t in $T; do curl -sf -o trees/$t.usd $B/Vegetation/Trees/$t.usd; done
$PY ../fetch_deps.py $(for t in $T; do echo $t.usd; done)
P="Cuban_Royal_Palm Hurricane_Palm Fan_Palm Agave"
for p in $P; do curl -sfL -o tropical/$p.usd $B/Vegetation/Plant_Tropical/$p.usd; done
$PY ../fetch_tropical.py $(for p in $P; do echo $p.usd; done)
