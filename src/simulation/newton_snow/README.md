# Newton snow / sand / water walks

Demo renders of the whole-body Pioneer walking (the arm-swing policy, `models/pioneer_whole_body_arm_swing/`) through
Newton implicit-MPM snow, sand and gravel, Newton XPBD leaves, and MPM muddy water, path traced with OVRTX.
The walking runs in MuJoCo as `sim2sim_mujoco.py`; the feet push the material, it doesn't push back.

## Setup (once)

```bash
cd src/simulation/newton_snow
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements.txt
robot/make_robot.sh   # the Oct 24 whole-body model (git d104f229) the policy was trained on, not main's (#468)
./fetch_assets.sh     # 1.2 GB of skies, trees and palms from NVIDIA's public Omniverse bucket
```

## Render

| scene | command | background |
|---|---|---|
| night snow | `./night.sh` | `night_scene.py` (moonlit sky, spruces, lamp) |
| beach sand | `./beach.sh` | `beach_scene.py` (palms) |
| granite gravel | `./granite.sh` | `granite_scene.py` |
| puddle | `./puddle.sh` | `puddle_scene.py` (autumn wood) |
| autumn leaves | `leaf_walk.py <policy> out/x.mp4 --bg out/autumn.usda` | `autumn_scene.py` (`CHERRY=1`: cherry blossom) |

Videos go to `out/`. The snow slab is `vx * seconds + 1.5` m long and each background is cut for its slab, so keep
`--seconds 12` with these backgrounds (a shorter run leaves a dark gap).

On an 8 GB GPU the MPM solver and the path tracer don't fit together: `night.sh` runs the physics with `--record DIR`
(no renderer), then path-traces the saved frames with `--replay DIR` (5 GB of frames). Same video as one pass.
