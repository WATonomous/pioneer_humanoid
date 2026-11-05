# Terrain mapping

Robot-centric elevation and traversability map from the chest D455, per the terrain mapping
project plan (tickets TR-1 to TR-18). Same layout as the SLAM stack: a ROS-free core library
(`terrain_core`), a thin ROS wrapper (later), and offline tools.

## Status

| Part | Ticket | State |
|------|--------|-------|
| Preprocessing: depth to points, crop, voxel downsample, into `odom` | TR-4 | Done, tested on TUM RGB-D |
| Self-filtering (remove arms and legs) | TR-5 | Next |
| Elevation grid, traversability, ROS node | TR-2/3, TR-7/8, TR-14 | Not started |

## Layout

```
terrain/
├── config/terrain.yaml     settings (preprocess: crop range, voxel size)
├── terrain_core/           pip-installable library, no ROS imports
│   ├── terrain_core/
│   │   └── preprocess/     depth.py, crop.py, downsample.py, transform.py, pipeline.py
│   └── tests/              python -m pytest terrain_core (no dataset needed, except one TUM test)
├── terrain_tools/          tum.py: TumDataset loads a TUM RGB-D sequence as slam_core Frames
├── prototype/              demo scripts with plots (take --data path/to/sequence)
└── data/                   datasets and bags (gitignored)
```

## Setup

```bash
cd src/perception/terrain
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e "terrain_core[dev]" -e ../slam/slam_core
python -m pytest terrain_core
```

`terrain_core` depends on `slam_core` (for `Frame`), which is not on PyPI, so always install
the two in the same `pip install` command as above: pip then takes `slam_core` from this repo.
`[dev]` adds what the tests and prototypes need (pytest, scipy, matplotlib, pillow).

## Using preprocessing

```python
from terrain_core.preprocess.pipeline import PreprocessConfig, preprocess

config = PreprocessConfig.from_yaml("config/terrain.yaml")
cloud = preprocess(frame, T_odom_cam, config)
# cloud.points: (N, 3) float32 in odom, metres; cloud.stamp: the depth image's timestamp
```

- `frame`: a `slam_core.types.Frame` with depth in metres and intrinsics `K`.
- `T_odom_cam`: 4x4 pose of the camera optical frame in `odom` **at `frame.stamp`**
  (`T_a_b` naming from `slam_core.types.pose`).

## Results on TUM RGB-D `freiburg1_desk`

Practice data until our D455 course bags exist (TR-10): depth images with ground-truth camera poses.

- **Speed:** `preprocess()` takes about 13 ms per 640x480 frame on a laptop (camera budget at 30 fps: 33 ms).
- **Size:** 2 cm voxels turn about 200k points into about 11k.
- **Alignment:** frames 0 and 5, both in the world frame: median distance between them 1.6 cm with
  the right poses, 3.7 cm with a pose 50 ms late, 4.7 cm with a stale pose
  (`tests/test_tum.py`; `tests/test_alignment.py` does the same check on a made-up scene, for CI).

To try it, download the sequence into `data/`:

```bash
mkdir -p data && cd data
curl -L -O https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar -xzf rgbd_dataset_freiburg1_desk.tgz && cd ..
python prototype/try_world.py                   # or: --data /somewhere/else/rgbd_dataset_freiburg1_desk
```

The TUM test in `tests/test_tum.py` runs when the sequence is in `data/` (or `TUM_DATA=/path` is set),
and is skipped otherwise.

## Not done yet

- Lens distortion is ignored (pinhole only); fine for the D455, noticeable on TUM.
- Not yet run on our own D455 bags or checked in Foxglove (needs TR-10 bags and the TR-13 runner).
