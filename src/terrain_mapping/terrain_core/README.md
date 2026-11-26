# terrain_core

The ROS-free part of terrain mapping (tickets TR-1 to TR-18): plain Python that can be run and tested
offline, with no ROS imports. The ROS side (launch files, `elevation_mapping_cupy`) is in
[`../terrain_mapping_bringup`](../terrain_mapping_bringup); see [`../README.md`](../README.md).

## Status

| Part | Ticket | State |
|------|--------|-------|
| Preprocessing: depth to points, crop, voxel downsample, into `odom` | TR-4 | Done, tested on TUM RGB-D |
| Self-filtering (remove arms and legs) | TR-5 | Next |
| Traversability layers: slope, step, roughness | TR-7 | Done offline; synthetic and noisy-map tested |
| Combined traversability decision | TR-8 | Available offline. Human review on course bags pending. |
| Elevation grid, ROS node | TR-2/3, TR-14 | In progress / not wired to our node |

## Layout

```
src/terrain_mapping/terrain_core/
├── pyproject.toml          pip-installable library, no ROS imports
├── COLCON_IGNORE           not a ROS package: keeps the terrain_mapping image's colcon build away
├── config/terrain.yaml     preprocessing plus TR-6/7 robot limits and windows
├── terrain_core/
│   ├── preprocess/         depth.py, crop.py, downsample.py, transform.py, pipeline.py
│   └── traversability/     slope.py, step.py, combined.py (TR-7 layers and offline score)
├── tests/                  python -m pytest (no dataset needed, except one TUM test)
├── terrain_tools/          tum.py plus eval_traversability_npy.py for saved elevation grids
├── prototype/              demo scripts with plots (take --data path/to/sequence)
└── data/                   datasets and bags (gitignored)
```

## Setup

From a checkout of this repository, use Python 3.10 (the version used in CI) and enter
`src/terrain_mapping/terrain_core`. Create and activate a virtual environment once.

**macOS or Linux**

```sh
python3.10 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt)**

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
```

**All platforms**, in the activated environment:

```sh
python -m pip install -e ".[dev]" -e ../../perception/slam/slam_core
python -m pytest
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

## TR-7: traversability layers

TR-7 starts with a 2D elevation grid: one height in metres per map cell, with `NaN` for a cell the camera has not
measured. It produces three same-shaped layers:

- **Slope** is the angle of the least-squares plane fitted around each cell. It answers “how tilted is the ground?”
- **Step** is the largest height jump to a nearby cell. The rise explained by a smooth ramp is removed, so a ramp is
  not falsely reported as stairs.
- **Roughness** is the RMS error left after fitting the plane. It distinguishes a smooth ramp from uneven ground.

`config/terrain.yaml` is the source of truth for the TR-6 limits and window sizes. The slope/roughness fit uses a
0.40 m window so 1–2 cm elevation noise does not look like a dangerous slope. Step detection uses only a 0.05 m
neighbour reach so a real edge remains local. Both settings are metres, not cell counts, so they preserve their
meaning for both 5 cm and ETH's 4 cm grids.

Run the focused regression suite:

```bash
python -m pytest tests/test_traversability.py -q
```

The suite uses clean ramps, steps, boxes, holes, and 1–2 cm seeded noise at both 5 cm and 4 cm resolution. In
particular it requires noisy flat floor and a 5 degree ramp to remain traversable, while a noisy 10 degree ramp and
10 cm step remain blocked.

### Generate the diagrams and check the numbers

From this directory after setup, run:

```sh
python prototype/simulate_tr8.py --report outputs/tr8-report.pdf
```

Open `outputs/tr8-report.pdf` in any PDF viewer. The simulator generates the input on your CPU.
Each page shows one scene's elevation, slope, step height, roughness, score, and final decision.
The counts at the top apply to the sampled surface named on that line. For example, the 14 cm
vertical edge should be blocked, and the 5° ramp up to a 14 cm landing should be walkable.
Gray cells have insufficient data. To repeat the experiment with 10 mm of simulated depth noise:

```sh
python prototype/simulate_tr8.py --noise-mm 10 --report outputs/tr8-noise-10mm.pdf
```

## TR-8 course review

The decision applies the configured limits to each cell. Smooth ground must pass the slope limit (7°).
At a discontinuity, the plane fit can cut across two levels, so the classifier uses the local height jump
to judge the edge. Step (8 cm) and roughness (3 cm, provisional) apply everywhere with sufficient data.
The exported grid uses 1 for traversable, 0 for blocked and NaN for unknown. A level box top can pass
even when its edge is blocked. The layer judges local terrain, not whether a route to the top exists.

### Check a recorded course

1. Export `elevation.npy` using [Score a run](../README.md#score-a-run). Keep each bag's grid and labels together.
2. Before looking at the prediction, identify recognizable floor, step and box edges, and ramp interiors
   from the recording and the elevation image. For the Nov 6 course the measured 14 cm step, 39 cm box edge,
   and roughly 9° ramp all exceed the configured limits. Include clear floor cells as positive examples.
3. Save your judgments on the same grid. To read coordinates, display the array with
   `plt.imshow(z, origin="lower", interpolation="nearest")` and `plt.show()`. The horizontal axis is the
   column and the vertical axis is the row. Replace the example indices below with cells you reviewed.

   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   z = np.load("elevation.npy")
   plt.imshow(z, origin="lower", interpolation="nearest")
   plt.show()
   labels = np.full(z.shape, np.nan)
   labels[20, 30] = 1  # example floor cell
   labels[40, 50] = 0  # example step edge
   np.save("labels.npy", labels)
   ```

4. Run from the terrain_core package directory, substituting the recorded map's cell size if different.

   ```bash
   python terrain_tools/eval_traversability_npy.py elevation.npy --resolution 0.04 \
     --labels labels.npy --decisions-out decisions.npy --out review.png
   ```

The report counts agreement across **all reviewed cells**, including unknown predictions in the denominator.
It lists missed obstacles and incorrectly blocked cells separately. Check the color-coded decision panel
against the elevation panel for each disagreement. Record the bag, labels, and counts together for sign-off.

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

The TUM test in `tests/test_tum.py` runs when the sequence is in `data/` (or `TUM_DATA=/path` is set), tand is skipped otherwise.

## Not done yet

- Lens distortion is ignored (pinhole only); fine for the D455, noticeable on TUM.
- Not yet run on our own D455 bags or checked in Foxglove (needs TR-10 bags and the TR-13 runner).
