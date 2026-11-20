# Trajectory evaluation

These scripts score already-exported trajectories. They do not read ROS bags,
launch RTAB-Map, or depend on ROS 2. Each trajectory must use
TUM format:

```text
timestamp tx ty tz qx qy qz qw
```

## Setup

Install `evo` in a Python environment:

```bash
python3 -m venv ~/.venvs/evo
source ~/.venvs/evo/bin/activate
python -m pip install evo
```

The scripts also work when `evo_ape`, `evo_rpe`, and `evo_traj` are already
available on `PATH`.

## Commands

Compare a trajectory with ground truth:

```bash
./eval_trajectory.sh groundtruth.txt estimated_trajectory.txt
```

This prints absolute positional and rotational RMSE, plus translational and
rotational relative-pose RMSE. RPE is measured over a fixed one-metre delta so
results are comparable between frame-based and keyframe trajectories. The evo
comparisons use rigid alignment, which preserves metric scale. For a
scale-arbitrary COLMAP reference, enable Sim(3) alignment with:

```bash
./eval_trajectory.sh groundtruth.txt colmap_trajectory.txt --align-scale
```

Use `--align-scale` only when scale is not meaningful; do not use it for
metric ground-truth benchmarks because it can hide scale errors.

To print only one CSV row for the spreadsheet, add `--csv`. The four fields
are ATE translation RMSE (m), ATE rotation RMSE (deg), RPE translation RMSE
(m), and RPE rotation RMSE (deg):

```bash
./eval_trajectory.sh groundtruth.txt estimated_trajectory.txt --csv
```

The options can be combined:

```bash
./eval_trajectory.sh groundtruth.txt colmap_trajectory.txt --align-scale --csv
```

Compute loop-closure error for a recording that starts and ends at the same
physical location:

```bash
./eval_loop.sh estimated_trajectory.txt
```

This reports the displacement and rotation between the first and last poses,
the total trajectory path length, and translation drift as a percentage of the
path length. Run it on the raw frontend odometry before loop-closure or backend
correction; otherwise the correction can hide the accumulated drift. It is a
loop-closure metric, not ground-truth accuracy.

For a one-row CSV result, add `--csv`. The fields are loop translation error
(m), loop rotation error (deg), total path length (m), and translation drift
(%):

```bash
./eval_loop.sh estimated_trajectory.txt --csv
```

Generate trajectory plots:

```bash
./plot_trajectory.sh groundtruth.txt estimated_trajectory.txt
```

By default, plots are saved in a `trajectory_plots/` directory beside the
ground-truth file. To choose a different output directory, pass it as the
third argument:

```bash
./plot_trajectory.sh groundtruth.txt estimated_trajectory.txt plots/run_01
```

