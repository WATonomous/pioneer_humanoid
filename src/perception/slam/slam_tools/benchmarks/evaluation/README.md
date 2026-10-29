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
rotational relative-pose RMSE. The evo comparisons use alignment.

Compute loop-closure error for a recording that starts and ends at the same
physical location:

```bash
./eval_loop.sh estimated_trajectory.txt
```

This reports the displacement and rotation between the first and last poses.
It is a loop-closure metric, not ground-truth accuracy.

Generate trajectory plots:

```bash
./plot_trajectory.sh groundtruth.txt estimated_trajectory.txt
```

Plots are saved in a `trajectory_plots/` directory beside the ground-truth
file.

