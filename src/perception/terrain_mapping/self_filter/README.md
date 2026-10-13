# Self-filter (terrain mapping input cleaning)

Cleans sensor point clouds before they reach the elevation map
(`elevation_mapping_cupy`). Inputs: the pelvis lidar and the chest camera
(moving to the head later).

1. Cut depth to the useful range and thin it out
2. Remove the robot's own arms and legs using the URDF + joint states
3. Keep timestamps and frames consistent (output keeps the input's header)

> Status: just started. Folder location may change once the terrain mapping
> team agrees on a layout.

## Setup (plain Python, no ROS needed)

```bash
cd src/perception/terrain_mapping/self_filter
python3.10 -m venv .venv
source .venv/bin/activate
```
