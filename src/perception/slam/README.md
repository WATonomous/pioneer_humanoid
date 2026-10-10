# SLAM

Custom SLAM system for the humanoid, built around an Intel RealSense D455 (left IR + depth + IMU).
Visual odometry frontend, GTSAM backend, loop closure, dense mapping.

> Status: skeleton. Only the VO frontend (`frontend/`, `io/`, `odometry_node.py`) and shared `types/` have placeholder files so far. Every other folder exists but is empty (`.gitkeep`): add your module's files there when you start on it.

## Scope

Must
- Visual odometry frontend
- GTSAM backend
- Loop closure
- Runs offline on recorded data
- Failure-case writeup (tracking loss and bad loops, with cause and fix)

Should
- Dense voxel map, rebuilt after loop closure
- Map save/load and same-sensor relocalization
- IMU preintegration (VIO)

Nice to have
- Live on the robot (tethered PC, onboard compute later)
- Phone-map-to-robot relocalization

## Architecture

A ROS-free core library (`slam_core`) plus a thin ROS 2 wrapper (`slam_ros`), the same split RTAB-Map uses (`rtabmap` + `rtabmap_ros`).

Why:
- Use the SLAM code without ROS. Every feature (VO, the backend graph, loop closure) can be run and debugged from a plain Python script, with no nodes to launch.
- Reproducible offline runs. `ros2 bag play` replays in real time, so a slow node drops or reorders messages and two runs on the same bag can give different results. `slam_tools/run_offline.py` instead reads datasets and bags directly and feeds every frame through the core, as fast as the code allows, with the same result every run. That is what makes failures reproducible and benchmarks fair.
- Easier testing. Plain `pytest` on any module, no ROS install or running nodes needed.
- Easier to contribute. People working on algorithms only touch `slam_core` and never deal with ROS overhead (topics, QoS, TF, launch files). ROS knowledge is only needed in `slam_ros`.
- One code path for datasets, our bags, the MuJoCo sim, and live data.

```
 camera / bag / dataset
          |
          v
 +------------------+   /vo/keyframe   +---------------------------+
 |  odometry_node   | ---------------> |        graph_node         |
 |  (frontend, VO)  |                  | backend + loop closure    |
 |  ~30 Hz          |                  | + map, ~1-5 Hz            |
 +------------------+                  +---------------------------+
   TF odom->camera_link                  TF map->odom, optimized poses
                                                   |
                                                   v
                                       +---------------------------+
                                       |       mapping_node        |
                                       | voxel map, point clouds,  |
                                       | 2D costmap                |
                                       +---------------------------+
```

Nodes are split by rate, not by module. Loop closure lives with the backend because it needs the keyframe database and adds factors straight into the graph.

Language: Python throughout. GTSAM via its Python bindings. Hot paths can be ported later without touching the ROS side.

## Layout

```
slam/
├── config/              calibration + parameters, shared by core and ROS
├── slam_core/           the SLAM library, NO ROS imports
│   ├── slam_core/
│   │   ├── types/          Frame, Keyframe, Pose, Camera, Imu  (agree on these first)
│   │   ├── io/             TUM, EuRoC, our ROS 2 bags -> Frame
│   │   ├── frontend/       features, matching, PnP, keyframe selection, local map, arm masking, tracker (VOCore)
│   │   ├── backend/        factor graph, factors, IMU preintegration, optimizer (iSAM2)
│   │   ├── loop_closure/   place recognition, geometric verification
│   │   ├── map/            keyframe database, save/load (one map format)
│   │   ├── relocalization/ localize in a saved map
│   │   └── dense/          voxel map, 2D costmap
│   └── tests/
├── slam_ros/            ROS 2 nodes, message conversions, launch files, Foxglove layout
├── slam_msgs/           Keyframe, LoopClosure, TrackingStats messages
├── slam_tools/          offline runner, evo evaluation, RTAB-Map baseline, benchmarks
├── prototype/           throwaway end-to-end prototypes; reference only, nothing imports from here
├── docs/                architecture, failure cases, benchmarks
└── data/                datasets and bags (gitignored, local only)
```
