# Pioneer - UWaterloo's First Humanoid Robot 

We are a team of undergraduate student from UWaterloo building a 160 cm humanoid robot. We are the first engineering design team in Canada building a humanoid!

Our humanoid robot will feature 2 arms + 2 legs and we are also developing a 22 DOF humanoid hand that can be modularly switchable with the arm gripper.

More info at Pioneer's website: https://watonomous.github.io/humanoid-docs/index.html

## Find your subteam

| Subteam | Start here |
|---|---|
| Software: manipulation | [`src/teleop`](src/teleop) · [`src/robot_learning`](src/robot_learning) · [`src/simulation`](src/simulation) |
| Software: locomotion | [`src/simulation/humanoid_rl`](src/simulation/humanoid_rl) · [`src/simulation/humanoid_rl_tasks`](src/simulation/humanoid_rl_tasks) |
| Firmware / interfacing | [`src/embedded`](src/embedded) · [`src/interfacing`](src/interfacing) |
| Perception | [`src/perception`](src/perception) |

## Quick start

`watod` is our Docker Compose wrapper: it spins up one dev container per module (interfacing, perception, simulation, …), so you only run the stack your subteam needs.

```bash
cp watod-config.sh watod-config.local.sh   # set your ACTIVE_MODULES
./watod up -d                 # start your modules
./watod -t <service>          # shell into one (interfacing, perception, simulation_mj, ...)
./watod build <service>       # rebuild after a Dockerfile / dependency change
./watod down                  # stop
```

Your `src/<module>` folder is mounted into the container, so edits on your machine show up inside it immediately without rebuild.

`watod-config.sh` is shared defaults and is CI-guarded, so don't commit personal changes to it. Everything local goes in `watod-config.local.sh` (gitignored).

| `ACTIVE_MODULES` | What it runs |
|------------------|--------------|
| `interfacing` | CAN / hardware interfacing, `joint_command` |
| `perception` | Perception (cameras, GPU), `voxel_grid` |
| `simulation_isaac` | **Isaac Lab 2.3.2** — SO101 robot learning, RL tasks, Quest teleop |
| `simulation_mj` | MuJoCo / mjlab RL |

**Isaac Lab sim (recommended):** see [docker/simulation/isaac_lab/README.md](docker/simulation/isaac_lab/README.md).

## Repo map

Two workflows share this repo:

- **Real robot:** `embedded` firmware ⇄ `interfacing` (CAN ⇄ ROS 2) ⇄ `perception` — driven live by `teleop`, or by a policy.
- **Sim / learning:** `teleop` collects demos in `simulation` scenes → `robot_learning` records datasets and trains policies, or `simulation/humanoid_rl` trains RL policies → deploy back through `interfacing`.

**`robot_learning` vs RL:** `src/robot_learning` learns policies from recorded datasets (sim or real), e.g. ACT, SmolVLA, pi0.5. RL is sim-based (Isaac Lab), so it lives in `src/simulation/humanoid_rl*`.

```
humanoid
├── watod, watod-config.sh   # container orchestrator + module config
├── modules/  docker/        # compose files + Dockerfiles, per module
├── src/
│   ├── pioneer_humanoid/    # the robot definition (imported everywhere)
│   ├── interfacing/         # CAN ⇄ ROS 2 bridge, real-arm bring-up
│   ├── embedded/            # STM32 / ESP32S3 motor firmware
│   ├── perception/          # cameras, voxel grid
│   ├── simulation/          # Isaac Lab + MuJoCo: RL tasks, teleop scenes
│   ├── teleop/              # drive the arm (sim or real)
│   ├── robot_learning/      # record demos, train / eval policies
│   └── common_msgs/         # shared ROS 2 messages
├── assets/                  # robot URDF / USD / meshes, scene props
├── models/                  # frozen RL policy baselines
├── outputs/                 # training runs (gitignored)
└── docs/                    # pointer to the docs site
```

Full detail — [src/simulation/README.md](src/simulation/README.md). Other areas: each has its own `README.md`.

## CAN / arm bring-up

Connect → calibrate → visualize → move, top to bottom: [src/interfacing/README.md](src/interfacing/README.md)

```bash
./src/interfacing/can/scripts/can_udev.sh install   # once per host → /dev/canable
```

## Things to know

- CI does **not** build `simulation_*` or `embedded`. A green check on those means nothing was built, so test them yourself.
- Set a unique `ROS_DOMAIN_ID` (0–232) in `watod-config.local.sh` if someone else runs ROS on the same subnet, or you will see each other's topics.
