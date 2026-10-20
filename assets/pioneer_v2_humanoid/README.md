# pioneer_v2_humanoid

The whole-body Pioneer v2 humanoid (graphite helmet, taller torso, hands D, 1010 legs; 69 joints) for MuJoCo.

| file | what it is |
|------|------------|
| `pioneer_v2.xml` | **Open this one** in MuJoCo's viewer (`python -m mujoco.viewer --mjcf pioneer_v2.xml`, or drag it in): the robot at a workbench with its CAD colours, the collision boxes hidden (group 3). Written by `python -m pioneer_whole_body.view --scene bare --export ...`. |
| `pioneer_humanoid_v2.urdf` | The URDF (from the Prometheus CAD, `convert.py`). Opened in MuJoCo's viewer directly it shows a robot of grey boxes: MuJoCo's URDF reader puts the collision boxes in group 0 next to the meshes and drops the colours. Use it through `pioneer_whole_body.robot.robot_spec()` (what the scenes do) or for other tools. |
| `meshes/` | The visual meshes, decimated to 32 MB by `convert.py`. |
| `convert.py` | Rebuilds the URDF + meshes from the Prometheus export. |

The scenes and their demos: `src/simulation/whole_body_scenes` (`python -m pioneer_whole_body.view --scene bearing_block --play` runs one live in the viewer).
