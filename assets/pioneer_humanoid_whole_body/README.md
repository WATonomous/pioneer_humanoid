# pioneer_humanoid_whole_body

The Pioneer Humanoid Whole Body (graphite helmet, taller torso, hands D, 1010 legs; 69 joints) for MuJoCo.

Not to be confused with `assets/old_leg_only_humanoid` (`old_leg_only_humanoid.urdf`, once `whole_body_humanoid`): the
old legs-only humanoid, two legs under a solid placeholder upper body with no arms, used for Isaac Lab locomotion
training (`pioneer_humanoid.whole_body`). This one is the whole robot as designed in CAD: head,
torso, arms, hands with every finger joint, a waist yaw and the legs. Its leg joints carry the same names, ranges and
directions as the team's model (`python assets/old_leg_only_humanoid/check_leg_directions.py` checks both).

| file | what it is |
|------|------------|
| `pioneer_humanoid_whole_body.xml` | **Open this one** in MuJoCo's viewer (`python -m mujoco.viewer --mjcf pioneer_humanoid_whole_body.xml`, or drag it in): the robot at a workbench with its CAD colours, the collision boxes hidden (group 3). Written by `python -m pioneer_whole_body.view --scene bare --export ...`. |
| `pioneer_humanoid_whole_body.urdf` | The URDF (from the Prometheus CAD, `convert.py`). Opened in MuJoCo's viewer directly it shows a robot of grey boxes: MuJoCo's URDF reader puts the collision boxes in group 0 next to the meshes and drops the colours. Use it through `pioneer_whole_body.robot.robot_spec()` (what the scenes do) or for other tools. |
| `meshes/` | The visual meshes, decimated to 41 MB by `convert.py` (each keeps at least 98 % of its surface: thinner, the shells open up). |
| `convert.py` | Rebuilds the URDF + meshes from the Prometheus export. |

The scenes and their demos: `src/simulation/whole_body_scenes` (`python -m pioneer_whole_body.view --scene bearing_block --play` runs one live in the viewer).
