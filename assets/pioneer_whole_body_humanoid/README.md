# pioneer_whole_body_humanoid

The Pioneer Whole Body Humanoid (graphite helmet, taller torso, hands D, 1010 legs; 69 joints) for MuJoCo.

Not to be confused with `assets/old_leg_only_humanoid` (`old_leg_only_humanoid.urdf`, once `whole_body_humanoid`): the
old legs-only humanoid, two legs under a solid placeholder upper body with no arms, used for Isaac Lab locomotion
training (`pioneer_humanoid.whole_body`). This one is the whole robot as designed in CAD: head,
torso, arms, hands with every finger joint, a waist yaw and the legs. Its leg joints carry the same names, ranges and
directions as the team's model (`python assets/old_leg_only_humanoid/check_leg_directions.py` checks both).

| file | what it is |
|------|------------|
| `pioneer_whole_body_humanoid.xml` | **Open this one** in MuJoCo's viewer (`python -m mujoco.viewer --mjcf pioneer_whole_body_humanoid.xml`, or drag it in): the robot at a workbench with its CAD colours, the collision boxes hidden (group 3). Written by `python -m pioneer_whole_body.view --scene bare --export ...`. |
| `pioneer_whole_body_humanoid.urdf` | The URDF (from the Prometheus CAD, `convert.py`). Opened in MuJoCo's viewer directly it shows a robot of grey boxes: MuJoCo's URDF reader puts the collision boxes in group 0 next to the meshes and drops the colours. Use it through `pioneer_whole_body.robot.robot_spec()` (what the scenes do) or for other tools. |
| `pioneer_whole_body_humanoid_gripper.xml` / `.urdf` | **The same robot with grippers**: each arm ends in its own wrist and two-jaw gripper, as armV2 was built, in place of the hands. Its arms are the team's arm model's exactly (`assets/pioneer_bimanual_arm`: the same CAD): joints `joint1`..`joint8` (right) and `joint1L`, `joint2l`..`joint8l` (left), with that model's axes and limits, `joint7` / `joint8` the sliding jaws, so code written for the bimanual arm drives these arms as they are. The head, torso, waist, neck and legs are the hand model's (same meshes, joints and limits). The MuJoCo model has one position actuator a joint and a fixed base: open it to move the joints by hand. Written by `to_mjcf.py`. |
| `meshes/` | The visual meshes, decimated to 41 MB by `convert.py` (each keeps at least 98 % of its surface: thinner, the shells open up). |
| `convert.py` | Rebuilds the URDF + meshes from the Prometheus export (`python convert.py <...>_gripper_mujoco.urdf pioneer_whole_body_humanoid_gripper` for the gripper model: it adds only its wrist and jaw meshes). |
| `to_mjcf.py` | A converted URDF to a MuJoCo model to open in the viewer (the gripper model's `.xml`). |

The scenes and their demos (in a follow-up PR; `pioneer_whole_body` above is their package): `src/simulation/whole_body_scenes` (`python -m pioneer_whole_body.view --scene bearing_block --play` runs one live in the viewer).
