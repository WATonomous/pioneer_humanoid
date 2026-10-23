# pioneer_leg_only_humanoid

The Pioneer Whole Body Humanoid (`../pioneer_whole_body_humanoid`) without its arms: the legs, pelvis, torso and head,
for learning to walk first. **12 moving joints**: the 6 leg joints a side, with the same names, ranges and directions as
`old_leg_only_humanoid` (`Hip_F`, `Hip_A`, `Hip_R`, `Knee`, `Ankle_P`, `Ankle_R`, `_L` / `_R`). The waist and neck are
locked (fixed joints). **32.2 kg** (an estimate: see the whole-body README's "Masses and joint limits"; no arms, hands or
shoulder motors; the torso keeps its shoulder frame). Its soles stand at z = 0.

| file | what it is |
|------|------------|
| `pioneer_leg_only_humanoid.xml` | **Open this one** in MuJoCo's viewer: the robot on a floor in its CAD colours, a fixed base, one position actuator a joint capped at its motor's limit. Written by `../pioneer_whole_body_humanoid/to_mjcf.py`. |
| `pioneer_leg_only_humanoid.urdf` | The URDF, for Isaac Lab and other tools. |
| `meshes/` | Only the torso's meshes (without the shoulder motors it differs from the whole-body robot's). Every other mesh is the whole-body robot's: the URDF points at `../pioneer_whole_body_humanoid/meshes/`, so keep the two folders side by side. |

Rebuilt with `../pioneer_whole_body_humanoid/convert.py` (its third form) from the Prometheus export
(`make_urdf.py` with `URDF_ARMS=none`). Checks: `python ../old_leg_only_humanoid/check_leg_directions.py`, and
`check_hold.py` / `check_motion.py` in the whole-body folder, given this URDF: all pass.
