# Pioneer humanoid: joint limits

Generated from `joint_specs.yaml` by `uv run python motor_test.py --limits JOINT_LIMITS.md`: edit the YAML, not this table. *Mass moved* is everything outboard of the joint, with the model's current masses.

| Group | Joint | Lower (°) | Upper (°) | Torque cap (Nm) | Speed limit (rad/s) | Rotor inertia (kg·m²) | Mass moved (kg) | Specs source | Limits source |
|---|---|---|---|---|---|---|---|---|---|
| Torso & head | `waist_yaw` | -90 | 90 | 10 | 6 | 0.01 | 3.89 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Torso & head | `neck_yaw` | -70 | 70 | 10 | 6 | 0.01 | 0.37 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Torso & head | `neck_pitch` | -20 | 30 | 10 | 6 | 0.01 | 0.26 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right arm | `r_shoulder_pitch` | -180 | 180 | 53 | 6 | 0.0081 | 1.15 | ASSUMED: previous Wato arm motor (AK10-9), confirm for armV2 | URDF |
| Right arm | `r_shoulder_roll` | -90 | 90 | 53 | 6 | 0.0081 | 0.81 | ASSUMED: previous Wato arm motor (AK10-9), confirm for armV2 | URDF |
| Right arm | `r_upper_arm_yaw` | -90 | 90 | 18 | 6 | 0.0049 | 0.63 | ASSUMED: previous Wato arm motor (AK80-9), confirm for armV2 | URDF |
| Right arm | `r_elbow` | -90 | 0 | 18 | 6 | 0.0049 | 0.42 | ASSUMED: previous Wato arm motor (AK80-9), confirm for armV2 | URDF (fixed Oct 22: was -90..90) |
| Right arm | `r_forearm_roll` | -90 | 90 | 18 | 6 | 0.0049 | 0.34 | ASSUMED: previous Wato arm motor (AK80-9), confirm for armV2 | URDF |
| Right arm | `r_wrist_dev` | -20 | 30 | 0.73 | 6 | 0.0005 | 0.26 | ASSUMED: previous Wato arm motor (GL40), confirm for armV2 | URDF |
| Right arm | `r_wrist_flex` | -60 | 60 | 0.73 | 6 | 0.0005 | 0.25 | ASSUMED: previous Wato arm motor (GL40), confirm for armV2 | URDF |
| Right hand | `r_abd` | -15 | 15 | 10 | 6 | 0.01 | 0.04 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_mcp` | 0 | 90 | 10 | 6 | 0.01 | 0.03 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_pip` | 0 | 90 | 10 | 6 | 0.01 | 0.02 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_i_abd` | -15 | 15 | 10 | 6 | 0.01 | 0.04 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_i_mcp` | 0 | 90 | 10 | 6 | 0.01 | 0.03 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_i_pip` | 0 | 90 | 10 | 6 | 0.01 | 0.02 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_r_abd` | -15 | 15 | 10 | 6 | 0.01 | 0.04 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_r_mcp` | 0 | 90 | 10 | 6 | 0.01 | 0.03 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_r_pip` | 0 | 90 | 10 | 6 | 0.01 | 0.02 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_l_abd` | -15 | 15 | 10 | 6 | 0.01 | 0.04 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_l_mcp` | 0 | 90 | 10 | 6 | 0.01 | 0.03 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_l_pip` | 0 | 90 | 10 | 6 | 0.01 | 0.02 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_th_c` | 0 | 50 | 10 | 6 | 0.01 | 0.04 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_th_f` | 0 | 60 | 10 | 6 | 0.01 | 0.03 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_th_mcp` | 0 | 80 | 10 | 6 | 0.01 | 0.02 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Right hand | `r_th_ip` | 0 | 80 | 10 | 6 | 0.01 | 0.01 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left arm | `l_shoulder_pitch` | -180 | 180 | 53 | 6 | 0.0081 | 1.14 | ASSUMED: previous Wato arm motor (AK10-9), confirm for armV2 | URDF |
| Left arm | `l_shoulder_roll` | -90 | 90 | 53 | 6 | 0.0081 | 0.81 | ASSUMED: previous Wato arm motor (AK10-9), confirm for armV2 | URDF |
| Left arm | `l_upper_arm_yaw` | -90 | 90 | 18 | 6 | 0.0049 | 0.63 | ASSUMED: previous Wato arm motor (AK80-9), confirm for armV2 | URDF |
| Left arm | `l_elbow` | -90 | 0 | 18 | 6 | 0.0049 | 0.42 | ASSUMED: previous Wato arm motor (AK80-9), confirm for armV2 | URDF (fixed Oct 22: was -90..90) |
| Left arm | `l_forearm_roll` | -90 | 90 | 18 | 6 | 0.0049 | 0.34 | ASSUMED: previous Wato arm motor (AK80-9), confirm for armV2 | URDF |
| Left arm | `l_wrist_dev` | -20 | 30 | 0.73 | 6 | 0.0005 | 0.26 | ASSUMED: previous Wato arm motor (GL40), confirm for armV2 | URDF |
| Left arm | `l_wrist_flex` | -60 | 60 | 0.73 | 6 | 0.0005 | 0.25 | ASSUMED: previous Wato arm motor (GL40), confirm for armV2 | URDF |
| Left hand | `l_abd` | -15 | 15 | 10 | 6 | 0.01 | 0.04 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_mcp` | 0 | 90 | 10 | 6 | 0.01 | 0.03 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_pip` | 0 | 90 | 10 | 6 | 0.01 | 0.02 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_i_abd` | -15 | 15 | 10 | 6 | 0.01 | 0.04 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_i_mcp` | 0 | 90 | 10 | 6 | 0.01 | 0.03 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_i_pip` | 0 | 90 | 10 | 6 | 0.01 | 0.02 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_r_abd` | -15 | 15 | 10 | 6 | 0.01 | 0.04 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_r_mcp` | 0 | 90 | 10 | 6 | 0.01 | 0.03 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_r_pip` | 0 | 90 | 10 | 6 | 0.01 | 0.02 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_l_abd` | -15 | 15 | 10 | 6 | 0.01 | 0.04 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_l_mcp` | 0 | 90 | 10 | 6 | 0.01 | 0.03 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_l_pip` | 0 | 90 | 10 | 6 | 0.01 | 0.02 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_th_c` | 0 | 50 | 10 | 6 | 0.01 | 0.04 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_th_f` | 0 | 60 | 10 | 6 | 0.01 | 0.03 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_th_mcp` | 0 | 80 | 10 | 6 | 0.01 | 0.02 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left hand | `l_th_ip` | 0 | 80 | 10 | 6 | 0.01 | 0.01 | PLACEHOLDER (URDF): no real spec yet | URDF |
| Left leg | `Hip_F_L` | -20 | 90 | 222 | 3.67 | 0.03 | 17.15 | team model (assets/whole_body_humanoid): real | URDF |
| Left leg | `Hip_A_L` | -10 | 60 | 120 | 20.94 | 0.025 | 15.29 | team model (assets/whole_body_humanoid): real | URDF |
| Left leg | `Hip_R_L` | -30 | 90 | 60 | 20.42 | 0.012 | 13.11 | team model (assets/whole_body_humanoid): real | URDF |
| Left leg | `Knee_L` | 0 | 65 | 222 | 3.67 | 0.03 | 6.72 | team model (assets/whole_body_humanoid): real | URDF (fixed Oct 22: was -65..0) |
| Left leg | `Ankle_P_L` | -30 | 50 | 120 | 20.42 | 0.024 | 2.15 | team model (assets/whole_body_humanoid): real | URDF |
| Left leg | `Ankle_R_L` | -15 | 15 | 120 | 20.42 | 0.024 | 1.89 | team model (assets/whole_body_humanoid): real | URDF |
| Right leg | `Hip_F_R` | -20 | 90 | 222 | 3.67 | 0.03 | 17.15 | team model (assets/whole_body_humanoid): real | URDF |
| Right leg | `Hip_A_R` | -10 | 60 | 120 | 20.94 | 0.025 | 15.29 | team model (assets/whole_body_humanoid): real | URDF |
| Right leg | `Hip_R_R` | -30 | 90 | 60 | 20.42 | 0.012 | 13.11 | team model (assets/whole_body_humanoid): real | URDF |
| Right leg | `Knee_R` | 0 | 65 | 222 | 3.67 | 0.03 | 6.72 | team model (assets/whole_body_humanoid): real | URDF (fixed Oct 22: was -65..0) |
| Right leg | `Ankle_P_R` | -30 | 50 | 120 | 20.42 | 0.024 | 2.15 | team model (assets/whole_body_humanoid): real | URDF |
| Right leg | `Ankle_R_R` | -15 | 15 | 120 | 20.42 | 0.024 | 1.89 | team model (assets/whole_body_humanoid): real | URDF |

**Finger DIP joints** (no motor; each follows its PIP joint, URDF `<mimic>`): `l_dip` = 1 × `l_pip`, `l_i_dip` = 1 × `l_i_pip`, `l_l_dip` = 1 × `l_l_pip`, `l_r_dip` = 1 × `l_r_pip`, `r_dip` = 1 × `r_pip`, `r_i_dip` = 1 × `r_i_pip`, `r_l_dip` = 1 × `r_l_pip`, `r_r_dip` = 1 × `r_r_pip`.

**Link masses set in `joint_specs.yaml`** (the rest are the URDF's placeholders): `l_hip_pitch_link` 1.86 kg, `l_hip_roll_link` 2.18 kg, `l_thigh` 6.39 kg, `l_shank_foot` 4.57 kg, `l_ankle_cross` 0.26 kg, `l_foot` 1.89 kg, `r_hip_pitch_link` 1.86 kg, `r_hip_roll_link` 2.18 kg, `r_thigh` 6.39 kg, `r_shank_foot` 4.57 kg, `r_ankle_cross` 0.26 kg, `r_foot` 1.89 kg.

Total robot mass: 38.2 kg.
