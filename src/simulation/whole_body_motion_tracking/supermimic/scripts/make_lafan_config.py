"""GMR IK config LAFAN1 BVH -> Pioneer, derived from GMR's LAFAN1 -> G1 config.

Pioneer's kinematic chain matches G1's link for link, and every Pioneer link frame is G1's convention turned +90 deg
about z (see make_retarget_mjcf.py), so each rotation offset is G1's times Rz(90). Then registers "pioneer" in GMR.

  python make_lafan_config.py <GMR>/general_motion_retargeting <retarget.xml> <a LAFAN1 bvh, for its T-pose first frame>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

GMR = Path(sys.argv[1])  # .../general_motion_retargeting
XML = Path(sys.argv[2]).resolve()

NAMES = {
    "pelvis": "pelvis", "torso_link": "torso",
    "left_hip_yaw_link": "l_thigh", "left_knee_link": "l_shank_foot", "left_ankle_roll_link": "l_foot",
    "right_hip_yaw_link": "r_thigh", "right_knee_link": "r_shank_foot", "right_ankle_roll_link": "r_foot",
    "left_shoulder_yaw_link": "l_arm_link3", "left_elbow_link": "l_arm_link4", "left_wrist_yaw_link": "l_hand_H",
    "right_shoulder_yaw_link": "r_arm_link3", "right_elbow_link": "r_arm_link4", "right_wrist_yaw_link": "r_hand_H",
}
RZ90 = R.from_euler("z", 90, degrees=True)

cfg = json.load(open(GMR / "ik_configs/bvh_lafan1_to_g1.json"))
# G1's 0.75 arm scale shrank punches short of full reach: Pioneer's arm (0.655 m shoulder to wrist) outreaches a LAFAN
# actor's (0.58 m at full extension)
for b in ("LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"):
    cfg["human_scale_table"][b] = 1.1
for table in ("ik_match_table1", "ik_match_table2"):
    new = {}
    for g1_name, (human, pw, rw, pos, quat) in cfg[table].items():
        q = (R.from_quat(quat, scalar_first=True) * RZ90).as_quat(scalar_first=True)
        new[NAMES[g1_name]] = [human, pw, rw, pos, np.round(q, 8).tolist()]
    cfg[table] = new
json.dump(cfg, open(GMR / "ik_configs/bvh_lafan1_to_pioneer.json", "w"), indent=4)

params = (GMR / "params.py").read_text()
if '"pioneer"' not in params:
    params = params.replace("ROBOT_XML_DICT = {", f'ROBOT_XML_DICT = {{\n    "pioneer": pathlib.Path("{XML}"),', 1)
    params = params.replace('"unitree_g1": IK_CONFIG_ROOT / "bvh_lafan1_to_g1.json",',
                            '"unitree_g1": IK_CONFIG_ROOT / "bvh_lafan1_to_g1.json",\n        "pioneer": IK_CONFIG_ROOT / "bvh_lafan1_to_pioneer.json",', 1)
    params = params.replace("ROBOT_BASE_DICT = {", 'ROBOT_BASE_DICT = {\n    "pioneer": "pelvis",', 1)
    params = params.replace("VIEWER_CAM_DISTANCE_DICT = {", 'VIEWER_CAM_DISTANCE_DICT = {\n    "pioneer": 3.0,', 1)
    (GMR / "params.py").write_text(params)
print("ok")

# The arms come out wrong with G1's offsets (they held the arms out sideways), so every offset except the pelvis is
# recalibrated on LAFAN's T-pose first frame: offset = human^-1 * robot, with the robot in a T-pose (shoulder roll +90)
# facing the way the G1-derived pelvis offset puts it.
import mujoco
from general_motion_retargeting.utils.lafan1 import load_bvh_file

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
human = load_bvh_file(sys.argv[3], format="lafan1")[0][0]
pelvis = R.from_quat(human["Hips"][1], scalar_first=True) * R.from_quat(cfg["ik_match_table1"]["pelvis"][4], scalar_first=True)
d.qpos[:] = 0
d.qpos[3:7] = pelvis.as_quat(scalar_first=True)
for side in "lr":
    d.qpos[m.jnt_qposadr[m.joint(f"{side}_shoulder_roll").id]] = np.pi / 2
mujoco.mj_forward(m, d)
for table in ("ik_match_table1", "ik_match_table2"):
    for body, entry in cfg[table].items():
        off = R.from_quat(human[entry[0]][1], scalar_first=True).inv() * R.from_quat(d.body(body).xquat, scalar_first=True)
        entry[4] = np.round(off.as_quat(scalar_first=True), 8).tolist()
json.dump(cfg, open(GMR / "ik_configs/bvh_lafan1_to_pioneer.json", "w"), indent=4)
print("calibrated on T-pose")
