"""SO-101 forward kinematics (LeHome's so101_new_calib.urdf, gripper_frame_link) in LeHome's world frame."""
import os
import re
from pathlib import Path

import mujoco
import numpy as np

URDF = Path(os.environ.get("LEHOME_ROOT", "/lehome")) / "assets/robots/so101_new_calib.urdf"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# LeHome garment_bi_cfg_v2.py base positions; their USD base frame is the URDF one turned 90 deg (checked: grasps land on the garment)
BASES = {"left": np.array([-0.23, -0.25, 0.5]), "right": np.array([0.23, -0.25, 0.5])}
R_BASE = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # URDF +x (forward) -> world +y, toward the garment


def _model():
    xml = URDF.read_text()
    xml = re.sub(r"<visual>.*?</visual>", "", xml, flags=re.S)
    xml = re.sub(r"<collision>.*?</collision>", "", xml, flags=re.S)
    xml = xml.replace("<robot ", '<robot xmlns:mujoco="http://mujoco.org"  ', 1)
    xml = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler discardvisual="true" fusestatic="false"/></mujoco>', xml, count=1)
    return mujoco.MjModel.from_xml_string(xml)


M = _model()
D = mujoco.MjData(M)
QADR = [M.jnt_qposadr[M.joint(j).id] for j in JOINTS]
# gripper_frame_link is a fixed child of gripper_link (fusestatic off keeps it a body)
EE = M.body("gripper_frame_link").id


def fk(q6, side):
    """q6 (T, 6) LeHome joint values (rad) -> tool point (T, 3) and rotation (T, 3, 3) in the world."""
    pos, rot = [], []
    for q in np.atleast_2d(q6):
        D.qpos[:] = 0.0
        D.qpos[QADR] = q
        mujoco.mj_kinematics(M, D)
        pos.append(BASES[side] + R_BASE @ D.xpos[EE])
        rot.append(R_BASE @ D.xmat[EE].reshape(3, 3))
    return np.array(pos), np.array(rot)
