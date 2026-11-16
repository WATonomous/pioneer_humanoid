"""python make_retarget_mjcf.py <out.xml>

Floating-base, finger-locked copy of the Pioneer whole-body MJCF for GMR retargeting.

The asset is fixed-base (legs and torso hang off the world, root at the soles, forward = -Y, left = +X). This wraps
everything in a free-floating "pelvis" at hip height turned +90 deg about z (forward = +X, left = +Y, as GMR expects),
drops the finger joints (29 DoF left: 12 legs, waist, 2 neck, 14 arms) and the actuators/equalities (kinematics only).
"""
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
from pathlib import Path

ASSET = Path(__file__).resolve().parents[5] / "assets/pioneer_whole_body_humanoid"
HIP_Z = 0.847917
KEEP_HAND_JOINTS = {"wrist_dev", "wrist_flex"}
# closed fists, as on the whole-body walker (pioneer_humanoid.leg_only.FIST), baked into the locked finger joints
FIST = {f"{side}_{finger}{joint}": angle for side in "lr" for finger in ("", "i_", "r_", "l_")
        for joint, angle in (("mcp", 1.4), ("pip", 1.5), ("dip", 1.2))}
FIST |= {f"{side}_th_{joint}": angle for side in "lr" for joint, angle in (("c", 0.6), ("f", 0.9), ("mcp", 0.8), ("ip", 0.9))}

src, out = ASSET / "pioneer_whole_body_humanoid.xml", Path(sys.argv[1])
root = ET.parse(src).getroot()
root.find("compiler").set("meshdir", str(ASSET))
for tag in ("actuator", "equality"):
    for el in root.findall(tag):
        root.remove(el)

world = root.find("worldbody")
pelvis = ET.Element("body", name="pelvis", pos=f"0 0 {HIP_Z}", quat="0.70710678 0 0 0.70710678")
ET.SubElement(pelvis, "freejoint", name="floating_base")
# base_link's inertia from the URDF (the fixed-base MJCF drops it), axes turned to the pelvis frame
ET.SubElement(pelvis, "inertial", pos="0 0 0.001065", mass="4.6735", diaginertia="0.0502475 0.00804196 0.050078")
inner = ET.SubElement(pelvis, "body", name="base_link", pos=f"0 0 {-HIP_Z}")
for child in list(world):
    if child.tag == "body" or (child.tag == "geom" and child.get("type") != "plane"):
        world.remove(child)
        inner.append(child)
world.append(pelvis)
for g in world.findall("geom"):
    if g.get("type") == "plane":
        g.set("size", "0 0 0.05")  # infinite floor: clips travel far

for body in root.iter("body"):
    if "_hand_" in body.get("name", ""):
        for j in body.findall("joint"):
            if j.get("name").split("_", 1)[1] not in KEEP_HAND_JOINTS:
                if j.get("name") in FIST:  # the joint sits at the body origin, so the curl is a rotation of the body
                    axis = np.array(j.get("axis").split(), dtype=float)
                    q = R.from_quat([float(v) for v in body.get("quat", "1 0 0 0").split()], scalar_first=True)
                    q = q * R.from_rotvec(FIST[j.get("name")] * axis / np.linalg.norm(axis))
                    body.set("quat", " ".join(f"{v:.8f}" for v in q.as_quat(scalar_first=True)))
                body.remove(j)
for j in root.iter("joint"):
    if j.get("name", "").endswith("_elbow"):
        j.set("range", f"{j.get('range').split()[0]} 0")  # the elbow bends one way only; the asset also allows +90 deg
ET.ElementTree(root).write(out)
print("wrote", out)
