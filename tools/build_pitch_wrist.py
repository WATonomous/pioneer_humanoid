#!/usr/bin/env python3
"""Build the pitch-wrist arm: the current gripper on the old forearm and pitch wrist bracket.

The current wrist (joint6) is a yaw: tipping the gripper needs a joint3/joint6 counter-twist. The arm
before #468 (5244df7) had a pitch wrist: an L-bracket on the forearm tip, joint6 axis parallel to the
elbow. This takes from that commit, for each arm:

- link5 (forearm + bracket side of the pitch motor), as it was;
- the top of link6 (the bracket the gripper hangs from), cut off above the old gripper;

and puts the current gripper (link6 below its yaw clevis, fingers link7/link8) where the old gripper
sat on it. joint6 gets the old pitch axis. Its origin stays on the forearm axis: the old origin sits
along that same pitch line (46 / 37 mm sideways, along the axis), so the motion is identical.

Writes urdf/pioneer_bimanual_arm_pitch_wrist.urdf and meshes/pitch_*.STL next to the current arm::

    python3 tools/build_pitch_wrist.py

Use it with PIONEER_ARM_WRIST=pitch (pioneer_humanoid.urdf_joint_limits).
"""
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

OLD_COMMIT = "5244df7"
ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "assets" / "pioneer_bimanual_arm"
MESH_URL = "package://armv2URDF/meshes/"
BRACKET_ABOVE = -0.062   # m, old link6 frame: above this is bracket, below it the old gripper
GRIPPER_BELOW = -0.055   # m, current link6 frame: below this is gripper, above it the yaw clevis
COLOURS = {"print": "0.925 0.929 0.937 1", "gl40": "0.227 0.235 0.251 1", "steel": "0.604 0.620 0.643 1"}

# suffix: (old joint6 origin minus the on-axis one, current gripper shift onto the old gripper's spot), link6 frame
SIDES = {
    "l": ((0.0, -0.046, 0.0), (0.0, 0.007, -0.009)),
    "": ((0.0, 0.037, 0.0), (0.0, -0.007, -0.009)),
}
_STL = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("x", "<u2")])


def read_stl(data: bytes) -> np.ndarray:
    n = int(np.frombuffer(data[80:84], "<u4")[0])
    return np.frombuffer(data[84:84 + n * 50], dtype=_STL)["v"].copy()


def write_stl(path: Path, tris: np.ndarray) -> None:
    out = np.zeros(len(tris), dtype=_STL)
    out["v"] = tris
    path.write_bytes(b"\0" * 80 + np.uint32(len(tris)).tobytes() + out.tobytes())


def old_file(rel: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{OLD_COMMIT}:{rel}"], check=True, capture_output=True).stdout


def shift(elem, d) -> None:
    elem.set("xyz", " ".join(str(a + b) for a, b in zip(map(float, elem.get("xyz").split()), d)))


def mesh_elem(parent, tag: str, name: str, colour: str | None = None) -> None:
    e = ET.SubElement(parent, tag)
    ET.SubElement(e, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(ET.SubElement(e, "geometry"), "mesh", filename=MESH_URL + name)
    if colour:
        ET.SubElement(ET.SubElement(e, "material", name=colour), "color", rgba=COLOURS[colour])


def main() -> None:
    text = (ARM / "urdf" / "pioneer_bimanual_arm.urdf").read_text()
    tree = ET.ElementTree(ET.fromstring(text))
    robot = tree.getroot()
    old = ET.fromstring(old_file("assets/pioneer_bimanual_arm/urdf/pioneer_bimanual_arm.urdf"))
    links = {e.get("name"): e for e in robot.findall("link")}
    joints = {e.get("name"): e for e in robot.findall("joint")}
    old_links = {e.get("name"): e for e in old.findall("link")}
    old_joints = {e.get("name"): e for e in old.findall("joint")}

    for side, (to_axis, onto_bracket) in SIDES.items():
        link5, link6 = f"link5{side}", f"link6{side}"
        # forearm: the old one, recoloured like the current print
        new5 = old_links[link5]
        for e in new5.findall("visual") + new5.findall("collision"):
            e.find("geometry/mesh").set("filename", MESH_URL + f"pitch_{link5}.STL")
        for e in new5.findall("visual"):
            e.remove(e.find("material"))
            ET.SubElement(ET.SubElement(e, "material", name="print"), "color", rgba=COLOURS["print"])
        robot.insert(list(robot).index(links[link5]), new5)
        robot.remove(links[link5])
        (ARM / "meshes" / f"pitch_{link5}.STL").write_bytes(old_file(f"assets/pioneer_bimanual_arm/meshes/{link5}.STL"))

        # wrist bracket + current gripper
        tris = read_stl(old_file(f"assets/pioneer_bimanual_arm/meshes/{link6}.STL"))
        write_stl(ARM / "meshes" / f"pitch_{link6}_bracket.STL", tris[tris.mean(1)[:, 2] > BRACKET_ABOVE] + to_axis)
        for part in ("_print", "_gl40", "_steel", ""):
            tris = read_stl((ARM / "meshes" / f"{link6}{part}.STL").read_bytes())
            write_stl(ARM / "meshes" / f"pitch_{link6}{part}.STL", tris[tris.mean(1)[:, 2] < GRIPPER_BELOW] + onto_bracket)
        body = links[link6]
        for e in body.findall("visual") + body.findall("collision"):
            body.remove(e)
        mesh_elem(body, "visual", f"pitch_{link6}_bracket.STL", "print")
        for part in ("print", "gl40", "steel"):
            mesh_elem(body, "visual", f"pitch_{link6}_{part}.STL", part)
        mesh_elem(body, "collision", f"pitch_{link6}_bracket.STL")
        mesh_elem(body, "collision", f"pitch_{link6}.STL")
        shift(body.find("inertial/origin"), onto_bracket)
        for finger in (f"joint7{side}", f"joint8{side}"):
            shift(joints[finger].find("origin"), onto_bracket)
        joints[f"joint6{side}"].find("axis").set("xyz", old_joints[f"joint6{side}"].find("axis").get("xyz"))

    ET.indent(tree)
    preamble = text[:text.index("<robot")]   # the manual-fix notes sit above <robot>
    (ARM / "urdf" / "pioneer_bimanual_arm_pitch_wrist.urdf").write_text(preamble + ET.tostring(robot, encoding="unicode") + "\n")


if __name__ == "__main__":
    main()
