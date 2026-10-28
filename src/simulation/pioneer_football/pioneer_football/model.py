"""Build a floating astronaut from the authoritative URDF, without editing it.

Phase one locks the upper body with slight shoulder abduction, retaining its
mass/inertia. The root frame is moved to the pelvis COM; the original CAD root
is at floor level. Leg gains start from the existing locomotion baseline and
are increased for the full body; these are not identified actuator dynamics.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[4]
URDF = REPO / "assets/pioneer_whole_body_humanoid/pioneer_whole_body_humanoid.urdf"
JOINTS = tuple(
    f"{name}_{side}"
    for side in ("L", "R")
    for name in ("Hip_F", "Hip_A", "Hip_R", "Knee", "Ankle_P", "Ankle_R")
)
# stiffness, damping, effort Nm, speed rad/s, action scale rad, rest rad
PARAMS = {
    "Hip_F": (100.0, 8.0, 222.0, 3.6652, 0.25, 0.08),
    "Hip_A": (80.0, 5.0, 120.0, 20.944, 0.06, 0.04),
    "Hip_R": (60.0, 4.0, 60.0, 20.42, 0.15, 0.0),
    "Knee": (100.0, 8.0, 222.0, 3.6652, 0.30, -0.22),
    "Ankle_P": (80.0, 6.0, 60.0, 20.42, 0.12, -0.14),
    "Ankle_R": (80.0, 6.0, 60.0, 20.42, 0.08, 0.04),
}
REST = {j: PARAMS[j.rsplit("_", 1)[0]][5] for j in JOINTS}
SCALES = {j: PARAMS[j.rsplit("_", 1)[0]][4] for j in JOINTS}


def audit_source():
    root = ET.parse(URDF).getroot()
    total = 0.0
    for link in root.findall("link"):
        i = link.find("inertial/inertia")
        mass = float(link.find("inertial/mass").get("value"))
        tensor = np.array(
            [
                [float(i.get(f"i{a}{b}" if a <= b else f"i{b}{a}")) for b in "xyz"]
                for a in "xyz"
            ]
        )
        eig = np.linalg.eigvalsh(tensor)
        if (
            not np.isfinite(eig).all()
            or mass <= 0
            or eig[0] <= 0
            or eig[2] > eig[:2].sum() + 1e-9
        ):
            raise ValueError(f"Invalid mass/inertia: {link.get('name')}")
        total += mass
    return total


def robot_spec(visuals=True):
    audit_source()
    root = ET.parse(URDF).getroot()
    pelvis_z = float(
        root.find("link[@name='base_link']/inertial/origin").get("xyz").split()[2]
    )
    # Shift only root-local quantities; all child-local frames stay unchanged.
    origins = list(root.findall("link[@name='base_link']/*/origin"))
    for joint in root.findall("joint"):
        if joint.find("parent").get("link") == "base_link":
            origins.append(joint.find("origin"))
        if joint.get("name") not in JOINTS:
            # Slight shoulder abduction clears the hands from the thighs.
            if joint.get("name") in ("l_shoulder_roll", "r_shoulder_roll"):
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                joint.find("origin").set("rpy", f"0 {axis[1] * 0.15} 0")
            joint.set("type", "fixed")
            for tag in ("limit", "mimic", "axis", "dynamics"):
                for child in joint.findall(tag):
                    joint.remove(child)
    for origin in origins:
        xyz = np.fromstring(origin.get("xyz"), sep=" ")
        xyz[2] -= pelvis_z
        origin.set("xyz", " ".join(map(str, xyz)))
    for link in root.findall("link"):
        for index, geom in enumerate(link.findall("collision")):
            geom.set("name", f"{link.get('name')}_collision_{index}")
        for visual in link.findall("visual"):
            if not visuals:
                link.remove(visual)
            else:
                mesh = visual.find("geometry/mesh")
                if mesh is not None:
                    mesh.set("filename", str(URDF.parent / mesh.get("filename")))
    compiler = root.find("mujoco/compiler")
    compiler.set("balanceinertia", "false")
    compiler.set("discardvisual", "false" if visuals else "true")
    ET.SubElement(root, "link", name="world")
    ET.SubElement(root, "link", name="base")
    canonical = ET.SubElement(root, "joint", name="canonical_frame", type="fixed")
    ET.SubElement(canonical, "parent", link="base")
    ET.SubElement(canonical, "child", link="base_link")
    # CAD +X is left, -Y is forward. Policy convention: +X forward, +Y left.
    ET.SubElement(canonical, "origin", xyz="0 0 0", rpy=f"0 0 {np.pi / 2}")
    free = ET.SubElement(root, "joint", name="root", type="floating")
    ET.SubElement(free, "parent", link="world")
    ET.SubElement(free, "child", link="base")
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.compiler.fusestatic = True
    # A valid compiled zero state prevents contact overflow before env reset.
    spec.body("base").pos = [0, 0, 0.9]
    spec.option.timestep = 0.002
    # The shank box surrounds the two-axis ankle assembly. These pairs are
    # mechanically adjacent through the ankle cross; their boxes overlap by design.
    for side in ("l", "r"):
        spec.add_exclude(bodyname1=f"{side}_shank_foot", bodyname2=f"{side}_foot")
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.contype = g.conaffinity = 0
            g.group = 2
        else:
            g.contype = g.conaffinity = 1
            g.group = 3
            g.friction = [0.8, 0.005, 0.0001]
            if g.name.startswith(("l_foot_collision", "r_foot_collision")):
                # Use the randomized sole friction for ground contacts, rather
                # than MuJoCo's equal-priority max(sole, floor) combination.
                g.priority = 1
    for name in JOINTS:
        joint = spec.joint(name)
        kp, kd, effort, _, _, _ = PARAMS[name.rsplit("_", 1)[0]]
        joint.armature = 0.01
        a = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=4 * kp, kv=2 * kd)
        a.ctrllimited, a.ctrlrange = True, list(joint.range)
        a.forcelimited, a.forcerange = True, [-effort, effort]
    return spec


def rest_data(model, ground_clearance=0.002):
    data = mujoco.MjData(model)
    for name, pos in REST.items():
        data.qpos[model.joint(name).qposadr[0]] = pos
        data.ctrl[model.actuator(name).id] = pos
    mujoco.mj_forward(model, data)
    feet = [
        g
        for g in range(model.ngeom)
        if model.geom(g).name.startswith(("l_foot_collision", "r_foot_collision"))
    ]
    low = min(
        data.geom_xpos[g, 2]
        - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g]
        for g in feet
    )
    data.qpos[2] += ground_clearance - low
    mujoco.mj_forward(model, data)
    return data


def spawn_height():
    model = robot_spec(visuals=False).compile()
    return float(rest_data(model).qpos[2])


def ground_spec(timestep=0.002, visuals=False):
    spec = robot_spec(visuals=visuals)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.timestep = timestep
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.1],
        friction=[0.8, 0.005, 0.0001],
    )
    return spec
