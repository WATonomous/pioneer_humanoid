"""Put the arm's new forearm and wrist gripper (ArmV2_NewForearms_NewWristGrippers.STEP) into this URDF: link5..link8
of each arm get new meshes and inertias, joint7/joint8 (the fingers) follow the new rail. Joints 1-5 are untouched:
their axes in the STEP match this URDF to 0.2 mm, which `check_new_gripper.py` tests.

The arm's zero has the forearm and gripper turned about joint5 from the STEP (JOINT5_ZERO_TURN: -90 deg right,
+90 left), so the fingers close side to side at joint5 = 0. joint6 turns with them: its axis is the old one turned
90 deg about joint5 (base -X on both arms), through the joint5 axis.

    python step_to_links.py ArmV2_NewForearms_NewWristGrippers.STEP

Needs build123d (the OCC kernel) and fast_simplification. The STEP is Y up, front -Z, right arm at +X (mm); this
URDF is Z up, front +X, right arm at -Y (m). The links below joint5 are:
    link5  forearm + cover + the wrist GL40's stator
    link6  wrist yoke, gripper base, the gripper GL40, its swivel and linkages, the MGN9 rail, the wrist GL40's rotor
    link7  the outer finger (in front of the rail in the STEP)
    link8  the inner finger
Masses are volume x 1 g/cm^3, the density the rest of this URDF was exported with.
"""
import functools
import os
import re
import sys

import build123d as bd
import fast_simplification
import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "urdf", "pioneer_bimanual_arm.urdf")
DENSITY = 1000.0                      # kg/m^3, as the exporter's
MAX_FACES = 60_000
# STEP mm -> URDF base_link m. The offset puts the STEP's joint axes on this URDF's: joint5/joint6 at X = 14.8 mm,
# joint4 at Z = -240.6, joint6 at Z = -476.1 (each within 0.2 mm).
R = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
OFFSET_MM = np.array([21.8, 0.0, -1179.65])
FINGER_TRAVEL = 0.041                 # m each: the STEP is fully open (82.1 mm between the fingers' inner faces)
JOINT5_ZERO_TURN = {"right": -90.0, "left": 90.0}   # deg about joint5 (base +Z), STEP pose -> the arm's zero
JOINT6_AXIS_STEP = {"right": (0.0, -1.0, 0.0), "left": (0.0, 1.0, 0.0)}   # joint6 in the STEP pose (the old axis)
RAIL_STEP = (1.0, 0.0, 0.0)           # the fingers' travel in the STEP pose, base frame

# STEP part names (after "<side>_new_" or the wrist GL40's "original_..." name) per link
LINKS = {
    "5": [r"forearm", r"cover", r"gl40iistat"],
    "6": [r"wrist_(01|02|03|07|08|09|10|11|12|13|14)_", r"gl40iirot"],
    "7": [r"wrist_(15|16|17)_"],
    "8": [r"wrist_(04|05|06)_"],
}


def side_of(label):
    m = re.match(r"(left|right)_new_", label)
    if m:
        return m.group(1)
    if "gl40ii" in label and "hum_" not in label:            # the wrist GL40: original_*, the left one mirrored
        return "left" if "mirror" in label else "right"
    return None


def link_of(label):
    for link, pats in LINKS.items():
        if any(re.search(p, label) for p in pats):
            return link
    return None


def turn(side):
    a = np.radians(JOINT5_ZERO_TURN[side])
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


@functools.lru_cache(maxsize=None)
def joint5_xy(side):
    """where the joint5 axis is (m, base X / Y), from this URDF"""
    return link_frames()["link5" if side == "right" else "link5l"][:2]


def to_urdf(p_mm, side):
    """STEP points (mm) -> base_link (m), turned about joint5 to the arm's zero"""
    p = (np.asarray(p_mm) @ R.T + OFFSET_MM) / 1000.0
    c = np.array([*joint5_xy(side), 0.0])
    return (p - c) @ turn(side).T + c


def mesh(shape):
    BRepMesh_IncrementalMesh(shape.wrapped, 0.08, False, 0.3, True)
    v, f, n = [], [], 0
    for face in shape.faces():
        loc = TopLoc_Location()
        t = BRep_Tool.Triangulation_s(face.wrapped, loc)
        if t is None:
            continue
        tr = loc.Transformation()
        v += [(p.X(), p.Y(), p.Z()) for p in (t.Node(k).Transformed(tr) for k in range(1, t.NbNodes() + 1))]
        rev = face.wrapped.Orientation() == TopAbs_REVERSED
        for k in range(1, t.NbTriangles() + 1):
            a, b, c = t.Triangle(k).Get()
            f.append((a - 1 + n, c - 1 + n, b - 1 + n) if rev else (a - 1 + n, b - 1 + n, c - 1 + n))
        n = len(v)
    return np.array(v, float), np.array(f, int)


def mass_props(shape):
    """(mass kg, centre m, inertia about the centre kg m^2), in the STEP's axes"""
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, g)
    vol = g.Mass() * 1e-9                                        # mm^3 -> m^3
    c = g.CentreOfMass()
    mi = g.MatrixOfInertia()                                     # about the centre, per unit density, mm^5
    inertia = np.array([[mi.Value(r, k) for k in (1, 2, 3)] for r in (1, 2, 3)]) * 1e-15
    return vol * DENSITY, np.array([c.X(), c.Y(), c.Z()]), inertia * DENSITY


def link_frames():
    """each link's origin in base_link (m); every frame of this URDF is axis-aligned with base_link"""
    text = open(URDF).read()
    parent, xyz = {}, {}
    for j in re.finditer(r"<joint\s+name=\"[^\"]+\"[\s\S]*?</joint>", text):
        b = j.group(0)
        o = re.search(r"<origin\s+xyz=\"([^\"]+)\"\s+rpy=\"([^\"]+)\"", b)
        assert all(abs(float(r)) < 1e-9 for r in o.group(2).split()), "a rotated joint frame: not handled"
        c = re.search(r"<child\s+link=\"([^\"]+)\"", b).group(1)
        parent[c] = re.search(r"<parent\s+link=\"([^\"]+)\"", b).group(1)
        xyz[c] = np.array([float(v) for v in o.group(1).split()])

    def pos(link):
        return np.zeros(3) if link not in parent else pos(parent[link]) + xyz[link]
    return {k: pos(k) for k in parent}


def link_name(side, k):
    return {"right": f"link{k}", "left": {"5": "link5l", "6": "link6l", "7": "link7l", "8": "link8l"}[k]}[side]


def fmt(x):
    return repr(float(x))


def link_block(name, mass, com, inertia):
    return f"""<link
    name="{name}">
    <inertial>
      <origin
        xyz="{fmt(com[0])} {fmt(com[1])} {fmt(com[2])}"
        rpy="0 0 0" />
      <mass
        value="{fmt(mass)}" />
      <inertia
        ixx="{fmt(inertia[0, 0])}"
        ixy="{fmt(inertia[0, 1])}"
        ixz="{fmt(inertia[0, 2])}"
        iyy="{fmt(inertia[1, 1])}"
        iyz="{fmt(inertia[1, 2])}"
        izz="{fmt(inertia[2, 2])}" />
    </inertial>
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://armv2URDF/meshes/{name}.STL" />
      </geometry>
      <material
        name="">
        <color
          rgba="0.752941176470588 0.752941176470588 0.752941176470588 1" />
      </material>
    </visual>
    <collision>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://armv2URDF/meshes/{name}.STL" />
      </geometry>
    </collision>
  </link>"""


def main(step):
    root = bd.import_step(step)
    frames = link_frames()
    parts = {}                                                   # (side, link) -> [shape]
    for c in root.children:
        side, link = side_of(c.label), link_of(c.label)
        if side and link:
            parts.setdefault((side, link), []).append(c)
    text = open(URDF).read()
    fingers, wrist = {}, {}
    for (side, k), shapes in sorted(parts.items()):
        name = link_name(side, k)
        # joint7/joint8 sit at their finger's middle on the rail, in link6's frame
        if k in ("7", "8"):
            boxes = [s.bounding_box() for s in shapes]
            ends = np.vstack([to_urdf([(x, y, z) for x in (b.min.X, b.max.X) for y in (b.min.Y, b.max.Y)
                                       for z in (b.min.Z, b.max.Z)], side) for b in boxes])
            lo, hi = np.min(ends, 0), np.max(ends, 0)
            fingers[name] = (lo + hi) / 2
            origin = fingers[name]
        elif k == "6":
            origin = wrist[side] = np.array([*joint5_xy(side), frames[name][2]])   # on the joint5 axis
        else:
            origin = frames[name]
        V, F, m_tot, mc, props = [], [], 0.0, np.zeros(3), []
        for s in shapes:
            v, f = mesh(s)
            F.append(f + sum(len(x) for x in V))
            V.append(to_urdf(v, side) - origin)
            m, c, inertia = mass_props(s)
            Rt = turn(side) @ R
            props.append((m, to_urdf(c, side) - origin, Rt @ inertia @ Rt.T))
            m_tot += m
            mc += m * (to_urdf(c, side) - origin)
        com = mc / m_tot
        I_tot = np.zeros((3, 3))
        for m, c, inertia in props:                              # parallel axes to the link's centre of mass
            d = c - com
            I_tot += inertia + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        tm = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=True)
        if len(tm.faces) > MAX_FACES:
            v2, f2 = fast_simplification.simplify(tm.vertices, tm.faces, 1.0 - MAX_FACES / len(tm.faces))
            tm = trimesh.Trimesh(v2, f2)
        tm.export(os.path.join(HERE, "meshes", f"{name}.STL"))
        text = re.sub(r"<link\s+name=\"%s\">[\s\S]*?</link>" % name, lambda _: link_block(name, m_tot, com, I_tot), text)
        print(f"{name}: {m_tot:.3f} kg, {len(tm.faces)} faces, com {np.round(com * 1000, 1)} mm")
    # joint6 through the joint5 axis, turned with the wrist; the fingers along the turned rail (base Y, as the old
    # gripper's): joint7 (outer) closes negative, joint8 (inner) positive, as before
    for side in ("right", "left"):
        l5, l7, l8 = (link_name(side, k) for k in "578")
        j6, j7, j8 = ("joint6", "joint7", "joint8") if side == "right" else ("joint6l", "joint7l", "joint8l")
        rail = turn(side) @ RAIL_STEP
        todo = [(j6, wrist[side] - frames[l5], turn(side) @ JOINT6_AXIS_STEP[side], None)]
        todo += [(j7, fingers[l7] - wrist[side], rail, (-FINGER_TRAVEL, 0.0)),
                 (j8, fingers[l8] - wrist[side], rail, (0.0, FINGER_TRAVEL))]
        for j, o, ax, lim in todo:
            ax = np.round(ax, 9) + 0.0

            def joint(mt, o=o, ax=ax, lim=lim):
                b = mt.group(0)
                b = re.sub(r"(<origin\s+xyz=\")[^\"]+(\")", lambda q: f"{q.group(1)}{fmt(o[0])} {fmt(o[1])} {fmt(o[2])}{q.group(2)}", b, count=1)
                b = re.sub(r"(<axis\s+xyz=\")[^\"]+(\")", lambda q: f"{q.group(1)}{ax[0]:g} {ax[1]:g} {ax[2]:g}{q.group(2)}", b)
                if lim is None:
                    return b
                b = re.sub(r"lower=\"[^\"]+\"", f"lower=\"{lim[0]}\"", b)
                return re.sub(r"upper=\"[^\"]+\"", f"upper=\"{lim[1]}\"", b)
            text, n = re.subn(r"<joint\s+name=\"%s\"[\s\S]*?</joint>" % j, joint, text)
            assert n == 1, j
    open(URDF, "w").write(text)


if __name__ == "__main__":
    main(sys.argv[1])
