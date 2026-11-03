"""Build this URDF's meshes from the arm's CAD (ArmV2_NewForearms_NewWristGrippers.STEP), every link in its colours:
link5..link8 of each arm (the new forearm and wrist gripper) get new meshes and inertias, joint6 and joint7/joint8
(the fingers) follow them; base_link..link4 get their meshes from the same CAD (the same parts as before, within
mesh resolution) and keep their inertias. Joints 1-5 are untouched: their axes in the STEP match this URDF to 0.2 mm,
which `check_new_gripper.py` tests.

Each link has one collision mesh (meshes/<link>.STL, every part) and a visual mesh a colour
(meshes/<link>_<colour>.STL): prints white, AK motors and the camera graphite, GL40s dark grey, the fingers orange,
the stand's extrusions aluminium, the rail and screws steel.

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
and above it (STEP_LINKS): base_link the stand, camera and joint1's AK10; link1 joint2's AK10 and the shoulder
bracket; link2 / link3 / link4 joint3's / joint4's / joint5's AK80 and the bracket it sits in.
Masses of link5..link8 are volume x 1 g/cm^3, the density the rest of this URDF was exported with.
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
from OCP.BRepTools import BRepTools
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "urdf", "pioneer_bimanual_arm.urdf")
DENSITY = 1000.0                      # kg/m^3, as the exporter's
MAX_FACES = 60_000                    # a mesh
# STEP mm -> URDF base_link m. The offset puts the STEP's joint axes on this URDF's: joint5/joint6 at X = 14.8 mm,
# joint4 at Z = -240.6, joint6 at Z = -476.1 (each within 0.2 mm).
R = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
OFFSET_MM = np.array([21.8, 0.0, -1179.65])
FINGER_TRAVEL = 0.041                 # m each: the STEP is fully open (82.1 mm between the fingers' inner faces)
JOINT5_ZERO_TURN = {"right": -90.0, "left": 90.0}   # deg about joint5 (base +Z), STEP pose -> the arm's zero
JOINT6_AXIS_STEP = {"right": (0.0, -1.0, 0.0), "left": (0.0, 1.0, 0.0)}   # joint6 in the STEP pose (the old axis)
RAIL_STEP = (1.0, 0.0, 0.0)           # the fingers' travel in the STEP pose, base frame

# colours (rgba) and which parts take them, first match wins; everything else is a print
COLOURS = {
    "finger": ("0.910 0.455 0.173 1", r"finger"),
    "gl40": ("0.227 0.235 0.251 1", r"gl40"),
    "motor": ("0.169 0.176 0.192 1", r"ak80|ak10"),
    "steel": ("0.604 0.620 0.643 1", r"mgn9|component2[45]|countersunk"),
    "aluminium": ("0.725 0.741 0.761 1", r"frame1|ext\d|extlong|4040"),
    "camera": ("0.063 0.071 0.078 1", r"realsense"),
    "print": ("0.925 0.929 0.937 1", r""),
}
# the parts above joint5 (the same as the old meshes'): name -> link, the motors by where they sit (STEP mm)
STEP_LINKS = [
    (r"3dpbracket|4040|frame1|ext\d|extlong|attachment|camera_mount|realsense|squaremount|shoulderbearing|arm12bearing",
     lambda c: "base"),
    (r"ak10", lambda c: "base" if abs(c[0]) < 200 else "1"),          # joint1's AK10 on the stand, joint2's on link1
    (r"arm12", lambda c: "1"),
    (r"arm23", lambda c: "2"),
    (r"arm34", lambda c: "3"),
    (r"arm45", lambda c: "4"),
    (r"ak80", lambda c: "2" if c[1] > 1050 else "3" if c[1] > 900 else "4"),   # joint3 / joint4 / joint5
]
KEEP_INERTIA = ("base", "1", "2", "3", "4")

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


def assign(label, centre):
    """(side, link) of a STEP part: side None for base_link; (None, None) for a part of no link"""
    side, link = side_of(label), link_of(label)
    if side and link:
        return side, link
    for pat, rule in STEP_LINKS:
        if re.search(pat, label):
            link = rule(centre)
            return (None if link == "base" else "right" if centre[0] > 0 else "left"), link
    return None, None


def colour_of(label):
    return next(k for k, (_, pat) in COLOURS.items() if re.search(pat, label.lower()))


def turn(side):
    if side is None:
        return np.eye(3)
    a = np.radians(JOINT5_ZERO_TURN[side])
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


@functools.lru_cache(maxsize=None)
def joint5_xy(side):
    """where the joint5 axis is (m, base X / Y), from this URDF"""
    return link_frames()["link5" if side == "right" else "link5l"][:2]


def to_urdf(p_mm, side, link="5"):
    """STEP points (mm) -> base_link (m); below joint5 (link5..8) turned about joint5 to the arm's zero"""
    p = (np.asarray(p_mm) @ R.T + OFFSET_MM) / 1000.0
    if side is None or link not in "5678":
        return p
    c = np.array([*joint5_xy(side), 0.0])
    return (p - c) @ turn(side).T + c


def mesh(shape):
    bb = shape.bounding_box()
    BRepMesh_IncrementalMesh(shape.wrapped, max(0.08, bb.diagonal / 2000.0), False, 0.3, True)   # the stand's 1.5 m bars coarser
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
    BRepTools.Clean_s(shape.wrapped)                             # the triangulation, copied: free it
    return np.array(v, float), np.array(f, int).reshape(-1, 3)


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
    if k == "base":
        return "base_link"
    return f"link{k}" if side == "right" else "link1L" if k == "1" else f"link{k}l"


def fmt(x):
    return repr(float(x))


def inertial_block(mass, com, inertia):
    return f"""<inertial>
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
    </inertial>"""


def link_block(name, inertial, colours):
    visuals = "".join(f"""
    <visual>
      <origin
        xyz="0 0 0"
        rpy="0 0 0" />
      <geometry>
        <mesh
          filename="package://armv2URDF/meshes/{name}_{c}.STL" />
      </geometry>
      <material
        name="{c}">
        <color
          rgba="{COLOURS[c][0]}" />
      </material>
    </visual>""" for c in colours)
    return f"""<link
    name="{name}">
    {inertial}{visuals}
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


def decimated(pieces, max_faces=MAX_FACES):
    """one mesh of [(vertices, faces)], at most max_faces"""
    V, F, n = [], [], 0
    for v, f in pieces:
        V.append(v)
        F.append(f + n)
        n += len(v)
    tm = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=True)
    if len(tm.faces) > max_faces:
        v2, f2 = fast_simplification.simplify(tm.vertices, tm.faces, 1.0 - max_faces / len(tm.faces))
        tm = trimesh.Trimesh(v2, f2)
    return tm


def main(step):
    root = bd.import_step(step)
    frames = link_frames()
    parts = {}                                                   # (side, link) -> [shape]
    for c in root.children:
        side, link = assign(c.label, tuple(c.bounding_box().center()))
        if link:
            parts.setdefault((side, link), []).append(c)
    text = open(URDF).read()
    frames["base_link"] = np.zeros(3)
    fingers, wrist = {}, {}
    for (side, k), shapes in sorted(parts.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        name = link_name(side, k)
        # joint7/joint8 sit at their finger's middle on the rail, in link6's frame
        if k in ("7", "8"):
            boxes = [s.bounding_box() for s in shapes]
            ends = np.vstack([to_urdf([(x, y, z) for x in (b.min.X, b.max.X) for y in (b.min.Y, b.max.Y)
                                       for z in (b.min.Z, b.max.Z)], side, k) for b in boxes])
            lo, hi = np.min(ends, 0), np.max(ends, 0)
            fingers[name] = (lo + hi) / 2
            origin = fingers[name]
        elif k == "6":
            origin = wrist[side] = np.array([*joint5_xy(side), frames[name][2]])   # on the joint5 axis
        else:
            origin = frames[name]
        groups, m_tot, mc, props = {}, 0.0, np.zeros(3), []
        for s in shapes:
            if k not in KEEP_INERTIA:
                m, c, inertia = mass_props(s)
                Rt = turn(side) @ R
                props.append((m, to_urdf(c, side, k) - origin, Rt @ inertia @ Rt.T))
                m_tot += m
                mc += m * (to_urdf(c, side, k) - origin)
            v, f = mesh(s)
            groups.setdefault(colour_of(s.label), []).append((to_urdf(v, side, k) - origin, f))
        old = re.search(r"<link\s+name=\"%s\">[\s\S]*?</link>" % name, text).group(0)
        if k in KEEP_INERTIA:
            inertial = re.search(r"<inertial>[\s\S]*?</inertial>", old).group(0)
        else:
            com = mc / m_tot
            I_tot = np.zeros((3, 3))
            for m, c, inertia in props:                          # parallel axes to the link's centre of mass
                d = c - com
                I_tot += inertia + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
            inertial = inertial_block(m_tot, com, I_tot)
        for f in os.listdir(os.path.join(HERE, "meshes")):           # the link's old visual meshes
            if re.fullmatch(r"%s_[a-z]+\.STL" % re.escape(name), f):
                os.remove(os.path.join(HERE, "meshes", f))
        colours = [c for c in COLOURS if c in groups]
        for c in colours:
            decimated(groups[c]).export(os.path.join(HERE, "meshes", f"{name}_{c}.STL"))
        tm = decimated([p for c in colours for p in groups[c]])
        tm.export(os.path.join(HERE, "meshes", f"{name}.STL"))
        text = text.replace(old, link_block(name, inertial, colours))
        print(f"{name}: {len(shapes)} parts, colours {colours}, collision {len(tm.faces)} faces")
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
