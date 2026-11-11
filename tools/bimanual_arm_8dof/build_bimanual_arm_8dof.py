"""Build assets/pioneer_bimanual_arm_8dof (URDF + meshes) from the owner's 1107 Arm Draft STEP.

    python tools/bimanual_arm_8dof/build_bimanual_arm_8dof.py ARM_DRAFT.step [assets/pioneer_bimanual_arm_8dof]

The STEP (wilsonchenghy/Prometheus, Backend/cad_bench/projects/arm_shell/source/arm_draft_1107.step.xz,
unpack with `xz -dk`) has geometry only. Joints, links and limits come from:
  - joint axes / pivots: the arm-shell session's arm_shell.py JOINTS (found from bearing and motor geometry);
  - link of every solid: its arm_draft.py LINK assignment (right arm keeps the STEP tree, left solids take their
    mirror twin's link, the middle wrist frame is its own link);
  - limits: the owner's joint chart (photo, Nov 2026), see LIMITS_DEG below.

Robot: the bimanual rig with its arms replaced: pioneer_bimanual_arm's base_link AS IS (aluminium stand, steel
shoulder bar, printed brackets, the two shoulder-pitch motors, head camera; its meshes, inertia and collision are
referenced from assets/pioneer_bimanual_arm) carrying two new 7-DoF arms + a 1-DoF gripper each (GL40 -> linkage
-> two fingers on an MGN9 rail; finger B mimics finger A), on the same shoulder-pitch pivots (old joint1/joint1L).
The STEP's own shoulder bar is not used. base_link is the old pioneer_bimanual_arm's base_link frame (x forward, y left, z up, metres), mapped
from the draft frame (mm, x right->left arm at x<0, y forward, z up) by
    urdf = ((y - 575)/1000 + 0.02157, -x/1000, (z - 679)/1000 + 0.06048)
(the arm-shell session's mapping: the pitch motors coincide with the old arm's joint1).

Zero pose = the CAD pose (arms hanging, forearm forward) EXCEPT the elbow, whose zero is straight (as the chart:
-5 = just past straight, +130 = flexion), so the CAD pose is elbow = 90 deg.
The right arm is the left mirrored in the mid-plane; its joint axes are chosen so the SAME angle gives the
mirror pose: both arms share the chart limits and signs (+ flexion raises the arm forward, + abduction out).

Masses are ESTIMATES (the STEP has no mass properties): motor masses plus structure, spread over each link's
convex hull. Replace with measured values.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.collections import Sequence_TDF_Label
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.Quantity import Quantity_Color
from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_ColorTool, XCAFDoc_DocumentTool

# ------------------------------------------------------------------ the robot definition (left arm, draft frame, mm)
# joint: (point on axis, axis, first link it turns, child link name)
JOINTS = [
    ("shoulder_pitch", (-163.325, 575.0, 679.0), (1, 0, 0), "link1"),  # AK10; = old joint1L origin
    ("shoulder_roll", (-279.0, 575.0, 679.0), (0, 1, 0), "link2"),   # AK10
    ("shoulder_yaw", (-279.0, 568.0, 563.0), (0, 0, 1), "link3"),    # AK80, upper-arm rotation
    ("elbow", (-279.0, 568.0, 378.0), (1, 0, 0), "link4"),           # AK80
    ("forearm_roll", (-279.0, 675.0, 379.0), (0, 1, 0), "link5"),    # AK80
    ("wrist_pitch", (-279.0, 729.0, 378.0), (1, 0, 0), "wrist1"),    # AK45-10
    ("wrist_yaw", (-279.0, 789.0, 378.0), (0, 0, 1), "gripper"),     # AK45-10
]
# Owner's joint chart (photo): degrees, same for both arms.
LIMITS_DEG = {"shoulder_pitch": (-90, 180), "shoulder_roll": (-20, 160), "shoulder_yaw": (-120, 120),
              "elbow": (-5, 130), "forearm_roll": (-110, 110), "wrist_pitch": (-60, 60), "wrist_yaw": (-90, 90)}
ELBOW_CAD_DEG = 90.0          # the CAD pose bends the elbow 90 deg; URDF elbow zero = straight
# effort (Nm, peak) / velocity (rad/s) for the URDF <limit>; the motors' datasheet peaks, not tested limits
MOTOR = {"shoulder_pitch": (53.0, 6.0), "shoulder_roll": (53.0, 6.0), "shoulder_yaw": (22.0, 6.0),
         "elbow": (22.0, 6.0), "forearm_roll": (22.0, 6.0), "wrist_pitch": (7.0, 6.0), "wrist_yaw": (7.0, 6.0)}
LINK_NAMES = {"link1": "link1", "link2": "link2", "link3": "link3", "link4": "link4", "link5": "link5",
              "wrist1": "link6", "gripper": "link7"}
# ESTIMATED masses (kg): carried motor + structure
MASS = {"base_link": 3.0, "link1": 1.15, "link2": 0.75, "link3": 0.85, "link4": 0.75, "link5": 0.45,
        "link6": 0.40, "link7": 0.40, "finger_a": 0.04, "finger_b": 0.04}
GRIPPER_STROKE = 0.041         # m per finger; CAD pose = open (inner faces 83 mm apart)
JAW_CENTRE_X = -279.0          # draft x of the jaw centre (left arm)
FINGER_PARTS = ("finger_carriage", "component188", "component189", "0_1_1_272_", "0_1_1_275_")
OLD_URDF = Path(__file__).resolve().parents[2] / "assets/pioneer_bimanual_arm/urdf/pioneer_bimanual_arm.urdf"
VISUAL_FACES = {}
NO_DECIMATION = ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "finger_a", "finger_b")   # all: decimation left spikes
DEFAULT_VISUAL_FACES = 14000
SMALL_PART_MM = 12.0   # visual meshes leave out parts smaller than this (fasteners); collision/inertia keep them
DEFAULT_RGB = (0.9, 0.9, 0.92)   # parts without a STEP colour: light print
# The fingers are orange, as on the rig's current arms (pioneer_bimanual_arm link7l/8l finger material); the STEP
# export colours them blue. Dark (motor) and grey (metal) parts keep their STEP colour.
FINGER_ORANGE = (0.910, 0.455, 0.173)


def finger_colour(link, rgb):
    r, g, b = rgb
    is_blue = b > r + 0.15 and b > g + 0.1
    is_metal_or_motor = max(rgb) - min(rgb) < 0.08 and max(rgb) < 0.7
    if is_blue or (link in ("finger_a", "finger_b") and not is_metal_or_motor):
        return FINGER_ORANGE
    return rgb

# ------------------------------------------------------------------ frames
def to_urdf(p_mm):
    p = np.asarray(p_mm, float)
    return np.stack([(p[..., 1] - 575.0) / 1000 + 0.02157, -p[..., 0] / 1000, (p[..., 2] - 679.0) / 1000 + 0.06048], -1)


def axis_to_urdf(a):
    a = np.asarray(a, float)
    return np.array([a[1], -a[0], a[2]])


def rot(axis, angle):
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


# ------------------------------------------------------------------ STEP -> solids (as arm_draft.py / step_to_links)
def read_solids(src):
    doc = TDocStd_Document(TCollection_ExtendedString("s"))
    rd = STEPCAFControl_Reader()
    rd.SetNameMode(True)
    rd.ReadFile(str(src))
    rd.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

    def color(l):
        c = Quantity_Color()
        for t in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
            for f in (lambda: XCAFDoc_ColorTool.GetColor_s(l, t, c), lambda: ct.GetColor(l, t, c)):
                try:
                    if f():
                        return (c.Red(), c.Green(), c.Blue())
                except TypeError:
                    pass
        return None

    def name(l):
        a = TDataStd_Name()
        return a.Get().ToExtString() if l.FindAttribute(TDataStd_Name.GetID_s(), a) else ""

    def tess(solid):
        BRepMesh_IncrementalMesh(solid, 0.15, False, 0.2, True)   # ~11 deg facets; MuJoCo smooths normals
        vs, fs = [], []
        ex = TopExp_Explorer(solid, TopAbs_FACE)
        while ex.More():
            face = TopoDS.Face(ex.Current())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is not None:
                tr = loc.Transformation()
                b = len(vs)
                for i in range(1, tri.NbNodes() + 1):
                    p = tri.Node(i).Transformed(tr)
                    vs.append((p.X(), p.Y(), p.Z()))
                rev = face.Orientation() == TopAbs_REVERSED
                for i in range(1, tri.NbTriangles() + 1):
                    a, bb, c = tri.Triangle(i).Get()
                    fs.append((b + a - 1, b + c - 1, b + bb - 1) if rev else (b + a - 1, b + bb - 1, b + c - 1))
            ex.Next()
        return np.array(vs), np.array(fs, dtype=int)

    solids, used = {}, {}

    def walk(lab, loc, nm, link="", depth=0, col=None):
        ref = lab
        col = color(lab) or col
        if st.IsReference_s(lab):
            ref = TDF_Label()
            st.GetReferredShape_s(lab, ref)
        col = color(ref) or col
        if st.IsAssembly_s(ref):
            kids = Sequence_TDF_Label()
            st.GetComponents_s(ref, kids, False)
            for i in range(1, kids.Length() + 1):
                k = kids.Value(i)
                walk(k, loc.Multiplied(st.GetLocation_s(k)), name(k) or nm, link or (name(k) if depth == 0 else ""),
                     depth + 1, col)
            return
        ex = TopExp_Explorer(st.GetShape_s(ref).Moved(loc), TopAbs_SOLID)
        while ex.More():
            b = re.sub(r"[^a-z0-9]+", "_", nm.lower()).strip("_") or "part"
            used[b] = used.get(b, 0) + 1
            v, f = tess(ex.Current())
            solids[f"{b}_{used[b]}"] = {"link": link, "v": v, "f": f, "color": col}
            ex.Next()

    roots = Sequence_TDF_Label()
    st.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), TopLoc_Location(), name(roots.Value(i)))
    return solids


def assign_links(solids):
    LEFT_OWN = {"0_1_1_277_17": "link1", "0_1_1_277_29": "link1", "0_1_1_277_32": "link1", "0_1_1_277_18": "link2",
                "0_1_1_277_19": "link3", "0_1_1_277_20": "link3", "0_1_1_277_21": "link3", "0_1_1_277_26": "link3",
                "0_1_1_277_22": "link4", "0_1_1_277_28": "link5", "0_1_1_277_30": "link5", "0_1_1_277_31": "link5",
                "0_1_1_277_23": "link5", "0_1_1_277_27": "link5", "0_1_1_277_24": "link5", "0_1_1_277_25": "link5",
                "0710_wrist_dual_support_6001_shouldered_none_mirror_1_2": "gripper"}
    WRIST1 = ("0_1_1_277_23", "0_1_1_277_24", "0_1_1_277_25", "0_1_1_277_27")
    spare = {n for n, s in solids.items() if len(s["v"]) and s["v"][:, 2].max() < 250}
    c = {n: (s["v"].min(0) + s["v"].max(0)) / 2 if len(s["v"]) else np.zeros(3) for n, s in solids.items()}

    def short(lk):
        m = re.match(r"(baseLink|link\d|Gripper V3)", lk)
        return m.group(1) if m else ""
    right = [n for n in solids if short(solids[n]["link"]) and c[n][0] > 0 and n not in spare]
    rc = np.array([c[n] * (-1, 1, 1) for n in right])
    link = {}
    for n in solids:
        if n in spare:
            continue
        if short(solids[n]["link"]):
            link[n] = short(solids[n]["link"])
        elif n in LEFT_OWN:
            link[n] = LEFT_OWN[n]
        elif c[n][0] < 0:
            link[n] = short(solids[right[int(np.argmin(np.linalg.norm(rc - c[n], axis=1)))]]["link"])
    link = {n: ("gripper" if v == "Gripper V3" else v) for n, v in link.items()}
    for n in WRIST1:
        link[n] = "wrist1"
    return link, c


# ------------------------------------------------------------------ build
def main(step, out):
    out = Path(out)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    (out / "urdf").mkdir(parents=True, exist_ok=True)
    print("reading STEP ...", flush=True)
    solids = read_solids(step)
    link, centre = assign_links(solids)

    # left-arm geometry (draft frame, mm) per link; fingers split from the gripper; duplicates dropped
    groups, seen = {}, set()
    for n, lk in link.items():
        s = solids[n]
        if not len(s["f"]):
            continue
        if lk == "baseLink":
            continue                                   # the rig's base_link comes from pioneer_bimanual_arm
        if centre[n][0] >= 0:
            continue                                   # right arm: rebuilt as the left's mirror
        key = (lk, tuple(np.round(s["v"].min(0), 1)), tuple(np.round(s["v"].max(0), 1)))
        if key in seen:
            continue                                   # the STEP has some solids twice
        seen.add(key)
        name = LINK_NAMES[lk]
        if lk == "gripper" and any(t in n for t in FINGER_PARTS):
            name = "finger_a" if centre[n][0] < JAW_CENTRE_X else "finger_b"
        rgb = finger_colour(name, s["color"] or DEFAULT_RGB)
        rgb = tuple(round(x, 3) for x in rgb)
        groups.setdefault(name, []).append((s["v"], s["f"], rgb))
    mesh = {k: trimesh.util.concatenate([trimesh.Trimesh(v, f, process=False) for v, f, _ in parts])
            for k, parts in groups.items()}
    # per link, one visual mesh per STEP colour (motors dark, prints white, metal grey ...), as the rig's base_link
    colour_mesh = {k: {} for k in groups}
    for k, parts in groups.items():
        for v, f, rgb in parts:
            if np.linalg.norm(v.max(0) - v.min(0)) < SMALL_PART_MM:
                continue          # screws, nuts, washers: invisible at viewing distance, undecimatable (visual only)
            colour_mesh[k].setdefault(rgb, []).append(trimesh.Trimesh(v, f, process=False))
    colour_mesh = {k: {rgb: trimesh.util.concatenate(ms) for rgb, ms in cm.items()} for k, cm in colour_mesh.items()}

    # joint frames in URDF coords; the elbow's downstream geometry rotated to the straight-elbow zero pose
    pts = {j: to_urdf(p) for j, p, a, l in JOINTS}
    axs = {j: axis_to_urdf(a) for j, p, a, l in JOINTS}
    downstream = ["link4", "link5", "link6", "link7", "finger_a", "finger_b"]
    Rz = rot(axs["elbow"], np.radians(-ELBOW_CAD_DEG))
    pe = pts["elbow"]

    def unbend(p):
        return (p - pe) @ Rz.T + pe

    def place(k, m):
        v = to_urdf(m.vertices)
        if k in downstream:
            v = unbend(v)
        return trimesh.Trimesh(v, m.faces, process=False)
    mesh = {k: place(k, m) for k, m in mesh.items()}
    colour_mesh = {k: {rgb: place(k, m) for rgb, m in cm.items()} for k, cm in colour_mesh.items()}
    for j in ("forearm_roll", "wrist_pitch", "wrist_yaw"):
        pts[j] = unbend(pts[j])
        axs[j] = Rz @ axs[j]
    # finger slide: along the rail (draft x -> URDF -y), closing toward the jaw centre
    rail = axis_to_urdf((1, 0, 0))
    rail = Rz @ rail
    finger_axes = {"finger_a": rail, "finger_b": -rail}
    origin = {}
    chain = [("link1", "shoulder_pitch"), ("link2", "shoulder_roll"), ("link3", "shoulder_yaw"), ("link4", "elbow"),
             ("link5", "forearm_roll"), ("link6", "wrist_pitch"), ("link7", "wrist_yaw")]
    for lk, j in chain:
        origin[lk] = pts[j]
    origin["finger_a"] = origin["finger_b"] = origin["link7"]

    def mirror(p):
        return np.asarray(p) * np.array([1, -1, 1])

    # meshes: left + mirrored right, link-local, decimated visual + convex-hull collision
    inertial, visuals = {}, {}
    for side in ("left", "right"):
        for k, m in mesh.items():
            v = m.vertices.copy()
            f = m.faces.copy()
            if side == "right":
                v = mirror(v)
                f = f[:, ::-1]
            v = v - (mirror(origin[k]) if side == "right" else origin[k])
            full = trimesh.Trimesh(v, f, process=True)
            n_faces = VISUAL_FACES.get(k, DEFAULT_VISUAL_FACES)
            fname = f"{side}_{k}"
            visuals[fname] = []
            total = sum(len(cm.faces) for cm in colour_mesh[k].values())
            for ci, (rgb, cm) in enumerate(sorted(colour_mesh[k].items(), key=lambda kv: -len(kv[1].faces))):
                cv, cf = cm.vertices.copy(), cm.faces.copy()
                if side == "right":
                    cv, cf = mirror(cv), cf[:, ::-1]
                cv = cv - (mirror(origin[k]) if side == "right" else origin[k])
                part = trimesh.Trimesh(cv, cf, process=True)
                budget = max(1500, int(n_faces * len(cm.faces) / total))
                # wrist and gripper links are seen up close and are small: never decimated (decimation left spikes)
                keep = k in NO_DECIMATION
                vis = part.simplify_quadric_decimation(face_count=budget) if len(part.faces) > budget and not keep else part
                vis.export(out / "meshes" / f"{fname}_c{ci}.stl")
                visuals[fname].append((f"{fname}_c{ci}", rgb))
            hull = full.convex_hull
            hull.export(out / "meshes" / f"{fname}_collision.stl")
            mass = MASS[k]
            hull_mass = hull.mass_properties
            scale = mass / max(hull.volume, 1e-9)
            inertial[fname] = (mass, hull.center_mass, hull.moment_inertia * scale)
            print(f"  {fname:16s} {len(visuals[fname])} colour meshes, mass {mass:.2f} kg (estimate)", flush=True)

    # URDF
    def old_base_link():
        """pioneer_bimanual_arm's base_link element verbatim (stand, bar, pitch motors, camera), meshes re-pathed."""
        import xml.etree.ElementTree as ET
        base = next(l for l in ET.parse(OLD_URDF).getroot().findall("link") if l.get("name") == "base_link")
        text = ET.tostring(base, encoding="unicode")
        return "  " + re.sub(r'package://[^/"]+/meshes/', "package://pioneer_bimanual_arm/meshes/", text).strip()

    def inertial_xml(fname):
        m, c, I = inertial[fname]
        return (f'<inertial><origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/><mass value="{m:.4f}"/>'
                f'<inertia ixx="{I[0,0]:.3e}" ixy="{I[0,1]:.3e}" ixz="{I[0,2]:.3e}" iyy="{I[1,1]:.3e}" '
                f'iyz="{I[1,2]:.3e}" izz="{I[2,2]:.3e}"/></inertial>')

    def link_xml(fname):
        vis = "".join(
            f'    <visual><geometry><mesh filename="package://pioneer_bimanual_arm_8dof/meshes/{mf}.stl"/></geometry>'
            f'<material name="{fname}_m{i}"><color rgba="{r:.3f} {g:.3f} {b:.3f} 1"/></material></visual>\n'
            for i, (mf, (r, g, b)) in enumerate(visuals[fname]))
        return (f'  <link name="{fname}">\n    {inertial_xml(fname)}\n{vis}'
                f'    <collision><geometry><mesh filename="package://pioneer_bimanual_arm_8dof/meshes/{fname}_collision.stl"/>'
                f'</geometry></collision>\n  </link>')

    xml = ['<?xml version="1.0"?>',
           '<!-- GENERATED by tools/bimanual_arm_8dof/build_bimanual_arm_8dof.py from the 1107 Arm Draft STEP; do not edit by hand. -->',
           '<!-- Two 7-DoF arms + 1-DoF grippers. Limits: owner\'s joint chart. Masses: ESTIMATES. -->',
           '<robot name="pioneer_bimanual_arm_8dof">', old_base_link()]
    joints_meta = {}
    for side in ("left", "right"):
        sgn = 1 if side == "left" else -1
        parent, parent_origin = "base_link", np.zeros(3)
        for lk, j in chain:
            o = origin[lk] if side == "left" else mirror(origin[lk])
            a = axs[j] if side == "left" else -mirror(axs[j])      # right: same angle = mirror pose
            rel = o - parent_origin
            lo, hi = np.radians(LIMITS_DEG[j])
            eff, vel = MOTOR[j]
            jn = f"{side}_{j}"
            xml.append(link_xml(f"{side}_{lk}"))
            xml.append(f'  <joint name="{jn}" type="revolute"><parent link="{parent}"/><child link="{side}_{lk}"/>'
                       f'<origin xyz="{rel[0]:.6f} {rel[1]:.6f} {rel[2]:.6f}" rpy="0 0 0"/>'
                       f'<axis xyz="{a[0]:.6f} {a[1]:.6f} {a[2]:.6f}"/>'
                       f'<limit lower="{lo:.6f}" upper="{hi:.6f}" effort="{eff}" velocity="{vel}"/></joint>')
            joints_meta[jn] = dict(parent=parent, child=f"{side}_{lk}", limit_deg=LIMITS_DEG[j])
            parent, parent_origin = f"{side}_{lk}", o
        for fk in ("finger_a", "finger_b"):
            a = finger_axes[fk] if side == "left" else mirror(finger_axes[fk])
            jn = f"{side}_gripper" if fk == "finger_a" else f"{side}_gripper_mimic"
            xml.append(link_xml(f"{side}_{fk}"))
            mimic = f'<mimic joint="{side}_gripper" multiplier="1" offset="0"/>' if fk == "finger_b" else ""
            xml.append(f'  <joint name="{jn}" type="prismatic"><parent link="{side}_link7"/><child link="{side}_{fk}"/>'
                       f'<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="{a[0]:.6f} {a[1]:.6f} {a[2]:.6f}"/>'
                       f'<limit lower="0" upper="{GRIPPER_STROKE}" effort="34" velocity="0.2"/>{mimic}</joint>')
    xml.append("</robot>")
    (out / "urdf" / "pioneer_bimanual_arm_8dof.urdf").write_text("\n".join(xml) + "\n")
    # camera mounts (link7 frame): between the fingers at the palm, looking along the gripper
    jaw = unbend(to_urdf((JAW_CENTRE_X, 905.0, 395.0))) - origin["link7"]
    meta = {"joints": joints_meta, "wrist_camera_left_link7": jaw.tolist(),
            "gripper_forward_link7": (Rz @ axis_to_urdf((0, 1, 0))).tolist()}
    (out / "build_info.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "assets/pioneer_bimanual_arm_8dof")
