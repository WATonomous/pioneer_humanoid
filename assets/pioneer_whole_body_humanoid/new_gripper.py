"""Put the new forearm and wrist gripper into the gripper model (pioneer_whole_body_humanoid_gripper.urdf): its arms are
the bimanual arm's, so r_arm_link5..8 / l_arm_link5..8 and joint6..8 / joint6l..8l are rebuilt from the same CAD as
assets/pioneer_bimanual_arm (step_to_links.py: the same parts, the same 90 deg turn about joint5, the same joint axes
and limits), in this model's frame and in its colours and mass rules:

    print parts, fingers   c0 white      0.65 g/cm^3
    GL40 motors            c18 graphite  107 g a motor (the wrist's and the gripper's)
    MGN9 rail, carriages   c19 grey      7.85 g/cm^3 (steel)

    python new_gripper.py ArmV2_NewForearms_NewWristGrippers.STEP     # then: python to_mjcf.py pioneer_whole_body_humanoid_gripper.urdf

Run after step_to_links.py (it reads the bimanual URDF's joint frames). Needs build123d and fast_simplification.
"""
import os
import re
import sys

import build123d as bd
import fast_simplification
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = os.path.join(HERE, "..", "pioneer_bimanual_arm")
sys.path.insert(0, ARM)
import step_to_links as s2l  # noqa: E402

URDF = os.path.join(HERE, "pioneer_whole_body_humanoid_gripper.urdf")
# the bimanual arm's base_link -> this model's world: the robot faces -Y with its right arm at -X (a quarter turn
# about Z), shoulders 1.2145 m up. Puts joint5 of each arm where this model has it (checked below).
Q = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
T = np.array([0.0, 0.0217, 1.2145])
COLOURS = {"print": ("c0", "0.925 0.929 0.937 1"), "motor": ("c18", "0.227 0.235 0.251 1"),
           "steel": ("c19", "0.604 0.620 0.643 1")}
PRINT_DENSITY, STEEL_DENSITY, GL40_KG = 650.0, 7850.0, 0.107
MAX_FACES = 60_000


def kind(label):
    if "gl40ii" in label:
        return "motor"
    if re.search(r"mgn9|component2[45]", label):
        return "steel"
    return "print"


def motor_of(side, label):
    """which GL40 a motor part belongs to: the wrist's (split over link5 / link6) or the gripper's"""
    return (side, "gripper" if "hum_gl40" in label else "wrist")


def bm_frames():
    """link5..8 of each arm in the bimanual arm's base_link (m) and the joint6..8 axes, from its URDF"""
    text = open(s2l.URDF).read()
    frames = s2l.link_frames()
    axes, limits = {}, {}
    for j in ("joint6", "joint7", "joint8", "joint6l", "joint7l", "joint8l"):
        b = re.search(r"<joint\s+name=\"%s\"[\s\S]*?</joint>" % j, text).group(0)
        axes[j] = np.array([float(v) for v in re.search(r"<axis\s+xyz=\"([^\"]+)\"", b).group(1).split()])
        limits[j] = tuple(float(v) for v in re.search(r"lower=\"([^\"]+)\"\s+upper=\"([^\"]+)\"", b).groups())
    return frames, axes, limits


def wb(p):
    return np.asarray(p) @ Q.T + T


def f6(x):
    return f"{x:.6f}"


def main(step):
    root = bd.import_step(step)
    frames, axes, limits = bm_frames()
    parts = {}                                                   # (side, link) -> [(label, shape)]
    for c in root.children:
        side, link = s2l.side_of(c.label), s2l.link_of(c.label)
        if side and link:
            parts.setdefault((side, link), []).append((c.label, c))
    motor_vol = {}                                               # each GL40's total volume, for its 107 g
    for (side, _), items in parts.items():
        for label, s in items:
            if kind(label) == "motor":
                motor_vol[motor_of(side, label)] = motor_vol.get(motor_of(side, label), 0.0) + s.volume
    text = open(URDF).read()
    origin = {}
    for side in ("right", "left"):
        p = "r" if side == "right" else "l"
        for k in "5678":
            origin[f"{p}_arm_link{k}"] = wb(frames[s2l.link_name(side, k)])
    # joint5 is unchanged: the bimanual arm's link5 must land on this model's
    for side, p in (("right", "r"), ("left", "l")):
        m = re.search(r"<link name=\"%s_arm_link5\">[\s\S]*?<visual><origin xyz=\"([^\"]+)\"" % p, text)
        have = -np.array([float(v) for v in m.group(1).split()])
        assert np.allclose(have, origin[f"{p}_arm_link5"], atol=2e-4), (side, have, origin[f"{p}_arm_link5"])
    for (side, k), items in sorted(parts.items()):
        p = "r" if side == "right" else "l"
        name = f"{p}_arm_link{k}"
        o = origin[name]
        Rt = Q @ s2l.turn(side) @ s2l.R
        groups, m_tot, mc, props = {}, 0.0, np.zeros(3), []
        for label, s in items:
            v, f = s2l.mesh(s)
            g = groups.setdefault(kind(label), ([], []))
            g[1].append(f + sum(len(x) for x in g[0]))
            g[0].append(wb(s2l.to_urdf(v, side)))
            vol_m3, c, inertia = s2l.mass_props(s)                # at 1000 kg/m^3
            vol_m3 /= s2l.DENSITY
            inertia = inertia / s2l.DENSITY
            kd = kind(label)
            rho = (PRINT_DENSITY if kd == "print" else STEEL_DENSITY if kd == "steel"
                   else GL40_KG / (motor_vol[motor_of(side, label)] * 1e-9))
            mass = rho * vol_m3
            com = wb(s2l.to_urdf(c, side)) - o
            props.append((mass, com, Rt @ (inertia * rho) @ Rt.T))
            m_tot += mass
            mc += mass * com
        com = mc / m_tot
        I_tot = np.zeros((3, 3))
        for mass, c, inertia in props:
            d = c - com
            I_tot += inertia + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        for f in os.listdir(os.path.join(HERE, "meshes")):           # the link's old meshes
            if re.fullmatch(r"%s_\d+\.stl" % name, f):
                os.remove(os.path.join(HERE, "meshes", f))
        visuals, lo, hi = [], np.full(3, np.inf), np.full(3, -np.inf)
        for i, kd in enumerate(sorted(groups, key=["print", "motor", "steel"].index)):
            V, F = groups[kd]
            tm = trimesh.Trimesh(np.vstack(V) * 1000.0, np.vstack(F), process=True)      # mm, world at zero
            if len(tm.faces) > MAX_FACES:
                v2, f2 = fast_simplification.simplify(tm.vertices, tm.faces, 1.0 - MAX_FACES / len(tm.faces))
                tm = trimesh.Trimesh(v2, f2)
            tm.export(os.path.join(HERE, "meshes", f"{name}_{i}.stl"))
            lo, hi = np.minimum(lo, tm.bounds[0] / 1000.0 - o), np.maximum(hi, tm.bounds[1] / 1000.0 - o)
            cname, rgba = COLOURS[kd]
            visuals.append(f'    <visual><origin xyz="{f6(-o[0])} {f6(-o[1])} {f6(-o[2])}"/><geometry><mesh filename="meshes/'
                           f'{name}_{i}.stl" scale="0.001 0.001 0.001"/></geometry><material name="{cname}"><color '
                           f'rgba="{rgba}"/></material></visual>')
        bc, bs = (lo + hi) / 2, hi - lo
        block = (f'<link name="{name}">\n'
                 f'    <inertial><origin xyz="{f6(com[0])} {f6(com[1])} {f6(com[2])}"/><mass value="{m_tot:.4f}"/><inertia '
                 f'ixx="{I_tot[0, 0]:.6g}" ixy="{I_tot[0, 1]:.6g}" ixz="{I_tot[0, 2]:.6g}" iyy="{I_tot[1, 1]:.6g}" '
                 f'iyz="{I_tot[1, 2]:.6g}" izz="{I_tot[2, 2]:.6g}"/></inertial>\n' + "\n".join(visuals) + "\n"
                 f'    <collision><origin xyz="{f6(bc[0])} {f6(bc[1])} {f6(bc[2])}"/><geometry><box size="{f6(bs[0])} '
                 f'{f6(bs[1])} {f6(bs[2])}"/></geometry></collision>\n  </link>')
        text, n = re.subn(r"<link name=\"%s\">[\s\S]*?</link>" % name, lambda _: block, text)
        assert n == 1, name
        print(f"{name}: {m_tot * 1000:.1f} g, {len(groups)} colours")
    for side, p, sfx in (("right", "r", ""), ("left", "l", "l")):
        for jn, parent, child in ((f"joint6{sfx}", 5, 6), (f"joint7{sfx}", 6, 7), (f"joint8{sfx}", 6, 8)):
            o = origin[f"{p}_arm_link{child}"] - origin[f"{p}_arm_link{parent}"]
            ax = Q @ axes[jn] + 0.0

            def joint(mt, o=o, ax=ax, lim=limits[jn], prismatic=child in (7, 8)):
                b = mt.group(0)
                b = re.sub(r"<origin xyz=\"[^\"]+\"/>", f'<origin xyz="{f6(o[0])} {f6(o[1])} {f6(o[2])}"/>', b, count=1)
                b = re.sub(r"<axis xyz=\"[^\"]+\"/>", f'<axis xyz="{f6(ax[0])} {f6(ax[1])} {f6(ax[2])}"/>', b)
                if prismatic:
                    b = re.sub(r"lower=\"[^\"]+\" upper=\"[^\"]+\"", f'lower="{lim[0]:.4f}" upper="{lim[1]:.4f}"', b)
                return b
            text, n = re.subn(r"<joint name=\"%s\"[\s\S]*?</joint>" % jn, joint, text)
            assert n == 1, jn
    open(URDF, "w").write(text)


if __name__ == "__main__":
    main(sys.argv[1])
