"""Build the gripper model's arms (pioneer_whole_body_humanoid_gripper.urdf) from the arm's CAD: its arms are the
bimanual arm's, so r_arm_link1..8 / l_arm_link1..8 take the same parts as assets/pioneer_bimanual_arm
(step_to_links.py: the same links, colours, 90 deg turn about joint5, joint axes and limits), in this model's frame.
link1..link4 (the same parts as before) get new meshes in colour and keep their inertias and collision boxes;
link5..link8 (the new forearm and wrist gripper) and joint6..8 are rebuilt, in this model's mass rules:

    prints                 c0 white      0.65 g/cm^3
    fingers                c6 orange     0.65 g/cm^3
    AK10 / AK80 motors     c2 graphite   940 g / 490 g a motor
    GL40 motors            c18 graphite  107 g a motor (the wrist's and the gripper's)
    MGN9 rail, carriages   c19 grey      7.85 g/cm^3 (steel)

    python new_arm.py ArmV2_NewForearms_NewWristGrippers.STEP     # then: python to_mjcf.py pioneer_whole_body_humanoid_gripper.urdf

Run after step_to_links.py (it reads the bimanual URDF's joint frames). Needs build123d and fast_simplification.
"""
import os
import re
import sys

import build123d as bd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = os.path.join(HERE, "..", "pioneer_bimanual_arm")
sys.path.insert(0, ARM)
import step_to_links as s2l  # noqa: E402

URDF = os.path.join(HERE, "pioneer_whole_body_humanoid_gripper.urdf")
# the bimanual arm's base_link -> this model's world: the robot faces -Y with its right arm at -X (a quarter turn
# about Z), shoulders 1.2145 m up. Puts joint5 of each arm where this model has it (checked below).
Q = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
T = np.array([0.0, 0.0217, 1.2145])
MATERIAL = {"print": "c0", "finger": "c6", "motor": "c2", "gl40": "c18", "steel": "c19", "aluminium": "c3", "camera": "c1"}
PRINT_DENSITY, STEEL_DENSITY = 650.0, 7850.0
MOTOR_KG = {"ak10": 0.940, "ak80": 0.490, "gl40": 0.107}


def motor_of(side, link, label):
    """which motor a part belongs to (one a link; the wrist GL40 is split over link5 / link6), and its type"""
    t = next((t for t in MOTOR_KG if t in label.lower()), None)
    if t is None:
        return None
    if "gl40ii" in label and "hum_" not in label:
        return (side, "wristgl40"), t
    return (side, link, t), t


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
        side, link = s2l.assign(c.label, tuple(c.bounding_box().center()))
        if side and link:                                        # (base_link, the stand, is not in this model)
            parts.setdefault((side, link), []).append((c.label, c))
    motor_vol = {}                                               # each motor's total volume, for its weight
    for (side, link), items in parts.items():
        for label, s in items:
            mo = motor_of(side, link, label)
            if mo:
                motor_vol[mo[0]] = motor_vol.get(mo[0], 0.0) + s.volume
    text = open(URDF).read()
    origin = {}
    for side in ("right", "left"):
        p = "r" if side == "right" else "l"
        for k in "12345678":
            origin[f"{p}_arm_link{k}"] = wb(frames[s2l.link_name(side, k)])
    # joints 1-5 are unchanged: the bimanual arm's link1..5 must land on this model's
    for side, p in (("right", "r"), ("left", "l")):
        for k in "12345":
            m = re.search(r"<link name=\"%s_arm_link%s\">[\s\S]*?<visual><origin xyz=\"([^\"]+)\"" % (p, k), text)
            have = -np.array([float(v) for v in m.group(1).split()])
            assert np.allclose(have, origin[f"{p}_arm_link{k}"], atol=2e-4), (side, k, have, origin[f"{p}_arm_link{k}"])
    for (side, k), items in sorted(parts.items()):
        p = "r" if side == "right" else "l"
        name = f"{p}_arm_link{k}"
        o = origin[name]
        Rt = Q @ s2l.turn(side) @ s2l.R
        groups, m_tot, mc, props = {}, 0.0, np.zeros(3), []
        keep = k in s2l.KEEP_INERTIA
        for label, s in items:
            v, f = s2l.mesh(s)
            groups.setdefault(s2l.colour_of(label), []).append((wb(s2l.to_urdf(v, side, k)), f))
            if keep:
                continue
            vol_m3, c, inertia = s2l.mass_props(s)                # at 1000 kg/m^3
            vol_m3 /= s2l.DENSITY
            inertia = inertia / s2l.DENSITY
            mo = motor_of(side, k, label)
            kd = s2l.colour_of(label)
            rho = (MOTOR_KG[mo[1]] / (motor_vol[mo[0]] * 1e-9) if mo else STEEL_DENSITY if kd == "steel"
                   else PRINT_DENSITY)
            mass = rho * vol_m3
            com = wb(s2l.to_urdf(c, side, k)) - o
            props.append((mass, com, Rt @ (inertia * rho) @ Rt.T))
            m_tot += mass
            mc += mass * com
        old = re.search(r"<link name=\"%s\">[\s\S]*?</link>" % name, text).group(0)
        com = mc / m_tot if m_tot else np.zeros(3)
        I_tot = np.zeros((3, 3))
        for mass, c, inertia in props:
            d = c - com
            I_tot += inertia + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        for f in os.listdir(os.path.join(HERE, "meshes")):           # the link's old meshes
            if re.fullmatch(r"%s_\d+\.stl" % name, f):
                os.remove(os.path.join(HERE, "meshes", f))
        visuals, lo, hi = [], np.full(3, np.inf), np.full(3, -np.inf)
        for i, kd in enumerate(c for c in s2l.COLOURS if c in groups):
            tm = s2l.decimated([(v * 1000.0, f) for v, f in groups[kd]])          # mm, world at zero
            tm.export(os.path.join(HERE, "meshes", f"{name}_{i}.stl"))
            lo, hi = np.minimum(lo, tm.bounds[0] / 1000.0 - o), np.maximum(hi, tm.bounds[1] / 1000.0 - o)
            cname, rgba = MATERIAL[kd], s2l.COLOURS[kd][0]
            visuals.append(f'    <visual><origin xyz="{f6(-o[0])} {f6(-o[1])} {f6(-o[2])}"/><geometry><mesh filename="meshes/'
                           f'{name}_{i}.stl" scale="0.001 0.001 0.001"/></geometry><material name="{cname}"><color '
                           f'rgba="{rgba}"/></material></visual>')
        bc, bs = (lo + hi) / 2, hi - lo
        if keep:                                                 # the same parts: the inertia and box as they were
            inertial = re.search(r"    <inertial>.*</inertial>", old).group(0)
            collision = "\n".join(re.findall(r"    <collision>.*</collision>", old))
        else:
            inertial = (f'    <inertial><origin xyz="{f6(com[0])} {f6(com[1])} {f6(com[2])}"/><mass value="{m_tot:.4f}"/>'
                        f'<inertia ixx="{I_tot[0, 0]:.6g}" ixy="{I_tot[0, 1]:.6g}" ixz="{I_tot[0, 2]:.6g}" '
                        f'iyy="{I_tot[1, 1]:.6g}" iyz="{I_tot[1, 2]:.6g}" izz="{I_tot[2, 2]:.6g}"/></inertial>')
            collision = (f'    <collision><origin xyz="{f6(bc[0])} {f6(bc[1])} {f6(bc[2])}"/><geometry><box size="{f6(bs[0])} '
                         f'{f6(bs[1])} {f6(bs[2])}"/></geometry></collision>')
        block = f'<link name="{name}">\n{inertial}\n' + "\n".join(visuals) + f"\n{collision}\n  </link>"
        text, n = re.subn(r"<link name=\"%s\">[\s\S]*?</link>" % name, lambda _: block, text)
        assert n == 1, name
        print(f"{name}: {'inertia kept' if keep else f'{m_tot * 1000:.1f} g'}, colours {list(groups)}")
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
