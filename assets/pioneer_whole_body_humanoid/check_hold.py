"""Can the robot hold itself up? Each whole-body model standing (its feet held to the floor, its base free) in a slight
crouch with its arms out in front, every joint's motor a position servo capped at the URDF's effort limit, 3 s under
gravity: how far each joint sags from where it's told to be, and which motors saturate (need their whole limit).

    python check_hold.py [model.urdf ...]          # default: both whole-body URDFs here

A joint that sags more than 2 degrees (or 2 mm for a jaw) fails: its motor can't carry the load at that pose.
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POSE = {"Hip_F": 0.25, "Knee": -0.5, "Ankle_P": -0.25,                       # a crouch, the soles level
        "shoulder_pitch": -0.8, "joint1": -0.8, "joint1L": -0.8}             # the arms out in front
SAG_RAD, SAG_M = np.radians(2.0), 0.002


def load(urdf):
    text = open(urdf).read()
    limits = {j.get("name"): float(j.find("limit").get("effort"))
              for j in ET.fromstring(text).iter("joint") if j.find("limit") is not None}
    mimics = {j.get("name") for j in ET.fromstring(text).iter("joint") if j.find("mimic") is not None}
    text = re.sub(r"<visual>.*?</visual>", "", text, flags=re.S)
    text = text.replace('discardvisual="false"', "")
    text = text.replace("</robot>", '  <link name="ground"/>\n  <joint name="base_free" type="floating"><parent link="ground"/>'
                        '<child link="base_link"/></joint>\n</robot>')       # the base free (MuJoCo: a free joint)
    spec = mujoco.MjSpec.from_string(text)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for g in spec.geoms:                                   # nothing collides: the feet are held, the rest is free
        g.contype = g.conaffinity = 0
    for side in ("l", "r"):                                # the feet held where they stand
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1=f"{side}_foot",
                          name2="world", solref=[0.005, 1])
    for j in spec.joints:
        if j.name in limits and j.name not in mimics:
            e = limits[j.name]
            slide = j.type == mujoco.mjtJoint.mjJNT_SLIDE
            kp = e / (0.002 if slide else 0.01)            # full torque at 0.6 deg (2 mm) error: a stiff servo
            a = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=kp, kv=0.05 * kp)
            a.forcelimited, a.forcerange = True, [-e, e]
            j.armature = 0.01
    return spec.compile(), limits


def target(name):
    j = name[2:] if name[:2] in ("l_", "r_") else name
    for k, v in POSE.items():
        if j == k or j.startswith(k + "_") or name == k:
            return v
    return 0.0


def hold(urdf, scale=1.0):
    m, limits = load(urdf)
    m.actuator_forcerange[:] *= scale
    d = mujoco.MjData(m)
    for a in range(m.nu):
        jn = m.joint(m.actuator_trnid[a][0]).name
        d.ctrl[a] = np.clip(target(jn), *m.jnt_range[m.actuator_trnid[a][0]])
        d.qpos[m.jnt_qposadr[m.actuator_trnid[a][0]]] = d.ctrl[a]
    mujoco.mj_forward(m, d)
    for e in range(m.neq):                 # the feet held where the pose puts them (not where they stand at zero)
        b = m.eq_obj1id[e]
        q = np.zeros(4)
        mujoco.mju_negQuat(q, d.xquat[b])
        m.eq_data[e][0:3] = 0.0
        m.eq_data[e][3:6] = -(d.xmat[b].reshape(3, 3).T @ d.xpos[b])     # the world's origin in the foot's frame
        m.eq_data[e][6:10] = q
    mujoco.mj_forward(m, d)
    residual = float(np.abs(d.efc_pos[d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]).max())
    assert residual < 1e-6, f"the feet's welds start {residual:.2g} off"
    peak = np.zeros(m.nu)
    while d.time < 3.0:
        mujoco.mj_step(m, d)
        if d.time > 1.0:
            peak = np.maximum(peak, np.abs(d.actuator_force))
    rows = []
    for a in range(m.nu):
        j = m.actuator_trnid[a][0]
        err = abs(d.qpos[m.jnt_qposadr[j]] - d.ctrl[a])
        slide = m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
        lim = m.actuator_forcerange[a][1]
        rows.append((m.joint(j).name, err, slide, peak[a], lim, err > (SAG_M if slide else SAG_RAD)))
    total = sum(m.body_mass)
    return rows, total


def main(paths):
    bad = 0
    for p in paths:
        rows, total = hold(p)
        worst = sorted(rows, key=lambda r: -r[3] / max(r[4], 1e-9))[:6]
        fails = [r for r in rows if r[5]]
        print(f"{os.path.relpath(p)}: {total:.1f} kg, {len(rows)} motors, {len(fails)} sag past 2 deg / 2 mm")
        for n, err, slide, pk, lim, f in fails + [r for r in worst if r not in fails]:
            unit = ("mm", 1000.0) if slide else ("deg", np.degrees(1.0))
            print(f"  {n:18s} sag {err * unit[1]:6.2f} {unit[0]:3s}  peak {pk:7.2f} of {lim:6.2f}  {'FAIL' if f else 'ok'}")
        bad += len(fails)
    print("every motor holds its joint" if not bad else f"{bad} joints sag")
    return 1 if bad else 0


if __name__ == "__main__":
    default = [os.path.join(HERE, f) for f in ("pioneer_whole_body_humanoid.urdf", "pioneer_whole_body_humanoid_gripper.urdf")]
    sys.exit(main(sys.argv[1:] or default))
