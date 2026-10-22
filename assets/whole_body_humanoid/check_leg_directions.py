"""Check that each leg joint moves the way a person's does, not just that it moves.

A mirror check can't catch a joint that is wrong on both sides alike: the knee bent the foot forward (a bird's knee)
on both legs until Oct 22. Each leg joint is turned alone to 60 % of its range, toward the side its range favours, and
the leg must move as follows:

    Hip_F   the knee forward        Hip_A    the knee outward
    Knee    the ankle back          Ankle_P  the toes down

Forward is taken from the model itself (from the ankle to the foot's centre of mass, the toes' side), outward from the
body's middle to the hip. Needs only MuJoCo (the meshes are stripped, so the package:// paths don't matter).

    python assets/whole_body_humanoid/check_leg_directions.py [model.urdf|model.xml ...]
"""
import os
import re
import sys
import tempfile

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = [os.path.join(HERE, "whole_body_humanoid.urdf"),
           os.path.join(HERE, "..", "pioneer_v2_humanoid", "pioneer_v2.xml")]
DIRECTIONS = {"Hip_F": ("knee", "forward"), "Hip_A": ("knee", "outward"), "Knee": ("ankle", "back"),
              "Ankle_P": ("toes", "down")}


def load(path):
    if not path.endswith(".urdf"):
        return mujoco.MjModel.from_xml_path(path)
    s = open(path).read()
    s = re.sub(r"<visual>.*?</visual>|<collision>.*?</collision>", "", s, flags=re.S)
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(s)
    try:
        return mujoco.MjModel.from_xml_path(f.name)
    finally:
        os.remove(f.name)


def check(m):
    d = mujoco.MjData(m)
    rows = []
    for side in ("L", "R"):
        jid = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{k}_{side}")
               for k in ("Hip_F", "Hip_A", "Knee", "Ankle_P", "Ankle_R")}
        if min(jid.values()) < 0:
            continue

        def points():
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            return {"knee": d.xanchor[jid["Knee"]].copy(), "ankle": d.xanchor[jid["Ankle_P"]].copy(),
                    "toes": d.xipos[m.jnt_bodyid[jid["Ankle_R"]]].copy()}
        d.qpos[:] = m.qpos0
        p0 = points()
        up = np.array([0, 0, 1.0])
        fwd = p0["toes"] - p0["ankle"]
        fwd -= up * (fwd @ up)
        fwd /= np.linalg.norm(fwd)
        root = d.xpos[m.body_rootid[m.jnt_bodyid[jid["Hip_F"]]]]
        out = d.xanchor[jid["Hip_A"]] - root
        out -= up * (out @ up) + fwd * (out @ fwd)
        out /= np.linalg.norm(out)
        way = {"forward": fwd, "back": -fwd, "outward": out, "down": -up}
        for k, (pt, w) in DIRECTIONS.items():
            j = jid[k]
            lo, hi = m.jnt_range[j]
            q = 0.6 * (hi if abs(hi) >= abs(lo) else lo)
            d.qpos[:] = m.qpos0
            d.qpos[m.jnt_qposadr[j]] = q
            moved = float((points()[pt] - p0[pt]) @ way[w])
            rows.append((f"{k}_{side}", np.degrees(q), f"{pt} {w}", moved * 1000, moved > 0.005))
    return rows


def main(paths):
    bad = 0
    for path in paths:
        rows = check(load(path))
        print(os.path.relpath(path))
        for name, q, want, mm, ok in rows:
            print(f"  {name:10s} {q:6.0f} deg  {want:13s} {mm:6.0f} mm  {'PASS' if ok else 'FAIL'}")
        bad += sum(not r[-1] for r in rows) + (not rows)
    print("all leg joints move the right way" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT))
