"""Find what passes through what in a qpos record: in this sim the arms collide with nothing and only the hands
collide with the scene, so the script has to keep the arms clear. Reports, per pair, the closest approach and
when it is under the threshold.

    python -m humanoid_hands_sim.check matcha.npy [--every 6] [--mm 5]

  arm links vs torso and thighs   closest mesh vertex (KD-tree of the fixed body's vertices)
  tools vs arm links              mj_geomDistance (the arm link's convex hull: an over-estimate of its volume)
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from .robot import TOOLS, make_model

ARM_LINKS = ("arm_link2", "arm_link3", "arm_link4", "forearm", "hand_W", "hand_H")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("traj")
    ap.add_argument("--every", type=int, default=6, help="check every Nth frame (30 Hz record)")
    ap.add_argument("--mm", type=float, default=5.0, help="report time spans closer than this")
    a = ap.parse_args()
    m = make_model()
    d = mujoco.MjData(m)
    traj = np.load(a.traj)
    bn = [m.body(i).name for i in range(m.nbody)]

    def verts(g, step):
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]][::step]
        return d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T

    meshes = lambda names: [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and bn[m.geom_bodyid[g]] in names]
    d.qpos[:] = traj[0]
    mujoco.mj_forward(m, d)
    body = cKDTree(np.vstack([verts(g, 2) for g in meshes(("base_link", "l_thigh", "r_thigh"))]))   # fixed base
    arm = {f"{s}_{l}": meshes((f"{s}_{l}",)) for s in "lr" for l in ARM_LINKS}
    tools = {t: [g for g in range(m.ngeom) if bn[m.geom_bodyid[g]] == t] for t in TOOLS}
    out = {}

    def note(key, dist, t):
        o = out.setdefault(key, [np.inf, 0.0, []])
        if dist < o[0]:
            o[0], o[1] = dist, t
        if dist < a.mm / 1000:
            o[2].append(t)

    for f in range(0, len(traj), a.every):
        d.qpos[:] = traj[f]
        mujoco.mj_forward(m, d)
        t = f / 30
        for link, gs in arm.items():
            if link.endswith("arm_link2"):
                continue                         # the shoulder's own housing sits against the torso
            note((link, "torso/thighs"), body.query(np.vstack([verts(g, 15) for g in gs]))[0].min(), t)
            for tool, tg in tools.items():
                best = np.inf
                for g in gs:
                    for h in tg:
                        if np.linalg.norm(d.geom_xpos[g] - d.geom_xpos[h]) < m.geom_rbound[g] + m.geom_rbound[h] + 0.05:
                            best = min(best, mujoco.mj_geomDistance(m, d, g, h, 0.05, None))
                if best < np.inf and not link.endswith(("hand_W", "hand_H")):   # the hands hold the tools
                    note((link, tool), best, t)
    bad = 0
    for (x, y), (dmin, tmin, close) in sorted(out.items(), key=lambda kv: kv[1][0]):
        if dmin > 0.03:
            continue
        span = f"{min(close):.1f}-{max(close):.1f} s ({len(close)} checks)" if close else "never"
        bad += bool(close)
        print(f"{x:16s} vs {y:13s} closest {dmin * 1000:+6.1f} mm at {tmin:5.1f} s;  under {a.mm:g} mm: {span}")
    print("OK: nothing passes through anything" if not bad else f"{bad} pair(s) closer than {a.mm:g} mm")


if __name__ == "__main__":
    main()
