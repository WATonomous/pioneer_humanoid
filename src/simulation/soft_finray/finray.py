"""Soft TPU Fin Ray finger for MuJoCo as a beam network.

The finger's TPU profile (link7_finger.STL cut at mid-width) is skeletonised into its walls and ribs. Each one becomes a
chain of short rigid links joined by hinges about x (the finger's width), with bending springs k = E I / l,
I = W t^3 / 12 from the local thickness t. Ribs close their loops with weld constraints; the profile above MOUNT_Z is
clamped to the carriage. The finger bends in its own plane only.

finger_xml(name, side) returns the body XML (children of the carriage "<name>_carriage") and the <equality> welds.
The model needs a default class "tpu" for the link geoms (friction, contype, conaffinity, density 0)."""
import os
import struct
from collections import deque

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt, label
from skimage.morphology import skeletonize

MESH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets", "pioneer_bimanual_arm", "meshes", "link7_finger.STL")
MOUNT_Z = 0.047  # above this the TPU is clamped to the carriage
TPU_E = 20e6  # Pa, printed TPU 95A (datasheets 20-30 MPa)
TPU_RHO = 1210.0
WIDTH = 0.0434  # finger width along x
BEAM_E = 0.625  # beam theory overstates a frame's stiffness where ribs meet walls; fitted to a 2D FEM of this profile
FACE_Y = 0.019  # the contact wall's centre line (its face is at y = 0.020)


def profile(h=0.0001):
    """The STL cut at x = 7 mm, rasterised: (inside[nz, ny], y centres, z centres)."""
    b = open(MESH, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    T = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))["v"].reshape(-1, 3, 3).astype(float)
    segs = []
    for tri in T:
        d = tri[:, 0] - 0.007
        pts = [tri[i] + d[i] / (d[i] - d[(i + 1) % 3]) * (tri[(i + 1) % 3] - tri[i]) for i in range(3) if d[i] * d[(i + 1) % 3] < 0]
        if len(pts) == 2:
            segs.append([pts[0][1:], pts[1][1:]])
    a, b = np.array(segs)[:, 0], np.array(segs)[:, 1]
    ys = np.arange(-0.0215, 0.0215, h) + h / 2
    zs = np.arange(-0.070, 0.062, h) + h / 2
    inside = np.zeros((len(zs), len(ys)), bool)
    for i, z in enumerate(zs):  # inside = odd number of crossings
        cross = (a[:, 1] - z) * (b[:, 1] - z) < 0
        yc = np.sort(a[cross, 0] + (z - a[cross, 1]) / (b[cross, 1] - a[cross, 1]) * (b[cross, 0] - a[cross, 0]))
        inside[i] = np.searchsorted(yc, ys) % 2 == 1
    return inside, ys, zs


def graph(h=0.0001):
    """Skeleton of the profile split at its junctions: edges (polyline, thickness, end junctions) and junction points."""
    I, ys, zs = profile(h)
    sk = skeletonize(I)
    th = 2 * distance_transform_edt(I) * h
    junc = sk & (convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1 >= 3)
    jl, nj = label(junc, np.ones((3, 3)))
    xy = lambda p: np.array([ys[p[1]], zs[p[0]]])
    J = [np.mean([xy(p) for p in np.argwhere(jl == k + 1)], 0) for k in range(nj)]
    rest = sk & ~junc
    near = lambda p: [(p[0] + i, p[1] + j) for i in (-1, 0, 1) for j in (-1, 0, 1)
                      if 0 <= p[0] + i < sk.shape[0] and 0 <= p[1] + j < sk.shape[1]]
    seen, edges = np.zeros_like(sk), []
    for start in map(tuple, np.argwhere(rest)):
        if seen[start]:
            continue
        comp, stack = [], [start]
        seen[start] = True
        while stack:  # the run of skeleton pixels between junctions
            p = stack.pop()
            comp.append(p)
            for q in near(p):
                if rest[q] and not seen[q]:
                    seen[q] = True
                    stack.append(q)
        ends = {jl[q] - 1 for p in comp for q in near(p) if jl[q]}
        cset = set(comp)
        cur = next((p for p in comp if sum(q in cset for q in near(p)) <= 2), comp[0])  # an end of the run
        path, cset = [cur], cset - {cur}
        while cset:
            nxt = [q for q in near(cur) if q in cset]
            if not nxt:
                break
            cur = nxt[0]
            path.append(cur)
            cset.discard(cur)
        pts = np.array([xy(p) for p in path])
        if len(ends) == 2:
            edges.append(dict(pts=pts, t=np.array([th[p] for p in path]), j=sorted(ends, key=lambda k: np.linalg.norm(J[k] - pts[0]))))
    return edges, J


def finger_xml(name, side=+1, dt=1e-2, seg=0.003, young=TPU_E, damping_ratio=0.3):
    """side=+1: contact face at +y of the finger frame (link7); -1: mirrored (link8). dt sets the hinges' rotor inertia."""
    E, J = graph()
    J = [j * [side, 1] for j in J]
    rigid = {k for k, j in enumerate(J) if j[1] > MOUNT_Z}
    node = {k: f"{name}_carriage" for k in rigid}  # junction -> body it is fixed to
    bodies = {f"{name}_carriage": dict(pos=np.zeros(2), parent=None, xml="")}
    pending, welds, n = deque(), [], 0
    for e in E:
        a, b = e["j"]
        pts = np.vstack([J[a], e["pts"] * [side, 1], J[b]])
        t = np.minimum(np.r_[e["t"][0], e["t"], e["t"][-1]], np.median(e["t"]))  # the skeleton's circle swells at junctions
        if a in rigid and b in rigid:
            continue
        if b in rigid:
            a, b, pts, t = b, a, pts[::-1], t[::-1]
        if a in rigid:  # cut at the clamp line
            k = np.argmax(pts[:, 1] <= MOUNT_Z)
            pts, t = pts[k:], t[k:]
        pending.append(dict(a=a, b=b, pts=pts, t=t))
    stall = 0
    while pending:  # grow the tree out from the clamp; an edge whose both ends exist closes a loop with a weld
        c = pending.popleft()
        if c["a"] not in node and c["b"] not in node:
            pending.append(c)
            stall += 1
            if stall > len(pending):
                raise RuntimeError("disconnected beam network")
            continue
        stall = 0
        if c["a"] not in node:
            c.update(a=c["b"], b=c["a"], pts=c["pts"][::-1], t=c["t"][::-1])
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c["pts"], axis=0), axis=1))]
        s = np.linspace(0, d[-1], max(1, round(d[-1] / seg)) + 1)
        P = np.c_[np.interp(s, d, c["pts"][:, 0]), np.interp(s, d, c["pts"][:, 1])]
        T, L = np.interp((s[:-1] + s[1:]) / 2, d, c["t"]), np.diff(s)
        parent = node[c["a"]]
        for i in range(len(L) + (c["b"] in node)):  # one extra hinge at a weld, so the rib end can bend too
            last = i == len(L)
            l, th = (L[-1], T[-1]) if last else (L[i], T[i])
            k = young * BEAM_E * WIDTH * th ** 3 / 12 / ((l + L[max(i - 1, 0)]) / 2)  # the hinge stands for the half links beside it
            arm = k * (2 * dt) ** 2  # rotor inertia: the hinge's spring period is ~12 steps (quasi-static grasping)
            joint = f'<joint type="hinge" axis="1 0 0" stiffness="{k:.5g}" armature="{arm:.4g}" damping="{2 * damping_ratio * np.sqrt(k * arm):.4g}"/>'
            if last:
                xml = joint + '<inertial pos="0 0 0" mass="1e-4" diaginertia="1e-9 1e-9 1e-9"/>'
            else:
                v = P[i + 1] - P[i]
                xml = (joint + f'<inertial pos="0 {v[0] / 2:.6f} {v[1] / 2:.6f}" mass="{WIDTH * th * l * TPU_RHO:.4g}" diaginertia="1e-8 1e-8 1e-8"/>'
                       f'<geom type="box" pos="0 {v[0] / 2:.6f} {v[1] / 2:.6f}" euler="{np.degrees(np.arctan2(v[1], v[0])):.4f} 0 0" '
                       f'size="{WIDTH / 2:.5f} {l / 2:.6f} {th / 2:.6f}" class="tpu"/>')
            bn = f"{name}_l{n}"
            n += 1
            bodies[bn] = dict(pos=P[min(i, len(L))], parent=parent, xml=xml)
            parent = bn
        if c["b"] in node:
            welds.append(f'<weld body1="{parent}" body2="{node[c["b"]]}" solref="{2 * dt} 1" solimp="0.999 0.9999 0.001"/>')
        else:
            jb = f"{name}_j{c['b']}"
            bodies[jb] = dict(pos=P[-1], parent=parent, xml='<inertial pos="0 0 0" mass="1e-4" diaginertia="1e-9 1e-9 1e-9"/>')
            node[c["b"]] = jb
    kids = {}
    for bn, b in bodies.items():
        kids.setdefault(b["parent"], []).append(bn)

    def emit(bn):  # bodies carry no rotation, so a body's position is its world offset from its parent
        b = bodies[bn]
        rel = b["pos"] - bodies[b["parent"]]["pos"]
        return f'<body name="{bn}" pos="0 {rel[0]:.6f} {rel[1]:.6f}">{b["xml"]}{"".join(emit(c) for c in kids.get(bn, []))}</body>'
    face = [bn for bn, b in bodies.items() if b["parent"] and abs(b["pos"][0] * side - FACE_Y) < 0.0008]
    return "".join(emit(c) for c in kids[f"{name}_carriage"]), "".join(welds), dict(links=n, welds=len(welds), face=face)
