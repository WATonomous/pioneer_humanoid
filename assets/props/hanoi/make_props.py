"""Writes the Tower of Hanoi props next to this file: lathe-turned rings, a peg and the hand-over tray.

  /workspace/isaaclab/_isaac_sim/python.sh assets/props/hanoi/make_props.py

Sizes and colours are the ones in humanoid_isaac_scenes/hanoi/scene.py.
"""
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})   # pxr comes with the app

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

SEG = 64   # segments around the axis


def fillet(corner, prev, nxt, r, n=5):   # replace a polygon corner by an arc of radius r
    c = np.asarray(corner, float)
    a, b = (np.asarray(prev, float) - c), (np.asarray(nxt, float) - c)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    centre = c + (a + b) * r
    p0, p1 = c + a * r - centre, c + b * r - centre
    t0, t1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
    dt = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
    return [centre + r * np.array([np.cos(t0 + dt * s), np.sin(t0 + dt * s)]) for s in np.linspace(0, 1, n)]


def rounded(poly, r, closed=True, keep=()):
    out = []
    for i, p in enumerate(poly):
        if i in keep or (not closed and i in (0, len(poly) - 1)):
            out.append(np.asarray(p, float))
        else:
            out += fillet(p, poly[i - 1], poly[(i + 1) % len(poly)], r)
    return out


def lathe(stage, path, prof, closed, color, rough=0.35):
    """Revolve a CCW (r, z) profile around +z; smooth around the axis, per-segment normals along the profile."""
    prof = [np.asarray(p, float) for p in prof]
    K = len(prof) - (0 if closed else 1)
    th = np.linspace(0, 2 * np.pi, SEG, endpoint=False)
    pts, nrm, idx, cnt = [], [], [], []
    for k in range(K):
        a, b = prof[k], prof[(k + 1) % len(prof)]
        d = b - a
        n2 = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-9)
        for j in range(SEG):
            for (p, t) in ((a, th[j]), (b, th[j]), (b, th[(j + 1) % SEG]), (a, th[(j + 1) % SEG])):
                idx.append(len(pts))
                pts.append((p[0] * np.cos(t), p[0] * np.sin(t), p[1]))
                nrm.append((n2[0] * np.cos(t), n2[0] * np.sin(t), n2[1]))
            cnt.append(4)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    m.CreateFaceVertexCountsAttr(cnt)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in nrm]))
    m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)   # quads above are wound clockwise seen from outside
    m.CreateDoubleSidedAttr(True)
    mat = UsdShade.Material.Define(stage, path + "_mat")
    sh = UsdShade.Shader.Define(stage, path + "_mat/s")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.3)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


def physics_material(stage, path, mu):
    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(mu)
    pm.CreateDynamicFrictionAttr(mu * 0.9)
    return mat


def new_stage(f):
    st = Usd.Stage.CreateNew(f)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/root")
    st.SetDefaultPrim(root.GetPrim())
    return st, root


def ring(f, ro, ri, h, color, mass=0.06, mu=1.0, nbox=20):
    """Ring lying on z=0: smooth visual, collision = nbox boxes around the annulus."""
    st, root = new_stage(f)
    lathe(st, "/root/visual", rounded([(ri, 0), (ro, 0), (ro, h), (ri, h)], 0.0025), True, color, 0.3)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
    pm = physics_material(st, "/root/pmat", mu)
    rm, w = (ro + ri) / 2, ro - ri
    L = 2 * ro * np.tan(np.pi / nbox) * 1.02
    for k in range(nbox):
        t = 2 * np.pi * k / nbox
        c = UsdGeom.Cube.Define(st, f"/root/col{k}")
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(rm * np.cos(t), rm * np.sin(t), h / 2))
        c.AddRotateZOp().Set(float(np.degrees(t)))
        c.AddScaleOp().Set(Gf.Vec3f(w, L, h))
        c.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    st.Save()


def static_lathe(f, prof, color, rough=0.45, mu=0.4):
    """Static fixture (peg / tray) turned from a profile, convex-hull collider."""
    st, root = new_stage(f)
    m = lathe(st, "/root/visual", prof, False, color, rough)
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(physics_material(st, "/root/pmat", mu),
                                                         UsdShade.Tokens.weakerThanDescendants, "physics")
    st.Save()


def peg(f, r, h, color):
    static_lathe(f, [(0, 0)] + rounded([(0, 0), (r, 0), (r, h), (0, h)], r * 0.9, closed=False, keep=(1,))[1:-1] + [(0, h)],
                 color, 0.4, 0.3)


def tray(f, r, h, color):
    static_lathe(f, [(0, 0), (r, 0)] + fillet((r, h), (r, 0), (0, h), 0.004) + [(0, h)], color, 0.5, 1.0)


if __name__ == "__main__":
    here = Path(__file__).parent
    H, HOLE_R, PEG_R, BOARD_H = 0.022, 0.011, 0.005, 0.02
    peg(str(here / "peg.usda"), PEG_R, 3 * H + 0.018, (0.80, 0.62, 0.40))
    tray(str(here / "tray.usda"), 0.036, BOARD_H, (0.62, 0.40, 0.22))
    for name, d, c in (("orange", 0.064, (0.96, 0.42, 0.03)), ("yellow", 0.054, (0.98, 0.80, 0.08)),
                       ("green", 0.044, (0.10, 0.62, 0.32))):
        ring(str(here / f"ring_{name}.usda"), d / 2, HOLE_R, H, c)
    app.close()
