"""Background USD for the path-traced sand walk: a tropical beach in late-afternoon sun. Clear-sky HDRI dome and a warm
sun (NVIDIA Omniverse Skies), palms and agaves (NVIDIA Omniverse Vegetation/Plant_Tropical), a sand ground that slopes
down into the sea on the far side, with a hole cut where the simulated sand is.

  python beach_scene.py <out.usda> <path length m> x0 x1 y0 y1 depth
"""
import os
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

HERE = Path(__file__).resolve().parent / "assets"
out, length = sys.argv[1], float(sys.argv[2])
x0, x1, y0, y1, depth = [float(v) for v in sys.argv[3:8]]
env = lambda k, d: float(os.environ.get(k, d))  # noqa: E731
stage = Usd.Stage.CreateNew(out)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
rng = np.random.default_rng(7)

dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
dome.CreateTextureFileAttr(str(HERE / "sky" / "syferfontein_18d_clear.hdr"))
dome.CreateTextureFormatAttr("latlong")
dome.CreateIntensityAttr(env("DOME", 1000))
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(env("SUN", 2500))
sun.CreateEnableColorTemperatureAttr(True)
sun.CreateColorTemperatureAttr(4300.0)
sun.CreateAngleAttr(0.8)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(62, 0, 40))  # low: long shadows in the footprints


def material(path, rgb, rough, metal=0.0, opacity=None):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def mesh(path, X, Y, Z, mat, hole=False):
    n0, n1 = X.shape
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    idx = np.arange(n0 * n1).reshape(n0, n1)
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    if hole:
        c = pts[quads].mean(1)
        quads = quads[~((c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1))]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    m.CreateFaceVertexCountsAttr([4] * len(quads))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(quads.flatten().astype(np.int32)))
    m.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)


# sand: rippled dunes behind the camera side, level with the simulated sand along the path, sloping into the sea at y > 9
n, half = 200, 50.0
xs = np.linspace(-half, half, n) + length / 2
ys = np.linspace(-half, 14.0, n)
X, Y = np.meshgrid(xs, ys, indexing="ij")
Z = sum(a * np.sin(f * X + rng.uniform(0, 6)) * np.cos(f * 0.7 * Y + rng.uniform(0, 6)) for a, f in ((0.25, 0.25), (0.06, 0.9), (0.012, 3.0)))
Z = np.where(np.abs(Y) < 3.0, Z * (np.abs(Y) / 3.0) ** 2, Z) + depth
Z = np.where(Y > 6.0, Z - 0.12 * (Y - 6.0), Z)
X = np.where(np.abs(X - x0) < half / n, x0, np.where(np.abs(X - x1) < half / n, x1, X))  # grid lines on the slab edges
Y = np.where(np.abs(Y - y0) < 0.5 * (14 + half) / n, y0, np.where(np.abs(Y - y1) < 0.5 * (14 + half) / n, y1, Y))
SAND = tuple(float(c) for c in os.environ.get("SAND", "0.6 0.47 0.31").split())
mesh("/World/Ground", X, Y, Z, material("/World/Looks/Sand", SAND, 0.95), hole=True)

# the sea: a large glossy plane from the shoreline out to the horizon
WY = 7.5
sx, sy = np.meshgrid(np.linspace(-200, 200, 2) + length / 2, np.linspace(WY, 400, 2), indexing="ij")
mesh("/World/Sea", sx, sy, np.full_like(sx, depth - 0.15), material("/World/Looks/Sea", (0.03, 0.2, 0.26), 0.04))


def ground_z(x, y):
    i = np.clip(np.searchsorted(xs, x), 0, n - 1)
    j = np.clip(np.searchsorted(ys, y), 0, n - 1)
    return float(Z[i, j])


# palms and agaves along the back of the beach and on the camera side, clear of the path and the camera
PLANTS = [("Cuban_Royal_Palm", 0.008, 0.011, 3.5), ("Hurricane_Palm", 0.009, 0.012, 2.5), ("Fan_Palm", 0.012, 0.016, 0.9),
          ("Agave", 0.009, 0.014, 0.5)]
k = 0
for _ in range(2000):
    name, s0, s1, crown = PLANTS[k % len(PLANTS)]
    x = rng.uniform(-14, length + 14)
    y = rng.uniform(-26, 7.0) if name != "Agave" else rng.uniform(-12, 6.0)
    if (y > 0 and y < 2.2 + crown) or (y < 0 and -y < 7.5 + crown):
        continue
    k += 1
    xf = UsdGeom.Xform.Define(stage, f"/World/Plants/P{k:02d}")
    xf.GetPrim().GetReferences().AddReference(str(HERE / "tropical" / f"{name}.usd"))
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, ground_z(x, y) - 0.03))
    xf.AddRotateZOp().Set(float(rng.uniform(0, 360)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(rng.uniform(s0, s1))] * 3))
    if k == int(env("NPLANTS", 28)):
        break
stage.GetRootLayer().Save()
print(f"{out}: {k} plants")
