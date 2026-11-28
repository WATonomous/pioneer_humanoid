"""Background USD for the granite-gravel walk (from beach_scene.py): a wide crushed-granite field in late-afternoon sun,
spruces around it, static granite pebbles scattered on the ground outside the simulated slab so the two match.

  python granite_scene.py <out.usda> <path length m> x0 x1 y0 y1 depth
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
ys = np.linspace(-half, half, n)
X, Y = np.meshgrid(xs, ys, indexing="ij")
Z = sum(a * np.sin(f * X + rng.uniform(0, 6)) * np.cos(f * 0.7 * Y + rng.uniform(0, 6)) for a, f in ((0.25, 0.25), (0.06, 0.9), (0.012, 3.0)))
Z = np.where(np.abs(Y) < 3.0, Z * (np.abs(Y) / 3.0) ** 2, Z) + depth
X = np.where(np.abs(X - x0) < half / n, x0, np.where(np.abs(X - x1) < half / n, x1, X))  # grid lines on the slab edges
Y = np.where(np.abs(Y - y0) < half / n, y0, np.where(np.abs(Y - y1) < half / n, y1, Y))
SAND = tuple(float(c) for c in os.environ.get("SAND", "0.3 0.28 0.26").split())
mesh("/World/Ground", X, Y, Z, material("/World/Looks/Granite", SAND, 0.95), hole=True)

def ground_z(x, y):
    """Bilinear height of the ground mesh (nearest-node sampling left visible terraces in the pebbles)."""
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    i = np.clip(np.searchsorted(xs, x) - 1, 0, n - 2)
    j = np.clip(np.searchsorted(ys, y) - 1, 0, n - 2)
    u = np.clip((x - xs[i]) / (xs[i + 1] - xs[i]), 0, 1)
    v = np.clip((y - ys[j]) / (ys[j + 1] - ys[j]), 0, 1)
    z = (1 - u) * (1 - v) * Z[i, j] + u * (1 - v) * Z[i + 1, j] + (1 - u) * v * Z[i, j + 1] + u * v * Z[i + 1, j + 1]
    return z if z.size > 1 else float(z[0])


# granite pebbles on the ground around the slab (a point instancer of grain-sized spheres, granite colours)
GRANITE = [(0.3, 0.29, 0.28), (0.42, 0.4, 0.38), (0.48, 0.34, 0.3), (0.09, 0.09, 0.09), (0.62, 0.6, 0.57)]
inst = UsdGeom.PointInstancer.Define(stage, "/World/Pebbles")
protos = []
for k, c in enumerate(GRANITE):
    sp = UsdGeom.Sphere.Define(stage, f"/World/Pebbles/Protos/P{k}")
    sp.CreateRadiusAttr(1.0)
    UsdShade.MaterialBindingAPI.Apply(sp.GetPrim()).Bind(material(f"/World/Looks/Pebble{k}", c, 0.6))
    protos.append(sp.GetPath())
inst.CreatePrototypesRel().SetTargets(protos)
N = int(env("PEBBLES", 1500000))
px, py = rng.uniform(x0 - 7, x1 + 8, N), rng.uniform(y0 - 6, y1 + 9, N)
keep = ~((px > x0 + 0.15) & (px < x1 - 0.15) & (py > y0 + 0.15) & (py < y1 - 0.15))  # overlap the slab's pinned rim
px, py = px[keep], py[keep]
R = float(env("GRAIN", 0.018))
pz = ground_z(px, py) + rng.uniform(-0.3, 0.6, len(px)) * R
inst.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(np.stack([px, py, pz], 1).astype(np.float32)))
inst.CreateScalesAttr(Vt.Vec3fArray.FromNumpy((R * rng.uniform(0.7, 1.25, (len(px), 1)) * [1.0, 0.85, 0.7]).astype(np.float32)))
inst.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(rng.choice(5, len(px), p=[0.35, 0.25, 0.2, 0.1, 0.1]).astype(np.int32)))

# spruces around the field, clear of the path and the cameras
k = 0
for _ in range(2000):
    x, y = rng.uniform(-14, length + 14), rng.uniform(-24, 24)
    if abs(y) < 5.0 or (y < 0 and -y < 9.0):
        continue
    k += 1
    xf = UsdGeom.Xform.Define(stage, f"/World/Trees/T{k:02d}")
    xf.GetPrim().GetReferences().AddReference(str(HERE / "trees" / "Norway_Spruce.usd"))
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, ground_z(x, y) - 0.05))
    xf.AddRotateZOp().Set(float(rng.uniform(0, 360)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(rng.uniform(0.0042, 0.0068))] * 3))
    if k == int(env("NTREES", 26)):
        break
stage.GetRootLayer().Save()
print(f"{out}: {k} spruces, {len(px)} pebbles")
