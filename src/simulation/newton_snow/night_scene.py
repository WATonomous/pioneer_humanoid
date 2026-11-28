"""Background USD for the path-traced snow walk: a snowy clearing at night. Moonlit-night HDRI dome (NVIDIA Omniverse
Skies), Norway spruces (NVIDIA Omniverse Vegetation) around the clearing, a snow-covered ground, and one warm lamp on a
post beside the path.

  python night_scene.py <out.usda> <path length m> [x0 x1 y0 y1 depth: the simulated snow slab, cut out of the ground]
"""
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

HERE = Path(__file__).resolve().parent / "assets"
out, length = sys.argv[1], float(sys.argv[2])
slab = [float(v) for v in sys.argv[3:8]] if len(sys.argv) >= 8 else None
stage = Usd.Stage.CreateNew(out)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

# moonlit sky: dim, cool
dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
dome.CreateTextureFileAttr(str(HERE / "sky" / "moonlit_golf.hdr"))
dome.CreateTextureFormatAttr("latlong")
dome.CreateIntensityAttr(float(__import__("os").environ.get("DOME", 300)))
dome.CreateColorAttr(Gf.Vec3f(0.75, 0.85, 1.0))
# moonlight: a faint cold key from high up
moon = UsdLux.DistantLight.Define(stage, "/World/Moon")
moon.CreateIntensityAttr(float(__import__("os").environ.get("MOON", 300)))
moon.CreateColorAttr(Gf.Vec3f(0.7, 0.8, 1.0))
moon.CreateAngleAttr(0.6)
UsdGeom.Xformable(moon).AddRotateXYZOp().Set(Gf.Vec3f(35, 0, 30))


def material(path, rgb, rough):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


# snow-covered ground: a large gently undulating mesh, just below the simulated snow's base
n, half = 160, 60.0
xs = np.linspace(-half, half, n)
X, Y = np.meshgrid(xs + length / 2, xs, indexing="ij")
rng = np.random.default_rng(1)
Z = sum(a * np.sin(f * X + rng.uniform(0, 6)) * np.cos(f * 0.8 * Y + rng.uniform(0, 6)) for a, f in ((0.06, 0.35), (0.03, 0.9), (0.012, 2.3)))
Z = np.where(np.abs(Y) < 2.5, Z * (np.abs(Y) / 2.5) ** 2, Z) - 0.002  # flat along the path
if slab:  # level with the simulated snow's surface, with a hole where the simulated snow is
    x0, x1, y0, y1, depth = slab
    Z = Z + depth
    # put grid lines exactly on the slab edges so the hole matches it
    X = np.where(np.abs(X - x0) < half / n, x0, np.where(np.abs(X - x1) < half / n, x1, X))
    Y = np.where(np.abs(Y - y0) < half / n, y0, np.where(np.abs(Y - y1) < half / n, y1, Y))
pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
idx = np.arange(n * n).reshape(n, n)
quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
if slab:
    c = pts[quads].mean(1)
    quads = quads[~((c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1))]
ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
ground.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
ground.CreateFaceVertexCountsAttr([4] * len(quads))
ground.CreateFaceVertexIndicesAttr([int(i) for i in quads.flatten()])
ground.CreateSubdivisionSchemeAttr("none")
UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material("/World/Looks/Snow", (0.88, 0.9, 0.94), 0.75))

# spruces around the clearing, none on the path or in front of the cameras
tree = str(HERE / "trees" / "Norway_Spruce.usd")
k = 0
for _ in range(400):
    x, y = rng.uniform(-14, length + 14), rng.uniform(-22, 22)
    if abs(y) < 4.5 or (y < 0 and abs(y) < 7.5):  # keep the camera side (-y) more open
        continue
    k += 1
    xf = UsdGeom.Xform.Define(stage, f"/World/Trees/Spruce_{k:02d}")
    xf.GetPrim().GetReferences().AddReference(tree)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, -0.05))
    xf.AddRotateZOp().Set(float(rng.uniform(0, 360)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(rng.uniform(0.0042, 0.0068))] * 3))
    if k == int(__import__("os").environ.get("NTREES", 34)):
        break

# a lamp post on the far side of the path, warm light pooling on the snow where the robot walks
lx = length * 0.55
post = UsdGeom.Cylinder.Define(stage, "/World/Lamp/Post")
post.CreateRadiusAttr(0.05); post.CreateHeightAttr(3.2); post.CreateAxisAttr("Z")
post.AddTranslateOp().Set(Gf.Vec3d(lx, 2.2, 1.6))
arm = UsdGeom.Cube.Define(stage, "/World/Lamp/Arm")
arm.CreateSizeAttr(1.0); arm.AddTranslateOp().Set(Gf.Vec3d(lx, 1.85, 3.18)); arm.AddScaleOp().Set(Gf.Vec3f(0.05, 0.7, 0.05))
dark = material("/World/Looks/Iron", (0.05, 0.05, 0.055), 0.5)
for p in ("Post", "Arm"):
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"/World/Lamp/{p}")).Bind(dark)
bulb = UsdLux.SphereLight.Define(stage, "/World/Lamp/Bulb")
bulb.CreateRadiusAttr(0.08)
bulb.CreateIntensityAttr(float(__import__("os").environ.get("LAMP", 200000)))
bulb.CreateEnableColorTemperatureAttr(True)
bulb.CreateColorTemperatureAttr(2900.0)
bulb.AddTranslateOp().Set(Gf.Vec3d(lx, 1.5, 3.05))
stage.GetRootLayer().Save()
print(f"{out}: {k} spruces, lamp at x {lx:.1f}")
