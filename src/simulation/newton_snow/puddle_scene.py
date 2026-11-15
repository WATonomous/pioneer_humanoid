"""Background USD for the puddle walk: a flooded trail through an autumn wood at golden hour (from autumn_scene.py). The
ground dips into a long puddle along the path (water box x0..x1, y0..y1, see puddle_walk.py); its banks rise above the
water level so they hide the box walls and make the shoreline. Evening HDRI dome and a low warm sun
(NVIDIA Omniverse Skies), autumn trees (NVIDIA Omniverse Vegetation: Japanese maple, scarlet oak, red ash, black oak),
a forest-floor ground, and static leaf litter (a point instancer of maple leaves) around the simulated strip.

  python puddle_scene.py <out.usda> <path length m> x0 x1 y0 y1
"""
import os
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

HERE = Path(__file__).resolve().parent / "assets"
out, length = sys.argv[1], float(sys.argv[2])
bx0, bx1, by0, by1 = [float(v) for v in sys.argv[3:7]]
env = lambda k, d: float(os.environ.get(k, d))  # noqa: E731
CHERRY = os.environ.get("THEME") == "cherry"  # spring: cherry trees in bloom, grass, petals, morning light
stage = Usd.Stage.CreateNew(out)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
rng = np.random.default_rng(5)

dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
dome.CreateTextureFileAttr(str(HERE / "sky" / ("signal_hill_sunrise.hdr" if CHERRY else "evening_road_01.hdr")))
dome.CreateTextureFormatAttr("latlong")
dome.CreateIntensityAttr(env("DOME", 1000))
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(env("SUN", 2500))
sun.CreateEnableColorTemperatureAttr(True)
sun.CreateColorTemperatureAttr(5200.0 if CHERRY else 3400.0)
sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(70, 0, 60))  # low, from behind-left of the side camera


def material(path, rgb, rough):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


# forest floor with the puddle: an outline wobbling along the path, banks ramping up to 12 cm over 30 cm
dense_x = np.arange(bx0 - 2.0, bx1 + 2.0, 0.025)
dense_y = np.arange(-3.0, 3.0, 0.025)
xs = np.unique(np.concatenate([np.linspace(-45, bx0 - 2.0, 40), dense_x, np.linspace(bx1 + 2.0, length + 45, 40)]))
ys = np.unique(np.concatenate([np.linspace(-50, -3.0, 40), dense_y, np.linspace(3.0, 50, 40)]))
X, Y = np.meshgrid(xs, ys, indexing="ij")
far = sum(a * np.sin(f * X + rng.uniform(0, 6)) * np.cos(f * 0.8 * Y + rng.uniform(0, 6)) for a, f in ((0.12, 0.3), (0.04, 1.1)))
far = np.where(np.abs(Y) < 3.0, far * (np.abs(Y) / 3.0) ** 2, far)
w = 0.7 + 0.15 * np.sin(1.1 * X + 0.5) + 0.08 * np.sin(2.7 * X + 2.0)  # puddle half-width
qx = np.maximum(bx0 + 0.25 - X, X - (bx1 - 0.25)) + 0.6  # outside distance from the outline, ends rounded (r 0.6 m)
qy = np.abs(Y) - w + 0.6
d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - 0.6
Z = np.clip(d / 0.3 + 1.0, 0.0, 1.0) ** 1.5 * 0.13 - 0.01 + far
pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
idx = np.arange(Z.size).reshape(Z.shape)
quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
ground.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
ground.CreateFaceVertexCountsAttr([4] * len(quads))
ground.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(quads.flatten().astype(np.int32)))
ground.CreateSubdivisionSchemeAttr("none")
UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material("/World/Looks/Soil", (0.085, 0.055, 0.032), 0.7))


def ground_z(x, y):
    i = np.clip(np.searchsorted(xs, x), 0, len(xs) - 1)
    j = np.clip(np.searchsorted(ys, y), 0, len(ys) - 1)
    return Z[i, j]


# static leaf litter everywhere but in the puddle
COLOURS = [(0.62, 0.09, 0.03), (0.78, 0.22, 0.04), (0.86, 0.42, 0.06), (0.84, 0.6, 0.1), (0.5, 0.26, 0.08), (0.36, 0.2, 0.08)]
inst = UsdGeom.PointInstancer.Define(stage, "/World/Litter")
protos = []
if CHERRY:
    COLOURS = [(0.95, 0.5, 0.66), (0.9, 0.42, 0.6), (0.98, 0.62, 0.75), (1.0, 0.72, 0.82), (0.86, 0.36, 0.55), (0.93, 0.48, 0.64)]
for k, c in enumerate(COLOURS):
    th = np.linspace(0, 2 * np.pi, 41)[:-1]
    if CHERRY:
        r = 0.0125 / np.sqrt(np.cos(th) ** 2 + (1.6 * np.sin(th)) ** 2)
    else:
        r = 0.045 * (0.42 + 0.58 * np.abs(np.cos(2.5 * (th - np.pi / 2))) ** 0.7) * (1 - 0.35 * np.clip(-np.sin(th), 0, 1) ** 3)
    v = np.concatenate([[[0, 0, 0]], np.stack([r * np.cos(th), r * np.sin(th), 1.2 * r ** 2], 1)]).astype(np.float32)
    tris = np.array([[0, 1 + i, 1 + (i + 1) % 40] for i in range(40)])
    m = UsdGeom.Mesh.Define(stage, f"/World/Litter/Protos/Leaf{k}")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    m.CreateFaceVertexCountsAttr([3] * len(tris))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.flatten().astype(np.int32)))
    m.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material(f"/World/Looks/Leaf{k}", c, 0.7))
    protos.append(m.GetPath())
inst.CreatePrototypesRel().SetTargets(protos)
N = int(env("LITTER", 60000))
lx = rng.uniform(-16, length + 16, N)
ly = rng.uniform(-14, 14, N)
keep = ground_z(lx, ly) > 0.1
lx, ly = lx[keep], ly[keep]
lz = ground_z(lx, ly) + rng.uniform(0.002, 0.03, len(lx))
inst.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(np.stack([lx, ly, lz], 1).astype(np.float32)))
yaw = rng.uniform(0, 2 * np.pi, len(lx))
inst.CreateOrientationsAttr([Gf.Quath(float(np.cos(a / 2)), Gf.Vec3h(0, 0, float(np.sin(a / 2)))) for a in yaw])
inst.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(np.repeat(rng.uniform(0.8, 1.25, (len(lx), 1)), 3, 1).astype(np.float32)))
inst.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(rng.integers(0, len(COLOURS), len(lx)).astype(np.int32)))

# autumn trees around the clearing (heights from the assets: maple 3.5 m, scarlet oak 8.7, red ash 6.0, black oak 16.3)
TREES = [("Japanese_Cherry", 0.012, 0.017)] if CHERRY else [("Japanese_Maple_Fall", 0.011, 0.016), ("Scarlet_Oak_fall", 0.009, 0.012), ("Red_Ash_Fall", 0.010, 0.014), ("Black_Oak_Fall", 0.005, 0.007)]
k = 0
for _ in range(500):
    x, y = rng.uniform(-12, length + 12), rng.uniform(-24, 24)
    name, s0, s1 = TREES[k % len(TREES)]
    crown = {"Japanese_Cherry": 3.5, "Japanese_Maple_Fall": 3.0, "Scarlet_Oak_fall": 4.5, "Red_Ash_Fall": 3.0, "Black_Oak_Fall": 7.0}[name]
    if (y > 0 and y < 2.0 + crown) or (y < 0 and -y < 8.0 + crown):  # canopies clear of the path and the cameras
        continue
    k += 1
    xf = UsdGeom.Xform.Define(stage, f"/World/Trees/T{k:02d}")
    xf.GetPrim().GetReferences().AddReference(str(HERE / "trees" / f"{name}.usd"))
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, float(ground_z(np.array([x]), np.array([y]))[0]) - 0.05))
    xf.AddRotateZOp().Set(float(rng.uniform(0, 360)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(rng.uniform(s0, s1))] * 3))
    if k == int(env("NTREES", 28)):
        break
stage.GetRootLayer().Save()
print(f"{out}: {k} trees, {len(lx)} litter leaves")
