"""Convert NASA's CTB mesh, scale, transforms and inertia to a self-contained rigid USD.

Run: python assets/nasa_cargo/build_usd.py (requires numpy and USD Python bindings).
The source meshes are inches; URDF mesh scales convert them to metres.
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
CARGO_MASS_KG = 0.1  # Lightweight training prop to reduce hardware payload demands.


def stl_points(path):
    data = path.read_bytes()
    count = int.from_bytes(data[80:84], "little")
    if len(data) != 84 + 50 * count:
        raise ValueError(f"Not a resolved binary STL: {path}")
    triangles = np.frombuffer(data, dtype=[("normal", "<f4", 3), ("vertices", "<f4", (3, 3)),
                                         ("attr", "<u2")], offset=84)
    return np.unique(triangles["vertices"].reshape(-1, 3), axis=0, return_inverse=True)


def build(output=ROOT / "ctb.usdc"):
    link = ET.parse(SOURCE / "ctb/urdf/ctb_macro.xacro").find(".//link")
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Cargo")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    source_mass = float(link.find("inertial/mass").get("value"))
    mass.CreateMassAttr(CARGO_MASS_KG)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    inertia = link.find("inertial/inertia")
    # Same geometry and mass distribution, with uniformly reduced density.
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*(
        float(inertia.get(a)) * CARGO_MASS_KG / source_mass for a in ("ixx", "iyy", "izz"))))
    stage.GetRootLayer().customLayerData = {
        "source": "NASA-JSC-Robotics/surface_robotics_mockups_description",
        "revision": "934f1129085450e99ac6c1cf3e7f9b20ad076469",
        "adaptation": f"Rigid CTB; mass reduced from {source_mass} to {CARGO_MASS_KG} kg with proportional inertia; metre-converted geometry",
    }
    contact_mat = UsdShade.Material.Define(stage, "/Cargo/Materials/Contact")
    contact = UsdPhysics.MaterialAPI.Apply(contact_mat.GetPrim())
    contact.CreateStaticFrictionAttr(1.5)
    contact.CreateDynamicFrictionAttr(1.2)
    contact.CreateRestitutionAttr(0.0)
    for i, visual in enumerate(link.findall("visual")):
        name = "body" if i == 0 else "handle"
        geometry = visual.find("geometry/mesh")
        points, indices = stl_points(SOURCE / geometry.get("filename").removeprefix("package://"))
        points = points.astype(np.float64) * np.array([float(v) for v in geometry.get("scale").split()])
        origin = visual.find("origin")
        rpy = [float(v) for v in origin.get("rpy").split()]
        # The source defines roll only, +/-1.57 rad. Bake the exact source transform.
        assert rpy[1:] == [0.0, 0.0]
        c, s = math.cos(rpy[0]), math.sin(rpy[0])
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        points = (points @ rot.T + np.array([float(v) for v in origin.get("xyz").split()])).astype(np.float32)
        mat = UsdShade.Material.Define(stage, f"/Cargo/Materials/{name}")
        shader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        rgba = visual.find("material/color").get("rgba").split()
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(float(v) for v in rgba[:3])))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        for collision in (False, True):
            mesh = UsdGeom.Mesh.Define(stage, f"/Cargo/{name}_{'collision' if collision else 'visual'}")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(indices) // 3)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices.astype(np.int32)))
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0).tolist()), Gf.Vec3f(*points.max(0).tolist())])
            if collision:
                mesh.CreateVisibilityAttr("invisible")
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
                    "convexHull" if name == "body" else "convexDecomposition")
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(contact_mat, materialPurpose="physics")
            else:
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        print(f"{name}: bounds {points.min(0)} .. {points.max(0)} m")
    stage.GetRootLayer().Save()
    return output


if __name__ == "__main__":
    print(build())
