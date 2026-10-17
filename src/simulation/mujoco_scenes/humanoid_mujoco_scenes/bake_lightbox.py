"""Bake ``assets/props/lightbox.usd`` (the Isaac workcell's CAD lightbox) into STL meshes for MuJoCo.

    pip install usd-core
    python -m humanoid_mujoco_scenes.bake_lightbox

Writes one binary STL per CAD material to ``assets/props/lightbox_mujoco/`` (colours in ``colors.json``), in the
USD's own frame: ``workcell.py`` applies the same pose and scale as Isaac's ``LightboxWorkcellCfg``. Only
needed again when the USD changes; the scenes load the committed STLs, so MuJoCo itself never needs pxr.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
USD = REPO / "assets" / "props" / "lightbox.usd"
OUT = REPO / "assets" / "props" / "lightbox_mujoco"
# CAD material -> mesh name; diffuse colours come from the USD's preview shaders.
PARTS = {"Diffuse": "brackets", "Diffuse_1": "frame", "Diffuse_2": "panels", "Diffuse_3": "table"}


def main() -> None:
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(USD))
    xf = UsdGeom.XformCache()
    verts = {p: [] for p in PARTS.values()}
    faces = {p: [] for p in PARTS.values()}
    colors = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        part = PARTS[material.GetPath().name]
        shader = UsdShade.Shader(stage.GetPrimAtPath(material.GetPath().AppendChild("PreviewSurface")))
        colors[part] = [round(float(c), 4) for c in shader.GetInput("diffuseColor").Get()]

        m = np.array(xf.GetLocalToWorldTransform(prim))           # row-vector convention: p @ m
        pts = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        pts = (np.c_[pts, np.ones(len(pts))] @ m)[:, :3]
        base = len(verts[part])
        verts[part].extend(pts)
        counts, idx = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        flip = np.linalg.det(m[:3, :3]) < 0                      # mirrored parts: keep faces outward
        k = 0
        for n in counts:
            poly = [base + i for i in idx[k:k + n]]
            for j in range(1, n - 1):                             # fan-triangulate
                tri = (poly[0], poly[j], poly[j + 1])
                faces[part].append(tri[::-1] if flip else tri)
            k += n

    OUT.mkdir(parents=True, exist_ok=True)
    for part in PARTS.values():
        tris = np.asarray(verts[part], dtype="<f4")[np.asarray(faces[part])]        # (n, 3 corners, xyz)
        rec = np.zeros(len(tris), dtype=[("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        rec["v"] = tris                                                           # MuJoCo computes its own normals
        header = f"{USD.name}: {part}, metres, USD frame".encode()[:80].ljust(80, b" ")
        (OUT / f"{part}.stl").write_bytes(header + struct.pack("<I", len(rec)) + rec.tobytes())
        print(f"{part}: {len(tris)} triangles")
    (OUT / "colors.json").write_text(json.dumps(colors, indent=2) + "\n")


if __name__ == "__main__":
    main()
