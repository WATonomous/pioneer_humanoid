# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the Onshape stair-course *terrain* USD from a STEP export.

Single entry point for the whole CAD -> terrain pipeline (it replaces the old
scatter of convert_cad / scale_usd / inspect_onshape_usd / merge_course_usd
scripts, all of which have been removed):

  1. HOOPS CAD convert   <step>            -> <stem>.usd
  2. bake unit scale     HOOPS emits the metre-declared STEP 1000x too small,
                         so a x1000 ``xformOp:scale`` is baked on the root
                         prim (auto-detected; ``--unit-scale`` overrides).
  3. inspect             per-part world bounds -> <stem>_bounds.json, which
                         also reports the ``marker`` part centre used for spawn.
  4. merge               every part *except* the ``marker`` -> ONE triangle
                         mesh, <stem>_terrain.usd, with exact triangle-soup
                         collision.
  5. verify              re-open the merged file and print its bounds.

Why one mesh: the Isaac Lab RayCaster height scanner reads only the **first**
Mesh under its prim path, so the multi-part HOOPS output is unusable as terrain
directly. The merged output also carries the model's own ground plane (the
``ground`` part of the Part Studio), so the task needs no added collision floor.

Usage:
    isaaclab.sh -p tools/onshape_course/build_course.py \\
        --step assets/staircase_onshape/course.step

    # reuse an existing conversion (skip step 1):
    isaaclab.sh -p tools/onshape_course/build_course.py \\
        --step assets/staircase_onshape/course.step --skip-convert
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess

# --------------------------------------------------------------------------- #
# CLI (parsed before the app boots: step 1 runs as a plain subprocess)
# --------------------------------------------------------------------------- #

parser = argparse.ArgumentParser(description="Build the Onshape course terrain USD.")
parser.add_argument("--step", required=True, help="Input STEP export from Onshape.")
parser.add_argument("--isaac-root", default=os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"),
                    help="Isaac Lab root holding _isaac_sim/ (default: $ISAACLAB_PATH or /workspace/isaaclab).")
parser.add_argument("--skip-convert", action="store_true",
                    help="Reuse the existing <stem>.usd instead of re-running HOOPS.")
parser.add_argument("--unit-scale", type=float, default=0.0,
                    help="Uniform scale to bake on the root prim (0 = auto-detect).")
parser.add_argument("--marker", default="marker", help="Part name to exclude from the merge (spawn marker).")
parser.add_argument("--keep-parts", default="",
                    help="Comma-separated extra part-name substrings to exclude from the merge.")
args_cli = parser.parse_args()

_STEP = os.path.abspath(args_cli.step)
_DIR = os.path.dirname(_STEP)
_STEM = os.path.splitext(os.path.basename(_STEP))[0]
_RAW_USD = os.path.join(_DIR, _STEM + ".usd")
_BOUNDS_JSON = os.path.join(_DIR, _STEM + "_bounds.json")
_TERRAIN_USD = os.path.join(_DIR, _STEM + "_terrain.usd")
_HOOPS_CFG = os.path.join(_DIR, "hoops_config.json")

# HOOPS CAD converter options (written next to the STEP each run so the
# pipeline is self-contained). dMetersPerUnit + iTessLOD are the load-bearing
# two; the rest mirror the settings this project was validated with.
_HOOPS_OPTIONS = {
    "bInstancing": False,
    "bGlobalXforms": False,
    "bConvertHidden": True,
    "bOptimize": False,
    "bDedup": False,
    "iTessLOD": 2,
    "bAccurateSurfaceCurvatures": True,
    "bUseNormals": True,
    "bUseMaterials": False,
    "bReportProgress": True,
    "bConvertCurves": False,
    "bConvertMetadata": False,
    "iUpAxis": 0,
    "dMetersPerUnit": 1.0,
}

# Parts that must never end up in the terrain mesh: the spawn marker (a flat
# sliver with no collision value) and the legacy "Part9" name from an older
# export of the same course.
_SKIP = [_ for _ in ["marker", "Part9"] + args_cli.keep_parts.split(",") if _]


# --------------------------------------------------------------------------- #
# Step 1 - HOOPS CAD conversion (separate kit process, no app in *this* one)
# --------------------------------------------------------------------------- #

def run_hoops() -> None:
    """Convert STEP -> USD with the HOOPS CAD converter app."""
    hoops_dirs = glob.glob(os.path.join(args_cli.isaac_root, "_isaac_sim/extscache/omni.kit.converter.hoops-*/"))
    if not hoops_dirs:
        raise RuntimeError(
            f"No omni.kit.converter.hoops extension under {args_cli.isaac_root}/_isaac_sim/extscache. "
            "Pass --isaac-root."
        )
    launcher = os.path.join(hoops_dirs[0], "omni/kit/converter/hoops/process/launch_hoops_app.py")
    if not os.path.isfile(launcher):
        raise RuntimeError(f"HOOPS launcher missing: {launcher}")

    with open(_HOOPS_CFG, "w") as handle:
        json.dump(_HOOPS_OPTIONS, handle, indent=4)

    # The HOOPS app is driven through kit's --exec, which takes the whole
    # launcher invocation as ONE argument, hence the embedded quotes.
    exec_arg = (
        f"'{launcher}' --config-path '{_HOOPS_CFG}' "
        f"--input-path '{_STEP}' --output-path '{_RAW_USD}'"
    )
    cmd = [
        os.path.join(args_cli.isaac_root, "_isaac_sim/kit/kit"),
        "--ext-folder", os.path.join(args_cli.isaac_root, "_isaac_sim/exts"),
        "--ext-folder", os.path.join(args_cli.isaac_root, "_isaac_sim/extscache"),
        "--enable", "omni.kit.converter.hoops_core",
        "--exec", exec_arg,
        "--info",
    ]
    # Kit refuses to run as root unless told otherwise. The parent isaaclab.sh
    # session only exports this once AppLauncher boots, which is *after* us, so
    # set it explicitly for the child.
    env = dict(os.environ)
    env["OMNI_KIT_ALLOW_ROOT"] = "1"

    print(f"[BUILD] step 1/5 HOOPS convert -> {_RAW_USD}", flush=True)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0 or not os.path.isfile(_RAW_USD):
        raise RuntimeError(f"HOOPS conversion failed (rc={result.returncode}, output exists={os.path.isfile(_RAW_USD)})")
    print("[BUILD] step 1/5 done", flush=True)


def main() -> None:
    # Step 1 first, deliberately: the HOOPS app is its own kit process, so it
    # must not run while this process holds a Kit app (GPU + USD stage clash).
    if not args_cli.skip_convert:
        run_hoops()

    # pxr is only importable once the Kit app has booted.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app
    try:
        import numpy as np
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        def mesh_world_points(prim) -> np.ndarray:
            """Mesh points in world space (bakes nested xforms, incl. the unit scale)."""
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            xform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            # USD column-vector convention: world = R @ local + t, t in column 3.
            rot = np.array([[xform[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)
            trn = np.array([xform[r][3] for r in range(3)], dtype=np.float64)
            return pts @ rot.T + trn

        def iter_meshes(stage):
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Mesh) or prim.GetTypeName() == "Mesh":
                    yield prim

        def stage_extent(stage) -> tuple:
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            for prim in iter_meshes(stage):
                pts = mesh_world_points(prim)
                if len(pts):
                    lo = np.minimum(lo, pts.min(axis=0))
                    hi = np.maximum(hi, pts.max(axis=0))
            return lo, hi

        # --- step 2: bake unit scale -------------------------------------- #
        def bake_scale() -> None:
            stage = Usd.Stage.Open(_RAW_USD)
            root = stage.GetDefaultPrim()
            if not root or not root.IsValid():
                raise RuntimeError(f"No default prim in {_RAW_USD}")

            # Idempotent: re-running against an already-fixed file (the usual
            # --skip-convert case) must not scale it a second time.
            xform = UsdGeom.Xformable(root)
            op = next((o for o in xform.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeScale), None)
            existing = float(op.Get()[0]) if op is not None else 1.0

            lo, hi = stage_extent(stage)
            extent = float(np.max(hi - lo))
            if args_cli.unit_scale > 0.0:
                scale, reason = args_cli.unit_scale, "explicit --unit-scale"
            elif extent < 1.0:
                # A real course is tens of metres; HOOPS emitting mm-as-m lands at ~1e-3.
                scale, reason = 1000.0, f"auto: raw extent {extent:.6g} m looks mm-as-m"
            elif op is not None:
                scale, reason = existing, f"auto: extent {extent:.6g} m already metric"
            else:
                scale, reason = 1.0, f"auto: extent {extent:.6g} m already metric"

            if op is None or scale != existing:
                print(f"[BUILD] step 2/5 baked scale x{scale:g} on {root.GetPath()} ({reason})", flush=True)
                if op is None:
                    op = xform.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")
                op.Set(Gf.Vec3d(scale, scale, scale))
                stage.GetRootLayer().Save()
            else:
                print(f"[BUILD] step 2/5 scale x{scale:g} already baked on {root.GetPath()} ({reason})", flush=True)

            lo, hi = stage_extent(stage)
            print(f"[BUILD] step 2/5 world bounds min={np.round(lo, 4).tolist()} "
                  f"max={np.round(hi, 4).tolist()}", flush=True)

        # --- step 3: inspect parts + locate the spawn marker -------------- #
        def inspect_parts() -> None:
            stage = Usd.Stage.Open(_RAW_USD)
            parts = []
            for prim in iter_meshes(stage):
                pts = mesh_world_points(prim)
                bmin, bmax = pts.min(axis=0), pts.max(axis=0)
                # Label from the tn__PartN_f5 ancestor when present, else the prim name.
                label = prim.GetName()
                parent = prim.GetParent()
                while parent and parent.GetPath() != "/":
                    if parent.GetName().startswith("tn__"):
                        label = parent.GetName()
                        break
                    parent = parent.GetParent()
                parts.append({
                    "name": prim.GetName(),
                    "label": label,
                    "path": str(prim.GetPath()),
                    "min": [float(v) for v in bmin],
                    "max": [float(v) for v in bmax],
                    "size": [float(v) for v in (bmax - bmin)],
                })
            with open(_BOUNDS_JSON, "w") as handle:
                json.dump(parts, handle, indent=2)
            print(f"[BUILD] step 3/5 inspected {len(parts)} meshes -> {_BOUNDS_JSON}", flush=True)
            for part in parts:
                print(f"[BUILD]   {part['label']:<20} y {part['min'][1]:9.3f}..{part['max'][1]:9.3f} "
                      f"z {part['min'][2]:6.3f}..{part['max'][2]:6.3f}", flush=True)

            marker = next((p for p in parts if args_cli.marker.lower() in p["path"].lower()), None)
            if marker is None:
                raise RuntimeError(f"No '{args_cli.marker}' part in {_RAW_USD}; cannot place the spawn.")
            centre = [(marker["min"][i] + marker["max"][i]) / 2 for i in range(3)]
            print(f"[BUILD] MARKER centre = x={centre[0]:.4f} y={centre[1]:.4f} "
                  f"z={marker['min'][2]:.4f} (size {np.round(marker['size'], 4).tolist()})", flush=True)

        # --- step 4: merge everything except the marker ------------------- #
        def merge_parts() -> None:
            stage = Usd.Stage.Open(_RAW_USD)
            all_pts, tris, offset, n_parts = [], [], 0, 0
            for prim in iter_meshes(stage):
                if any(skip.lower() in str(prim.GetPath()).lower() for skip in _SKIP):
                    print(f"[BUILD]   skipping {prim.GetPath()}", flush=True)
                    continue
                pts = mesh_world_points(prim)
                counts = np.asarray(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get(), dtype=np.int64)
                indices = np.asarray(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
                start = 0
                for count in counts:
                    fan = indices[start:start + count]
                    for k in range(1, len(fan) - 1):
                        tris.append((fan[0] + offset, fan[k] + offset, fan[k + 1] + offset))
                    start += count
                all_pts.append(pts)
                offset += len(pts)
                n_parts += 1

            verts = np.vstack(all_pts).astype(np.float32)
            tri = np.asarray(tris, dtype=np.int32)
            print(f"[BUILD] step 4/5 merged {n_parts} parts: {len(verts)} verts, {len(tri)} tris", flush=True)

            # Averaged vertex normals from the authored winding (CAD parts carry
            # no meaningful materials to light from).
            normals = np.zeros_like(verts)
            v0, v1, v2 = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
            face_n = np.cross(v1 - v0, v2 - v0)
            lens = np.linalg.norm(face_n, axis=1, keepdims=True)
            lens[lens == 0.0] = 1.0
            face_n /= lens
            for k in range(3):
                np.add.at(normals, tri[:, k], face_n)
            lens = np.linalg.norm(normals, axis=1, keepdims=True)
            lens[lens == 0.0] = 1.0
            normals /= lens

            out = Usd.Stage.CreateNew(_TERRAIN_USD)
            UsdGeom.SetStageUpAxis(out, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(out, 1.0)
            # Xform root + Mesh child named "mesh": the terrain importer
            # references the default prim while the RayCaster resolves the first
            # Mesh child.
            root = out.DefinePrim("/course", "Xform")
            out.SetDefaultPrim(root)
            mesh_prim = out.DefinePrim("/course/mesh", "Mesh")
            mesh = UsdGeom.Mesh(mesh_prim)
            mesh.CreatePointsAttr(verts)
            mesh.CreateFaceVertexCountsAttr(np.full(len(tri), 3, dtype=np.int32))
            mesh.CreateFaceVertexIndicesAttr(tri.reshape(-1))
            mesh.CreateNormalsAttr(normals)
            mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            bmin, bmax = verts.min(axis=0), verts.max(axis=0)
            mesh.CreateExtentAttr([tuple(float(v) for v in bmin), tuple(float(v) for v in bmax)])
            # Collision recipe mirrored from Isaac Lab's create_prim_from_mesh:
            # the CollisionAPI plus an (empty) PhysX collision API, no
            # approximation override, i.e. exact triangle-soup at cook time.
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
            UsdPhysics.CollisionAPI(mesh_prim).CreateCollisionEnabledAttr(True)
            PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
            out.GetRootLayer().Save()
            print(f"[BUILD] step 4/5 wrote {_TERRAIN_USD} "
                  f"min={np.round(bmin, 4).tolist()} max={np.round(bmax, 4).tolist()}", flush=True)

        # --- step 5: verify ----------------------------------------------- #
        def verify() -> None:
            stage = Usd.Stage.Open(_TERRAIN_USD)
            prim = stage.GetPrimAtPath("/course/mesh")
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            normals = np.asarray(mesh.GetNormalsAttr().Get(), dtype=np.float64)
            print(f"[BUILD] step 5/5 verify: valid={prim.IsValid()} verts={len(pts)} "
                  f"collision={bool(UsdPhysics.CollisionAPI(prim))} "
                  f"normals_finite={bool(np.isfinite(normals).all())} "
                  f"min={np.round(pts.min(axis=0), 4).tolist()} "
                  f"max={np.round(pts.max(axis=0), 4).tolist()}", flush=True)

            # Height profile along the course axis (+Y): max z per bin. This is
            # what the task's goal_distance is set from -- read the summit and
            # the shape (ascend/descend per pyramid) straight off the log.
            y, z = pts[:, 1], pts[:, 2]
            edges = np.linspace(y.min(), y.max(), 61)
            profile = np.full(60, np.nan)
            bins = np.clip(np.digitize(y, edges) - 1, 0, 59)
            for k in range(60):
                sel = z[bins == k]
                if len(sel):
                    profile[k] = sel.max()
            peak = float(z.max())
            summit = int(np.argmax(z))
            print(f"[BUILD] PROFILE y->max z (summit y={y[summit]:.3f} z={peak:.3f})", flush=True)
            for k in range(60):
                centre = 0.5 * (edges[k] + edges[k + 1])
                if np.isnan(profile[k]):
                    print(f"[BUILD]   y={centre:7.2f}     -  ", flush=True)
                else:
                    bar = "#" * max(1, int(round(profile[k] / peak * 44)))
                    print(f"[BUILD]   y={centre:7.2f} z={profile[k]:5.3f} {bar}", flush=True)

        bake_scale()
        inspect_parts()
        merge_parts()
        verify()
        print("[BUILD] done", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
