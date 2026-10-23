"""Extract a spiral staircase's geometry from the merged Onshape USD.

A spiral staircase is a helix of wedge-shaped treads: every tread is the same
shape, rotated a fixed angle and lifted by a fixed riser. That makes the
staircase a one-parameter family, and the task's *path-relative* rewards
(progress, facing, out-of-bounds, the clearance band) all need exactly those
parameters:

    centre             the axis of the helix (x, y)
    walking_radius     the radius the robot walks at (the spawn marker's)
    riser              the height of one step
    angular_step       the angle from one tread to the next (radians)
    treads_per_turn    2*pi / angular_step
    start_angle        the angle of the spawn marker
    first_tread_angle  the angle of the first (lowest) tread
    z_top              the height of the top tread
    band               the swing clearance each height demands

The centre is taken from the *floor*: the user modelled a square floor around
the staircase, so the floor's centroid is the helix axis. The treads' angles
are measured about that centre, using only the *outer* part of each wedge
(the inner tip sits close to the axis, where a small position error swings its
angle wildly).

    isaaclab.sh -p tools/onshape_course/spiral_profile.py \\
        --usd assets/staircase_onshape/spiral_terrain.usd \\
        --raw assets/staircase_onshape/spiral.usd \\
        --out assets/staircase_onshape/spiral_geometry.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

parser = argparse.ArgumentParser(description="Extract a spiral staircase's geometry from its USD.")
parser.add_argument("--usd", required=True, help="Merged terrain USD (the /course/mesh prim).")
parser.add_argument("--raw", default="", help="Raw (unmerged) USD, used to find the spawn marker.")
parser.add_argument("--out", required=True, help="Output JSON.")
parser.add_argument("--log", default="/tmp/spiral_profile.log", help="Log file (Kit swallows stdout).")
parser.add_argument("--band-margin", type=float, default=0.03, help="Foot-to-foot margin the term subtracts.")
parser.add_argument("--band-min", type=float, default=0.06, help="Band floor (a natural flat step).")
args_cli = parser.parse_args()

_LOG = None


def log(msg: str) -> None:
    if _LOG is not None:
        with open(_LOG, "a") as handle:
            handle.write(msg + "\n")
            handle.flush()
    print(msg, flush=True)


def _marker_from_raw(raw_usd: str):
    """The spawn marker's (x, y): the small flat sliver in the raw export."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(raw_usd)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) and prim.GetTypeName() != "Mesh":
            continue
        pts = _world_points(prim)
        if len(pts) == 0:
            continue
        extent = pts.max(axis=0) - pts.min(axis=0)
        # The marker is the small, flat piece: tiny in x/y and thin in z.
        if extent[2] < 0.05 and extent[0] < 0.5 and extent[1] < 0.5:
            return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    return None


def _world_points(prim):
    import numpy as np
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    if len(pts) == 0:
        return pts
    xform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    rot = np.array([[xform[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)
    trn = np.array([xform[r][3] for r in range(3)], dtype=np.float64)
    return pts @ rot.T + trn


def main() -> None:
    global _LOG
    _LOG = args_cli.log
    open(_LOG, "w").close()
    log("[SPIRAL] starting")
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app
    try:
        import numpy as np
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(args_cli.usd)
        prim = stage.GetPrimAtPath("/course/mesh")
        if not prim or not prim.IsValid():
            raise RuntimeError(f"/course/mesh not found in {args_cli.usd}")
        pts = _world_points(prim)
        counts = np.asarray(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get(), dtype=np.int64)

        tris = []
        offset = 0
        for count in counts:
            face = indices[offset : offset + count]
            offset += count
            for k in range(1, count - 1):
                tris.append((face[0], face[k], face[k + 1]))
        tris = np.asarray(tris)
        p = pts[tris]
        normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

        # --- the helix axis --------------------------------------------------
        # The floor's centroid used to be taken as the axis, which works only
        # while the floor is a square centred on the staircase. It is not
        # (the newer export's floor centroid is 14 cm off), so the axis is
        # fitted from the treads' *outer edge* instead: those points lie on a
        # circle about the true axis, so a least-squares circle fit through
        # them gives the centre, and it is iterated twice (the outer edge is
        # re-picked from the new centre each time).
        treads_all = p[normals[:, 2] > 0.85].reshape(-1, 3)
        treads_all = treads_all[treads_all[:, 2] > 0.06]
        floor = p[normals[:, 2] > 0.85].reshape(-1, 3)
        floor = floor[floor[:, 2] < 0.06]
        if len(floor) < 10:
            raise RuntimeError("no floor found -- the spiral's centre cannot be located")
        centre = np.array([floor[:, 0].mean(), floor[:, 1].mean()])
        log(f"[SPIRAL] floor centroid (the starting guess) = ({centre[0]:.4f}, {centre[1]:.4f})")
        for _ in range(3):
            radius = np.hypot(treads_all[:, 0] - centre[0], treads_all[:, 1] - centre[1])
            edge = treads_all[radius >= np.percentile(radius, 92.0)]
            # Least-squares circle fit (Kasa) about the current guess.
            ex = edge[:, 0] - centre[0]
            ey = edge[:, 1] - centre[1]
            a = np.column_stack([ex, ey, np.ones(len(ex))])
            b = 0.5 * (ex**2 + ey**2)
            sol, *_ = np.linalg.lstsq(a, b, rcond=None)
            centre = centre + sol[:2]
            log(f"[SPIRAL] axis fit -> ({centre[0]:.4f}, {centre[1]:.4f}) "
                f"from {len(edge)} outer-edge points")

        # --- the treads ------------------------------------------------------
        treads = p[normals[:, 2] > 0.85].reshape(-1, 3)
        treads = treads[treads[:, 2] > 0.06]
        x, y, z = treads[:, 0], treads[:, 1], treads[:, 2]
        radius = np.hypot(x - centre[0], y - centre[1])
        angle = np.arctan2(y - centre[1], x - centre[0])
        r_outer = float(radius.max())
        log(f"[SPIRAL] tread radius range {radius.min():.3f} .. {r_outer:.3f} m")

        levels = []
        for value in np.unique(np.round(z, 3)):
            if not levels or value - levels[-1][-1] > 0.03:
                levels.append([value])
            else:
                levels[-1].append(value)
        level_z = np.array([float(np.mean(lev)) for lev in levels])
        riser = float(np.median(np.diff(level_z))) if len(level_z) > 1 else 0.0
        log(f"[SPIRAL] {len(level_z)} tread levels, riser {riser:.4f} m, z {level_z[0]:.3f} .. {level_z[-1]:.3f}")

        # The angle of each tread, from its OUTER part only (the inner tip is
        # ill-conditioned near the axis).
        mid_angles = []
        for lev in levels:
            value = float(np.mean(lev))
            sel = np.abs(z - value) < 0.012
            if sel.sum() < 12:
                continue
            outer = sel & (radius > 0.6 * r_outer)
            if outer.sum() < 8:
                outer = sel
            angles = np.sort(angle[outer])
            mid = float(np.median(angles))
            # Unwrap to the previous tread's neighbourhood (the staircase winds
            # one way; the step is well under pi).
            if mid_angles:
                prev = mid_angles[-1]
                while mid - prev > math.pi:
                    mid -= 2 * math.pi
                while mid - prev < -math.pi:
                    mid += 2 * math.pi
            mid_angles.append(mid)
        steps = np.diff(np.array(mid_angles))
        # Drop the levels whose sector was mis-detected (a big jump or a
        # backwards step) and use the median of what is left.
        good = steps[(steps > 0.05) & (steps < 1.2)]
        angular_step = float(np.median(good))
        log(f"[SPIRAL] angular step {angular_step:.4f} rad ({math.degrees(angular_step):.2f} deg), "
            f"{2 * math.pi / angular_step:.2f} treads/turn ({len(good)}/{len(steps)} levels used)")

        # --- the marker (the spawn) ------------------------------------------
        marker = None
        if args_cli.raw and os.path.isfile(args_cli.raw):
            marker = _marker_from_raw(args_cli.raw)
        if marker is None:
            marker = (0.4936, -0.7161)
            log("[SPIRAL] marker not found in the raw USD -- using the known default")
        walking_radius = float(math.hypot(marker[0] - centre[0], marker[1] - centre[1]))
        start_angle = float(math.atan2(marker[1] - centre[1], marker[0] - centre[0]))
        log(f"[SPIRAL] marker {marker[0]:.4f},{marker[1]:.4f} -> walking radius {walking_radius:.3f} m, "
            f"start angle {math.degrees(start_angle):.1f} deg")

        first_tread_angle = float(mid_angles[0])
        log(f"[SPIRAL] first tread at {math.degrees(first_tread_angle):.1f} deg, "
            f"z {level_z[0]:.3f}")

        # --- the clearance band, as a function of the path coordinate (height)
        band = []
        for value in level_z:
            # On a spiral every riser is the same, so the demanded clearance is
            # the same everywhere on the staircase; only the flat approach
            # (below the first tread) needs less.
            if value <= level_z[0] + 1e-6:
                band.append([float(value), args_cli.band_min])
            else:
                band.append([float(value), float(min(riser + args_cli.band_margin, 0.30))])
        # Deduplicate the heights (two treads can share a level) and make sure
        # the table is sorted so the interpolation can assume monotonicity.
        merged = {}
        for value, b in band:
            merged[round(value, 4)] = b
        band = [[k, merged[k]] for k in sorted(merged)]
        # The flat band up to the first step.
        band.insert(0, [float(level_z[0]) - 0.5, args_cli.band_min])
        log(f"[SPIRAL] clearance band: {band[0][1]:.3f} m on the approach, {band[-1][1]:.3f} m on the stairs")

        geometry = {
            "usd": os.path.basename(args_cli.usd),
            "centre": [float(centre[0]), float(centre[1])],
            "walking_radius": walking_radius,
            "riser": riser,
            "angular_step": angular_step,
            "treads_per_turn": float(2 * math.pi / angular_step),
            "n_treads": int(len(level_z)),
            "z_top": float(level_z[-1]),
            "tread_outer_radius": r_outer,
            "marker": [float(marker[0]), float(marker[1])],
            "start_angle": start_angle,
            "first_tread_angle": first_tread_angle,
            "tread_angles": [float(a) for a in mid_angles],
            "tread_heights": [float(v) for v in level_z],
            "band": band,
        }
        with open(args_cli.out, "w") as handle:
            json.dump(geometry, handle, indent=1)
        log(f"[SPIRAL] wrote {args_cli.out}")
    except Exception:
        import traceback

        log("[SPIRAL] FAILED\n" + traceback.format_exc())
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
