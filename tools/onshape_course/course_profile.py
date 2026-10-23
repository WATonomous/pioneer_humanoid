"""Extract the course's terrain height profile and the required swing clearance.

The Onshape course is a static mesh, so the height of the stair surface along
the walking line is a *constant* of the task, not something the policy has to
rediscover. This script reads it straight off the merged USD and writes a
lookup table that the reward can interpolate:

    <stem>_profile.json  ->  {"heights": [[y, z], ...], "band": [[y, band], ...]}

``heights`` is the terrain height (the top surface at the walking line) and
``band`` is the *required* swing-foot clearance at that y: the height the next
step has to clear, i.e. the tallest tread within ``--lookahead`` metres minus
the tread the robot stands on, plus the term's own margin, clamped into
[band_min, band_max].

Why it matters: ``foot_swing_clearance`` used a single global band, so the
policy was paid the full band (a high knee) even on the flat approach, while
over-lifting past ``max_clearance`` cost nothing. A per-position band pays
only for the clearance the course actually demands.

The merged course is a low-poly prism soup (a few hundred triangles whose
treads are single big quads), so the height has to come from the *triangles*,
not from the vertices: a tread quad's corners are 4 m apart and the verts
alone say nothing about the surface in between.

Run inside the container (needs the Isaac Sim USD bindings):

    isaaclab.sh -p tools/onshape_course/course_profile.py \\
        --usd assets/staircase_onshape/course_easy_terrain.usd \\
        --out assets/staircase_onshape/course_easy_profile.json
"""

from __future__ import annotations

import argparse
import json
import os

parser = argparse.ArgumentParser(description="Extract the Onshape course height profile and clearance band.")
parser.add_argument(
    "--usd",
    default="/workspace/humanoid/assets/staircase_onshape/course_easy_terrain.usd",
    help="Merged course USD (the /course/mesh prim).",
)
parser.add_argument("--walk-x", type=float, default=4.163, help="Walking line (world x).")
parser.add_argument("--half-width", type=float, default=0.30, help="Half width sampled around --walk-x.")
parser.add_argument("--bin", type=float, default=0.05, help="y bin size (m).")
parser.add_argument("--lookahead", type=float, default=0.45, help="How far ahead a riser must be paid for (m).")
parser.add_argument("--margin", type=float, default=0.03, help="Foot-to-foot margin the term already subtracts.")
parser.add_argument("--band-min", type=float, default=0.06, help="Floor for the band (a natural flat step).")
parser.add_argument("--band-max", type=float, default=0.22, help="Ceiling for the band (the tallest riser + margin).")
parser.add_argument("--y-min", type=float, default=-4.0, help="First sample (the flat approach before the course).")
parser.add_argument("--y-max", type=float, default=23.0, help="Last sample (past the last part).")
parser.add_argument(
    "--out",
    default="/workspace/humanoid/assets/staircase_onshape/course_easy_profile.json",
    help="Output JSON.",
)
parser.add_argument("--probe-x", type=float, nargs="*", default=[3.5, 4.163], help="Extra x lines to print for checking.")
parser.add_argument("--log", default="/tmp/course_profile.log", help="Log file (Kit swallows stdout).")

args_cli = parser.parse_args()

import numpy as np  # noqa: E402  (after isaaclab.sh -p sets up numpy)

_LOG = None


def log(msg: str) -> None:
    """Write to the log file *and* stdout (Kit redirects stdout after boot)."""
    if _LOG is not None:
        with open(_LOG, "a") as handle:
            handle.write(msg + "\n")
            handle.flush()
    print(msg, flush=True)


def _tri_max_z_at_y(tri_yz: np.ndarray, y: float) -> float:
    """Highest z of a triangle at plane y (the triangle's cross-section top)."""
    best = -1e9
    for i in range(3):
        ya, za = tri_yz[i]
        yb, zb = tri_yz[(i + 1) % 3]
        if abs(yb - ya) < 1e-9:
            if abs(y - ya) < 1e-9:
                best = max(best, za, zb)
            continue
        lo, hi = (ya, yb) if ya < yb else (yb, ya)
        if lo - 1e-9 <= y <= hi + 1e-9:
            t = (y - ya) / (yb - ya)
            best = max(best, za + t * (zb - za))
    return best if best > -1e8 else float("nan")


def main() -> None:
    global _LOG
    _LOG = args_cli.log
    open(_LOG, "w").close()
    log("[PROFILE] starting")
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app
    log("[PROFILE] kit app up")
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(args_cli.usd)
        prim = stage.GetPrimAtPath("/course/mesh")
        if not prim or not prim.IsValid():
            raise RuntimeError(f"/course/mesh not found in {args_cli.usd}")
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        xform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        rot = np.array([[xform[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)
        trn = np.array([xform[r][3] for r in range(3)], dtype=np.float64)
        pts = pts @ rot.T + trn
        log(f"[PROFILE] {len(pts)} verts, {len(counts)} faces, world x=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
            f"y=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] z=[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")

        # Triangulate (fan per face).
        tris = []
        offset = 0
        for cnt in counts:
            face = indices[offset : offset + cnt]
            offset += cnt
            for k in range(1, cnt - 1):
                tris.append((face[0], face[k], face[k + 1]))
        tris = np.asarray(tris, dtype=np.int64)
        log(f"[PROFILE] {len(tris)} triangles")

        ys = np.arange(args_cli.y_min, args_cli.y_max + args_cli.bin, args_cli.bin)
        centres = 0.5 * (ys[:-1] + ys[1:])
        n_bins = len(centres)

        def height_along(line_x: float) -> np.ndarray:
            h = np.zeros(n_bins)
            for t in tris:
                p = pts[t]
                if p[:, 0].max() < line_x - args_cli.half_width or p[:, 0].min() > line_x + args_cli.half_width:
                    continue
                ylo, yhi = p[:, 1].min(), p[:, 1].max()
                if yhi < ys[0] or ylo > ys[-1]:
                    continue
                # Sample the triangle at its vertex y's and at every bin edge
                # inside its span, then keep the max per bin.
                lo_edge = np.ceil(ylo / args_cli.bin) * args_cli.bin
                cands = list(p[:, 1]) + list(np.arange(lo_edge, yhi + 1e-9, args_cli.bin))
                for yc in cands:
                    if yc < ys[0] or yc > ys[-1]:
                        continue
                    z = _tri_max_z_at_y(p[:, 1:3], yc)
                    if np.isnan(z):
                        continue
                    k = int(np.clip(np.searchsorted(ys, yc) - 1, 0, n_bins - 1))
                    if z > h[k]:
                        h[k] = z
            return h

        h_walk = height_along(args_cli.walk_x)
        for line_x in args_cli.probe_x:
            h = height_along(line_x)
            log(f"[PROFILE] walk line x={line_x:.3f}: max z={h.max():.3f} "
                f"matches x={args_cli.walk_x:.3f}: {np.allclose(h, h_walk, atol=1e-3)}")

        # The required clearance: the tallest tread within the lookahead minus
        # the tread underfoot, plus the term's margin.
        band = np.empty(n_bins)
        for k in range(n_bins):
            hi = k
            limit = centres[k] + args_cli.lookahead
            while hi + 1 < n_bins and centres[hi + 1] <= limit:
                hi += 1
            ahead = h_walk[k + 1 : hi + 1]
            needed = (ahead.max() if len(ahead) else h_walk[k]) - h_walk[k]
            band[k] = float(np.clip(needed + args_cli.margin, args_cli.band_min, args_cli.band_max))

        log("[PROFILE] y -> terrain z | required band")
        for k in range(n_bins):
            if h_walk[k] > 0.001 or band[k] > args_cli.band_min + 1e-6:
                bar = "#" * max(1, int(round(h_walk[k] / max(h_walk.max(), 1e-6) * 40)))
                log(f"[PROFILE]   y={centres[k]:7.2f} z={h_walk[k]:5.3f} band={band[k]:5.3f} {bar}")

        out = {
            "usd": os.path.basename(args_cli.usd),
            "walk_x": args_cli.walk_x,
            "bin": args_cli.bin,
            "lookahead": args_cli.lookahead,
            "margin": args_cli.margin,
            "heights": [[round(float(c), 4), round(float(h), 4)] for c, h in zip(centres, h_walk)],
            "band": [[round(float(c), 4), round(float(b), 4)] for c, b in zip(centres, band)],
        }
        with open(args_cli.out, "w") as handle:
            json.dump(out, handle, indent=1)
        log(f"[PROFILE] wrote {args_cli.out} ({n_bins} samples)")
    except Exception:
        import traceback

        log("[PROFILE] FAILED\n" + traceback.format_exc())
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
