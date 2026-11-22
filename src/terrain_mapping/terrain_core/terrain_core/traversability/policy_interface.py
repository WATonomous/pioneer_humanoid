import numpy as np


def build_height_scan(
    elevation,              # float (H, W), NaN = unknown, metres
    world_to_rc,            # callable (x, y) -> (row, col) ints, in the elevation map frame
    base_xy, base_yaw, base_z,
    scan_resolution=0.05,
    scan_size=(1.6, 1.0),
    height_offset=0.5,      # the "- 0.5" in foothold.py
):
    """Policy-style height scan, shape (num_y, num_x). NaN where unknown or off-map.

    Value = base_z - terrain_z - height_offset, sampled nearest-cell on a
    yaw-aligned grid centred on the base (x forward), as in foothold.py.
    Flattened with .reshape(-1) it matches the policy's ray-hit order (x fastest).
    """
    num_x = int(round(scan_size[0] / scan_resolution)) + 1
    num_y = int(round(scan_size[1] / scan_resolution)) + 1
    xs = -scan_size[0] / 2.0 + scan_resolution * np.arange(num_x)
    ys = -scan_size[1] / 2.0 + scan_resolution * np.arange(num_y)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")          # shape (num_y, num_x)

    c, s = np.cos(base_yaw), np.sin(base_yaw)
    wx = base_xy[0] + c * gx - s * gy
    wy = base_xy[1] + s * gx + c * gy

    h, w = elevation.shape
    scan = np.full((num_y, num_x), np.nan, dtype=np.float64)
    for iy, ix in np.ndindex(num_y, num_x):
        r, col = world_to_rc(wx[iy, ix], wy[iy, ix])
        if 0 <= r < h and 0 <= col < w:
            scan[iy, ix] = base_z - elevation[r, col] - height_offset
    return scan


def foothold_observation(
    height_scan,            # (num_y, num_x) from build_height_scan(), NaN allowed
    footholds_b,            # (2, 2) predicted footholds, base frame XY
    scan_resolution=0.05,
    scan_size=(1.6, 1.0),
    patch_size=7,
    fallback=0.0,           # used only if the whole scan is NaN. Explicit choice, see PR.
):
    """The 102-value actor foothold observation: per foot [x, y, 49 heights].

    Mirrors foothold_scan(): round-to-nearest anchor, clamp at the border, patch
    flattened x-slow / y-fast. NaN is filled with the patch median, then the
    scan median, then `fallback`. No clipping or noise is applied here.
    """
    num_y, num_x = height_scan.shape
    origin_x, origin_y = -scan_size[0] / 2.0, -scan_size[1] / 2.0
    half = patch_size // 2
    k = np.arange(-half, half + (patch_size % 2))
    off_x, off_y = np.meshgrid(k, k, indexing="ij")
    off_x, off_y = off_x.flatten(), off_y.flatten()

    flat = height_scan.reshape(-1)                       # y-major, x fastest
    scan_median = np.nanmedian(flat) if np.isfinite(flat).any() else fallback

    out = []
    for fx, fy in np.asarray(footholds_b, dtype=float).reshape(-1, 2):
        ax = int(np.rint((fx - origin_x) / scan_resolution))   # half-to-even, like torch.round
        ay = int(np.rint((fy - origin_y) / scan_resolution))
        sx = np.clip(ax + off_x, 0, num_x - 1)
        sy = np.clip(ay + off_y, 0, num_y - 1)
        patch = flat[sy * num_x + sx]
        nan = ~np.isfinite(patch)
        if nan.any():
            valid = patch[~nan]
            patch = np.where(nan, np.median(valid) if valid.size else scan_median, patch)
        out.append(np.concatenate([[fx, fy], patch]))
    return np.concatenate(out).astype(np.float32)