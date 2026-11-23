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
    height_scan,
    footholds_b,
    scan_resolution=0.05,
    scan_size=(1.6, 1.0),
    patch_size=7,
    fallback=0.0,
    return_nan_fraction=False,
):
    """Build the 102-value actor foothold observation.

    Returns the observation as a float32 array by default.
    If return_nan_fraction=True, also returns one NaN fraction per foot.

    NaN is filled with the patch median, then the scan median, then fallback.
    NaN fractions are measured before filling.
    """
    num_y, num_x = height_scan.shape

    expected_x = int(round(scan_size[0] / scan_resolution)) + 1
    expected_y = int(round(scan_size[1] / scan_resolution)) + 1

    if (num_y, num_x) != (expected_y, expected_x):
        raise ValueError(
            f"height_scan shape {(num_y, num_x)} does not match "
            f"scan_size/resolution expected shape {(expected_y, expected_x)}"
        )

    origin_x, origin_y = -scan_size[0] / 2.0, -scan_size[1] / 2.0
    half = patch_size // 2
    k = np.arange(-half, half + (patch_size % 2))
    off_x, off_y = np.meshgrid(k, k, indexing="ij")
    off_x, off_y = off_x.flatten(), off_y.flatten()

    flat = height_scan.reshape(-1)
    scan_median = (
        np.nanmedian(flat) if np.isfinite(flat).any() else fallback
    )

    out = []
    nan_fractions = []

    for fx, fy in np.asarray(footholds_b, dtype=float).reshape(-1, 2):
        ax = int(np.rint((fx - origin_x) / scan_resolution))
        ay = int(np.rint((fy - origin_y) / scan_resolution))

        sx = np.clip(ax + off_x, 0, num_x - 1)
        sy = np.clip(ay + off_y, 0, num_y - 1)

        patch = flat[sy * num_x + sx]

        nan = ~np.isfinite(patch)
        nan_fractions.append(float(nan.mean()))

        if nan.any():
            valid = patch[~nan]
            fill_value = np.median(valid) if valid.size else scan_median
            patch = np.where(nan, fill_value, patch)

        out.append(np.concatenate([[fx, fy], patch]))

    observation = np.concatenate(out).astype(np.float32)
    nan_fractions = np.asarray(nan_fractions, dtype=np.float32)

    if return_nan_fraction:
        return observation, nan_fractions

    return observation
