
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_footholds(
    traversable: np.ndarray,
    resolution_m: float,
    foot_radius_m: float = 0.10,
    edge_margin_m: float = 0.05,
    score_saturation_m: float = 0.30,
    traversability: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute a good-foothold mask and clearance-based score.

    A cell is a good foothold if its nearest non-traversable or unknown
    cell is at least foot_radius_m + edge_margin_m away. The map border
    counts as non-traversable.
    """
    trav = np.asarray(traversable, dtype=bool)

    padded = np.pad(trav, 1, mode="constant", constant_values=False)
    clearance_m = (
        distance_transform_edt(padded)[1:-1, 1:-1] * resolution_m
    )

    good = trav & (clearance_m >= foot_radius_m + edge_margin_m)

    score = np.clip(clearance_m / score_saturation_m, 0.0, 1.0)
    if traversability is not None:
        score = np.minimum(
            score, np.nan_to_num(traversability, nan=0.0)
        )

    score = np.where(trav, score, 0.0)

    return {
        "foothold": good,
        "foothold_score": score.astype(np.float32),
    }


def score_footholds_at(
    foothold_score,
    foothold_mask,
    world_to_rc,
    resolution_m,
    footholds_base_xy,
    base_xy,
    base_yaw,
    patch_m=0.35,
):
    """Look up predicted footholds in the foothold layer.

    Returns per-foothold arrays. A foothold outside the map or on an
    unknown cell gets score 0 and good=False.
    """
    pts = np.asarray(footholds_base_xy, dtype=float).reshape(-1, 2)

    c, s = np.cos(base_yaw), np.sin(base_yaw)
    rot = np.array([[c, -s], [s, c]])
    world = pts @ rot.T + np.asarray(base_xy, dtype=float)

    half = max(
        int(np.floor(patch_m / 2.0 / resolution_m + 1e-9)), 0
    )

    h, w = foothold_score.shape
    n = len(world)

    out = {
        "in_map": np.zeros(n, dtype=bool),
        "good": np.zeros(n, dtype=bool),
        "score": np.zeros(n, dtype=np.float32),
        "patch_min_score": np.zeros(n, dtype=np.float32),
    }

    for i, (x, y) in enumerate(world):
        r, col = world_to_rc(x, y)

        if not (0 <= r < h and 0 <= col < w):
            continue

        out["in_map"][i] = True
        out["good"][i] = foothold_mask[r, col]
        out["score"][i] = foothold_score[r, col]

        patch = foothold_score[
            max(r - half, 0):r + half + 1,
            max(col - half, 0):col + half + 1,
        ]
        out["patch_min_score"][i] = patch.min()

    return out
