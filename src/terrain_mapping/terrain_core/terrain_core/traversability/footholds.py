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