from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def precision_recall(
    reference_coords: np.ndarray,
    predicted_coords: np.ndarray,
    dist_cutoff: float,
) -> dict[str, float]:
    """Precision and recall for predicted water positions against a reference set.

    recall    = fraction of reference points with ≥1 predicted point within dist_cutoff
    precision = fraction of predicted points within dist_cutoff of any reference point
    """
    ref_tree = cKDTree(reference_coords)
    pred_tree = cKDTree(predicted_coords)

    covered_refs = set(
        idx
        for matches in ref_tree.query_ball_point(predicted_coords, r=dist_cutoff)
        for idx in matches
    )
    recall = len(covered_refs) / len(reference_coords)

    covered_preds = set(
        idx
        for matches in pred_tree.query_ball_point(reference_coords, r=dist_cutoff)
        for idx in matches
    )
    precision = len(covered_preds) / len(predicted_coords)

    return {"precision": precision, "recall": recall}
