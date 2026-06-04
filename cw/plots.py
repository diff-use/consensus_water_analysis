from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree


def precision_recall(
    reference_coords: np.ndarray,
    predicted_coords: np.ndarray,
    dist_cutoff: float,
) -> dict[str, float]:
    """Precision and recall for predicted water positions against a reference set.

    recall    = fraction of reference points with ≥1 predicted point within dist_cutoff
    precision = fraction of predicted points within dist_cutoff of any reference point
    f1        = harmonic mean of precision and recall (0.0 when both are 0)
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

    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def matched_precision_recall(
    reference_coords: np.ndarray,
    predicted_coords: np.ndarray,
    dist_cutoff: float,
) -> dict[str, float]:
    """One-to-one matched precision/recall — each water matched to at most one counterpart.

    Unlike precision_recall, a single predicted water cannot be credited for
    covering multiple references: maximum bipartite matching pairs references and
    predictions that are within dist_cutoff with no vertex reused. n_matched is
    the size of that matching, so precision = recall = 1 only when both sets pair
    up exactly.
    """
    ref_tree = cKDTree(reference_coords)
    pred_tree = cKDTree(predicted_coords)

    dmat = ref_tree.sparse_distance_matrix(pred_tree, dist_cutoff, output_type="coo_matrix")
    adj = csr_matrix((np.ones(dmat.nnz, dtype=bool), (dmat.row, dmat.col)), shape=dmat.shape)

    matching = maximum_bipartite_matching(adj, perm_type="column")
    n_matched = int((matching >= 0).sum())

    return {
        "precision": n_matched / len(predicted_coords),
        "recall": n_matched / len(reference_coords),
    }


def chamfer_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbor distance between two water sets (Å).

    Continuous cousin of precision_recall: instead of thresholding the NN
    distance at a cutoff and counting, it averages the raw NN distance in each
    direction and sums them. This is the mean-NN variant (not summed-squared).
    Lower is closer.
    """
    tree_a = cKDTree(coords_a)
    tree_b = cKDTree(coords_b)
    d_ab, _ = tree_b.query(coords_a)
    d_ba, _ = tree_a.query(coords_b)
    return float(d_ab.mean() + d_ba.mean())
