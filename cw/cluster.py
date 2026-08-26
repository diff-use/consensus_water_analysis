from __future__ import annotations

import warnings

import hdbscan as hdbscan_lib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import] (compiled re-export)

from cw.metrics import consensus_water_mask


def run_hdbscan(
    coords: np.ndarray,
    *,
    min_cluster_size: int = 20,
    min_samples: int | None = None,
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Cluster (x, y, z) coordinates with HDBSCAN. Returns integer labels (-1 = noise).

    min_samples defaults to min_cluster_size (HDBSCAN default behavior) when None.
    """
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples if min_samples is not None else min_cluster_size,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        core_dist_n_jobs=n_jobs if n_jobs is not None else 1,
    )
    return clusterer.fit_predict(coords)


def _build_members_df(
    records: pd.DataFrame,
    labels: np.ndarray,
    radius: float,
) -> pd.DataFrame:
    """Return all water records with cluster_id (raw HDBSCAN label) and within_cutoff columns.

    within_cutoff is True only when cluster_id >= 0 and dist_to_center <= radius.
    Noise points (cluster_id == -1) always have within_cutoff = False.
    """
    coords = records[["x", "y", "z"]].to_numpy()
    labels = np.asarray(labels, dtype=int)

    within_cutoff = np.zeros(len(records), dtype=bool)

    for cluster_id in np.unique(labels):
        if cluster_id < 0:
            continue
        mask = labels == cluster_id
        center = coords[mask].mean(axis=0)
        dists = np.linalg.norm(coords[mask] - center, axis=1)
        idx_in_cluster = np.where(mask)[0]
        within_cutoff[idx_in_cluster] = dists <= radius

    df = records.copy()
    df.insert(0, "cluster_id", labels)
    df.insert(1, "within_cutoff", within_cutoff)
    return df


def _build_clusters_df(
    members_df: pd.DataFrame,
    n_total_structures: int,
) -> pd.DataFrame:
    """Return per-cluster summary stats derived from members_df.

    Center is the mean of all HDBSCAN-assigned members (within_cutoff or not).
    cluster_occupancy and n_structures are computed from within_cutoff members only.
    Clusters where no member passes the radius filter are omitted.

    clusters columns:
      cluster_id, center_x, center_y, center_z, std_x, std_y, std_z,
      cluster_occupancy, n_radius_rejected

    std_x/y/z are the per-axis standard deviations of all HDBSCAN-assigned
    members (before the radius cutoff), matching the coordinate set used for
    the center.
    """
    cluster_rows: list[dict] = []

    for cluster_id, group in members_df[members_df["cluster_id"] >= 0].groupby("cluster_id"):
        coords = group[["x", "y", "z"]].to_numpy()
        center = coords.mean(axis=0)
        std = coords.std(axis=0)
        n_radius_rejected = int((~group["within_cutoff"]).sum())

        kept = group[group["within_cutoff"]]
        if len(kept) == 0:
            continue

        n_structures = int(kept["pdb_id"].nunique())
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "std_x": float(std[0]),
                "std_y": float(std[1]),
                "std_z": float(std[2]),
                "cluster_occupancy": n_structures / n_total_structures,
                "n_radius_rejected": n_radius_rejected,
            }
        )

    clusters_df = pd.DataFrame(cluster_rows)
    if not clusters_df.empty:
        clusters_df = clusters_df.sort_values("cluster_id").reset_index(drop=True)
    return clusters_df


def build_cluster_tables(
    records: pd.DataFrame,
    labels: np.ndarray,
    *,
    radius: float,
    n_total_structures: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply radius filter and produce two output DataFrames.

    Returns (cluster_members, clusters):

    cluster_members — one row per water, all input rows preserved:
      cluster_id    raw HDBSCAN label (-1 = noise)
      within_cutoff True when cluster_id >= 0 and dist_to_center <= radius
      pdb_id, chain_id, res_id, altloc, x, y, z, b_factor, occupancy, edia

    clusters — one row per valid cluster (summary stats):
      cluster_id, center_x, center_y, center_z, std_x, std_y, std_z,
      cluster_occupancy, n_radius_rejected
    """
    members_df = _build_members_df(records, labels, radius)
    clusters_df = _build_clusters_df(members_df, n_total_structures)
    return members_df, clusters_df


def find_clusters_too_close(
    centers: np.ndarray,
    min_separation: float,
) -> list[tuple[int, int, float]]:
    """Index pairs of cluster centers closer than min_separation, with distance.

    A single water can fall within r of two centers only when those centers are
    within 2*r of each other, so pass min_separation = 2 * match_cutoff to flag
    the centers that make precision/recall matching ambiguous. Returns
    (i, j, distance) tuples sorted by distance (closest first).
    """
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=min_separation)
    out = [(i, j, float(np.linalg.norm(centers[i] - centers[j]))) for i, j in pairs]
    return sorted(out, key=lambda t: t[2])


def _cluster_sizes(cluster_ids: np.ndarray) -> dict[int, int]:
    """Point count per non-noise cluster id."""
    return {
        int(cluster_id): int((cluster_ids == cluster_id).sum())
        for cluster_id in np.unique(cluster_ids)
        if cluster_id >= 0
    }


def _mean_best_jaccard(
    base_labels: np.ndarray,
    alternate_clusterings: list[tuple[np.ndarray, dict[int, int]]],
) -> dict[int, float]:
    """Per-cluster mean best-Jaccard of base clusters against alternate clusterings.

    Each alternate is (labels, sizes) with sizes from _cluster_sizes. Returns cluster_id →
    mean over the alternates of the best Jaccard overlap with any of that alternate's clusters;
    1.0 for a base cluster when there are no alternates to compare against.
    """
    stability: dict[int, float] = {}
    for cluster_id in np.unique(base_labels):
        if cluster_id < 0:
            continue
        base_points = np.flatnonzero(base_labels == cluster_id)
        base_size = base_points.size
        best_jaccard_per_alternate: list[float] = []
        for alternate_labels, alternate_sizes in alternate_clusterings:
            best_jaccard = 0.0
            overlapping_ids, overlaps = np.unique(alternate_labels[base_points], return_counts=True)
            for alternate_id, overlap in zip(overlapping_ids, overlaps, strict=True):
                alternate_id = int(alternate_id)
                if alternate_id < 0:
                    continue
                union = base_size + alternate_sizes[alternate_id] - int(overlap)
                best_jaccard = max(best_jaccard, overlap / union)
            best_jaccard_per_alternate.append(best_jaccard)
        stability[int(cluster_id)] = (
            float(np.mean(best_jaccard_per_alternate)) if best_jaccard_per_alternate else 1.0
        )
    return stability


def _hdbscan_candidate_grid(n_total_structures: int) -> list[tuple[int, int]]:
    """(min_cluster_size, min_samples) candidates for the HDBSCAN hyperparameter grid search.

    min_cluster_size is an absolute detection floor in waters (~= structures agreeing), not a
    fraction of the cohort — else the cluster_occupancy distribution would be pre-truncated. It
    caps at 20 and drops 3 for larger cohorts, and never exceeds the cohort size.
    min_samples estimates local core density, and capped at min_cluster_size is HDBSCAN's most
    conservative default, as larger values are more likely to add noise.
    """
    floors = (3, 5, 10, 15, 20) if n_total_structures <= 30 else (5, 10, 15, 20)
    min_cluster_sizes = sorted(v for v in floors if v <= n_total_structures) or [
        max(2, n_total_structures)
    ]
    ladder = (2, 3, 5, 10, 15, 20)
    grid: list[tuple[int, int]] = []
    for min_cluster_size in min_cluster_sizes:
        grid.extend(
            (min_cluster_size, min_samples)
            for min_samples in ladder
            if 2 <= min_samples <= min_cluster_size
        )
    return grid


def _fit_candidate_grid(
    records: pd.DataFrame,
    *,
    radius: float,
    n_total_structures: int,
) -> tuple[dict[tuple[int, int], np.ndarray], pd.DataFrame]:
    """Fit HDBSCAN once per grid candidate.

    Returns the labels keyed by (min_cluster_size, min_samples) and one diagnostics row per
    candidate. Geometry columns (dbcv, median_spread, ...) are filled only for well-formed fits
    (>=2 clusters); mean_stability is left NaN for the caller to score.
    """
    coords = records[["x", "y", "z"]].to_numpy()
    labels_by_candidate: dict[tuple[int, int], np.ndarray] = {}
    rows: list[dict] = []
    for min_cluster_size, min_samples in _hdbscan_candidate_grid(n_total_structures):
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method="eom",
            gen_min_span_tree=True,
        ).fit(coords)
        labels = clusterer.labels_
        labels_by_candidate[(min_cluster_size, min_samples)] = labels
        n_clusters = int(len(set(labels[labels >= 0])))

        row = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "dbcv": float("nan"),
            "median_spread": float("nan"),
            "radius_reject_frac": float("nan"),
            "n_occ_ge_0_3": 0,
            "frac_water_in_occ": 0.0,
            "mean_stability": float("nan"),
        }
        if n_clusters >= 2:
            row["dbcv"] = float(clusterer.relative_validity_)
            members_df, clusters_df = build_cluster_tables(
                records, labels, radius=radius, n_total_structures=n_total_structures
            )
            assigned = members_df["cluster_id"] >= 0
            n_assigned = int(assigned.sum())
            n_rejected = int((assigned & ~members_df["within_cutoff"]).sum())
            spreads = np.sqrt(
                clusters_df["std_x"] ** 2 + clusters_df["std_y"] ** 2 + clusters_df["std_z"] ** 2
            )
            occ_ids = clusters_df.loc[clusters_df["cluster_occupancy"] >= 0.3, "cluster_id"]
            in_occ = consensus_water_mask(members_df, clusters_df, 0.3)
            row["median_spread"] = float(spreads.median())
            row["radius_reject_frac"] = (n_rejected / n_assigned) if n_assigned else float("nan")
            row["n_occ_ge_0_3"] = int(len(occ_ids))
            row["frac_water_in_occ"] = float(in_occ.sum() / len(records))
        rows.append(row)

    return labels_by_candidate, pd.DataFrame(rows)


def _score_candidate_stability(
    diagnostics: pd.DataFrame,
    labels_by_candidate: dict[tuple[int, int], np.ndarray],
    rows_to_score: pd.Index,
) -> None:
    """Fill mean_stability for `rows_to_score`, in place.

    A candidate's stability is the mean best-Jaccard of its clusters against its grid siblings
    (same min_cluster_size, other min_samples) — all already fit, so no extra clustering. It is
    cheap enough to score every well-formed candidate rather than gating first.
    """
    for idx in rows_to_score:
        min_cluster_size = int(diagnostics.at[idx, "min_cluster_size"])
        min_samples = int(diagnostics.at[idx, "min_samples"])
        siblings = [
            (labels, _cluster_sizes(labels))
            for (candidate_size, candidate_samples), labels in labels_by_candidate.items()
            if candidate_size == min_cluster_size and candidate_samples != min_samples
        ]
        stability = _mean_best_jaccard(
            labels_by_candidate[(min_cluster_size, min_samples)], siblings
        )
        diagnostics.at[idx, "mean_stability"] = (
            float(np.mean(list(stability.values()))) if stability else float("nan")
        )


def select_hdbscan_params(
    records: pd.DataFrame,
    *,
    n_total_structures: int,
    radius: float,
) -> dict:
    """Pick (min_cluster_size, min_samples) for a cohort's waters clustering by grid search.

    One HDBSCAN fit per candidate. Candidates must be well-formed (>=2 clusters) and compact
    (median spread <= `radius`; relaxed if none qualify). Every survivor is scored on two axes
    that pull in opposite directions: separation (DBCV, which favors crisp low-min_samples
    clusterings) and reproducibility (stability vs its min_samples siblings, which favors coarse
    high-min_samples ones). The winner minimizes its worse rank across the two (min-max-rank) —
    the balance point of that tension — so neither bias can run away, because a candidate that is
    extreme on one axis is capped by its poor rank on the other.

    `records` has one row per pooled water (x/y/z, pdb_id). Returns the chosen params, winner
    `labels`, its `dbcv_rank`/`stab_rank`, per-candidate `diagnostics` (with both ranks and a
    `frac_water_in_occ` column), and `guard_relaxed`.
    """
    if n_total_structures < 10:
        warnings.warn(
            f"cohort has only {n_total_structures} structures (<10): cluster consensus is "
            "statistically weak and the selected HDBSCAN params may not be robust",
            stacklevel=2,
        )
    labels_by_candidate, diagnostics = _fit_candidate_grid(
        records, radius=radius, n_total_structures=n_total_structures
    )

    # Hard guards, evaluated on the cheap fits: well-formed (>=2 clusters) and compact.
    valid = diagnostics["dbcv"].notna()
    compact = diagnostics["median_spread"] <= radius
    guard_relaxed = not (valid & compact).any()
    guard = valid if guard_relaxed else (valid & compact)

    _score_candidate_stability(diagnostics, labels_by_candidate, diagnostics.index[valid])

    # Rank the guard-passers on both axes (1 = best) and pick the candidate whose worse rank is
    # best; ties break toward better stability then better separation. Non-guard rows stay NaN.
    ranked = diagnostics.loc[guard]
    diagnostics["dbcv_rank"] = ranked["dbcv"].rank(ascending=False, method="min")
    diagnostics["stab_rank"] = ranked["mean_stability"].rank(ascending=False, method="min")
    diagnostics["max_rank"] = diagnostics[["dbcv_rank", "stab_rank"]].max(axis=1)
    winner = (
        diagnostics.loc[guard]
        .sort_values(["max_rank", "stab_rank", "dbcv_rank"], kind="stable")
        .iloc[0]
    )
    chosen_pair = (int(winner["min_cluster_size"]), int(winner["min_samples"]))

    return {
        "min_cluster_size": chosen_pair[0],
        "min_samples": chosen_pair[1],
        "dbcv": float(winner["dbcv"]),
        "mean_stability": float(winner["mean_stability"]),
        "median_spread": float(winner["median_spread"]),
        "radius_reject_frac": float(winner["radius_reject_frac"]),
        "n_clusters": int(winner["n_clusters"]),
        "n_occ_ge_0_3": int(winner["n_occ_ge_0_3"]),
        "dbcv_rank": int(winner["dbcv_rank"]),
        "stab_rank": int(winner["stab_rank"]),
        "labels": labels_by_candidate[chosen_pair],
        "diagnostics": diagnostics,
        "guard_relaxed": guard_relaxed,
    }
