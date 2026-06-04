# compute_cluster_stability ported from Vratin's porting_reference/cluster.py.
from __future__ import annotations

import hdbscan as hdbscan_lib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def occupancy_to_min_cluster_size(min_occupancy: float, n_total_structures: int) -> int:
    """Convert a minimum occupancy fraction to the integer min_cluster_size for HDBSCAN."""
    return max(2, round(min_occupancy * n_total_structures))


def run_hdbscan(
    coords: np.ndarray,
    *,
    min_cluster_size: int = 20,
    min_samples: int | None = None,
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
) -> np.ndarray:
    """Cluster (x, y, z) coordinates with HDBSCAN. Returns integer labels (-1 = noise).

    min_samples defaults to min_cluster_size (HDBSCAN default behaviour) when None.
    """
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples if min_samples is not None else min_cluster_size,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
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
    radius: float = 1.4,
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


def compute_cluster_stability(
    all_coords: np.ndarray,
    base_labels: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int | None = None,
    n_stability_runs: int = 6,
) -> dict[int, float]:
    """Stability score for each cluster via min_samples sweep.

    Holds min_cluster_size fixed and scans min_samples log-uniformly from 1 to
    min_cluster_size (n_stability_runs candidates), excluding the base value.
    min_samples defaults to min_cluster_size when None. Returns a dict of
    cluster_id → mean best-Jaccard score across the alternate clusterings.
    """
    cluster_ids = [int(c) for c in np.unique(base_labels) if c >= 0]
    if not cluster_ids:
        return {}

    base_ms = min_samples if min_samples is not None else min_cluster_size
    ms_values = {int(round(v)) for v in np.geomspace(1, min_cluster_size, n_stability_runs)} - {
        base_ms
    }

    if not ms_values:
        return {c: 1.0 for c in cluster_ids}

    alt_clusterings: list[tuple[np.ndarray, dict[int, int]]] = []
    for ms in sorted(ms_values):
        alt_labels = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=ms,
            cluster_selection_method="eom",
        ).fit_predict(all_coords)
        alt_sizes = {int(c): int((alt_labels == c).sum()) for c in np.unique(alt_labels) if c >= 0}
        alt_clusterings.append((alt_labels, alt_sizes))

    stability: dict[int, float] = {}
    for cluster_id in cluster_ids:
        base_idx = np.flatnonzero(base_labels == cluster_id)
        if base_idx.size == 0:
            stability[cluster_id] = 0.0
            continue
        base_size = base_idx.size
        run_scores: list[float] = []
        for alt_labels, alt_sizes in alt_clusterings:
            best_jaccard = 0.0
            overlap_ids, counts = np.unique(alt_labels[base_idx], return_counts=True)
            for alt_id, count in zip(overlap_ids, counts, strict=True):
                alt_id = int(alt_id)
                if alt_id < 0:
                    continue
                union = base_size + alt_sizes[alt_id] - int(count)
                best_jaccard = max(best_jaccard, count / union)
            run_scores.append(best_jaccard)
        stability[cluster_id] = float(np.mean(run_scores)) if run_scores else 1.0

    return stability
