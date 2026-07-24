"""One-off cohort summary — prints the headline numbers for a (sub)cohort.

Reads the pre-computed artifacts of one cohort folder:
  metadata.csv                    (structure-level: resolution, R-free, num_water)
  clusters.csv / cluster_members.csv   (the recommended-params clustering)
  clustering_hyperparameters.csv  (the grid search, for the across-params range)

Conserved clusters are those with cluster_occupancy >= 0.3; a water counts as
conserved only when it is within the membership radius (1.0 A) of a conserved
cluster's center (the `within_cutoff` flag written by the clustering stage).

Usage:
    uv run experiments/summary_cohort.py <cohort>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from cw.metrics import consensus_centers, per_structure_consensus_pr


def _find_metadata(cohort_dir: Path, cohort: str, data_dir: Path) -> Path | None:
    """metadata.csv in the cohort folder, else the parent cohort (name minus the
    '_iso...' suffix)."""
    local = cohort_dir / "metadata.csv"
    if local.exists():
        return local
    if "_iso" in cohort:
        parent = cohort.split("_iso")[0]
        parent_meta = data_dir / parent / "metadata.csv"
        if parent_meta.exists():
            return parent_meta
    return None


def _fmt_range(series: pd.Series, fmt: str = ".2f") -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return "n/a"
    return f"{vals.min():{fmt}} - {vals.max():{fmt}}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a headline summary for a cohort.")
    parser.add_argument("cohort", help="Cohort folder name under DATA_DIR")
    parser.add_argument("--data-dir", type=Path, default=Path(config.DATA_DIR))
    parser.add_argument("--occupancy-cutoff", type=float, default=0.3)
    parser.add_argument("--match-radius", type=float, default=config.CLUSTER_MEMBER_RADIUS)
    parser.add_argument(
        "--pdb-redo-rfree",
        action="store_true",
        help="Report the re-refined (PDB-REDO) R-free column instead of the "
        "default deposited (RCSB) R-free.",
    )
    args = parser.parse_args()

    rfree_col = "r_free" if args.pdb_redo_rfree else "deposited_r_free"

    cutoff = args.occupancy_cutoff
    radius = args.match_radius
    cohort_dir = args.data_dir / args.cohort

    clusters = pd.read_csv(cohort_dir / "clusters.csv")
    cluster_members = pd.read_csv(cohort_dir / "cluster_members.csv")

    # --- items 1 & 2: structure-level stats over the clustered structures ------
    members_pdbs = set(cluster_members["pdb_id"].unique())
    n_total_water = len(cluster_members)

    meta_path = _find_metadata(cohort_dir, args.cohort, args.data_dir)
    print(f"cohort:        {args.cohort}")
    print(f"cohort dir:    {cohort_dir}")
    print(f"metadata.csv:  {meta_path}")
    print()

    if meta_path is not None:
        metadata = pd.read_csv(meta_path)
        meta = metadata[metadata["pdb_id"].isin(members_pdbs)]
        print(
            f"1. There are {len(meta)} PDBs, "
            f"the resolution range is {_fmt_range(meta['resolution'])}, "
            f"the R-free range is {_fmt_range(meta[rfree_col], '.3f')}, "
            f"and num_water range is {_fmt_range(meta['num_water'], '.0f')}."
        )
    else:
        print(
            f"1. There are {len(members_pdbs)} PDBs "
            "(metadata.csv not found; resolution/R-free/num_water unavailable)."
        )

    print(f"2. There are {n_total_water} total waters.")

    # --- item 3: conserved-water fraction across all hyperparameters ----------
    grid_path = cohort_dir / "clustering_hyperparameters.csv"
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        well_formed = grid[grid["dbcv"].notna()]
        frac = well_formed["frac_water_in_occ"]
        print(
            f"3. Across all hyperparameters ({len(well_formed)} well-formed candidates), "
            f"the fraction of waters in conserved clusters ranges from "
            f"{frac.min():.1%} - {frac.max():.1%}."
        )
    else:
        grid = None
        print("3. clustering_hyperparameters.csv not found — across-params range unavailable.")

    # --- item 4: the best (recommended) hyperparameter ------------------------
    conserved_ids = set(clusters.loc[clusters["cluster_occupancy"] >= cutoff, "cluster_id"])
    is_conserved_water = (
        cluster_members["within_cutoff"] & cluster_members["cluster_id"].isin(conserved_ids)
    )
    conserved_frac = is_conserved_water.sum() / n_total_water
    n_noise = int((cluster_members["cluster_id"] == -1).sum())

    best = ""
    if grid is not None and grid["recommended"].any():
        rec = grid[grid["recommended"]].iloc[0]
        best = (
            f" (min_cluster_size={int(rec['min_cluster_size'])}, "
            f"min_samples={int(rec['min_samples'])})"
        )
    print(
        f"4. The best hyperparameter{best} identified {conserved_frac:.1%} waters "
        f"in conserved clusters, {n_noise} noise waters."
    )

    # --- item 5: per-structure precision/recall vs consensus centers ----------
    centers = consensus_centers(clusters, cutoff)
    pr_df = per_structure_consensus_pr(cluster_members, centers, radius)

    # num_water-binned curve — 10 quantile bins, 1/99% tails folded into the ends
    # (identical to notebooks/01_cluster_analysis.py's Pareto cell).
    edges = np.unique(np.quantile(pr_df["num_water"], np.linspace(0.01, 0.99, 11)))
    if np.allclose(pr_df["num_water"], np.round(pr_df["num_water"])):
        edges = np.unique(np.rint(edges))
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    bins = pd.cut(pr_df["num_water"], bins=edges).rename("num_water_bin")
    curve = (
        pr_df.groupby(bins, observed=True)[["recall", "precision", "num_water"]]
        .mean()
        .sort_values("num_water")
    )
    lo, hi = curve.iloc[0], curve.iloc[-1]

    knee = pr_df.loc[pr_df["f1"].idxmax()]

    print(
        "5. P/R per structure against conserved clusters, binned by num_water:\n"
        f"   a) low-end bin  (num_water={int(round(lo['num_water']))}): "
        f"recall={lo['recall']:.2f}, precision={lo['precision']:.2f}\n"
        f"   b) high-end bin (num_water={int(round(hi['num_water']))}): "
        f"recall={hi['recall']:.2f}, precision={hi['precision']:.2f}\n"
        f"   c) Pareto knee (max F1): recall={knee['recall']:.2f}, "
        f"precision={knee['precision']:.2f}, F1={knee['f1']:.2f}, "
        f"num_water={int(round(knee['num_water']))}"
    )


if __name__ == "__main__":
    main()
