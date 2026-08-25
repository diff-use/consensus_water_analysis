"""Grid-search HDBSCAN (min_cluster_size, min_samples) for a cohort and persist the scores.

Fits the candidate grid once, scores every well-formed candidate on separation (DBCV) and
reproducibility (stability), and picks the candidate whose worse rank across the two is best
(min-max-rank). Writes the full per-candidate scores table to clustering_hyperparameters.csv
with a `recommended` column marking the winner, and prints a human summary.

The recommended candidate is already clustered during the grid search, so by default this
script also writes clusters.csv and cluster_members.csv from its labels — identical to running
cluster_waters.py at the recommended --min-cluster-size/--min-samples, just without the extra
fit. Pass --no-write-clusters to only write the scores table (leaving any existing
clusters.csv / cluster_members.csv untouched). Re-run cluster_waters.py only to cluster at
different (non-recommended) params.

Reads pooled water oxygens straight from the aligned CIFs (same collection as cluster_waters.py),
so it depends only on the alignment stage.

Usage:
    uv run scripts/find_clustering_hyperparameters.py <cohort.txt> [--input-dir <dir>] [-o <dir>]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

import config
from cw.cluster import build_cluster_tables, select_hdbscan_params
from cw.io import collect_aligned_waters, read_cohort, resolve_aligned_water_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search HDBSCAN params for a cohort and write clustering_hyperparameters.csv."
    )
    parser.add_argument("cohort", type=Path, help="Cohort .txt file (one member ID per line)")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory of aligned CIFs (default: config.DATA_DIR/<cohort_id>/aligned_pdbs/)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the scores CSV (default: config.DATA_DIR/<cohort_id>/)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=config.CLUSTER_MEMBER_RADIUS,
        metavar="Å",
        help=f"Cluster-membership radius (default: config.CLUSTER_MEMBER_RADIUS = {config.CLUSTER_MEMBER_RADIUS})",
    )
    parser.add_argument(
        "--no-write-clusters",
        action="store_true",
        help="Only write clustering_hyperparameters.csv; skip auto-saving clusters.csv / "
        "cluster_members.csv for the recommended params (default: auto-save them)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Show debug output")
    verbosity.add_argument("--quiet", action="store_true", help="Show warnings and errors only")
    args = parser.parse_args()

    logger.remove()
    if args.quiet:
        logger.add(sys.stderr, level="WARNING")
    elif args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    cohort_path: Path = args.cohort
    if not cohort_path.exists():
        logger.error(f"Cohort file not found: {cohort_path}")
        sys.exit(1)

    cohort_id = cohort_path.stem
    data_dir = Path(config.DATA_DIR) / cohort_id
    aligned_dir = args.input_dir or data_dir / "aligned_pdbs"
    out_dir = args.output_dir or data_dir

    member_ids = read_cohort(cohort_path)

    cif_json_pairs = resolve_aligned_water_inputs(
        member_ids,
        aligned_dir,
        edia_dir=config.ALL_PDB_REDO_DIR,
        edia_template=config.EDIA_TEMPLATE,
    )

    n_total = len(member_ids)
    n_found = len(cif_json_pairs)
    logger.info(f"Cohort:         {cohort_id}")
    logger.info(
        f"Members:        {n_total} total — {n_found} aligned CIFs found, {n_total - n_found} missing"
    )
    logger.info(f"Input:          {aligned_dir}")
    logger.info(f"Output:         {out_dir}")
    logger.info(f"Cluster radius: {args.radius} Å")

    if n_found == 0:
        logger.error(f"No aligned CIFs found in {aligned_dir}")
        sys.exit(1)

    logger.info("Collecting water records...")
    waters = collect_aligned_waters(cif_json_pairs)
    logger.info(f"  {len(waters):,} water oxygens across {n_found} structures")

    if waters.empty:
        logger.error("No water records found — check aligned CIFs")
        sys.exit(1)

    # Denominator for cluster_occupancy — count found aligned CIFs, matching cluster_waters.py
    # (not waters["pdb_id"].nunique()) so the two scripts produce identical clusters.csv. this
    # would count structures without waters.
    n_total_structures = n_found

    logger.info("Grid-searching HDBSCAN params (one fit per candidate)...")
    result = select_hdbscan_params(
        waters, n_total_structures=n_total_structures, radius=args.radius
    )
    diagnostics = result["diagnostics"]

    recommended = (diagnostics["min_cluster_size"] == result["min_cluster_size"]) & (
        diagnostics["min_samples"] == result["min_samples"]
    )
    scores = diagnostics.copy()
    scores["recommended"] = recommended
    scores["guard_relaxed"] = result["guard_relaxed"]

    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "clustering_hyperparameters.csv"
    scores.to_csv(scores_path, index=False)
    logger.info(f"clustering_hyperparameters.csv: {len(scores)} candidates  →  {scores_path}")

    if args.no_write_clusters:
        logger.info("Skipping clusters.csv / cluster_members.csv (--no-write-clusters)")
    else:
        logger.info(f"Applying {args.radius} Å radius filter to the recommended clustering...")
        members_df, clusters_df = build_cluster_tables(
            waters,
            result["labels"],
            radius=args.radius,
            n_total_structures=n_total_structures,
        )
        members_path = out_dir / "cluster_members.csv"
        clusters_path = out_dir / "clusters.csv"
        members_df.to_csv(members_path, index=False)
        clusters_df.to_csv(clusters_path, index=False)
        logger.info(f"cluster_members.csv: {len(members_df):,} rows  →  {members_path}")
        logger.info(f"clusters.csv:        {len(clusters_df):,} rows  →  {clusters_path}")

    _print_summary(waters, n_total_structures, result)


def _print_summary(waters, n_total_structures: int, result: dict) -> None:
    """Print the two single-axis extremes, the conserved-site range, and the recommendation."""
    diagnostics = result["diagnostics"]
    well_formed = diagnostics[diagnostics["dbcv"].notna()]

    dbcv_best = well_formed.loc[well_formed["dbcv"].idxmax()]
    stab_best = well_formed.loc[well_formed["mean_stability"].idxmax()]
    n_occ, frac = well_formed["n_occ_ge_0_3"], well_formed["frac_water_in_occ"]

    print(
        f"\n{len(waters):,} pooled waters / {n_total_structures} structures  "
        f"({len(well_formed)} well-formed candidates)\n"
    )
    print(
        f"Conserved sites (consensus>0.3): {int(n_occ.min())}-{int(n_occ.max())} across candidates; "
        f"{frac.min():.0%}-{frac.max():.0%} of all pooled waters lie in them"
    )
    print(
        f"DBCV best      : {int(dbcv_best.min_cluster_size)}/{int(dbcv_best.min_samples)} "
        f"(dbcv={dbcv_best.dbcv:.3f})  -- crispest, lowest min_samples"
    )
    print(
        f"Stability best : {int(stab_best.min_cluster_size)}/{int(stab_best.min_samples)} "
        f"(stability={stab_best.mean_stability:.3f})  -- coarsest, highest min_samples"
    )
    print(
        f"\nRECOMMEND (min-max-rank): min_cluster_size={result['min_cluster_size']}, "
        f"min_samples={result['min_samples']}   "
        f"(DBCV rank #{result['dbcv_rank']}, stability rank #{result['stab_rank']}; "
        f"n_clusters={result['n_clusters']}, consensus>0.3={result['n_occ_ge_0_3']})"
    )
    if result["guard_relaxed"]:
        note = "WARNING: no params kept clusters within the membership radius; guard relaxed."
    elif result["dbcv_rank"] == 1 and result["stab_rank"] == 1:
        note = "single best on both separation and reproducibility -> criteria agree, low-risk."
    else:
        note = "balance point of the DBCV (crisp) <-> stability (coarse) tension; neither extreme."
    print(f"  {note}")


if __name__ == "__main__":
    main()
