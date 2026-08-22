"""Cluster conserved waters from aligned structures and write cluster_members.csv / clusters.csv.

Usage:
    uv run scripts/cluster_waters.py <cohort.txt> [--input-dir <dir>] [-o <output_dir>]

Reads aligned CIFs from data/<cohort_id>/aligned_pdbs/ by default.
Writes cluster_members.csv and clusters.csv to data/<cohort_id>/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

import config
from cw.cluster import (
    build_cluster_tables,
    find_clusters_too_close,
    run_hdbscan,
)
from cw.io import (
    collect_aligned_waters,
    read_cohort,
    resolve_aligned_water_inputs,
    write_cluster_cif,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster conserved waters from aligned structures."
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
        help="Output directory for CSVs (default: config.DATA_DIR/<cohort_id>/)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        metavar="N",
        help="Override HDBSCAN min_cluster_size (default: config.HDBSCAN_MIN_CLUSTER_SIZE)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        metavar="N",
        help="Override HDBSCAN min_samples (default: config.HDBSCAN_MIN_SAMPLES)",
    )
    parser.add_argument(
        "--write-cif",
        action="store_true",
        help="Write cluster centers to clusters.cif after computing CSVs",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Include noise waters (cluster_id == -1) in clusters.cif (requires --write-cif)",
    )
    parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=None,
        metavar="JOBS",
        help="Number of jobs to run in parallel (default: None means 1)",
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
        muse_dir=config.MUSE_DIR,
        muse_template=config.MUSE_TEMPLATE,
        cohort_id=cohort_id,
    )
    n_muse = sum(1 for _, _, muse in cif_json_pairs if muse is not None)

    n_total = len(member_ids)
    n_found = len(cif_json_pairs)
    min_cluster_size = (
        args.min_cluster_size
        if args.min_cluster_size is not None
        else config.HDBSCAN_MIN_CLUSTER_SIZE
    )
    min_samples = args.min_samples if args.min_samples is not None else config.HDBSCAN_MIN_SAMPLES

    logger.info(f"Cohort:           {cohort_id}")
    logger.info(
        f"Members:          {n_total} total — {n_found} aligned CIFs found, {n_total - n_found} missing"
    )
    logger.info(f"MUSE scores:      {n_muse}/{n_found} structures have a MUSE CSV")
    logger.info(f"Input:            {aligned_dir}")
    logger.info(f"Output:           {out_dir}")
    min_cluster_size_source = "override" if args.min_cluster_size is not None else "config"
    min_samples_source = "override" if args.min_samples is not None else "config"
    logger.info(f"Min cluster size: {min_cluster_size}  ({min_cluster_size_source})")
    logger.info(f"Min samples:      {min_samples}  ({min_samples_source})")
    logger.info(f"Cluster radius:   {config.CLUSTER_MEMBER_RADIUS} Å")

    if n_found == 0:
        logger.error(f"No aligned CIFs found in {aligned_dir}")
        sys.exit(1)

    logger.info("Collecting water records...")
    waters = collect_aligned_waters(cif_json_pairs)
    logger.info(f"  {len(waters):,} water oxygens across {n_found} structures")

    if waters.empty:
        logger.error("No water records found — check aligned CIFs")
        sys.exit(1)

    coords = waters[["x", "y", "z"]].to_numpy()

    logger.info("Running HDBSCAN...")
    logger.info(f"  n_jobs: {args.n_jobs}")
    labels = run_hdbscan(
        coords,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        n_jobs=args.n_jobs,
    )
    n_clusters = len(set(labels[labels >= 0]))
    n_noise = int((labels == -1).sum())
    logger.info(f"  {n_clusters} clusters, {n_noise:,} noise points")

    logger.info(f"Applying {config.CLUSTER_MEMBER_RADIUS} Å radius filter...")
    members_df, clusters_df = build_cluster_tables(
        waters,
        labels,
        radius=config.CLUSTER_MEMBER_RADIUS,
        n_total_structures=n_found,
    )
    n_rejected = (
        int(((members_df["cluster_id"] >= 0) & ~members_df["within_cutoff"]).sum())
        if not members_df.empty
        else 0
    )
    logger.info(
        f"  {n_rejected:,} members demoted (outside {config.CLUSTER_MEMBER_RADIUS} Å of center)"
    )

    min_separation = 2 * config.CLUSTER_MEMBER_RADIUS
    if not clusters_df.empty:
        centers = clusters_df[["center_x", "center_y", "center_z"]].to_numpy()
        close_pairs = find_clusters_too_close(centers, min_separation)
        logger.info(
            f"  {len(close_pairs)} center pair(s) within {min_separation} Å "
            f"(a single water can match both — inflates recall)"
        )
        for i, j, dist in close_pairs:
            cid_i = int(clusters_df.iloc[i]["cluster_id"])
            cid_j = int(clusters_df.iloc[j]["cluster_id"])
            logger.debug(f"    clusters {cid_i} & {cid_j}: {dist:.2f} Å apart")

    out_dir.mkdir(parents=True, exist_ok=True)
    members_path = out_dir / "cluster_members.csv"
    clusters_path = out_dir / "clusters.csv"

    members_df.to_csv(members_path, index=False)
    clusters_df.to_csv(clusters_path, index=False)

    logger.info(f"cluster_members.csv: {len(members_df):,} rows  →  {members_path}")
    logger.info(f"clusters.csv:        {len(clusters_df):,} rows  →  {clusters_path}")

    if args.write_cif:
        cif_path = out_dir / "clusters.cif"
        write_cluster_cif(
            cif_path,
            clusters_df=clusters_df,
            members_df=members_df,
            include_noise=args.include_noise,
        )
        logger.info(f"clusters.cif:        {len(clusters_df):,} centers  →  {cif_path}")


if __name__ == "__main__":
    main()
