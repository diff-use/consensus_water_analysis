"""Filter waters by distance to protein for every structure in a cohort .txt.

Usage:
    uv run scripts/filter_waters_by_distance.py <cohort.txt> [--cutoff 4.0]

Writes filtered CIFs to data/<cohort_id>/filtered_pdbs/<member_id>.cif.
Writes filtering_report_{cutoff}.csv to the same output directory.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import biotite.structure.io.pdbx as pdbx
import gemmi
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

import config
from cw.filter import filter_by_distance
from cw.io import cif_path_for, read_cohort, write_filtered_cif


def _filter_one(
    member_id: str, cif_path: Path, out_path: Path, cutoff: float
) -> tuple[dict | None, str | None]:
    """Filter one structure's waters and write the result. Returns (report_row, error)."""
    try:
        st = gemmi.read_structure(str(cif_path))
        cell = st.cell
        sg = st.find_spacegroup() or gemmi.SpaceGroup("P 1")

        cif_file = pdbx.CIFFile.read(cif_path)
        atoms = pdbx.get_structure(
            cif_file,
            model=1,
            altloc="all",
            extra_fields=["b_factor", "occupancy"],
        )
        filtered, n_water, n_removed, n_moved, keep_mask = filter_by_distance(
            atoms, cell, sg, cutoff
        )
        write_filtered_cif(cif_file, keep_mask, filtered, out_path)
        return (
            {
                "pdb_id": member_id,
                "n_waters_before": n_water,
                "n_waters_moved": n_moved,
                "n_waters_removed": n_removed,
                "n_waters_remaining": n_water - n_removed,
            },
            None,
        )
    except Exception as exc:
        return (None, f"{member_id}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter waters by protein distance for a cohort.")
    parser.add_argument("cohort", type=Path, help="Cohort .txt file (one member ID per line)")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=config.WATER_PROT_DIST_CUTOFF,
        help=f"Distance cutoff in Å (default: {config.WATER_PROT_DIST_CUTOFF})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: config.DATA_DIR/<cohort_id>/filtered_pdbs/)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count(),
        metavar="N",
        help="Parallel worker processes (default: all CPUs)",
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
    out_dir: Path = args.output_dir or Path(config.DATA_DIR) / cohort_id / "filtered_pdbs"

    member_ids = read_cohort(cohort_path)
    cif_paths = {
        m: cif_path_for(m, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE) for m in member_ids
    }
    missing = [m for m, p in cif_paths.items() if not p.exists()]
    found = [m for m, p in cif_paths.items() if p.exists()]

    logger.info(f"Cohort:  {cohort_id}")
    logger.info(f"Members: {len(member_ids)} total — {len(found)} found, {len(missing)} missing")
    if missing:
        logger.warning(f"Missing CIFs: {', '.join(missing)}")
    logger.info(f"Output:  {out_dir}")
    logger.info(f"Cutoff:  {args.cutoff} Å")
    logger.info(f"Jobs:    {args.jobs}")

    out_dir.mkdir(parents=True, exist_ok=True)

    results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(_filter_one)(m, cif_paths[m], out_dir / f"{m}.cif", args.cutoff)
        for m in found
    )

    rows, errors = [], []
    for row, error in tqdm(results, total=len(found), desc="filtering"):
        if error is not None:
            errors.append(error)
        else:
            rows.append(row)
            logger.debug(
                f"  {row['pdb_id']}: {row['n_waters_before']} waters  "
                f"moved {row['n_waters_moved']}  removed {row['n_waters_removed']}  "
                f"→ {row['n_waters_remaining']} remaining"
            )

    rows.sort(key=lambda r: r["pdb_id"])

    cutoff_str = f"{args.cutoff:.2f}A"
    report_path = out_dir / f"filtering_report_{cutoff_str}.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdb_id",
                "n_waters_before",
                "n_waters_moved",
                "n_waters_removed",
                "n_waters_remaining",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for error in errors:
        logger.error(f"  {error}")
    logger.info(f"Done: {len(rows)} written, {len(errors)} errors")
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
