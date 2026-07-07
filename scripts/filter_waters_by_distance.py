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
from cw.filter import filter_waters
from cw.io import cif_path_for, load_edia_all_altlocs, read_cohort, write_filtered_cif


def _filter_one(
    member_id: str,
    cif_path: Path,
    out_path: Path,
    cutoff: float,
    edia_json: Path | None,
    edia_cutoff: float | None,
    drop_if_no_edia_json: bool,
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

        edia_lists: dict | None = None
        edia_applied = False
        if edia_cutoff is not None:
            if edia_json is not None and edia_json.exists():
                edia_lists = load_edia_all_altlocs(edia_json)
                edia_applied = True
            elif drop_if_no_edia_json:
                edia_lists = {}  # no scores → every water fails the cutoff
                edia_applied = True

        filtered, stats, keep_mask = filter_waters(
            atoms, cell, sg, cutoff, edia_lists=edia_lists, edia_cutoff=edia_cutoff
        )
        write_filtered_cif(cif_file, keep_mask, filtered, out_path)
        return (
            {
                "pdb_id": member_id,
                "n_waters_before": stats["n_water"],
                "n_waters_moved": stats["n_moved"],
                "n_waters_removed": stats["n_removed_distance"],
                "n_waters_removed_edia": stats["n_removed_edia"],
                "n_waters_remaining": stats["n_water"]
                - stats["n_removed_distance"]
                - stats["n_removed_edia"],
                "edia_applied": edia_applied,
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
        "--edia-cutoff",
        type=float,
        default=None,
        metavar="X",
        help="Minimum EDIAm to keep a water (e.g. 0.8). Off by default (distance only).",
    )
    parser.add_argument(
        "--drop-if-no-edia-json",
        action="store_true",
        help="With --edia-cutoff, drop all waters of a structure whose EDIA JSON is "
        "missing (default: keep them and warn).",
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

    edia_paths = {
        m: cif_path_for(m, config.ALL_PDB_REDO_DIR, config.EDIA_TEMPLATE) for m in found
    }

    logger.info(f"Cohort:  {cohort_id}")
    logger.info(f"Members: {len(member_ids)} total — {len(found)} found, {len(missing)} missing")
    if missing:
        logger.warning(f"Missing CIFs: {', '.join(missing)}")
    logger.info(f"Output:  {out_dir}")
    logger.info(f"Cutoff:  {args.cutoff} Å")
    if args.edia_cutoff is not None:
        logger.info(f"EDIA:    min {args.edia_cutoff}")
        no_json = [m for m, p in edia_paths.items() if not p.exists()]
        if no_json:
            action = "drop all their waters" if args.drop_if_no_edia_json else "keep their waters"
            logger.warning(f"Missing EDIA JSON for {len(no_json)} structure(s) — will {action}: {', '.join(no_json)}")
    logger.info(f"Jobs:    {args.jobs}")

    out_dir.mkdir(parents=True, exist_ok=True)

    results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(_filter_one)(
            m,
            cif_paths[m],
            out_dir / f"{m}.cif",
            args.cutoff,
            edia_paths[m],
            args.edia_cutoff,
            args.drop_if_no_edia_json,
        )
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
                f"edia-removed {row['n_waters_removed_edia']}  "
                f"→ {row['n_waters_remaining']} remaining"
            )

    rows.sort(key=lambda r: r["pdb_id"])

    cutoff_str = f"{args.cutoff:.2f}A"
    fieldnames = [
        "pdb_id",
        "n_waters_before",
        "n_waters_moved",
        "n_waters_removed",
        "n_waters_remaining",
    ]
    if args.edia_cutoff is not None:
        cutoff_str += f"_edia{args.edia_cutoff:.2f}"
        fieldnames[4:4] = ["n_waters_removed_edia"]
        fieldnames.append("edia_applied")
    report_path = out_dir / f"filtering_report_{cutoff_str}.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    for error in errors:
        logger.error(f"  {error}")
    logger.info(f"Done: {len(rows)} written, {len(errors)} errors")
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
