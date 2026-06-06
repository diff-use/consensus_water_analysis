"""Filter waters by distance to protein for every phenix cross-refinement CIF.

Usage:
    uv run scripts/filter_waters_phenix.py <cohort.txt> [--cutoff 4.0]

The cohort .txt is used only to locate data/<cohort_id>_phenix/. Every refined
CIF found under refinement_results/<A>/refined_by_<B>_<variant>/*_001.cif is
filtered (self-pairs included). Writes filtered CIFs to
data/<cohort_id>_phenix/filtered_pdbs/<A>_refined_by_<B>_<variant>.cif and a
filtering_report_{cutoff}.csv to the same output directory.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import biotite.structure.io.pdbx as pdbx
import gemmi
from loguru import logger

import config
from cw.filter import filter_by_distance
from cw.io import parse_identity, read_phenix_cif, write_filtered_cif


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter waters by protein distance for phenix cross-refinement CIFs."
    )
    parser.add_argument("cohort", type=Path, help="Cohort .txt file (locates the phenix dir)")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=config.WATER_PROT_DIST_CUTOFF,
        help=f"Distance cutoff in Å (default: {config.WATER_PROT_DIST_CUTOFF})",
    )
    parser.add_argument(
        "--phenix-dir",
        type=Path,
        default=None,
        help="Phenix root dir (default: config.DATA_DIR/<cohort_id>_phenix/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <phenix-dir>/filtered_pdbs/)",
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

    cohort_id = args.cohort.stem
    phenix_dir: Path = args.phenix_dir or Path(config.DATA_DIR) / f"{cohort_id}_phenix"
    out_dir: Path = args.output_dir or phenix_dir / "filtered_pdbs"

    refinement_dir = phenix_dir / "refinement_results"
    if not refinement_dir.exists():
        logger.error(f"refinement_results/ not found under: {phenix_dir}")
        sys.exit(1)

    cif_paths = sorted(refinement_dir.glob("*/refined_by_*/*_001.cif"))

    logger.info(f"Cohort:  {cohort_id}")
    logger.info(f"Phenix:  {phenix_dir}")
    logger.info(f"Refined CIFs: {len(cif_paths)}")
    logger.info(f"Output:  {out_dir}")
    logger.info(f"Cutoff:  {args.cutoff} Å")

    rows, n_ok, n_err = [], 0, 0
    for cif_path in cif_paths:
        identity = cif_path.stem.removesuffix("_001")
        try:
            origin, refined_by, variant = parse_identity(identity)

            st = gemmi.read_structure(str(cif_path))
            cell = st.cell
            sg = st.find_spacegroup() or gemmi.SpaceGroup("P 1")

            cif_file = read_phenix_cif(cif_path)
            atoms = pdbx.get_structure(
                cif_file,
                model=1,
                altloc="all",
                extra_fields=["b_factor", "occupancy"],
            )
            filtered, n_water, n_removed, n_moved, keep_mask = filter_by_distance(
                atoms, cell, sg, args.cutoff
            )

            out_path = out_dir / f"{identity}.cif"
            write_filtered_cif(cif_file, keep_mask, filtered, out_path)

            rows.append(
                {
                    "identity": identity,
                    "origin": origin,
                    "refined_by": refined_by,
                    "variant": variant,
                    "n_waters_before": n_water,
                    "n_waters_moved": n_moved,
                    "n_waters_removed": n_removed,
                    "n_waters_remaining": n_water - n_removed,
                }
            )
            n_ok += 1
            logger.info(
                f"  {identity}: {n_water} waters  "
                f"moved {n_moved}  removed {n_removed}  "
                f"→ {n_water - n_removed} remaining"
            )
        except Exception as exc:
            n_err += 1
            logger.error(f"  {identity}: {exc}")

    cutoff_str = f"{args.cutoff:.2f}A"
    report_path = out_dir / f"filtering_report_{cutoff_str}.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "identity",
                "origin",
                "refined_by",
                "variant",
                "n_waters_before",
                "n_waters_moved",
                "n_waters_removed",
                "n_waters_remaining",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Done: {n_ok} written, {n_err} errors")
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
