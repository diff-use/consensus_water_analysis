"""Pairwise Cα alignment of all cohort members onto a reference structure.

Usage:
    uv run scripts/align_structures.py <cohort.txt> [--reference <pdb_id>]
                                       [--input-dir <dir>] [--raw] [-o <output_dir>]

By default reads CIFs from data/<cohort_id>/filtered_pdbs/ (Stage 2 output).
Pass --raw to align from the original CIFs in ALL_PDB_REDO_DIR instead.
Writes aligned CIFs to data/<cohort_id>/aligned_pdbs/ and alignment_report_{ref_pdb_id}.csv.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

import config
from cw.align import align_to_reference
from cw.io import cif_path_for, load_protein, read_cohort

# Per-worker cache so each loky worker loads the reference protein only once,
# rather than pickling it across for every one of ~1000 tasks.
_REF_CACHE: dict[str, object] = {}


def _ref_protein(ref_cif: Path):
    key = str(ref_cif)
    protein = _REF_CACHE.get(key)
    if protein is None:
        protein, _ = load_protein(ref_cif)
        _REF_CACHE[key] = protein
    return protein


def _align_one(
    member_id: str, in_cif: Path, out_cif: Path, ref_cif: Path
) -> tuple[dict | None, str | None, str | None]:
    """Align one structure onto the reference. Returns (report, skip_id, error)."""
    try:
        report = align_to_reference(in_cif, _ref_protein(ref_cif), out_path=out_cif)
        if report is None:
            return (None, member_id, None)
        return (report, None, None)
    except Exception as exc:
        return (None, None, f"{member_id}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align cohort structures onto a reference.")
    parser.add_argument("cohort", type=Path, help="Cohort .txt file (one member ID per line)")
    parser.add_argument(
        "--reference",
        default=None,
        metavar="PDB_ID",
        help="Reference PDB ID (default: config.REF_PDB_ID)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory of input CIFs (default: config.DATA_DIR/<cohort_id>/filtered_pdbs/)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Align from original CIFs in ALL_PDB_REDO_DIR, ignoring filtered_pdbs/",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for aligned CIFs (default: config.DATA_DIR/<cohort_id>/aligned_pdbs/)",
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
    ref_id = args.reference or config.REF_PDB_ID
    data_dir = Path(config.DATA_DIR) / cohort_id

    out_dir = args.output_dir or data_dir / "aligned_pdbs"
    report_path = out_dir / f"alignment_report_{ref_id}.csv"

    # Reference CIF (original, altloc labels intact); workers load it once each.
    ref_cif = cif_path_for(ref_id, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE)
    if not ref_cif.exists():
        logger.error(f"Reference CIF not found: {ref_cif}")
        sys.exit(1)

    # Decide input source
    member_ids = read_cohort(cohort_path)

    if args.raw:
        use_filtered = False
    elif args.input_dir is not None:
        use_filtered = True
        filtered_dir = args.input_dir
    else:
        filtered_dir = data_dir / "filtered_pdbs"
        use_filtered = filtered_dir.exists()

    if use_filtered:
        logger.info(f"Input:   {filtered_dir}  (filtered)")
    else:
        logger.info(f"Input:   {config.ALL_PDB_REDO_DIR}  (raw)")

    def _input_cif(member_id: str) -> Path:
        if use_filtered:
            return filtered_dir / f"{member_id}.cif"
        return cif_path_for(member_id, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE)

    cif_paths = {m: _input_cif(m) for m in member_ids}
    missing = [m for m, p in cif_paths.items() if not p.exists()]
    found = [m for m, p in cif_paths.items() if p.exists()]

    logger.info(f"Cohort:  {cohort_id}")
    logger.info(f"Members: {len(member_ids)} total — {len(found)} found, {len(missing)} missing")
    logger.info(f"Ref:     {ref_id}")
    logger.info(f"Output:  {out_dir}")
    logger.info(f"Jobs:    {args.jobs}")

    out_dir.mkdir(parents=True, exist_ok=True)

    results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(_align_one)(m, cif_paths[m], out_dir / f"{m}.cif", ref_cif)
        for m in found
    )

    rows, skipped, errors = [], [], []
    for report, skip_id, error in tqdm(results, total=len(found), desc="aligning"):
        if error is not None:
            errors.append(error)
        elif skip_id is not None:
            skipped.append(skip_id)
            logger.debug(f"  {skip_id}: skipped (too few common Cα)")
        else:
            rows.append(report)
            logger.debug(
                f"  {report['pdb_id']}: {report['n_common_ca']} Cα  "
                f"RMSD {report['rmsd_before']} → {report['rmsd_after']} Å"
            )

    rows.sort(key=lambda r: r["pdb_id"])

    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pdb_id", "n_common_ca", "rmsd_before", "rmsd_after"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    for error in errors:
        logger.error(f"  {error}")
    logger.info(f"Aligned {len(rows)} structures → {out_dir}")
    logger.info(f"Report:  {report_path}")
    if skipped:
        logger.warning(f"{len(skipped)} skipped (< 10 common Cα)")
    if errors:
        logger.warning(f"{len(errors)} errors (see above)")


if __name__ == "__main__":
    main()
