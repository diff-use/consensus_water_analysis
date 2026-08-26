"""Pairwise water-set agreement metrics between aligned structures.

For every ordered pair (a, b) of cohort members, b is aligned onto a (Cα Kabsch)
and their water oxygens are compared: precision/recall/f1 (many-to-one), matched
precision/recall (one-to-one), chamfer distance, plus rmsd_after and
max_cell_diff. The cutoff is applied uniformly, so it lives in the filename
rather than a column; a failed alignment leaves NaN.

Input CIFs are read already distance-filtered (Stage 2 output).

Usage:
    uv run scripts/pairwise_water_metrics.py <cohort.txt> [--input-dir <dir>]
                                             [--cutoff <Å>] [-o <csv>]

    Input  (required): <cohort.txt>  — one PDB id per line.
    Input  (CIFs):     --input-dir/<id>.cif   [default: data/<cohort_id>/filtered_pdbs/]
    Output:            -o <csv>               [default: data/<cohort_id>/pairwise_metrics_<cutoff>.csv]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gemmi
import numpy as np
from loguru import logger

import config
from cw.align import align_to_reference
from cw.io import load_protein, load_structure_waters, read_cohort
from cw.metadata import max_cell_diff
from cw.metrics import chamfer_distance, matched_precision_recall, precision_recall

METRIC_FIELDS = [
    "n_common_ca",
    "rmsd_after",
    "precision",
    "recall",
    "f1",
    "matched_precision",
    "matched_recall",
    "chamfer",
]
FIELDNAMES = [
    "structure_ref",
    "structure_mobile",
    "n_water_ref",
    "n_water_mobile",
    *METRIC_FIELDS,
    "max_cell_diff",
]


def compute_pair_metrics(ref_coords, ref_protein, predictor_cif, predictor_coords, cutoff) -> dict:
    """Align predictor_cif's protein onto ref_protein (Cα Kabsch), transform
    predictor_coords into the reference frame, and return {METRIC_FIELDS: value}.

    Values are NaN-filled when the alignment is skipped (too few common Cα) or
    either water set is empty.
    """
    metrics = {k: float("nan") for k in METRIC_FIELDS}
    report = align_to_reference(predictor_cif, ref_protein, out_path=None)
    if report is None:
        return metrics

    metrics["n_common_ca"] = report["n_common_ca"]
    metrics["rmsd_after"] = report["rmsd_after"]
    if len(ref_coords) == 0 or len(predictor_coords) == 0:
        return metrics

    moved = (report["R"] @ predictor_coords.T).T + report["t"]
    pr = precision_recall(ref_coords, moved, cutoff)
    mpr = matched_precision_recall(ref_coords, moved, cutoff)
    metrics["precision"] = pr["precision"]
    metrics["recall"] = pr["recall"]
    metrics["f1"] = pr["f1"]
    metrics["matched_precision"] = mpr["precision"]
    metrics["matched_recall"] = mpr["recall"]
    metrics["chamfer"] = chamfer_distance(ref_coords, moved)
    return metrics


def all_pairs_rows(ids, proteins, coords, cells, mobile_cifs, cutoff) -> list[dict]:
    """Compute metrics for every ordered pair (ref, mobile) drawn from `ids`.

    proteins/coords/cells are {id: value} loaded once per structure; mobile_cifs
    is {id: cif_path}. Returns rows in the FIELDNAMES schema.
    """
    rows = []
    for structure_ref in ids:
        for structure_mobile in ids:
            metrics = compute_pair_metrics(
                coords[structure_ref],
                proteins[structure_ref],
                mobile_cifs[structure_mobile],
                coords[structure_mobile],
                cutoff,
            )
            rows.append(
                {
                    "structure_ref": structure_ref,
                    "structure_mobile": structure_mobile,
                    "n_water_ref": len(coords[structure_ref]),
                    "n_water_mobile": len(coords[structure_mobile]),
                    **metrics,
                    "max_cell_diff": max_cell_diff(cells[structure_ref], cells[structure_mobile]),
                }
            )
            if np.isnan(metrics["n_common_ca"]):
                logger.warning(
                    f"  ref {structure_ref} <- mobile {structure_mobile}: alignment skipped (too few common Cα)"
                )
            else:
                logger.info(
                    f"  ref {structure_ref} <- mobile {structure_mobile}: "
                    f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} CD={metrics['chamfer']:.3f}"
                )
    return rows


def run_cohort(args) -> None:
    cohort_path: Path = args.cohort
    if not cohort_path.exists():
        logger.error(f"Cohort file not found: {cohort_path}")
        sys.exit(1)

    cohort_id = cohort_path.stem
    data_dir = Path(config.DATA_DIR) / cohort_id
    input_dir = args.input_dir or data_dir / "filtered_pdbs"
    out_path = args.output or data_dir / f"pairwise_metrics_{args.cutoff}.csv"

    pdb_ids = read_cohort(cohort_path)
    cif_paths = {pdb_id: input_dir / f"{pdb_id}.cif" for pdb_id in pdb_ids}
    found = [pdb_id for pdb_id in pdb_ids if cif_paths[pdb_id].exists()]
    missing = [pdb_id for pdb_id in pdb_ids if not cif_paths[pdb_id].exists()]

    logger.info(f"Cohort:  {cohort_id}")
    logger.info(f"Input:   {input_dir}")
    logger.info(f"PDB IDs: {len(pdb_ids)} total — {len(found)} found, {len(missing)} missing")
    logger.info(f"Cutoff:  {args.cutoff} Å")
    logger.info(f"Output:  {out_path}")
    if missing:
        logger.warning(f"Missing CIFs: {missing}")

    # Load each structure once: protein (alignment reference), water coords, cell.
    proteins = {pdb_id: load_protein(cif_paths[pdb_id])[0] for pdb_id in found}
    coords = {
        pdb_id: load_structure_waters(cif_paths[pdb_id])[["x", "y", "z"]].to_numpy()
        for pdb_id in found
    }
    cells = {pdb_id: gemmi.read_structure(str(cif_paths[pdb_id])).cell for pdb_id in found}

    rows = all_pairs_rows(found, proteins, coords, cells, cif_paths, args.cutoff)
    _write_csv(out_path, FIELDNAMES, rows)


def _write_csv(out_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise water-set agreement metrics.")
    parser.add_argument("cohort", type=Path, help="Cohort .txt file")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory of input CIFs (default: config.DATA_DIR/<cohort_id>/filtered_pdbs/)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=config.CLUSTER_MEMBER_RADIUS,
        help="Match distance in Å (default: config.CLUSTER_MEMBER_RADIUS)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: config.DATA_DIR/<cohort_id>/pairwise_metrics_<cutoff>.csv)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Show debug output")
    verbosity.add_argument("--quiet", action="store_true", help="Show warnings and errors only")
    args = parser.parse_args()

    logger.remove()
    level = "DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO"
    logger.add(sys.stderr, level=level)

    run_cohort(args)


if __name__ == "__main__":
    main()
