"""Pairwise water-set agreement metrics between aligned structures.

Usage:
    # cohort mode — every ordered pair of cohort members
    uv run scripts/pairwise_water_metrics.py <cohort.txt> [--input-dir <dir>]
                                             [--cutoff <Å>] [-o <csv>]

    # phenix mode — each original reference vs its cross-refinement predictors
    uv run scripts/pairwise_water_metrics.py --phenix [--ref-dir <dir>]
                                             [--phenix-dir <dir>] [--variant V]
                                             [--cutoff <Å>]

Cohort mode: for every ordered pair (a, b) of cohort members, aligns b onto a
(Cα Kabsch) and compares their water oxygens — a is the reference / ground truth,
b the predicted set. One row per ordered pair → data/<cohort_id>/pairwise_metrics_{cutoff}.csv.

Phenix mode: for refinement variant V and entry (reference a, predictor b), the
reference is a's original waters (ref-dir/<a>.cif) and the predictor is
<b>_refined_by_<a>_<V>.cif (phenix-dir). Aligns each predictor onto its reference
and compares. One CSV per variant → data/hewls_65_subsampled_phenix/phenix_pairwise_metrics_{V}_{cutoff}.csv.

Both modes: precision/recall/f1 (many-to-one), matched precision/recall
(one-to-one) and chamfer distance. The cutoff is applied uniformly, so it lives in
the filename rather than a column; a failed alignment leaves NaN in rmsd_after and
the metric columns.
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
from cw.io import load_protein, load_structure_waters, parse_identity, read_cohort
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
# Reference (cohort) output keeps rmsd_after and adds a unit-cell difference
# column; phenix output drops rmsd_after.
FIELDNAMES = ["structure_ref", "structure_mobile", "n_water_ref", "n_water_mobile", *METRIC_FIELDS, "max_cell_diff"]
PHENIX_FIELDNAMES = [
    "reference",
    "predictor",
    "n_water_ref",
    "n_water_pred",
    *[k for k in METRIC_FIELDS if k != "rmsd_after"],
]
VARIANTS = ["auto", "fixed", "stripped"]


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


def run_cohort(args) -> None:
    cohort_path: Path = args.cohort
    if cohort_path is None or not cohort_path.exists():
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
        pdb_id: load_structure_waters(cif_paths[pdb_id])[["x", "y", "z"]].to_numpy() for pdb_id in found
    }
    cells = {pdb_id: gemmi.read_structure(str(cif_paths[pdb_id])).cell for pdb_id in found}

    rows = []
    for structure_ref in found:
        for structure_mobile in found:
            metrics = compute_pair_metrics(
                coords[structure_ref],
                proteins[structure_ref],
                cif_paths[structure_mobile],
                coords[structure_mobile],
                args.cutoff,
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
                logger.warning(f"  ref {structure_ref} <- mobile {structure_mobile}: alignment skipped (too few common Cα)")
            else:
                logger.info(
                    f"  ref {structure_ref} <- mobile {structure_mobile}: "
                    f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} CD={metrics['chamfer']:.3f}"
                )

    _write_csv(out_path, FIELDNAMES, rows)


def run_phenix(args) -> None:
    ref_dir = args.ref_dir or Path(config.DATA_DIR) / "hewls_65" / "filtered_pdbs"
    phenix_dir = args.phenix_dir or Path(config.DATA_DIR) / "hewls_65_subsampled_phenix" / "filtered_pdbs"
    out_dir = phenix_dir.parent
    variants = [args.variant] if args.variant else VARIANTS

    logger.info(f"Reference dir: {ref_dir}")
    logger.info(f"Phenix dir:    {phenix_dir}")
    logger.info(f"Variants:      {variants}")
    logger.info(f"Cutoff:        {args.cutoff} Å")

    for variant in variants:
        files = sorted(phenix_dir.glob(f"*_refined_by_*_{variant}.cif"))
        if not files:
            logger.warning(f"[{variant}] no predictor CIFs found in {phenix_dir}")
            continue

        # Each reference's original protein + waters are loaded once and reused.
        references = sorted({parse_identity(f.stem)[1] for f in files})
        ref_cifs = {a: ref_dir / f"{a}.cif" for a in references}
        present = [a for a in references if ref_cifs[a].exists()]
        absent = [a for a in references if not ref_cifs[a].exists()]
        if absent:
            logger.warning(f"[{variant}] missing reference CIFs (skipped): {absent}")
        ref_proteins = {a: load_protein(ref_cifs[a])[0] for a in present}
        ref_coords = {
            a: load_structure_waters(ref_cifs[a])[["x", "y", "z"]].to_numpy() for a in present
        }

        rows = []
        for f in files:
            mtz_source, starting_model, _ = parse_identity(f.stem)
            if starting_model not in ref_proteins:
                continue
            pred_coords = load_structure_waters(f)[["x", "y", "z"]].to_numpy()
            metrics = compute_pair_metrics(
                ref_coords[starting_model], ref_proteins[starting_model], f, pred_coords, args.cutoff
            )
            rows.append(
                {
                    "reference": starting_model,
                    "predictor": mtz_source,
                    "n_water_ref": len(ref_coords[starting_model]),
                    "n_water_pred": len(pred_coords),
                    **metrics,
                }
            )
            if np.isnan(metrics["n_common_ca"]):
                logger.warning(f"  [{variant}] ref {starting_model} <- {mtz_source}: alignment skipped (too few common Cα)")
            else:
                logger.info(
                    f"  [{variant}] ref {starting_model} <- {mtz_source}: "
                    f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} CD={metrics['chamfer']:.3f}"
                )

        out_path = (
            args.output
            if (args.output and len(variants) == 1)
            else out_dir / f"phenix_pairwise_metrics_{variant}_{args.cutoff}.csv"
        )
        _write_csv(out_path, PHENIX_FIELDNAMES, rows)


def _write_csv(out_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise water-set agreement metrics.")
    parser.add_argument("cohort", type=Path, nargs="?", help="Cohort .txt file (cohort mode)")
    parser.add_argument("--phenix", action="store_true", help="Phenix cross-refinement mode")
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="(phenix) original reference CIF dir (default: DATA_DIR/hewls_65/filtered_pdbs)",
    )
    parser.add_argument(
        "--phenix-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="(phenix) refined CIF dir (default: DATA_DIR/hewls_65_subsampled_phenix/filtered_pdbs)",
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default=None,
        help="(phenix) variant to run (default: all three)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="(cohort) directory of input CIFs (default: config.DATA_DIR/<cohort_id>/filtered_pdbs/)",
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
        help="Output CSV (cohort mode, or phenix mode with a single --variant)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Show debug output")
    verbosity.add_argument("--quiet", action="store_true", help="Show warnings and errors only")
    args = parser.parse_args()

    logger.remove()
    level = "DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO"
    logger.add(sys.stderr, level=level)

    if args.phenix:
        run_phenix(args)
    else:
        run_cohort(args)


if __name__ == "__main__":
    main()
