"""Pairwise water-set agreement metrics between aligned structures.

Usage:
    # cohort mode — every ordered pair of cohort members
    uv run scripts/pairwise_water_metrics.py <cohort.txt> [--input-dir <dir>]
                                             [--cutoff <Å>] [-o <csv>]

    # phenix mode — each reference vs its cross-refinement predictors
    uv run scripts/pairwise_water_metrics.py --phenix [--ref-dir <dir>]
                                             [--phenix-dir <dir> | --results-dir <dir>]
                                             [--ref-from-self-refined]
                                             [--variant V] [--cutoff <Å>]

    # self-refined mode — re-refined reference: every ordered pair of the
    # self-refined diagonal structures (<X>_refined_by_<X>_<variant>)
    uv run scripts/pairwise_water_metrics.py --self-refined --results-dir <dir>
                                             [--variant V] [--cutoff <Å>]

Cohort mode: for every ordered pair (a, b) of cohort members, aligns b onto a
(Cα Kabsch) and compares their water oxygens — a is the reference / ground truth,
b the predicted set. One row per ordered pair → data/<cohort_id>/pairwise_metrics_{cutoff}.csv.

Phenix mode: for refinement variant V and entry (reference a, predictor b), the
predictor is <b>_refined_by_<a>_<V>.cif (phenix-dir). The reference (ground truth)
is a's original waters (ref-dir/<a>.cif) by default, or — with --ref-from-self-refined
— a's self-refinement <a>_refined_by_<a>_<V>, which keeps the section-B re-refined Δ
clean (ground truth fixed). Aligns each predictor onto its reference and compares.
One CSV per variant → data/hewls_65_subsampled_phenix/phenix_pairwise_metrics[_selfref]_{V}_{cutoff}.csv.

Self-refined mode: a re-refined alternative to the cohort reference. For variant V
and ordered pair (a, b), the reference is <a>_refined_by_<a>_<V> and the predictor
is <b>_refined_by_<b>_<V> — i.e. both structures refined against their own model.
Output uses the same columns as cohort mode → self_refined_pairwise_metrics_{V}_{cutoff}.csv.

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
from cw.io import load_protein, load_structure_waters, parse_identity, read_cohort, read_phenix_cif
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


def flat_predictors(phenix_dir: Path, variant: str) -> list[tuple[Path, str, str]]:
    """Predictors from a flat dir of <mtz_source>_refined_by_<starting_model>_<variant>.cif."""
    records = []
    for cif_path in sorted(phenix_dir.glob(f"*_refined_by_*_{variant}.cif")):
        mtz_source, starting_model, _ = parse_identity(cif_path.stem)
        records.append((cif_path, mtz_source, starting_model))
    return records


def nested_predictors(results_dir: Path, variant: str) -> list[tuple[Path, str, str]]:
    """Predictors from the phenix refinement_results tree, one dir per refinement:
    <results_dir>/<mtz_source>/refined_by_<starting_model>_<variant>/<mtz_source>_refined_by_<starting_model>_<variant>_*.cif
    """
    records = []
    for target_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        mtz_source = target_dir.name
        for refinement_dir in sorted(target_dir.glob(f"refined_by_*_{variant}")):
            if not refinement_dir.is_dir():
                continue
            starting_model = refinement_dir.name[len("refined_by_") :].rsplit("_", 1)[0]
            cifs = sorted(refinement_dir.glob(f"{mtz_source}_refined_by_{starting_model}_{variant}_*.cif"))
            if cifs:
                records.append((cifs[-1], mtz_source, starting_model))
            else:
                logger.warning(f"[{variant}] no refined CIF under {refinement_dir}")
    return records


def discover_variants(results_dir: Path) -> list[str]:
    """Variants present in a refinement_results tree (from refined_by_*_<variant> dirs)."""
    variants = set()
    for target_dir in results_dir.iterdir():
        if not target_dir.is_dir():
            continue
        for refinement_dir in target_dir.glob("refined_by_*"):
            if refinement_dir.is_dir():
                variants.add(refinement_dir.name.rsplit("_", 1)[-1])
    return sorted(variants)


def all_pairs_rows(ids, proteins, coords, cells, mobile_cifs, cutoff, tag="") -> list[dict]:
    """Compute metrics for every ordered pair (ref, mobile) drawn from `ids`.

    proteins/coords/cells are {id: value} loaded once per structure; mobile_cifs
    is {id: cif} where cif is whatever compute_pair_metrics' alignment accepts (a
    Path, or an in-memory CIFFile for cleaned phenix output). Returns rows in the
    FIELDNAMES schema. `tag` only prefixes the per-pair log lines.
    """
    prefix = f"[{tag}] " if tag else ""
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
                logger.warning(f"  {prefix}ref {structure_ref} <- mobile {structure_mobile}: alignment skipped (too few common Cα)")
            else:
                logger.info(
                    f"  {prefix}ref {structure_ref} <- mobile {structure_mobile}: "
                    f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} CD={metrics['chamfer']:.3f}"
                )
    return rows


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

    rows = all_pairs_rows(found, proteins, coords, cells, cif_paths, args.cutoff)
    _write_csv(out_path, FIELDNAMES, rows)


def self_refined_cifs(results_dir: Path, variant: str) -> dict[str, Path]:
    """{structure -> <X>_refined_by_<X>_<variant> CIF} — the diagonal of the
    cross-refinement tree (each structure's data refined against its own model).
    """
    out: dict[str, Path] = {}
    for target_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        s = target_dir.name
        refinement_dir = target_dir / f"refined_by_{s}_{variant}"
        if not refinement_dir.is_dir():
            continue
        cifs = sorted(refinement_dir.glob(f"{s}_refined_by_{s}_{variant}_*.cif"))
        if cifs:
            out[s] = cifs[-1]
        else:
            logger.warning(f"[{variant}] no self-refined CIF under {refinement_dir}")
    return out


def run_self_refined(args) -> None:
    """Re-refined reference: pairwise metrics among the self-refined structures
    (<X>_refined_by_<X>_<variant>), one CSV per variant. Same FIELDNAMES schema as
    the cohort reference, so it is a drop-in alternative baseline to compare the
    cross-refinement matrices against."""
    results_dir: Path = args.results_dir
    if results_dir is None or not results_dir.is_dir():
        logger.error(f"--self-refined needs --results-dir pointing at a refinement_results tree: {results_dir}")
        sys.exit(1)

    out_dir = results_dir.parent
    variants = [args.variant] if args.variant else discover_variants(results_dir)
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Variants:    {variants}")
    logger.info(f"Cutoff:      {args.cutoff} Å")

    for variant in variants:
        cif_paths = self_refined_cifs(results_dir, variant)
        if not cif_paths:
            logger.warning(f"[{variant}] no self-refined CIFs found under {results_dir}")
            continue
        ids = sorted(cif_paths)

        # Raw phenix CIFs are cleaned on the fly; the cleaned CIFFile is reused as
        # both the alignment reference (via load_protein) and the mobile.
        cleaned = {s: read_phenix_cif(cif_paths[s]) for s in ids}
        proteins = {s: load_protein(cleaned[s])[0] for s in ids}
        coords = {
            s: load_structure_waters(cleaned[s], pdb_id=s)[["x", "y", "z"]].to_numpy() for s in ids
        }
        cells = {s: gemmi.read_structure(str(cif_paths[s])).cell for s in ids}

        rows = all_pairs_rows(ids, proteins, coords, cells, cleaned, args.cutoff, tag=variant)
        out_path = (
            args.output
            if (args.output and len(variants) == 1)
            else out_dir / f"self_refined_pairwise_metrics_{variant}_{args.cutoff}.csv"
        )
        _write_csv(out_path, FIELDNAMES, rows)


def run_phenix(args) -> None:
    ref_dir = args.ref_dir or Path(config.DATA_DIR) / "hewls_65" / "filtered_pdbs"

    # Two predictor layouts: a flat dir of cleaned CIFs (--phenix-dir, legacy), or
    # the nested refinement_results tree of raw phenix output (--results-dir). Raw
    # CIFs are cleaned on the fly via read_phenix_cif so no filtered_pdbs dir is needed.
    if args.results_dir:
        results_dir = args.results_dir
        out_dir = results_dir.parent
        predictor_source_desc = results_dir
        variants = [args.variant] if args.variant else discover_variants(results_dir)
        nested = True
    else:
        phenix_dir = args.phenix_dir or Path(config.DATA_DIR) / "hewls_65_subsampled_phenix" / "filtered_pdbs"
        out_dir = phenix_dir.parent
        predictor_source_desc = phenix_dir
        variants = [args.variant] if args.variant else VARIANTS
        nested = False

    ref_source = "self-refined <a>_refined_by_<a>" if args.ref_from_self_refined else f"deposited {ref_dir}"
    logger.info(f"Reference:      {ref_source}")
    logger.info(f"Predictor dir:  {predictor_source_desc}")
    logger.info(f"Variants:       {variants}")
    logger.info(f"Cutoff:         {args.cutoff} Å")

    for variant in variants:
        records = nested_predictors(results_dir, variant) if nested else flat_predictors(phenix_dir, variant)
        if not records:
            logger.warning(f"[{variant}] no predictor CIFs found in {predictor_source_desc}")
            continue

        # Each reference's protein + waters are loaded once and reused. By default
        # the reference is the original deposited structure (ref_dir/<a>.cif); with
        # --ref-from-self-refined it is the self-refinement <a>_refined_by_<a>_<variant>,
        # so a re-refined-reference Δ in section B holds the ground truth fixed.
        references = sorted({starting_model for _, _, starting_model in records})
        if args.ref_from_self_refined:
            if nested:
                ref_srcs = self_refined_cifs(results_dir, variant)
                ref_clean = True  # raw phenix CIFs, clean on the fly
            else:
                ref_srcs = {a: phenix_dir / f"{a}_refined_by_{a}_{variant}.cif" for a in references}
                ref_clean = False  # flat dir already holds cleaned CIFs
        else:
            ref_srcs = {a: ref_dir / f"{a}.cif" for a in references}
            ref_clean = False
        present = [a for a in references if a in ref_srcs and ref_srcs[a].exists()]
        absent = [a for a in references if a not in present]
        if absent:
            logger.warning(f"[{variant}] missing reference CIFs (skipped): {absent}")
        ref_objs = {a: (read_phenix_cif(ref_srcs[a]) if ref_clean else ref_srcs[a]) for a in present}
        ref_proteins = {a: load_protein(ref_objs[a])[0] for a in present}
        ref_coords = {
            a: load_structure_waters(ref_objs[a], pdb_id=a)[["x", "y", "z"]].to_numpy() for a in present
        }

        rows = []
        for cif_path, mtz_source, starting_model in records:
            if starting_model not in ref_proteins:
                continue
            predictor = read_phenix_cif(cif_path) if nested else cif_path
            pred_coords = load_structure_waters(predictor, pdb_id=mtz_source)[["x", "y", "z"]].to_numpy()
            metrics = compute_pair_metrics(
                ref_coords[starting_model], ref_proteins[starting_model], predictor, pred_coords, args.cutoff
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

        _kind = "phenix_pairwise_metrics_selfref" if args.ref_from_self_refined else "phenix_pairwise_metrics"
        out_path = (
            args.output
            if (args.output and len(variants) == 1)
            else out_dir / f"{_kind}_{variant}_{args.cutoff}.csv"
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
        "--self-refined",
        dest="self_refined",
        action="store_true",
        help="Re-refined reference mode: pairwise metrics among the self-refined "
        "diagonal CIFs (<X>_refined_by_<X>_<variant>). Needs --results-dir; one CSV "
        "per variant → self_refined_pairwise_metrics_<variant>_<cutoff>.csv.",
    )
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
        help="(phenix) flat refined CIF dir (default: DATA_DIR/hewls_65_subsampled_phenix/filtered_pdbs)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="(phenix) nested refinement_results tree; overrides --phenix-dir. Raw phenix "
        "CIFs are read straight from here and cleaned on the fly (no filtered_pdbs needed). "
        "Output CSVs go to its parent dir.",
    )
    parser.add_argument(
        "--ref-from-self-refined",
        dest="ref_from_self_refined",
        action="store_true",
        help="(phenix) use the self-refinement <a>_refined_by_<a>_<variant> as the "
        "ground-truth reference instead of ref-dir/<a>.cif, so a re-refined-reference "
        "Δ holds the ground truth fixed. Output → phenix_pairwise_metrics_selfref_<variant>_<cutoff>.csv.",
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

    if args.self_refined:
        run_self_refined(args)
    elif args.phenix:
        run_phenix(args)
    else:
        run_cohort(args)


if __name__ == "__main__":
    main()
