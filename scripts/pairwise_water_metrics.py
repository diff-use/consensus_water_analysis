"""Pairwise water-set agreement metrics between aligned structures.

Every mode Cα-aligns a mobile/predictor structure onto a reference (Kabsch) and compares their
water oxygens: precision/recall/f1 (many-to-one), matched precision/recall (one-to-one) and
chamfer distance. The cutoff is uniform, so it lives in the filename, not a column; a failed
alignment leaves NaN. Raw phenix waters (phenix / self-refined / partition modes) are
symmetry-aware distance-filtered to within --filter-cutoff Å of protein before comparison —
matching the deposited references — unless --no-filter; cohort mode reads already-filtered CIFs.

Four modes (purpose · command · where the output is viewed):

  cohort — every ordered pair of a cohort's filtered CIFs (deposited-vs-deposited), for
    cohort-homogeneity / reference-pick diagnostics.
      pairwise_water_metrics.py <cohort.txt> [--input-dir DIR] [--cutoff Å] [-o CSV]
      → data/<cohort>/pairwise_metrics_<cutoff>.csv, viewed in
        notebooks/optional_find_isomorphous_subset_and_align_ref.py, Part 2 (§5–§7);
        with -o .../reference_pairwise_metrics_<cutoff>.csv it is the deposited baseline in
        notebooks/02_refinement_water_heatmaps.py (Summary + Section B).

  --self-refined — pairwise metrics among the self-refined diagonal CIFs
    (<X>_refined_by_<X>_<V>), a re-refined-reference baseline.
      pairwise_water_metrics.py --self-refined --results-dir DIR [--variant V] [--no-filter] [-o CSV]
      → <results-dir parent>/self_refined_pairwise_metrics_<V>_<cutoff>.csv, viewed in
        notebooks/02_refinement_water_heatmaps.py (Section B).

  --phenix — each reference vs its cross-refinement predictors (<b>_refined_by_<a>_<V>),
    grounded on deposited waters or, with --ref-from-self-refined, on <a>_refined_by_<a>.
      pairwise_water_metrics.py --phenix --results-dir DIR (--ref-dir DIR | --ref-from-self-refined) [--variant V] [--no-filter] [-o CSV]
      → <results-dir parent>/phenix_pairwise_metrics[_selfref]_<V>_<cutoff>.csv, viewed in
        notebooks/02_refinement_water_heatmaps.py (Section B).

  --partition — starting-model-bias decomposition: per off-diagonal pair, the cross-refinement's
    recall of the shared / b_only / a_only / c_only water groups (see run_partition). The groups
    come from the two self-refinements, or from the deposited structures with --ref-dir, which
    writes to a separate ..._deposited_... CSV.
      pairwise_water_metrics.py --partition --results-dir DIR [--ref-dir DIR] [--variant V] [--no-filter]
      → <results-dir parent>/starting_model_partition[_deposited]_<V>_<cutoff>.csv, viewed in
        experiments/starting_model_partition_recall.py (gitignored, local-only).

Output paths: -o overrides only with a single --variant. Column schemas: FIELDNAMES
(cohort / self-refined), PHENIX_FIELDNAMES, PARTITION_FIELDNAMES.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import biotite.structure.io.pdbx as pdbx
import gemmi
import numpy as np
from loguru import logger
from scipy.spatial import cKDTree

import config
from cw.align import align_to_reference
from cw.filter import filter_waters
from cw.io import (
    load_protein,
    load_structure_waters,
    read_cohort,
    read_phenix_cif,
    water_oxygen_mask,
)
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
PARTITION_FIELDNAMES = [
    "reference",
    "donor",
    "n_shared",
    "n_b_only",
    "n_a_only",
    "n_c_only",
    "recall_shared",
    "recall_b_only",
    "recall_a_only",
    "recall_c_only",
    "n_pred",
    "n_pred_b",
    "n_pred_a_only",
    "n_pred_orphan",
    "n_pred_near_shared",
    "n_pred_near_b_only",
    "n_pred_near_a_only",
]


def clean_phenix_waters(raw_path, pdb_id, *, distance_filter, filter_cutoff):
    """Clean a raw phenix CIF and return (cleaned_cif, water_coords).

    read_phenix_cif normalises the raw phenix output so biotite can read it. When
    distance_filter, the waters are then symmetry-aware distance-filtered to within
    filter_cutoff Å of protein (cw.filter.filter_waters, each structure's own
    cell / space group) — matching how the deposited references were filtered — so
    phenix-added solvent far from protein is dropped. Otherwise every water is kept.
    """
    cleaned = read_phenix_cif(raw_path)
    if not distance_filter:
        return cleaned, load_structure_waters(cleaned, pdb_id=pdb_id)[["x", "y", "z"]].to_numpy()
    st = gemmi.read_structure(str(raw_path))
    sg = st.find_spacegroup() or gemmi.SpaceGroup("P 1")
    atoms = pdbx.get_structure(cleaned, model=1, altloc="all", extra_fields=["b_factor", "occupancy"])
    filtered, *_ = filter_waters(atoms, st.cell, sg, filter_cutoff)
    return cleaned, filtered.coord[water_oxygen_mask(filtered)].astype(float)


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
    distance_filter = not args.no_filter
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Variants:    {variants}")
    logger.info(f"Cutoff:      {args.cutoff} Å")
    logger.info(f"Water filter: {f'on (≤ {args.filter_cutoff} Å to protein)' if distance_filter else 'off'}")

    for variant in variants:
        cif_paths = self_refined_cifs(results_dir, variant)
        if not cif_paths:
            logger.warning(f"[{variant}] no self-refined CIFs found under {results_dir}")
            continue
        ids = sorted(cif_paths)

        # Raw phenix CIFs are cleaned (and their waters distance-filtered unless
        # --no-filter) on the fly; the cleaned CIFFile is reused as both the alignment
        # reference (via load_protein) and the mobile.
        cleaned, coords = {}, {}
        for s in ids:
            cleaned[s], coords[s] = clean_phenix_waters(
                cif_paths[s], s, distance_filter=distance_filter, filter_cutoff=args.filter_cutoff
            )
        proteins = {s: load_protein(cleaned[s])[0] for s in ids}
        cells = {s: gemmi.read_structure(str(cif_paths[s])).cell for s in ids}

        rows = all_pairs_rows(ids, proteins, coords, cells, cleaned, args.cutoff, tag=variant)
        out_path = (
            args.output
            if (args.output and len(variants) == 1)
            else out_dir / f"self_refined_pairwise_metrics_{variant}_{args.cutoff}.csv"
        )
        _write_csv(out_path, FIELDNAMES, rows)


def run_phenix(args) -> None:
    # Original-grounded mode reads deposited reference CIFs from ref_dir/<a>.cif;
    # --ref-from-self-refined instead reads references from the refinement tree.
    if not args.ref_from_self_refined and args.ref_dir is None:
        logger.error(
            "--phenix (original-grounded) needs --ref-dir <cohort>/filtered_pdbs "
            "(or pass --ref-from-self-refined to ground on the self-refinements)"
        )
        sys.exit(1)
    ref_dir = args.ref_dir

    # Predictors come from the nested refinement_results tree of raw phenix output
    # (--results-dir); each CIF is cleaned (read_phenix_cif) and its waters
    # distance-filtered on the fly, so no pre-filtered dir is needed.
    if args.results_dir is None or not args.results_dir.is_dir():
        logger.error(f"--phenix needs --results-dir pointing at a refinement_results tree: {args.results_dir}")
        sys.exit(1)
    results_dir = args.results_dir
    out_dir = results_dir.parent
    variants = [args.variant] if args.variant else discover_variants(results_dir)

    distance_filter = not args.no_filter
    ref_source = "self-refined <a>_refined_by_<a>" if args.ref_from_self_refined else f"deposited {ref_dir}"
    logger.info(f"Reference:      {ref_source}")
    logger.info(f"Predictor dir:  {results_dir}")
    logger.info(f"Variants:       {variants}")
    logger.info(f"Cutoff:         {args.cutoff} Å")
    logger.info(f"Water filter:   {f'on (≤ {args.filter_cutoff} Å to protein)' if distance_filter else 'off'}")

    for variant in variants:
        records = nested_predictors(results_dir, variant)
        if not records:
            logger.warning(f"[{variant}] no predictor CIFs found in {results_dir}")
            continue

        # Each reference's protein + waters are loaded once and reused. By default
        # the reference is the original deposited structure (ref_dir/<a>.cif); with
        # --ref-from-self-refined it is the self-refinement <a>_refined_by_<a>_<variant>,
        # so a re-refined-reference Δ in section B holds the ground truth fixed.
        references = sorted({starting_model for _, _, starting_model in records})
        if args.ref_from_self_refined:
            ref_srcs = self_refined_cifs(results_dir, variant)
            ref_clean = True  # raw phenix CIFs, clean + distance-filter on the fly
        else:
            ref_srcs = {a: ref_dir / f"{a}.cif" for a in references}
            ref_clean = False
        present = [a for a in references if a in ref_srcs and ref_srcs[a].exists()]
        absent = [a for a in references if a not in present]
        if absent:
            logger.warning(f"[{variant}] missing reference CIFs (skipped): {absent}")
        # ref_clean marks a raw self-refinement CIF (clean + distance-filter on the fly);
        # a deposited reference is read as-is (already distance-filtered on disk).
        ref_objs, ref_coords = {}, {}
        for a in present:
            if ref_clean:
                ref_objs[a], ref_coords[a] = clean_phenix_waters(
                    ref_srcs[a], a, distance_filter=distance_filter, filter_cutoff=args.filter_cutoff
                )
            else:
                ref_objs[a] = ref_srcs[a]
                ref_coords[a] = load_structure_waters(ref_srcs[a], pdb_id=a)[["x", "y", "z"]].to_numpy()
        ref_proteins = {a: load_protein(ref_objs[a])[0] for a in present}

        rows = []
        for cif_path, mtz_source, starting_model in records:
            if starting_model not in ref_proteins:
                continue
            # Raw phenix output → clean + distance-filter on the fly.
            predictor, pred_coords = clean_phenix_waters(
                cif_path, mtz_source, distance_filter=distance_filter, filter_cutoff=args.filter_cutoff
            )
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


def _within_cutoff_mask(points: np.ndarray, others: np.ndarray, cutoff: float) -> np.ndarray:
    """Boolean per point in `points`: is any point in `others` within cutoff?"""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    if len(others) == 0:
        return np.zeros(len(points), dtype=bool)
    distances, _ = cKDTree(others).query(points, distance_upper_bound=cutoff)
    return np.isfinite(distances)


def _group_recall(group_coords: np.ndarray, predictor_coords: np.ndarray, cutoff: float) -> tuple[float, int]:
    """Fraction of `group_coords` with a `predictor_coords` point within cutoff, and the group size.

    NaN recall when the group is empty (undefined); 0.0 when the group is non-empty but the
    predictor set is.
    """
    n = len(group_coords)
    if n == 0:
        return float("nan"), 0
    return float(_within_cutoff_mask(group_coords, predictor_coords, cutoff).sum()) / n, n


def run_partition(args) -> None:
    """Starting-model-bias decomposition. For each ordered off-diagonal pair (A = starting
    model, B = data donor, A != B), the cross-refinement <B>_refined_by_<A> is the predictor;
    self-refinements <B>_refined_by_<B> and <A>_refined_by_<A> are the ground truth. Everything
    is Cα-aligned into B's frame, then B's reference union is split into disjoint groups and the
    fraction of each the predictor recovers within the cutoff (recall) is reported:
        shared  — B's waters that also sit on A   (conserved; recall ~ ceiling)
        b_only  — B's waters absent from A        (B's data supports; high recall = data wins)
        a_only  — A's waters absent from B        (template-only; recall here = bias signal)
        c_only  — every unrelated C's waters absent from B, pooled (conserved-water chance floor;
                  a_only above c_only is bias above chance)
    Each group also carries a precision numerator (n_pred_near_<group>), the predictor waters
    within cutoff of it, so any of the three can be scored as a standalone ground truth against
    the shared denominator n_pred. One CSV per variant."""
    results_dir: Path = args.results_dir
    if results_dir is None or not results_dir.is_dir():
        logger.error(f"--partition needs --results-dir pointing at a refinement_results tree: {results_dir}")
        sys.exit(1)

    out_dir = results_dir.parent
    ref_dir: Path | None = args.ref_dir
    variants = [args.variant] if args.variant else discover_variants(results_dir)
    distance_filter = not args.no_filter
    logger.info(f"Results dir:  {results_dir}")
    logger.info(f"Ground truth: {f'deposited {ref_dir}' if ref_dir else 'self-refined <x>_refined_by_<x>'}")
    logger.info(f"Variants:     {variants}")
    logger.info(f"Cutoff:       {args.cutoff} Å")
    logger.info(f"Water filter: {f'on (≤ {args.filter_cutoff} Å to protein)' if distance_filter else 'off'}")

    for variant in variants:
        self_cifs = self_refined_cifs(results_dir, variant)
        if not self_cifs:
            logger.warning(f"[{variant}] no self-refined CIFs found under {results_dir}")
            continue
        ids = sorted(self_cifs)

        # Ground truth for A and B: cleaned CIF + water coords (own frame) + protein (alignment
        # target). By default the two self-refinements, which are variant-specific — the groups
        # move when the protocol does. With --ref-dir the deposited structures stand in, holding
        # the target fixed across protocols, so a per-pair auto/stripped difference is a pure
        # predictor effect. Deposited CIFs are read as-is (already distance-filtered on disk);
        # raw phenix output is cleaned and filtered on the fly.
        cleaned, coords = {}, {}
        if ref_dir is None:
            for structure in ids:
                cleaned[structure], coords[structure] = clean_phenix_waters(
                    self_cifs[structure], structure, distance_filter=distance_filter, filter_cutoff=args.filter_cutoff
                )
        else:
            absent = [structure for structure in ids if not (ref_dir / f"{structure}.cif").exists()]
            if absent:
                logger.error(f"[{variant}] missing deposited reference CIFs under {ref_dir}: {absent}")
                sys.exit(1)
            for structure in ids:
                cleaned[structure] = ref_dir / f"{structure}.cif"
                coords[structure] = (
                    load_structure_waters(cleaned[structure], pdb_id=structure)[["x", "y", "z"]].to_numpy()
                )
        proteins = {structure: load_protein(cleaned[structure])[0] for structure in ids}

        # Cross-refinement predictors keyed by (starting_model, donor), off-diagonal only.
        predictors = {
            (starting_model, mtz_source): cif
            for cif, mtz_source, starting_model in nested_predictors(results_dir, variant)
            if starting_model != mtz_source
        }

        # Every self-refinement's waters aligned into every donor's frame (reused across pairs
        # that share a donor). aligned[donor][structure] is None when the alignment is skipped.
        aligned: dict[str, dict[str, np.ndarray | None]] = {donor: {} for donor in ids}
        for donor in ids:
            for structure in ids:
                if structure == donor:
                    aligned[donor][structure] = coords[donor]
                    continue
                report = align_to_reference(cleaned[structure], proteins[donor], out_path=None, pdb_id=structure)
                aligned[donor][structure] = (
                    None if report is None else (report["R"] @ coords[structure].T).T + report["t"]
                )

        rows = []
        for (starting_model, donor), cif in sorted(predictors.items()):
            if starting_model not in ids or donor not in ids:
                continue
            donor_coords = coords[donor]
            template_coords = aligned[donor][starting_model]
            if template_coords is None:
                logger.warning(f"  [{variant}] {donor}_refined_by_{starting_model}: template A alignment skipped")
                continue

            predictor_cif, predictor_raw = clean_phenix_waters(
                cif, donor, distance_filter=distance_filter, filter_cutoff=args.filter_cutoff
            )
            report = align_to_reference(predictor_cif, proteins[donor], out_path=None, pdb_id=f"{donor}_by_{starting_model}")
            if report is None:
                logger.warning(f"  [{variant}] {donor}_refined_by_{starting_model}: predictor alignment skipped")
                continue
            predictor_coords = (report["R"] @ predictor_raw.T).T + report["t"]

            donor_near_template = _within_cutoff_mask(donor_coords, template_coords, args.cutoff)
            shared = donor_coords[donor_near_template]
            b_only = donor_coords[~donor_near_template]
            a_only = template_coords[~_within_cutoff_mask(template_coords, donor_coords, args.cutoff)]

            unrelated_only = [
                aligned[donor][other][~_within_cutoff_mask(aligned[donor][other], donor_coords, args.cutoff)]
                for other in ids
                if other not in (starting_model, donor) and aligned[donor][other] is not None
            ]
            c_only = np.concatenate(unrelated_only) if unrelated_only else np.zeros((0, 3))

            recall_shared, n_shared = _group_recall(shared, predictor_coords, args.cutoff)
            recall_b_only, n_b_only = _group_recall(b_only, predictor_coords, args.cutoff)
            recall_a_only, n_a_only = _group_recall(a_only, predictor_coords, args.cutoff)
            recall_c_only, n_c_only = _group_recall(c_only, predictor_coords, args.cutoff)

            # Predictor-side (precision) composition: classify the cross-refinement's OWN waters
            # by which self-refinement supports them. "orphan" = near neither B nor A — waters the
            # cross-refinement generated that neither independent refinement places (candidate
            # artifacts, e.g. from a retained starting water).
            pred_near_b = _within_cutoff_mask(predictor_coords, donor_coords, args.cutoff)
            pred_near_a = _within_cutoff_mask(predictor_coords, template_coords, args.cutoff)
            n_pred = len(predictor_coords)
            n_pred_b = int(pred_near_b.sum())
            n_pred_a_only = int((pred_near_a & ~pred_near_b).sum())
            n_pred_orphan = int((~pred_near_a & ~pred_near_b).sum())

            # Precision numerators against each of the three disjoint groups, so shared / b_only /
            # a_only can each be scored as its own ground truth. These are NOT the composition
            # counts above: n_pred_b lumps shared and b_only together, and n_pred_a_only excludes
            # anything near B. Scoring against a group directly also lets one predictor water count
            # for two groups — two ground-truth waters more than a cutoff apart can both be in
            # range of it — so these three need not sum to n_pred.
            n_pred_near_shared = int(_within_cutoff_mask(predictor_coords, shared, args.cutoff).sum())
            n_pred_near_b_only = int(_within_cutoff_mask(predictor_coords, b_only, args.cutoff).sum())
            n_pred_near_a_only = int(_within_cutoff_mask(predictor_coords, a_only, args.cutoff).sum())

            rows.append(
                {
                    "reference": starting_model,
                    "donor": donor,
                    "n_shared": n_shared,
                    "n_b_only": n_b_only,
                    "n_a_only": n_a_only,
                    "n_c_only": n_c_only,
                    "recall_shared": recall_shared,
                    "recall_b_only": recall_b_only,
                    "recall_a_only": recall_a_only,
                    "recall_c_only": recall_c_only,
                    "n_pred": n_pred,
                    "n_pred_b": n_pred_b,
                    "n_pred_a_only": n_pred_a_only,
                    "n_pred_orphan": n_pred_orphan,
                    "n_pred_near_shared": n_pred_near_shared,
                    "n_pred_near_b_only": n_pred_near_b_only,
                    "n_pred_near_a_only": n_pred_near_a_only,
                }
            )
            logger.info(
                f"  [{variant}] {donor}_refined_by_{starting_model}: "
                f"shared={recall_shared:.2f} b_only={recall_b_only:.2f} "
                f"a_only={recall_a_only:.2f} c_only={recall_c_only:.2f}"
            )

        _grounding = "_deposited" if ref_dir is not None else ""
        out_path = out_dir / f"starting_model_partition{_grounding}_{variant}_{args.cutoff}.csv"
        _write_csv(out_path, PARTITION_FIELDNAMES, rows)


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
        "--partition",
        action="store_true",
        help="Starting-model-bias decomposition: for each off-diagonal pair (A=starting model, "
        "B=donor), report the cross-refinement's recall of shared / b_only / a_only / c_only water "
        "groups. Needs --results-dir; one CSV per variant → starting_model_partition_<variant>_<cutoff>.csv.",
    )
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
        help="(phenix, original-grounded) deposited reference CIF dir <cohort>/filtered_pdbs; "
        "required unless --ref-from-self-refined",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="(phenix/self-refined) nested refinement_results tree. Raw phenix CIFs are read "
        "straight from here and cleaned + distance-filtered on the fly. Output CSVs go to its "
        "parent dir.",
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
        "--filter-cutoff",
        type=float,
        default=config.WATER_PROT_DIST_CUTOFF,
        metavar="Å",
        help="(phenix/self-refined) water→protein distance cutoff for the on-the-fly filter "
        f"of raw phenix waters (default: config.WATER_PROT_DIST_CUTOFF = {config.WATER_PROT_DIST_CUTOFF})",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="(phenix/self-refined) skip the on-the-fly distance filter; compare every refined water as-is",
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

    if args.partition:
        run_partition(args)
    elif args.self_refined:
        run_self_refined(args)
    elif args.phenix:
        run_phenix(args)
    else:
        run_cohort(args)


if __name__ == "__main__":
    main()
