"""Survey protein sequence mutations across any cohort .txt.

Usage:
    uv run scripts/wip/mutation_survey.py <cohort.txt> --reference PDB_ID
                                          [--identity-cutoff 0.90] [-o OUTPUT_DIR]

Extracts the ordered protein sequence (highest-occupancy altloc, longest chain)
from each structure's mmCIF and globally aligns it to a reference member.
Aligning rather than comparing by auth_seq_id removes residue-numbering
artifacts. Reports the identity distribution (which separates point-mutants of
one protein from any sequence-distinct constructs that share the crystal form),
the per-structure substitution-count distribution, the recurring mutation
signatures, and the per-position hotspots.

Writes sequence_mutation_counts.csv and sequence_mutation_hotspots.csv to
config.DATA_DIR/<cohort_id>/ by default.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from biotite.sequence import ProteinSequence
from loguru import logger

import config
from cw.align import paired_alignment_trace
from cw.io import cif_path_for, load_protein, read_cohort

# Modified residues biotite does not map to a standard one-letter code.
EXTRA_THREE_TO_ONE = {"MSE": "M", "SEC": "C", "PYL": "K", "MLY": "K", "CSO": "C"}


def three_to_one(residue_name: str) -> str:
    if residue_name in EXTRA_THREE_TO_ONE:
        return EXTRA_THREE_TO_ONE[residue_name]
    try:
        return ProteinSequence.convert_letter_3to1(residue_name)
    except KeyError:
        return "X"


def extract_sequence(cif_path: Path) -> tuple[str, list[int], int]:
    """Return the ordered one-letter sequence, deposited auth_seq_ids, and protein
    chain count of every Cα, in order.

    Assumes one protein chain: all chains are concatenated, so with >1 chain the
    reported positions merge chains and become ambiguous (caller warns).
    """
    protein, offset = load_protein(
        cif_path
    )  # highest-occupancy altloc; res_ids shifted to start at 1
    c_alpha = protein[protein.atom_name == "CA"]
    if c_alpha.array_length() == 0:
        raise ValueError("no CA atoms")
    letters = "".join(three_to_one(name) for name in c_alpha.res_name)
    # undo load_protein's offset (only applied when >0) so mutations are reported
    # in deposited numbering
    shift = offset if offset > 0 else 0
    res_ids = [int(r) + shift for r in c_alpha.res_id]
    return letters, res_ids, len(set(c_alpha.chain_id))


def to_protein_sequence(sequence: str) -> ProteinSequence:
    valid = ProteinSequence.alphabet
    return ProteinSequence("".join(c if c in valid else "X" for c in sequence))


def load_sequences(cif_paths: dict[str, Path]) -> tuple[dict, dict, list, list]:
    """Parse every member; return {id: seq}, {id: resids}, [(id, reason)] failures,
    and [id] of structures with >1 protein chain."""
    sequences, residue_ids, failed, multichain = {}, {}, [], []
    for index, (member_id, cif_path) in enumerate(cif_paths.items(), start=1):
        try:
            sequence, resids, n_chains = extract_sequence(cif_path)
            sequences[member_id] = sequence
            residue_ids[member_id] = resids
            if n_chains > 1:
                multichain.append(member_id)
        except Exception as exc:
            failed.append((member_id, str(exc)[:80]))
        if index % 150 == 0:
            logger.debug(f"  parsed {index}/{len(cif_paths)}")
    return sequences, residue_ids, failed, multichain


def substitutions_vs_reference(
    sequence: str,
    reference_sequence: str,
    reference_residue_ids: list[int],
) -> tuple[float, float, list[tuple[int, str, str]]]:
    """Global-align a sequence to the reference; return (identity, coverage, substitutions).

    identity = matched / aligned columns.
    coverage = aligned columns / reference length. Guards against a spuriously
        high identity computed over only a handful of paired residues — an
        unrelated reference that barely overlaps the cohort pairs few residues,
        so a couple of coincidental matches would otherwise read as ~100%.
    A substitution is (reference_auth_seq_id, reference_aa, mutant_aa). Terminal
    gaps (missing density) are not penalized and never counted as mutations.
    """
    paired = paired_alignment_trace(
        to_protein_sequence(reference_sequence), to_protein_sequence(sequence)
    )
    identical = 0
    substitutions = []
    for reference_index, query_index in paired:
        reference_aa = reference_sequence[reference_index]
        query_aa = sequence[query_index]
        if reference_aa == query_aa:
            identical += 1
        elif "X" not in (reference_aa, query_aa):
            substitutions.append((reference_residue_ids[reference_index], reference_aa, query_aa))
    identity = identical / len(paired) if len(paired) else 0.0
    coverage = len(paired) / len(reference_sequence) if reference_sequence else 0.0
    return identity, coverage, substitutions


def survey(
    sequences: dict[str, str],
    reference_sequence: str,
    reference_residue_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, list]]:
    """Align every structure to the reference; return a per-structure table plus
    the substitution list for each member."""
    rows, substitutions_by_structure = [], {}
    for member_id, sequence in sequences.items():
        identity, coverage, substitutions = substitutions_vs_reference(
            sequence, reference_sequence, reference_residue_ids
        )
        substitutions_by_structure[member_id] = substitutions
        signature = " ".join(f"{wt}{pos}{mut}" for pos, wt, mut in sorted(substitutions))
        rows.append(
            {
                "pdb_id": member_id,
                "n_modeled": len(sequence),
                "identity": identity,
                "coverage": coverage,
                "n_substitutions": len(substitutions),
                "signature": signature,
            }
        )
    table = pd.DataFrame(rows).sort_values("n_substitutions", ascending=False, ignore_index=True)
    return table, substitutions_by_structure


def mutation_hotspots(
    member_ids: list[str], substitutions_by_structure: dict[str, list]
) -> pd.DataFrame:
    """Tally, per reference position, how many members carry a non-reference residue."""
    variants = defaultdict(Counter)
    reference_aa = {}
    for member_id in member_ids:
        for position, wt, mut in substitutions_by_structure[member_id]:
            variants[position][mut] += 1
            reference_aa[position] = wt
    columns = ["position", "wt", "n_mutant", "n_variants", "variants"]
    rows = [
        {
            "position": position,
            "wt": reference_aa[position],
            "n_mutant": sum(counter.values()),
            "n_variants": len(counter),
            "variants": ";".join(f"{aa}:{n}" for aa, n in counter.most_common()),
        }
        for position, counter in variants.items()
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        "n_mutant", ascending=False, ignore_index=True
    )


# --- reporting (stdout, so it survives --quiet) ------------------------------


def report_identity(table: pd.DataFrame, identity_cutoff: float) -> None:
    print("\n=== sequence identity to reference ===")
    edges = [0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0001]
    labels = ["<50%", "50-80%", "80-90%", "90-95%", "95-98%", "98-99%", ">=99%"]
    binned = pd.cut(table.identity, bins=edges, labels=labels, right=False, include_lowest=True)
    for label, count in binned.value_counts().reindex(labels).items():
        count = int(count or 0)
        print(f"  {label:8s}: {count:4d}  {'#' * min(count, 80)}")
    distinct = table[table.identity < identity_cutoff]
    print(f"\n  point-mutant group (identity>={identity_cutoff:.0%}): {len(table) - len(distinct)}")
    print(f"  sequence-distinct group (identity <{identity_cutoff:.0%}): {len(distinct)}")
    if len(distinct):
        examples = ", ".join(distinct.sort_values("identity").pdb_id.head(6))
        print(
            f"  distinct identity range: {distinct.identity.min():.2f}-"
            f"{distinct.identity.max():.2f}  (e.g. {examples})"
        )


def report_distribution(mutant_group: pd.DataFrame) -> None:
    print("\n=== #substitutions per structure (point-mutant group) ===")
    for count, n in mutant_group.n_substitutions.value_counts().sort_index().items():
        print(f"  {count:2d} sub: {n:4d}  {'#' * min(int(n), 80)}")
    n_wild_type = int((mutant_group.n_substitutions == 0).sum())
    total = len(mutant_group)
    print(
        f"\n  mean={mutant_group.n_substitutions.mean():.2f}  "
        f"median={mutant_group.n_substitutions.median():.0f}  "
        f"max={mutant_group.n_substitutions.max()}"
    )
    print(
        f"  wild-type (0 sub): {n_wild_type} ({100 * n_wild_type / total:.1f}%)  "
        f"engineered (>=1): {total - n_wild_type} ({100 * (total - n_wild_type) / total:.1f}%)"
    )


def report_signatures(mutant_group: pd.DataFrame, limit: int = 10) -> None:
    print("\n=== most common mutation signatures ===")
    signatures = Counter(sig or "(wild-type)" for sig in mutant_group.signature)
    for signature, n in signatures.most_common(limit):
        print(f"  {n:4d}x  {signature}")


def report_hotspots(hotspots: pd.DataFrame, limit: int = 25) -> None:
    print("\n=== mutation hotspots (reference numbering) ===")
    print(f"  {len(hotspots)} positions mutated in >=1 structure")
    print("  pos  wt  n_mut  variants(count)")
    for _, row in hotspots.head(limit).iterrows():
        print(f"  {row.position:4d}  {row.wt}   {row.n_mutant:4d}   {row.variants}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Survey sequence mutations across a cohort.")
    parser.add_argument("cohort", type=Path, help="Cohort .txt file (one PDB ID per line)")
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference PDB ID (need not be a cohort member; if absent "
        "from the cohort it is loaded directly and not counted)",
    )
    parser.add_argument(
        "--identity-cutoff",
        type=float,
        default=0.90,
        help="Members below this identity are reported as sequence-distinct",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.80,
        help="Structures whose alignment covers less than this fraction of "
        "the reference are excluded (their identity is unreliable)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: config.DATA_DIR/<cohort_id>)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Show debug output")
    verbosity.add_argument("--quiet", action="store_true", help="Show warnings and errors only")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING" if args.quiet else "DEBUG" if args.verbose else "INFO")

    if not args.cohort.exists():
        logger.error(f"Cohort file not found: {args.cohort}")
        sys.exit(1)

    cohort_id = args.cohort.stem
    output_dir = args.output_dir or Path(config.DATA_DIR) / cohort_id

    member_ids = read_cohort(args.cohort)
    cif_paths = {
        member_id: cif_path_for(member_id, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE)
        for member_id in member_ids
    }
    found = {m: p for m, p in cif_paths.items() if p.exists()}
    missing = [m for m in member_ids if m not in found]
    logger.info(f"Cohort:  {cohort_id}")
    logger.info(f"Members: {len(member_ids)} total — {len(found)} found, {len(missing)} missing")
    if missing:
        logger.warning(f"Missing CIFs: {', '.join(missing)}")

    sequences, residue_ids, failed, multichain = load_sequences(found)
    logger.info(f"Parsed:  {len(sequences)} ok, {len(failed)} failed")
    for member_id, reason in failed:
        logger.warning(f"  {member_id}: {reason}")
    if multichain:
        listed = ", ".join(multichain[:10]) + (
            f" (+{len(multichain) - 10} more)" if len(multichain) > 10 else ""
        )
        logger.warning(
            f"{len(multichain)} structure(s) have >1 protein chain; chains are "
            f"concatenated so reported positions may be ambiguous: {listed}"
        )
    if not sequences:
        logger.error("No sequences parsed; nothing to survey.")
        sys.exit(1)

    reference_id = args.reference.lower()
    if reference_id in sequences:
        reference_sequence = sequences[reference_id]
        reference_ids = residue_ids[reference_id]
    else:
        # Reference lives outside the cohort — parse it on its own so it anchors
        # the alignment without being counted as a cohort member.
        logger.warning(
            f"Reference {reference_id!r} was not parsed from the cohort; "
            f"loading it directly from ALL_PDB_REDO_DIR (not counted as a member)."
        )
        reference_cif = cif_path_for(reference_id, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE)
        if not reference_cif.exists():
            logger.error(f"Reference CIF not found: {reference_cif}")
            sys.exit(1)
        try:
            reference_sequence, reference_ids, ref_chains = extract_sequence(reference_cif)
        except Exception as exc:
            logger.error(f"Failed to parse reference {reference_id!r}: {exc}")
            sys.exit(1)
        if ref_chains > 1:
            logger.warning(
                f"Reference {reference_id!r} has >1 protein chain; "
                f"reported positions may be ambiguous."
            )
    logger.info(
        f"Reference: {reference_id} "
        f"({len(reference_sequence)} residues, "
        f"auth_seq_id {min(reference_ids)}-{max(reference_ids)})"
    )

    # The reference anchors the alignment; it is never a surveyed data point, so
    # a self-comparison (0 substitutions) does not pollute the distributions.
    members = {m: s for m, s in sequences.items() if m != reference_id}
    table, substitutions_by_structure = survey(members, reference_sequence, reference_ids)

    logger.info(
        f"Coverage vs reference: min={table.coverage.min():.2f} "
        f"median={table.coverage.median():.2f} mean={table.coverage.mean():.2f}"
    )
    below = table[table.coverage < args.min_coverage].sort_values("coverage")
    if len(below):
        listed = ", ".join(
            f"{r.pdb_id} (cov {r.coverage:.2f})" for r in below.head(10).itertuples()
        )
        more = f" (+{len(below) - 10} more)" if len(below) > 10 else ""
        logger.warning(
            f"{len(below)} structure(s) below --min-coverage "
            f"{args.min_coverage:.2f}, excluded: {listed}{more}"
        )
        table = table[table.coverage >= args.min_coverage].reset_index(drop=True)
    if table.empty:
        logger.error(
            "No structures meet --min-coverage; reference likely incompatible with the cohort."
        )
        sys.exit(1)

    mutant_group = table[table.identity >= args.identity_cutoff]
    hotspots = mutation_hotspots(list(mutant_group.pdb_id), substitutions_by_structure)

    report_identity(table, args.identity_cutoff)
    report_distribution(mutant_group)
    report_signatures(mutant_group)
    report_hotspots(hotspots)

    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "sequence_mutation_counts.csv", index=False)
    hotspots.to_csv(output_dir / "sequence_mutation_hotspots.csv", index=False)
    logger.info(f"Wrote {output_dir}/sequence_mutation_counts.csv")
    logger.info(f"Wrote {output_dir}/sequence_mutation_hotspots.csv")


if __name__ == "__main__":
    main()
