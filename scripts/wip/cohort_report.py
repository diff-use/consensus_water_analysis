"""Cohort criteria report: one cohort .txt in, one combined summary table out.

Runs the four metadata-checking stages over a cohort and joins them into a single
per-structure table, then reports how many structures survive each combination of
criteria — the number you need before committing a cohort to alignment and
clustering.

Stages (each is an existing script; artifacts are reused unless --force):
  1. scripts/build_metadata.py          -> metadata.csv        resolution, waters, temp
  2. scripts/wip/apo_holo_<protein>.py  -> apo_holo.csv        ligand state, site occupancy
  3. scripts/wip/mutation_survey.py     -> sequence_mutation_counts.csv   wildtype vs mutant
  4. scripts/wip/apo_text_check.py      -> apo_text_check.csv  deposition free-text check

The apo/holo stage is protein-specific because the active-site anchor is (a metal,
a catalytic dyad, a nucleophilic cysteine), so --protein selects which
``apo_holo_*.py`` runs. Everything else is protein-agnostic. If that script emits a
``state*`` column (DJ-1 emits ``state106``, the Cys106 redox/adduct state), it is
carried through and broken down as the active-site-state axis.

The joined table is written to ``cohort_criteria.csv`` so any combination not
tabulated here can be filtered directly.

Usage:
    uv run scripts/wip/cohort_report.py <cohort.txt> --protein dj1 --reference 9yfr
                                        [-o OUTDIR] [--force] [--resolution 1.8]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # config, cw

import pandas as pd

import config
from cw.io import read_cohort

REPO = Path(__file__).parent.parent.parent
PROTEIN_SCRIPTS = {
    "dj1": "apo_holo_dj1.py",
    "ca": "apo_holo.py",
    "hewl": "apo_holo_hewl.py",
    "endothiapepsin": "apo_holo_endothiapepsin.py",
}

CRYO_MAX = 150.0      # K: at or below this the solvent is vitrified
ROOM_MIN = 250.0      # K: at or above this it is a room-temperature dataset


def run(command: list[str], label: str) -> None:
    print(f"[{label}] {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=REPO)
    if result.returncode != 0:
        sys.exit(f"[{label}] failed with exit code {result.returncode}")


def stage_artifacts(cohort: Path, protein: str, reference: str, outdir: Path,
                    force: bool) -> dict[str, Path]:
    """Run any stage whose artifact is missing; return the artifact paths."""
    paths = {
        "metadata": outdir / "metadata.csv",
        "apo_holo": outdir / "apo_holo.csv",
        "mutations": outdir / "sequence_mutation_counts.csv",
        "text": outdir / "apo_text_check.csv",
    }
    python = sys.executable

    # A metadata.csv written before diffrn_temp existed would silently report every
    # temperature as unknown, so a stale schema counts as a missing artifact.
    stale_metadata = paths["metadata"].exists() and "diffrn_temp" not in pd.read_csv(
        paths["metadata"], nrows=0
    ).columns
    if force or not paths["metadata"].exists() or stale_metadata:
        if stale_metadata:
            print("[metadata] existing CSV predates the diffrn_temp column; rebuilding")
        run([python, "scripts/build_metadata.py", str(cohort), "-o", str(paths["metadata"])],
            "metadata")
    if force or not paths["apo_holo"].exists():
        run([python, f"scripts/wip/{PROTEIN_SCRIPTS[protein]}", "--cohort", str(cohort),
             "--csv", str(paths["apo_holo"])], "apo/holo")
    if force or not paths["mutations"].exists():
        run([python, "scripts/wip/mutation_survey.py", str(cohort), "--reference", reference,
             "-o", str(outdir), "--quiet"], "mutations")
    if force or not paths["text"].exists():
        run([python, "scripts/wip/apo_text_check.py", str(paths["apo_holo"]),
             "--csv", str(paths["text"])], "text-check")
    return paths


def temp_class(value) -> str:
    if pd.isna(value):
        return "unknown"
    if value <= CRYO_MAX:
        return "cryo"
    if value >= ROOM_MIN:
        return "room-temp"
    return "intermediate"


def evidence_class(row) -> str:
    """How well the ligand-free claim holds up: pocket occupancy beats free text."""
    if row.ligand_state not in ("apo", "apo-peripheral-ligand"):
        return "n/a (holo)"
    if row.get("site_het") == 1:
        return "pocket-occupied"
    if row.get("text_verdict") == "review":
        return "text-claims-ligand"
    return "clean"


def build_table(paths: dict[str, Path], cohort_ids: list[str], reference: str) -> pd.DataFrame:
    metadata = pd.read_csv(paths["metadata"])
    apo_holo = pd.read_csv(paths["apo_holo"])
    mutations = pd.read_csv(paths["mutations"])

    keep = ["pdb_id", "call", "lig_comp", "lig_dist", "site_metal", "het_comp", "het_dist",
            "site_het", "holo"]
    state_columns = [c for c in apo_holo.columns if c.startswith("state")]
    apo_holo = apo_holo[[c for c in keep + state_columns if c in apo_holo.columns]]
    apo_holo = apo_holo.rename(columns={"call": "ligand_state"})
    if state_columns:
        apo_holo = apo_holo.rename(columns={state_columns[0]: "site_state"})

    table = (
        pd.DataFrame({"pdb_id": cohort_ids})
        .merge(metadata[[c for c in ("pdb_id", "resolution", "num_water", "r_free",
                                     "diffrn_temp", "space_group", "starting_model")
                         if c in metadata.columns]], on="pdb_id", how="left")
        .merge(apo_holo, on="pdb_id", how="left")
        .merge(mutations[["pdb_id", "n_substitutions", "identity", "signature"]],
               on="pdb_id", how="left")
    )

    if paths["text"].exists():
        text = pd.read_csv(paths["text"])[["pdb_id", "verdict", "title"]]
        table = table.merge(text.rename(columns={"verdict": "text_verdict"}),
                            on="pdb_id", how="left")
    else:
        table["text_verdict"] = pd.NA
        table["title"] = pd.NA

    if "diffrn_temp" not in table.columns:
        # metadata.csv predates the diffrn_temp column; --force regenerates it.
        table["diffrn_temp"] = pd.NA
    table["diffrn_temp"] = pd.to_numeric(table.diffrn_temp, errors="coerce")
    table["temp_class"] = table.diffrn_temp.map(temp_class)
    # The mutation survey excludes its own reference (a self-comparison would be a
    # spurious 0-substitution data point), which would otherwise leave the reference
    # with no sequence call and drop it from every wildtype filter. It defines the
    # sequence standard, so it is 0 substitutions by construction — flagged in
    # is_reference so the assumption stays visible. Rows still unlabelled here were
    # dropped by the survey's coverage filter and are genuinely unknown.
    table["is_reference"] = table.pdb_id == reference.lower()
    table.loc[table.is_reference & table.n_substitutions.isna(), "n_substitutions"] = 0
    table["sequence_class"] = table.n_substitutions.map(
        lambda n: "unknown" if pd.isna(n) else ("wildtype" if n == 0 else "mutant")
    )
    table["ligand_class"] = table.ligand_state.map(
        lambda c: "unknown" if not isinstance(c, str)
        else ("holo" if c.startswith("holo") else "apo")
    )
    table["evidence_class"] = table.apply(evidence_class, axis=1)
    return table


# --- reporting -------------------------------------------------------------
def axis(table: pd.DataFrame, column: str, title: str) -> None:
    counts = table[column].value_counts(dropna=False)
    print(f"  {title}")
    for value, n in counts.items():
        print(f"    {str(value):<26}{n:>5}")
    print()


def report(table: pd.DataFrame, cohort_id: str, resolution_cutoff: float | None) -> None:
    total = len(table)
    print("=" * 78)
    print(f"COHORT {cohort_id} — {total} structures")
    print("=" * 78)
    print()
    print("--- Per-axis breakdown ---")
    axis(table, "ligand_state", "ligand state (geometric)")
    axis(table, "sequence_class", "sequence")
    axis(table, "temp_class", "temperature class")
    if table.diffrn_temp.notna().any():
        print("  reported temperatures (K)")
        for value, n in table.diffrn_temp.value_counts(dropna=False).sort_index().items():
            print(f"    {value if pd.notna(value) else 'unknown':<26}{n:>5}")
        print()
    if "site_state" in table.columns and table.site_state.notna().any():
        axis(table, "site_state", "active-site state")
    axis(table, "evidence_class", "apo evidence quality")

    print("--- Combined: ligand x sequence x temperature ---")
    print(f"  {'ligand':<10}{'sequence':<12}{'temp':<16}{'n':>5}   "
          f"{'med.res':>8}{'med.waters':>12}")
    grouped = table.groupby(["ligand_class", "sequence_class", "temp_class"], dropna=False)
    for (ligand, sequence, temperature), group in grouped:
        print(f"  {ligand:<10}{sequence:<12}{temperature:<16}{len(group):>5}   "
              f"{group.resolution.median():>8.2f}{group.num_water.median():>12.0f}")
    print()

    print("--- Cumulative filter cascade (how many PDB IDs survive) ---")
    steps = [
        ("all structures in cohort", pd.Series(True, index=table.index)),
        ("apo (incl. peripheral-ligand-only)", table.ligand_class == "apo"),
        ("+ no peripheral ligand anywhere", table.ligand_state == "apo"),
        ("+ wildtype sequence", table.sequence_class == "wildtype"),
        ("+ cryo (<=%gK)" % CRYO_MAX, table.temp_class == "cryo"),
        ("+ empty pocket (no het within site cutoff)", table.site_het != 1),
        ("+ no ligand language in deposition text", table.text_verdict != "review"),
    ]
    if resolution_cutoff is not None:
        steps.append((f"+ resolution <= {resolution_cutoff:g} A",
                      table.resolution <= resolution_cutoff))

    mask = pd.Series(True, index=table.index)
    for label, condition in steps:
        mask = mask & condition.fillna(False)
        surviving = table[mask]
        listed = ""
        if 0 < len(surviving) <= 12:
            listed = "  [" + ", ".join(sorted(surviving.pdb_id)) + "]"
        print(f"  {label:<44}{len(surviving):>5}{listed}")
    print()
    print("  Note: the cascade is one ordering of the same criteria. Reorder or relax any "
          "step by filtering cohort_criteria.csv directly.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cohort", type=Path, help="Cohort .txt (one PDB ID per line)")
    parser.add_argument("--protein", required=True, choices=sorted(PROTEIN_SCRIPTS),
                        help="Which apo_holo_*.py defines the active site")
    parser.add_argument("--reference", required=True,
                        help="Reference PDB ID for the mutation survey")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Artifact directory (default: config.DATA_DIR/<cohort_id>)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute every stage instead of reusing existing artifacts")
    parser.add_argument("--resolution", type=float, default=None,
                        help="Add a resolution cutoff as the final cascade step")
    args = parser.parse_args()

    if not args.cohort.exists():
        sys.exit(f"Cohort file not found: {args.cohort}")

    cohort = args.cohort.resolve()   # stages run with cwd=REPO
    cohort_id = cohort.stem
    outdir = (args.output_dir or Path(config.DATA_DIR) / cohort_id).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    paths = stage_artifacts(cohort, args.protein, args.reference, outdir, args.force)
    table = build_table(paths, read_cohort(cohort), args.reference)

    out_csv = outdir / "cohort_criteria.csv"
    table.to_csv(out_csv, index=False)
    print()
    report(table, cohort_id, args.resolution)
    print(f"wrote {len(table)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
