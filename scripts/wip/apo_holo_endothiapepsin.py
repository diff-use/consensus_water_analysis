"""Apo / holo classification for an endothiapepsin cohort.

Endothiapepsin is an aspartic protease with no catalytic metal, so — like the
HEWL script (``apo_holo_hewl.py``) and unlike carbonic anhydrase (``apo_holo.py``)
— the active site is anchored on the catalytic dyad, here Asp35 / Asp219. The two
carboxylates face each other across the substrate cleft; a genuinely bound
inhibitor/fragment sits between them. The dyad is present in every model, so each
structure is analysed in its OWN frame (raw CIF); no aligned reference is needed.

A structure is HOLO when a ligand-of-interest atom sits within SITE_CUTOFF of the
dyad centroid. This cohort is largely a crystallographic fragment screen, so the
ligands of interest are small drug-like fragments in the cleft; everything in the
shared additive / cofactor / cryo sets is filtered out.

Shared geometry / ligand primitives live in ``apo_holo_lib.py``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))          # apo_holo_lib
sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # config, cw

import apo_holo_lib as lib

import config
from cw.io import cif_path_for, read_cohort

# --- configuration ---------------------------------------------------------
DEFAULT_COHORT = Path("data/endothiapepsin_000240_iso.txt")

# Endothiapepsin catalytic dyad (mature numbering). Both carboxylates must be
# present for the anchor to be defined.
CATALYTIC_SCHEMES = [
    {("ASP", 35): ("OD1", "OD2"), ("ASP", 219): ("OD1", "OD2")},
]

SITE_CUTOFF = 6.0    # A: ligand-of-interest atom within this of the dyad -> holo
CLEFT_CUTOFF = 10.0  # A: outer edge of the substrate cleft (breakdown only)

NON_LIGAND = lib.ADDITIVES | lib.COFACTORS | lib.HEAVY_ATOM_REAGENTS


def catalytic_anchor(model):
    return lib.catalytic_dyad_anchor(model, CATALYTIC_SCHEMES)


# --- per-structure analysis ------------------------------------------------
@dataclass
class Result:
    pdb_id: str
    method: str                      # ok | no-dyad | missing
    inventory: Counter = field(default_factory=Counter)  # all non-water het
    ligands: list[str] = field(default_factory=list)     # ligands of interest
    lig_dist: float | None = None    # nearest ligand-of-interest -> dyad
    lig_comp: str | None = None

    @property
    def holo(self) -> bool:
        return self.lig_dist is not None and self.lig_dist <= SITE_CUTOFF

    @property
    def call(self) -> str:
        if self.method != "ok":
            return self.method
        if self.holo:
            return "holo"
        if self.ligands:  # present but outside the catalytic site
            return "apo-peripheral-ligand"
        return "apo"


def analyze(pdb_id: str) -> Result:
    cif = cif_path_for(pdb_id, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE)
    if not cif.exists():
        return Result(pdb_id, "missing")

    model = lib.read_model(cif)
    inventory = lib.het_inventory(model)
    anchor = catalytic_anchor(model)
    if anchor is None:
        return Result(pdb_id, "no-dyad", inventory=inventory)

    comps = lib.ligands_of_interest(model, NON_LIGAND)
    dist, comp = lib.nearest_ligand(model, comps, anchor) if comps else (None, None)
    return Result(pdb_id, "ok", inventory=inventory, ligands=sorted(comps),
                  lig_dist=dist, lig_comp=comp)


# --- reporting -------------------------------------------------------------
def report(results: list[Result]):
    calls = Counter(r.call for r in results)
    usable = [r for r in results if r.method == "ok"]

    print(f"structures        : {len(results)}")
    print(f"  analyzed (ok)   : {len(usable)}")
    print(f"  no catalytic dyad: {calls['no-dyad']}   missing: {calls['missing']}")
    print()
    print(f"--- Call (holo = ligand of interest <= {SITE_CUTOFF:g} A of Asp35/Asp219) ---")
    print(f"  APO                      : {calls['apo']}")
    print(f"  APO (peripheral ligand)  : {calls['apo-peripheral-ligand']}")
    print(f"  HOLO                     : {calls['holo']}")
    print()

    print("  nearest ligand-to-dyad distance breakdown:")
    buckets = [(0.0, SITE_CUTOFF, f"<={SITE_CUTOFF:g} active site"),
               (SITE_CUTOFF, CLEFT_CUTOFF, f"{SITE_CUTOFF:g}-{CLEFT_CUTOFF:g} cleft edge"),
               (CLEFT_CUTOFF, float("inf"), f">{CLEFT_CUTOFF:g} surface")]
    for lo, hi, name in buckets:
        sel = [r for r in usable if r.lig_dist is not None and lo < r.lig_dist <= hi]
        print(f"     {name:<22}{len(sel):>6}")
    no_lig = sum(r.lig_dist is None for r in usable)
    print(f"     {'no ligand of interest':<22}{no_lig:>6}")
    print()

    print("--- Cohort-wide het inventory (from coordinates, not the nonpoly loop) ---")
    cohort_het: Counter = Counter()
    for r in results:
        cohort_het.update(set(r.inventory))  # count structures, not copies
    for comp, n in sorted(cohort_het.items(), key=lambda kv: (-kv[1], kv[0])):
        if comp in lib.MONATOMIC_IONS:
            tag = "ion"
        elif comp in NON_LIGAND:
            tag = "additive/cofactor"
        else:
            tag = "LIGAND-OF-INTEREST"
        print(f"  {comp:<6} in {n:>3} structures   [{tag}]")


def write_csv(results: list[Result], path: Path):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "pdb_id", "call", "method", "all_het", "ligands_of_interest",
            "lig_dist", "lig_comp", "holo",
        ])
        for r in results:
            all_het = "|".join(f"{c}:{n}" for c, n in sorted(r.inventory.items()))
            w.writerow([
                r.pdb_id, r.call, r.method, all_het, "|".join(r.ligands),
                f"{r.lig_dist:.3f}" if r.lig_dist is not None else "",
                r.lig_comp or "", int(r.holo),
            ])
    print(f"wrote {len(results)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, default=DEFAULT_COHORT,
                    help=f"cohort .txt (default: {DEFAULT_COHORT})")
    ap.add_argument("--csv", type=Path, help="write per-structure rows to this CSV")
    args = ap.parse_args()

    pdb_ids = read_cohort(args.cohort)
    results = [analyze(p) for p in pdb_ids]
    report(results)
    if args.csv:
        write_csv(results, args.csv)


if __name__ == "__main__":
    main()
