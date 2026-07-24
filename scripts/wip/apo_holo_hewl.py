"""Apo / holo classification for a hen-egg-white-lysozyme (HEWL) cohort.

HEWL has no catalytic metal, so — unlike the carbonic-anhydrase script
(``apo_holo.py``) — there is no zinc to anchor on. Instead the active site is
anchored on the catalytic dyad Glu35 / Asp52, which sits between substrate subsites
D and E; a genuinely bound substrate/inhibitor always occupies the catalytic
subsites. Because the dyad is present in every HEWL model, each structure is
analysed in its OWN frame (raw CIF) and no aligned reference is needed.

A structure is HOLO when a ligand-of-interest atom sits within SITE_CUTOFF of the
dyad centroid. Two HEWL-specific choices differ from the carbonic-anhydrase script:
  - Sugars (NAG, NDG, chitooligosaccharides, ...) are the natural HEWL
    substrate/inhibitor, so they are LIGANDS OF INTEREST here, not additives.
  - An independent second opinion comes from the RCSB Data API
    (``nonpolymer_bound_components`` and ``rcsb_binding_affinity``); disagreements
    between the local geometric call and RCSB are flagged for manual review. RCSB
    reflects the deposited PDB, not the re-refined CIF.

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

import config
from cw.io import cif_path_for, read_cohort
from cw.metadata import _fetch_rcsb_entry

import apo_holo_lib as lib

# --- configuration ---------------------------------------------------------
DEFAULT_COHORT = Path("data/hewls_65.txt")

# HEWL catalytic dyad, tried in order. Mature numbering (Glu35/Asp52) is the
# textbook scheme; some deposits use UniProt/precursor numbering that includes
# the 18-residue signal peptide, shifting the dyad to Glu53/Asp70. The first
# scheme whose GLU and ASP are both present is used.
CATALYTIC_SCHEMES = [
    {("GLU", 35): ("OE1", "OE2"), ("ASP", 52): ("OD1", "OD2")},   # mature
    {("GLU", 53): ("OE1", "OE2"), ("ASP", 70): ("OD1", "OD2")},   # precursor (+18)
]

SITE_CUTOFF = 6.0    # A: ligand-of-interest atom within this of the dyad -> holo
CLEFT_CUTOFF = 10.0  # A: outer edge of the substrate cleft (breakdown only)

# Sugars are deliberately NOT excluded here (they are the HEWL substrate analogs).
NON_LIGAND = lib.ADDITIVES | lib.COFACTORS | lib.HEAVY_ATOM_REAGENTS


# --- HEWL-specific helpers -------------------------------------------------
def catalytic_anchor(model):
    return lib.catalytic_dyad_anchor(model, CATALYTIC_SCHEMES)


def rcsb_ligand_info(pdb_id: str) -> dict | None:
    """RCSB's independent view: bound non-polymer components of interest and
    whether any binding-affinity data is deposited. Returns None if the entry
    could not be fetched."""
    entry = _fetch_rcsb_entry(pdb_id)
    if entry is None:
        return None
    info = entry.get("rcsb_entry_info") or {}
    bound = list(info.get("nonpolymer_bound_components") or [])
    of_interest = [
        c for c in bound
        if c not in lib.WATER and c not in NON_LIGAND and c not in lib.MONATOMIC_IONS
    ]
    has_affinity = bool(entry.get("rcsb_binding_affinity"))
    return {"bound": bound, "of_interest": of_interest, "has_affinity": has_affinity}


# --- per-structure analysis ------------------------------------------------
@dataclass
class Result:
    pdb_id: str
    method: str                      # ok | no-dyad | missing
    inventory: Counter = field(default_factory=Counter)  # all non-water het
    ligands: list[str] = field(default_factory=list)     # ligands of interest
    lig_dist: float | None = None    # nearest ligand-of-interest -> dyad
    lig_comp: str | None = None
    rcsb_of_interest: list[str] = field(default_factory=list)
    rcsb_affinity: bool = False
    rcsb_fetched: bool = False

    @property
    def holo_local(self) -> bool:
        return self.lig_dist is not None and self.lig_dist <= SITE_CUTOFF

    @property
    def rcsb_holo(self) -> bool:
        return bool(self.rcsb_of_interest) or self.rcsb_affinity

    @property
    def call(self) -> str:
        if self.method != "ok":
            return self.method
        if self.holo_local:
            return "holo"
        if self.ligands:  # present but outside the catalytic site
            return "apo-peripheral-ligand"
        return "apo"

    @property
    def disagree(self) -> bool:
        """Local says apo (no ligand in the site) but RCSB reports a bound
        ligand of interest / affinity data — worth a manual look."""
        return self.rcsb_fetched and not self.holo_local and self.rcsb_holo


def analyze(pdb_id: str, use_rcsb: bool) -> Result:
    cif = cif_path_for(pdb_id, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE)
    if not cif.exists():
        return Result(pdb_id, "missing")

    model = lib.read_model(cif)
    inventory = lib.het_inventory(model)
    anchor = catalytic_anchor(model)
    if anchor is None:
        r = Result(pdb_id, "no-dyad", inventory=inventory)
    else:
        comps = lib.ligands_of_interest(model, NON_LIGAND)
        dist, comp = lib.nearest_ligand(model, comps, anchor) if comps else (None, None)
        r = Result(pdb_id, "ok", inventory=inventory, ligands=sorted(comps),
                   lig_dist=dist, lig_comp=comp)

    if use_rcsb:
        rcsb = rcsb_ligand_info(pdb_id)
        if rcsb is not None:
            r.rcsb_fetched = True
            r.rcsb_of_interest = rcsb["of_interest"]
            r.rcsb_affinity = rcsb["has_affinity"]
    return r


# --- reporting -------------------------------------------------------------
def report(results: list[Result], use_rcsb: bool):
    calls = Counter(r.call for r in results)
    usable = [r for r in results if r.method == "ok"]

    print(f"structures        : {len(results)}")
    print(f"  analyzed (ok)   : {len(usable)}")
    print(f"  no catalytic dyad: {calls['no-dyad']}   missing: {calls['missing']}")
    print()
    print(f"--- Call (holo = ligand of interest <= {SITE_CUTOFF:g} A of Glu35/Asp52) ---")
    print(f"  APO                      : {calls['apo']}")
    print(f"  APO (peripheral ligand)  : {calls['apo-peripheral-ligand']}")
    print(f"  HOLO                     : {calls['holo']}")
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
    print()

    holo = [r for r in usable if r.holo_local]
    if holo:
        print("--- HOLO structures ---")
        for r in sorted(holo, key=lambda r: r.lig_dist):
            print(f"  {r.pdb_id}  {r.lig_comp:<4} @ {r.lig_dist:.2f} A   "
                  f"(all LOI: {'|'.join(r.ligands)})")
        print()

    peripheral = [r for r in usable if not r.holo_local and r.ligands]
    if peripheral:
        print("--- APO but ligand-of-interest present outside the site ---")
        for r in peripheral:
            d = f"{r.lig_dist:.2f} A" if r.lig_dist is not None else "n/a"
            print(f"  {r.pdb_id}  {'|'.join(r.ligands)}  nearest {d}")
        print()

    if use_rcsb:
        disagree = [r for r in results if r.disagree]
        print("--- RCSB cross-check ---")
        print(f"  entries fetched          : {sum(r.rcsb_fetched for r in results)}")
        print(f"  local-apo but RCSB-holo  : {len(disagree)}  (review these)")
        for r in disagree:
            aff = " +affinity" if r.rcsb_affinity else ""
            print(f"    {r.pdb_id}  RCSB bound-of-interest={'|'.join(r.rcsb_of_interest)}{aff}")
        print()


def write_csv(results: list[Result], path: Path):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "pdb_id", "call", "method", "all_het", "ligands_of_interest",
            "lig_dist", "lig_comp", "holo_local",
            "rcsb_of_interest", "rcsb_has_affinity", "rcsb_holo", "disagree",
        ])
        for r in results:
            all_het = "|".join(f"{c}:{n}" for c, n in sorted(r.inventory.items()))
            w.writerow([
                r.pdb_id, r.call, r.method, all_het, "|".join(r.ligands),
                f"{r.lig_dist:.3f}" if r.lig_dist is not None else "",
                r.lig_comp or "", int(r.holo_local),
                "|".join(r.rcsb_of_interest), int(r.rcsb_affinity),
                int(r.rcsb_holo), int(r.disagree),
            ])
    print(f"wrote {len(results)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, default=DEFAULT_COHORT,
                    help=f"cohort .txt (default: {DEFAULT_COHORT})")
    ap.add_argument("--csv", type=Path, help="write per-structure rows to this CSV")
    ap.add_argument("--no-rcsb", action="store_true",
                    help="skip the RCSB ligand-of-interest cross-check")
    args = ap.parse_args()

    use_rcsb = not args.no_rcsb
    pdb_ids = read_cohort(args.cohort)
    results = [analyze(p, use_rcsb) for p in pdb_ids]
    report(results, use_rcsb)
    if args.csv:
        write_csv(results, args.csv)


if __name__ == "__main__":
    main()
