"""Apo / holo classification for a DJ-1 (PARK7) cohort.

DJ-1 has no catalytic metal and no dyad — the active site is organized around a
single nucleophilic cysteine, Cys106, so the anchor here is that residue's Cβ
(``apo_holo.py`` uses a catalytic zinc; ``apo_holo_hewl.py`` and
``apo_holo_endothiapepsin.py`` use a two-carboxylate centroid). Cys106 is present
in every model, so each structure is analysed in its OWN frame (raw CIF); no
aligned reference is needed.

Cys106 is the cohort's second axis of variation and it is *modelled as the residue
itself*: the deposited comp_id at seqid 106 is CYS when reduced, an oxidation
state (CSO / CSD / SNC / …) when modified, a standard amino acid when the
construct is a C106X mutant, and a fused ligand comp_id when a compound is bound
covalently. So the residue name at 106 is read first, and a covalent adduct there
is reported as its own call rather than being counted as an ordinary het ligand.

A structure is HOLO when either (a) residue 106 is a covalent ligand adduct, or
(b) a non-covalent ligand-of-interest atom sits within SITE_CUTOFF of Cβ106.
Metals (Cu / Zn, which DJ-1 binds at the Cys106 site) are monatomic and therefore
never ligands-of-interest; their distance to the anchor is reported separately.

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

import gemmi

import config
from cw.io import cif_path_for, read_cohort

import apo_holo_lib as lib

# --- configuration ---------------------------------------------------------
DEFAULT_COHORT = Path("data/C000602.txt")

CATALYTIC_SEQID = 106      # DJ-1 nucleophile, mature numbering (chain is 1-188)
SITE_CUTOFF = 6.0          # A: non-covalent ligand atom within this of Cb106 -> holo
PERIPHERAL_CUTOFF = 12.0   # A: outer edge of the active-site pocket (breakdown only)
METAL_CUTOFF = 6.0         # A: metal within this of Cb106 -> site metal

# Cys106 states that are chemical modifications of the cysteine, not bound
# compounds: oxidation ladder (sulfenic / sulfinic / sulfonic), S-nitrosylation,
# and the iodoacetate/iodoacetamide reagent adducts.
CYS_MODIFICATIONS = {
    "CSO", "CSD", "CSX", "OCS", "CSW", "CSS", "CSU", "CAS", "YCM",
    "SNC", "CME", "CMT", "CCS", "SCY", "CSA",
}

NON_LIGAND = lib.ADDITIVES | lib.COFACTORS | lib.HEAVY_ATOM_REAGENTS


def catalytic_residue(model):
    """The residue at CATALYTIC_SEQID that is part of the polymer, or None.

    Selects on backbone atoms rather than on seqid alone — waters and het groups
    can carry the same seqid in a different chain.
    """
    for chain in model:
        for res in chain:
            if res.seqid.num != CATALYTIC_SEQID:
                continue
            names = {atom.name for atom in res}
            if {"N", "CA", "C", "CB"} <= names:
                return res
    return None


def catalytic_anchor(res):
    """Cb of the catalytic residue — the one side-chain atom every Cys106 state,
    C106X mutant, and covalent adduct shares."""
    for atom in res:
        if atom.name == "CB":
            return atom.pos
    return None


def cys106_state(res_name: str) -> str:
    if res_name == "CYS":
        return "reduced"
    if res_name in CYS_MODIFICATIONS:
        return "modified"
    info = gemmi.find_tabulated_residue(res_name)
    if info and info.is_amino_acid():
        return "mutant"
    return "covalent-ligand"


def nearest_metal(model, anchor):
    """(distance, element) of the metal atom closest to the anchor."""
    best_d, best_e = None, None
    for chain in model:
        for res in chain:
            for atom in res:
                if not atom.element.is_metal:
                    continue
                d = atom.pos.dist(anchor)
                if best_d is None or d < best_d:
                    best_d, best_e = d, atom.element.name
    return best_d, best_e


def nearest_het(model, anchor, skip: set[str]):
    """(distance, comp_id) of the closest atom of ANY non-water het species —
    additives, cryoprotectants and ions included.

    An "apo" structure whose catalytic site is occupied by glycerol, DTT or
    tartrate is not an empty site for water-network purposes, so the site has to
    be auditable independently of what counts as a ligand-of-interest.
    """
    best_d, best_c = None, None
    for chain in model:
        for res in chain:
            if res.name in lib.WATER or res.name in skip:
                continue
            info = gemmi.find_tabulated_residue(res.name)
            if info and (info.is_amino_acid() or info.is_nucleic_acid()):
                continue
            for atom in res:
                d = atom.pos.dist(anchor)
                if best_d is None or d < best_d:
                    best_d, best_c = d, res.name
    return best_d, best_c


# --- per-structure analysis ------------------------------------------------
@dataclass
class Result:
    pdb_id: str
    method: str                      # ok | no-cys106 | missing
    res106: str = ""                 # comp_id at seqid 106
    state106: str = ""               # reduced | modified | mutant | covalent-ligand
    inventory: Counter = field(default_factory=Counter)  # all non-water het
    ligands: list[str] = field(default_factory=list)     # non-covalent ligands of interest
    lig_dist: float | None = None    # nearest non-covalent ligand -> Cb106
    lig_comp: str | None = None
    metal_dist: float | None = None
    metal_name: str | None = None
    het_dist: float | None = None     # nearest het of any kind (additives, ions) -> Cb106
    het_comp: str | None = None

    @property
    def covalent(self) -> bool:
        return self.state106 == "covalent-ligand"

    @property
    def holo(self) -> bool:
        return self.covalent or (self.lig_dist is not None and self.lig_dist <= SITE_CUTOFF)

    @property
    def call(self) -> str:
        if self.method != "ok":
            return self.method
        if self.covalent:
            return "holo-covalent"
        if self.lig_dist is not None and self.lig_dist <= SITE_CUTOFF:
            return "holo"
        if self.ligands:  # present but outside the active-site pocket
            return "apo-peripheral-ligand"
        return "apo"

    @property
    def site_metal(self) -> bool:
        return self.metal_dist is not None and self.metal_dist <= METAL_CUTOFF

    @property
    def site_het(self) -> bool:
        """Any non-water het species in the active site, ligand-of-interest or not."""
        return self.het_dist is not None and self.het_dist <= SITE_CUTOFF


def analyze(pdb_id: str) -> Result:
    cif = cif_path_for(pdb_id, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE)
    if not cif.exists():
        return Result(pdb_id, "missing")

    model = lib.read_model(cif)
    inventory = lib.het_inventory(model)
    res = catalytic_residue(model)
    anchor = catalytic_anchor(res) if res is not None else None
    if anchor is None:
        return Result(pdb_id, "no-cys106", inventory=inventory)

    state = cys106_state(res.name)
    # A covalent adduct is the residue at 106, so it also shows up in the het
    # inventory; drop it from the non-covalent search to keep the two calls distinct.
    comps = lib.ligands_of_interest(model, NON_LIGAND) - {res.name}
    dist, comp = lib.nearest_ligand(model, comps, anchor) if comps else (None, None)
    metal_dist, metal_name = nearest_metal(model, anchor)
    het_dist, het_comp = nearest_het(model, anchor, skip={res.name})
    return Result(pdb_id, "ok", res106=res.name, state106=state, inventory=inventory,
                  ligands=sorted(comps), lig_dist=dist, lig_comp=comp,
                  metal_dist=metal_dist, metal_name=metal_name,
                  het_dist=het_dist, het_comp=het_comp)


# --- reporting -------------------------------------------------------------
def report(results: list[Result]):
    calls = Counter(r.call for r in results)
    usable = [r for r in results if r.method == "ok"]

    print(f"structures         : {len(results)}")
    print(f"  analyzed (ok)    : {len(usable)}")
    print(f"  no Cys106        : {calls['no-cys106']}   missing: {calls['missing']}")
    print()
    print(f"--- Call (holo = ligand of interest <= {SITE_CUTOFF:g} A of Cb106, "
          f"or covalent at 106) ---")
    print(f"  APO                      : {calls['apo']}")
    print(f"  APO (peripheral ligand)  : {calls['apo-peripheral-ligand']}")
    print(f"  HOLO (non-covalent)      : {calls['holo']}")
    print(f"  HOLO (covalent at 106)   : {calls['holo-covalent']}")
    print()

    print("  nearest non-covalent ligand-to-Cb106 distance breakdown:")
    buckets = [(0.0, SITE_CUTOFF, f"<={SITE_CUTOFF:g} active site"),
               (SITE_CUTOFF, PERIPHERAL_CUTOFF, f"{SITE_CUTOFF:g}-{PERIPHERAL_CUTOFF:g} pocket edge"),
               (PERIPHERAL_CUTOFF, float("inf"), f">{PERIPHERAL_CUTOFF:g} surface")]
    for lo, hi, name in buckets:
        sel = [r for r in usable if r.lig_dist is not None and lo < r.lig_dist <= hi]
        print(f"     {name:<24}{len(sel):>6}")
    no_lig = sum(r.lig_dist is None for r in usable)
    print(f"     {'no ligand of interest':<24}{no_lig:>6}")
    print()

    print("--- Cys106 state (comp_id at seqid 106) ---")
    states = Counter(r.state106 for r in usable)
    for state in ("reduced", "modified", "mutant", "covalent-ligand"):
        names = Counter(r.res106 for r in usable if r.state106 == state)
        listed = ", ".join(f"{c}:{n}" for c, n in names.most_common())
        print(f"  {state:<16}{states[state]:>4}   {listed}")
    print()

    print("--- Call x Cys106 state ---")
    order = [c for c in ("apo", "apo-peripheral-ligand", "holo", "holo-covalent") if calls[c]]
    state_order = [s for s in ("reduced", "modified", "mutant", "covalent-ligand") if states[s]]
    print(f"  {'':<24}" + "".join(f"{s:>17}" for s in state_order))
    for call in order:
        row = [sum(1 for r in usable if r.call == call and r.state106 == s) for s in state_order]
        print(f"  {call:<24}" + "".join(f"{n:>17}" for n in row))
    print()

    site_metal = [r for r in usable if r.site_metal]
    print(f"--- Metals (<= {METAL_CUTOFF:g} A of Cb106) ---")
    print(f"  structures with a site metal: {len(site_metal)}")
    for r in site_metal:
        print(f"     {r.pdb_id}  {r.metal_name} at {r.metal_dist:.2f} A   ({r.call})")
    print()

    apo_with_het = [r for r in usable if r.call.startswith("apo") and r.site_het]
    print(f"--- Apo structures whose site is occupied by a non-ligand het "
          f"(<= {SITE_CUTOFF:g} A of Cb106) ---")
    print(f"  {len(apo_with_het)} of {sum(r.call.startswith('apo') for r in usable)} apo structures")
    for r in sorted(apo_with_het, key=lambda r: r.het_dist):
        print(f"     {r.pdb_id}  {r.het_comp} at {r.het_dist:.2f} A   ({r.call}, 106={r.res106})")
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
        elif comp in CYS_MODIFICATIONS:
            tag = "Cys106 modification"
        else:
            tag = "LIGAND-OF-INTEREST"
        print(f"  {comp:<8} in {n:>3} structures   [{tag}]")


def write_csv(results: list[Result], path: Path):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "pdb_id", "call", "method", "res106", "state106", "all_het",
            "ligands_of_interest", "lig_dist", "lig_comp", "metal_dist", "metal_name",
            "site_metal", "het_dist", "het_comp", "site_het", "holo",
        ])
        for r in results:
            all_het = "|".join(f"{c}:{n}" for c, n in sorted(r.inventory.items()))
            w.writerow([
                r.pdb_id, r.call, r.method, r.res106, r.state106, all_het,
                "|".join(r.ligands),
                f"{r.lig_dist:.3f}" if r.lig_dist is not None else "",
                r.lig_comp or "",
                f"{r.metal_dist:.3f}" if r.metal_dist is not None else "",
                r.metal_name or "", int(r.site_metal),
                f"{r.het_dist:.3f}" if r.het_dist is not None else "",
                r.het_comp or "", int(r.site_het), int(r.holo),
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
