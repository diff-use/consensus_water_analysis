"""Shared apo/holo classification primitives (protein-agnostic).

The per-protein scripts (``apo_holo.py`` for carbonic anhydrase, ``apo_holo_hewl.py``
for lysozyme) import from here. Everything in this module is pure geometry / ligand
bookkeeping over a gemmi model — no I/O, no cohort logic, no ``__main__``.

The apo/holo pattern is the same for every protein:
  1. locate the active-site anchor (a catalytic metal, or a catalytic-residue centroid),
  2. enumerate the ligands of interest actually modeled in the coordinates,
  3. call holo when a ligand of interest sits within a cutoff of the anchor.

What differs per protein is only (a) which anchor finder to use and (b) which het
groups count as ligands of interest — so those two choices stay in the per-protein
scripts, and each builds its own ``NON_LIGAND`` set from the shared building blocks below.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import gemmi

# --- ligand-filtering building blocks ---------------------------------------
# Protein-agnostic. Each script composes NON_LIGAND from these plus its own extras
# (carbonic anhydrase also excludes sugars/gases; lysozyme keeps sugars as ligands).
WATER = {"HOH", "DOD", "WAT", "H2O", "D2O"}

ADDITIVES = {
    # polyols / cryo
    "GOL", "EDO", "MPD", "MRD", "BU3", "PGO", "PDO", "1BO", "TBU", "IPA", "MOH", "EOH",
    # PEGs
    "PEG", "PG4", "PGE", "P6G", "1PE", "2PE", "7PE", "12P", "15P", "PE3", "PE4", "PE5",
    "PE8", "M2M", "MME", "EGL", "P33", "PG6", "XPE",
    # buffer / salt anions and small organics
    "SO4", "PO4", "PI", "2PO", "NO3", "CO3", "ACT", "ACY", "FMT", "OXL", "SIN", "TLA",
    "TAR", "MLA", "MLI", "MAE", "FLC", "CIT", "CAC", "SCN", "AZI", "BCT", "DMS", "DMF",
    "BME", "MES", "EPE", "TRS", "TAM", "BTB", "BIS", "MPO", "NHE", "PIN", "IMD", "IPH",
    "BEZ", "ACN", "GAI", "URE", "NH4", "DTT", "DTU", "TCE", "TFA", "CXS", "MLT", "POP",
    "BCN", "EEE", "TEG", "144", "DIO", "DOX", "GLY", "BEN",
}

COFACTORS = {
    "HEM", "HEC", "HEA", "HEB", "HAS", "DHE", "SRM", "VER",
    "FAD", "FMN", "FDA", "FNR",
    "NAD", "NAP", "NDP", "NAI", "NAH",
    "PLP", "PMP", "LLP",
    "SAM", "SAH",
    "COA", "ACO", "COZ",
    "BTN", "TPP", "TDP", "PQQ", "B12", "COB", "MGD", "F43", "GSH",
}

# Heavy-atom / phasing / Cys reagents — real het groups but not functional ligands.
HEAVY_ATOM_REAGENTS = {"MBO", "HGB", "MMC", "MAC"}

# Common monatomic ions, for filtering the RCSB bound-component list (where there
# are no coordinates to count heavy atoms).
MONATOMIC_IONS = {
    "NA", "CL", "K", "CA", "MG", "ZN", "MN", "FE", "FE2", "CU", "CU1", "NI", "CO",
    "CD", "HG", "BR", "IOD", "F", "LI", "CS", "RB", "SR", "BA", "PB", "AU", "PT",
    "AG", "TL", "XE", "KR",
}


# --- parsing / geometry helpers --------------------------------------------
def read_model(path: Path):
    return gemmi.read_structure(str(path))[0]


def heavy_atom_counts(model) -> dict[str, int]:
    """comp_id -> number of distinct heavy (non-H/D) atom names."""
    names: dict[str, set] = {}
    for chain in model:
        for res in chain:
            bucket = names.setdefault(res.name, set())
            for atom in res:
                if atom.element.name not in ("H", "D"):
                    bucket.add(atom.name)
    return {comp: len(atoms) for comp, atoms in names.items()}


def het_inventory(model) -> Counter:
    """Every non-water, non-polymer comp_id present in the coordinates.

    Ground truth of what is modeled — independent of the ``_pdbx_entity_nonpoly``
    loop, which undercounts in the re-refined CIFs. Ions and additives included so
    the full inventory is visible.
    """
    counts: Counter = Counter()
    for chain in model:
        for res in chain:
            name = res.name
            if name in WATER:
                continue
            info = gemmi.find_tabulated_residue(name)
            if info and (info.is_amino_acid() or info.is_nucleic_acid()):
                continue
            counts[name] += 1
    return counts


def ligands_of_interest(model, non_ligand: set[str]) -> set[str]:
    """Het comp_ids that are neither polymer, water, monatomic ion, nor a member of
    the caller's ``non_ligand`` set (additives / cofactors / protein-specific extras)."""
    heavy = heavy_atom_counts(model)
    out = set()
    for chain in model:
        for res in chain:
            name = res.name
            if name in WATER or name in non_ligand:
                continue
            info = gemmi.find_tabulated_residue(name)
            if info and (info.is_amino_acid() or info.is_nucleic_acid()):
                continue
            if heavy.get(name, 0) <= 1:  # monatomic ion / metal
                continue
            out.add(name)
    return out


def nearest_ligand(model, comp_ids: set[str], anchor):
    """(distance, comp_id) of the ligand-of-interest atom closest to anchor."""
    best_d, best_c = None, None
    for chain in model:
        for res in chain:
            if res.name in comp_ids:
                d = min(a.pos.dist(anchor) for a in res)
                if best_d is None or d < best_d:
                    best_d, best_c = d, res.name
    return best_d, best_c


# --- active-site anchors ----------------------------------------------------
def zinc_positions(model) -> list:
    return [a.pos for chain in model for res in chain for a in res
            if a.element.name == "Zn"]


def catalytic_zinc(model, zn_his_dist: float = 2.6):
    """Position of the zinc coordinated by the most His imidazole N (>=2), i.e. the
    active-site zinc, or None. Used for metalloenzymes (carbonic anhydrase)."""
    his_n = [a.pos for chain in model for r in chain if r.name == "HIS"
             for a in r if a.name in ("ND1", "NE2")]
    best, best_n = None, 0
    for z in zinc_positions(model):
        n = sum(1 for p in his_n if z.dist(p) <= zn_his_dist)
        if n > best_n:
            best, best_n = z, n
    return best if best_n >= 2 else None


def catalytic_dyad_anchor(model, schemes: list[dict]):
    """Centroid of the catalytic-residue carboxylate oxygens for the first numbering
    ``scheme`` whose residues are all present, or None. Used for metal-free active
    sites (lysozyme Glu35/Asp52). Each scheme maps (resname, seqid) -> atom names;
    a residue with none of its named atoms falls back to its full heavy-atom set."""
    for scheme in schemes:
        per_residue: dict = {}
        for chain in model:
            for res in chain:
                key = (res.name, res.seqid.num)
                if key not in scheme:
                    continue
                picked = [a.pos for a in res if a.name in scheme[key]]
                if not picked:
                    picked = [a.pos for a in res if a.element.name not in ("H", "D")]
                per_residue.setdefault(key, []).extend(picked)
        if len(per_residue) < len(scheme):  # need every residue of the scheme
            continue
        positions = [p for group in per_residue.values() for p in group]
        n = len(positions)
        return gemmi.Position(sum(p.x for p in positions) / n,
                              sum(p.y for p in positions) / n,
                              sum(p.z for p in positions) / n)
    return None


# --- ligand chemistry -------------------------------------------------------
def _residue_has_sulfonamide(res) -> bool:
    """An S bonded to >=2 O and >=1 N within the residue (sulfonamide/sulfamate)."""
    atoms = list(res)
    for s in (a for a in atoms if a.element.name == "S"):
        n_o = sum(1 for a in atoms if a is not s and a.element.name == "O"
                  and s.pos.dist(a.pos) <= 1.65)
        n_n = sum(1 for a in atoms if a is not s and a.element.name == "N"
                  and s.pos.dist(a.pos) <= 1.95)
        if n_o >= 2 and n_n >= 1:
            return True
    return False


def has_sulfonamide(model, comp_ids: set[str]) -> bool:
    return any(res.name in comp_ids and _residue_has_sulfonamide(res)
               for chain in model for res in chain)
