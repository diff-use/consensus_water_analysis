"""Apo / holo classification for a carbonic-anhydrase cohort.

A structure is HOLO when a ligand-of-interest sits within HOLO_CUTOFF of the
catalytic-zinc active site. The site is located two ways:

  primary  (aligned-ref) : anchor on the REFERENCE structure's catalytic zinc in
                           the aligned frame and measure every aligned structure
                           against that one point. Works even for metal-free
                           structures and reads het groups straight from the
                           coordinates (the _pdbx_entity_nonpoly loop undercounts).
  fallback (his)         : for structures with no aligned CIF, find that
                           structure's own catalytic zinc intrinsically via
                           His-triad coordination and measure in its own frame.

Structures with neither an aligned CIF nor a His-coordinated zinc are reported
as undetermined. A ligand of interest is any het group that survives water /
monatomic-ion / additive / cofactor / borderline (sugar / substrate / heavy-atom
reagent) filtering.

Shared geometry / ligand primitives live in ``apo_holo_lib.py``. This script keeps
only the carbonic-anhydrase choices: the zinc anchor, sulfonamide flagging, the
aligned-ref-vs-His anchoring, and the apo cohort cut (``--select-apo``).
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Heavy imports (config, cw.io -> biotite, apo_holo_lib -> gemmi) are deferred into
# the analysis path so the CSV-only cohort cut (--from-csv) runs on the stdlib alone.
DEFAULT_COHORT = Path("data/carbonicanhydrase_000562_iso.txt")

HOLO_CUTOFF = 3.0    # A: ligand atom within this of the Zn anchor -> holo
CLEFT_CUTOFF = 6.0   # A: outer edge of the active-site cleft (apo cut + breakdown)
ZN_HIS_DIST = 2.6    # A: Zn-His(N) bond length, for catalytic-zinc detection
ZN_ON_SITE = 2.0     # A: own zinc this close to the ref zinc -> on canonical site

# Carbonic anhydrase adds sugars (cryo / glyco-conjugate) and the substrate gases
# to the non-ligand set, on top of the shared additives / cofactors / reagents.
SUGARS_AND_GASES = {
    "BGC", "GLC", "GAL", "MAN", "FUC", "NAG", "BMA", "SUC", "TRE", "CBI",
    "CO2", "CMO", "OXY",
}


def _non_ligand(lib) -> set[str]:
    return lib.ADDITIVES | lib.COFACTORS | lib.HEAVY_ATOM_REAGENTS | SUGARS_AND_GASES


# --- per-structure analysis ------------------------------------------------
@dataclass
class Result:
    pdb_id: str
    method: str                 # aligned-ref | his | undetermined | missing
    has_zn: bool
    his_catalytic: bool
    zn_offset: float | None     # own zinc -> ref site (aligned-ref only)
    lig_dist: float | None      # nearest ligand -> active-site anchor
    lig_comp: str | None
    has_sulfonamide: bool

    @property
    def holo(self) -> bool:
        return self.lig_dist is not None and self.lig_dist <= HOLO_CUTOFF


def is_apo(holo: bool, his_catalytic: bool, lig_dist: float | None) -> bool:
    """Apo cohort predicate: intact catalytic Zn, empty active-site cleft.

    Keeps the His-triad zinc (organizing center of the active-site water network)
    and requires no ligand of interest inside the ~6 A cleft, so the inhibitor is
    the only variable versus the sulfonamide-holo cohort."""
    return (not holo) and his_catalytic and (lig_dist is None or lig_dist > CLEFT_CUTOFF)


def analyze(pdb_id: str, ref_point, aligned_dir: Path, lib, non_ligand: set[str]) -> Result:
    import config
    from cw.io import cif_path_for

    aligned = aligned_dir / f"{pdb_id}.cif"
    raw = cif_path_for(pdb_id, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE)

    if aligned.exists():
        model, method, anchor = lib.read_model(aligned), "aligned-ref", ref_point
    elif raw.exists():
        model = lib.read_model(raw)
        method, anchor = "his", lib.catalytic_zinc(model, ZN_HIS_DIST)
    else:
        return Result(pdb_id, "missing", False, False, None, None, None, False)

    zincs = lib.zinc_positions(model)
    his_cat = lib.catalytic_zinc(model, ZN_HIS_DIST) is not None
    if anchor is None:  # His fallback found no catalytic zinc
        return Result(pdb_id, "undetermined", bool(zincs), his_cat,
                      None, None, None, False)

    zn_offset = (min(z.dist(ref_point) for z in zincs)
                 if zincs and method == "aligned-ref" else None)
    comps = lib.ligands_of_interest(model, non_ligand)
    if comps:
        dist, comp = lib.nearest_ligand(model, comps, anchor)
        sulfo = lib.has_sulfonamide(model, comps)
    else:
        dist, comp, sulfo = None, None, False
    return Result(pdb_id, method, bool(zincs), his_cat, zn_offset, dist, comp, sulfo)


# --- reporting -------------------------------------------------------------
def report(results: list[Result]):
    usable = [r for r in results if r.method in ("aligned-ref", "his")]
    parsed = len(usable)
    method_counts = Counter(r.method for r in results)

    print(f"  structures        : {len(results)}")
    print(f"  analyzed          : {parsed}  "
          f"(aligned-ref={method_counts['aligned-ref']}, his-fallback={method_counts['his']})")
    print(f"  undetermined      : {method_counts['undetermined']}  "
          f"missing={method_counts['missing']}")
    print()

    with_zn = [r for r in usable if r.has_zn]
    on_site = [r for r in with_zn if r.zn_offset is not None and r.zn_offset <= ZN_ON_SITE]
    offsets = sorted(r.zn_offset for r in with_zn if r.zn_offset is not None)
    print("--- Catalytic zinc ---")
    print(f"  with a Zn atom              : {len(with_zn)}")
    print(f"  Zn on canonical site (<= {ZN_ON_SITE} A) : {len(on_site)}")
    if offsets:
        print(f"  Zn-to-ref offset            : median={statistics.median(offsets):.2f} A, "
              f"max={offsets[-1]:.2f} A")
    print(f"  His-triad catalytic Zn      : {sum(r.his_catalytic for r in usable)}  (fallback method)")
    print(f"  metal-free                  : {sum(not r.has_zn for r in usable)}")
    print()

    holo = [r for r in usable if r.holo]
    print(f"--- Holo (ligand <= {HOLO_CUTOFF} A of the zinc) ---")
    print(f"  HOLO                        : {len(holo)}")
    print(f"     sulfonamide / sulfamate  : {sum(r.has_sulfonamide for r in holo)}")
    print(f"     non-sulfonamide          : {sum(not r.has_sulfonamide for r in holo)}")
    print(f"  APO                         : {parsed - len(holo)}")
    print(f"     apo cohort (empty cleft) : {sum(is_apo(r.holo, r.his_catalytic, r.lig_dist) for r in usable)}")
    print()

    print("  nearest ligand-to-zinc distance breakdown:")
    print(f"     {'bucket':<22}{'total':>6}{'sulfon':>8}{'non-sulf':>9}")
    buckets = [(0.0, HOLO_CUTOFF, f"<={HOLO_CUTOFF:g} coordinating"),
               (HOLO_CUTOFF, CLEFT_CUTOFF, f"{HOLO_CUTOFF:g}-{CLEFT_CUTOFF:g} cleft"),
               (CLEFT_CUTOFF, float("inf"), f">{CLEFT_CUTOFF:g} surface")]
    for lo, hi, name in buckets:
        sel = [r for r in usable if r.lig_dist is not None and lo < r.lig_dist <= hi]
        s = sum(r.has_sulfonamide for r in sel)
        print(f"     {name:<22}{len(sel):>6}{s:>8}{len(sel) - s:>9}")
    no_lig = sum(r.lig_dist is None for r in usable)
    print(f"     {'no ligand':<22}{no_lig:>6}")


def write_csv(results: list[Result], path: Path):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb_id", "method", "has_zn", "his_catalytic", "zn_offset",
                    "holo", "lig_dist", "lig_comp", "has_sulfonamide"])
        for r in results:
            w.writerow([r.pdb_id, r.method, int(r.has_zn), int(r.his_catalytic),
                        f"{r.zn_offset:.3f}" if r.zn_offset is not None else "",
                        int(r.holo),
                        f"{r.lig_dist:.3f}" if r.lig_dist is not None else "",
                        r.lig_comp or "", int(r.has_sulfonamide)])
    print(f"wrote {len(results)} rows -> {path}")


# --- apo cohort cut --------------------------------------------------------
def _apo_ids_from_results(results: list[Result]) -> list[str]:
    return sorted(r.pdb_id for r in results
                  if is_apo(r.holo, r.his_catalytic, r.lig_dist))


def _apo_ids_from_csv(path: Path) -> list[str]:
    def num(x):
        return float(x) if x not in ("", "None") else None
    ids = []
    for row in csv.DictReader(path.open()):
        holo = int(row["holo"] or 0) == 1
        his = int(row["his_catalytic"] or 0) == 1
        if is_apo(holo, his, num(row["lig_dist"])):
            ids.append(row["pdb_id"])
    return sorted(ids)


def write_cohort(ids: list[str], path: Path):
    path.write_text("\n".join(ids) + "\n")
    print(f"wrote {len(ids)} apo structures -> {path}  (3ks3 present: {'3ks3' in ids})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, default=DEFAULT_COHORT,
                    help=f"cohort .txt (default: {DEFAULT_COHORT})")
    ap.add_argument("--aligned-dir", type=Path, default=None,
                    help="directory of aligned reference-frame CIFs "
                         "(default: <DATA_DIR>/carbonicanhydrase_000562_iso/aligned_pdbs)")
    ap.add_argument("--reference", default=None,
                    help="reference PDB ID for the aligned anchor (default: config.REF_PDB_ID)")
    ap.add_argument("--csv", type=Path, help="write per-structure rows to this CSV")
    ap.add_argument("--select-apo", type=Path,
                    help="write the apo cohort .txt (empty-cleft predicate)")
    ap.add_argument("--from-csv", type=Path,
                    help="cut --select-apo from an existing classification CSV "
                         "instead of re-analysing the CIFs")
    args = ap.parse_args()

    # Fast path: cut the apo cohort straight from a saved CSV (no CIFs / pod needed).
    if args.from_csv is not None:
        if args.select_apo is None:
            raise SystemExit("--from-csv requires --select-apo")
        write_cohort(_apo_ids_from_csv(args.from_csv), args.select_apo)
        return

    sys.path.insert(0, str(Path(__file__).parent))          # apo_holo_lib
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # config, cw
    import config
    from cw.io import read_cohort
    import apo_holo_lib as lib

    aligned_dir = args.aligned_dir or (
        Path(config.DATA_DIR) / "carbonicanhydrase_000562_iso" / "aligned_pdbs")
    reference = args.reference or config.REF_PDB_ID
    non_ligand = _non_ligand(lib)

    ref_point = lib.catalytic_zinc(lib.read_model(aligned_dir / f"{reference}.cif"), ZN_HIS_DIST)
    if ref_point is None:
        raise SystemExit(f"no catalytic zinc found in reference {reference}")

    pdb_ids = read_cohort(args.cohort)
    print(f"Cohort: {args.cohort.name}")
    results = [analyze(p, ref_point, aligned_dir, lib, non_ligand) for p in pdb_ids]
    report(results)
    if args.csv:
        write_csv(results, args.csv)
    if args.select_apo:
        write_cohort(_apo_ids_from_results(results), args.select_apo)


if __name__ == "__main__":
    main()
