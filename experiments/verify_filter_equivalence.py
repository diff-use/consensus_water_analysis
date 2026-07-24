"""One-off: confirm the filter refactor is byte-identical to HEAD for distance-only.

Reconstructs HEAD's cw.filter.filter_by_distance (pre-refactor 5-tuple, distance-only)
and compares it against the refactored cw.filter.filter_waters on every hewls_65 member:
- in-memory: keep_mask, filtered coords, and the per-structure counts
- on-disk: the actual written CIF bytes

Also reports the edia_cutoff=-inf case (full EDIA lists, most-permissive cutoff) so we
can see where it does / doesn't coincide with distance-only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import biotite.structure.io.pdbx as pdbx
import gemmi
import numpy as np

import config
from cw.filter import filter_waters
from cw.io import (
    cif_path_for,
    load_edia_all_altlocs,
    read_cohort,
    water_oxygen_mask,
    write_filtered_cif,
)

CUTOFF = config.WATER_PROT_DIST_CUTOFF


def _head_filter_by_distance():
    snapshot = Path(__file__).parent / "_filter_head_snapshot.py"
    spec = importlib.util.spec_from_file_location("filter_head", snapshot)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.filter_by_distance


def _load(cif: Path):
    return pdbx.get_structure(
        pdbx.CIFFile.read(str(cif)),
        model=1,
        altloc="all",
        extra_fields=["b_factor", "occupancy"],
    )


def main() -> None:
    cohort = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(config.DATA_DIR) / "hewls_65.txt"
    if not cohort.exists():
        cohort = Path("data/hewls_65.txt")
    head_filter = _head_filter_by_distance()

    out_new = Path("/tmp/filter_cmp/new")
    out_old = Path("/tmp/filter_cmp/old")
    out_new.mkdir(parents=True, exist_ok=True)
    out_old.mkdir(parents=True, exist_ok=True)

    members = read_cohort(cohort)
    checked = mismatches = missing = 0
    inf_diverged = []

    for m in members:
        cif = cif_path_for(m, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE)
        if not cif.exists():
            missing += 1
            continue
        st = gemmi.read_structure(str(cif))
        sg = st.find_spacegroup() or gemmi.SpaceGroup("P 1")

        atoms_new = _load(cif)
        atoms_old = _load(cif)
        f_new, stats, mask_new = filter_waters(atoms_new, st.cell, sg, CUTOFF)
        f_old, n_water, n_removed, n_moved, mask_old = head_filter(atoms_old, st.cell, sg, CUTOFF)

        ok = (
            np.array_equal(mask_new, mask_old)
            and f_new.array_length() == f_old.array_length()
            and np.array_equal(f_new.coord, f_old.coord)
            and stats["n_water"] == n_water
            and stats["n_removed_distance"] == n_removed
            and stats["n_moved"] == n_moved
        )

        write_filtered_cif(cif, mask_new, f_new, out_new / f"{m}.cif")
        write_filtered_cif(cif, mask_old, f_old, out_old / f"{m}.cif")
        same_bytes = (out_new / f"{m}.cif").read_bytes() == (out_old / f"{m}.cif").read_bytes()

        if not (ok and same_bytes):
            mismatches += 1
            print(f"MISMATCH {m}: in_mem_ok={ok} same_bytes={same_bytes}")

        # bonus: most-permissive edia cutoff with full lists
        edia = load_edia_all_altlocs(cif_path_for(m, config.ALL_PDB_REDO_DIR, config.EDIA_TEMPLATE))
        _, stats_inf, mask_inf = filter_waters(
            _load(cif), st.cell, sg, CUTOFF, edia_lists=edia, edia_cutoff=-np.inf
        )
        if not np.array_equal(mask_inf, mask_old):
            inf_diverged.append((m, stats_inf["n_removed_edia"]))

        checked += 1

    print(f"\nchecked={checked} missing={missing} mismatches={mismatches}")
    print(f"distance-only refactor identical to HEAD: {mismatches == 0}")
    if inf_diverged:
        print(
            f"\nedia_cutoff=-inf diverged from distance-only for {len(inf_diverged)} structures "
            "(waters lacking any EDIA score are dropped even at -inf):"
        )
        for m, n in inf_diverged:
            print(f"  {m}: {n} extra dropped")
    else:
        print("\nedia_cutoff=-inf identical to distance-only for all structures.")


if __name__ == "__main__":
    main()
