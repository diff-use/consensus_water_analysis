"""Rigid-body align one starting model onto the MTZ-source structure.

The Phenix re-refinement matrix refines a pair (PDBID, REF_PDBID): PDBID's MTZ
data, REF_PDBID's model as the starting model. The MTZ lives in PDBID's deposited
crystal frame, but the REF_PDBID model is in its own frame — an origin/orientation
offset that rigid-body refinement may not recover, giving poor results (e.g. 5kxn).

This script aligns the REF_PDBID model (``--mobile``) onto PDBID's deposited model
(``--reference``) using the pipeline's BLOSUM62-paired Cα Kabsch superposition
(cw.align.align_to_reference): only protein Cα drive the fit, but the whole model
is transformed. It then adopts PDBID's crystal symmetry (unit cell + space group,
read from ``--reference-mtz``) so the aligned model and the reflection data agree
on the lattice. The downstream phenix.pdbtools call preserves this while dropping
deposited TLS.

Pure project-env Python (biotite/numpy/gemmi) — no Phenix dependency, so it runs
before the Phenix env is sourced.

Usage:
    uv run scripts/phenix/align_starting_model.py \
        --mobile <REF_PDBID cif> --reference <PDBID cif> --out <aligned cif> \
        [--reference-mtz <PDBID mtz>] [--pdb-id <PDBID>] [--ref-pdb-id <REF_PDBID>]

Emits one CSV line to stdout:
    pdb_id,ref_pdbid,n_common_ca,rmsd_before,rmsd_after,status

Exit codes: 0 aligned, 2 skipped (too few common Cα — caller should fall back to
the raw model), 1 on error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import gemmi

from cw.align import align_to_reference
from cw.io import load_protein


def adopt_crystal_symmetry(aligned_cif: Path, reference_mtz: Path) -> None:
    """Stamp the reference MTZ's unit cell + space group onto the aligned cif.

    The Cα alignment placed the model's coordinates in the MTZ-source Cartesian
    frame, but the cif header still carries the mobile model's lattice. phenix
    interprets fractional coordinates and symmetry with the model's cell, so it
    must match the reflection data. A gemmi round-trip rewrites the lattice; the
    later phenix.pdbtools call re-emits the model (dropping deposited TLS).
    """
    mtz = gemmi.read_mtz_file(str(reference_mtz))
    st = gemmi.read_structure(str(aligned_cif))
    st.cell = mtz.cell
    st.spacegroup_hm = mtz.spacegroup.hm
    st.make_mmcif_document().write_file(str(aligned_cif))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mobile", type=Path, required=True, help="Starting-model cif (REF_PDBID), transformed")
    parser.add_argument("--reference", type=Path, required=True, help="MTZ-source cif (PDBID), alignment target")
    parser.add_argument("--out", type=Path, required=True, help="Output path for the aligned cif")
    parser.add_argument(
        "--reference-mtz",
        type=Path,
        default=None,
        help="MTZ-source reflections (PDBID); its cell + space group are adopted into the aligned model",
    )
    parser.add_argument("--pdb-id", default=None, help="MTZ-source id, for the report line")
    parser.add_argument("--ref-pdb-id", default=None, help="Starting-model id, for the report line")
    args = parser.parse_args()

    pdb_id = args.pdb_id or args.reference.stem.removesuffix("_final")
    ref_pdb_id = args.ref_pdb_id or args.mobile.stem.removesuffix("_final")

    def report(n_common_ca, rmsd_before, rmsd_after, status) -> None:
        print(f"{pdb_id},{ref_pdb_id},{n_common_ca},{rmsd_before},{rmsd_after},{status}")

    try:
        ref_protein, _ = load_protein(args.reference)
        result = align_to_reference(args.mobile, ref_protein, out_path=args.out, pdb_id=ref_pdb_id)
    except Exception as exc:  # noqa: BLE001 — surface any failure as a report line + exit 1
        report("", "", "", f"error:{type(exc).__name__}")
        print(f"{ref_pdb_id} -> {pdb_id}: {exc}", file=sys.stderr)
        return 1

    if result is None:
        # Too few common Cα — leave --out unwritten so the caller falls back to raw.
        report("", "", "", "skipped")
        return 2

    if args.reference_mtz is not None:
        adopt_crystal_symmetry(args.out, args.reference_mtz)

    report(result["n_common_ca"], result["rmsd_before"], result["rmsd_after"], "aligned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
