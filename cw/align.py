"""Alignment math: sequence-guided Kabsch superposition of protein structures.

All file I/O (load_protein, write_transformed_cif) lives in cw.io.
"""

from __future__ import annotations

from pathlib import Path

import biotite.sequence as bseq
import biotite.sequence.align as bseq_align
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np

from cw.io import load_protein, write_transformed_cif

GAP_PENALTY = (-10, -1)


def paired_alignment_trace(seq_a: bseq.ProteinSequence, seq_b: bseq.ProteinSequence) -> np.ndarray:
    """Global BLOSUM62 alignment of two protein sequences.

    Returns the (n, 2) array of trace rows where neither sequence has a gap;
    column 0 indexes seq_a, column 1 indexes seq_b. Terminal gaps are free, so
    ragged termini (missing density) are not penalized.
    """
    matrix = bseq_align.SubstitutionMatrix.std_protein_matrix()
    alignment = bseq_align.align_optimal(  # type: ignore
        seq_a, seq_b, matrix, gap_penalty=GAP_PENALTY, terminal_penalty=False
    )[0]
    trace = alignment.trace
    return trace[(trace[:, 0] != -1) & (trace[:, 1] != -1)]


def get_ca_coords_and_sequence(protein: struc.AtomArray) -> tuple[np.ndarray, str]:
    """Extract Cα positions and one-letter sequence in matching order.

    Non-standard residues without a known one-letter code are skipped.
    All protein chains are concatenated (deliberate: superposition uses every Cα).
    """
    ca = protein[protein.atom_name == "CA"]
    coords, seq_chars = [], []
    for coord, res_name in zip(ca.coord, ca.res_name, strict=True):
        try:
            letter = bseq.ProteinSequence.convert_letter_3to1(res_name)
        except KeyError:
            continue
        coords.append(coord)
        seq_chars.append(letter)
    return np.array(coords), "".join(seq_chars)


def get_paired_ca_positions(
    mobile: struc.AtomArray, ref: struc.AtomArray
) -> tuple[np.ndarray, np.ndarray, int]:
    """BLOSUM62 pairwise alignment → paired Cα coordinate arrays.

    Returns (mobile_ca_coords, ref_ca_coords, n_paired).
    """
    mob_coords, mob_seq_str = get_ca_coords_and_sequence(mobile)
    ref_coords, ref_seq_str = get_ca_coords_and_sequence(ref)

    paired = paired_alignment_trace(
        bseq.ProteinSequence(mob_seq_str), bseq.ProteinSequence(ref_seq_str)
    )
    return mob_coords[paired[:, 0]], ref_coords[paired[:, 1]], len(paired)


def kabsch(mobile: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch rotation + translation aligning mobile onto fixed.

    Returns (R, t) such that R @ mobile.T + t[:,None] ≈ fixed.T.
    """
    mobile_center = mobile.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    centered_mobile = mobile - mobile_center
    centered_fixed = fixed - fixed_center
    covariance = centered_mobile.T @ centered_fixed
    u_mat, _, v_t = np.linalg.svd(covariance)
    det = np.linalg.det(v_t.T @ u_mat.T)
    rotation = v_t.T @ np.diag([1.0, 1.0, det]) @ u_mat.T
    translation = fixed_center - rotation @ mobile_center
    return rotation, translation


def align_to_reference(
    mobile_cif: Path | pdbx.CIFFile,
    ref_protein: struc.AtomArray,
    *,
    out_path: Path | None = None,
    min_common_ca: int = 10,
    pdb_id: str | None = None,
) -> dict | None:
    """Align one mobile CIF onto the reference protein.

    Steps: load mobile protein → BLOSUM62 paired Cα → Kabsch R,t. The transform
    is always computed and returned. Writing the transformed CIF is optional:
    when ``out_path`` is given, R,t is applied to ALL atoms (preserving altlocs)
    and the result is written there; when it is None (e.g. on-the-fly pairwise
    analysis), no file is written and the caller transforms coordinates in
    memory via ``(R @ xyz.T).T + t``.

    Returns a report dict {pdb_id, n_common_ca, rmsd_before, rmsd_after, R, t},
    or None if n_common_ca < min_common_ca (skipped with a warning).
    """
    if isinstance(mobile_cif, Path):
        pdb_id = pdb_id if pdb_id is not None else mobile_cif.stem.removesuffix("_final")
        mobile_cif_file = pdbx.CIFFile.read(mobile_cif)
    else:
        mobile_cif_file = mobile_cif
        pdb_id = pdb_id if pdb_id is not None else "structure"
    mobile_protein, _ = load_protein(mobile_cif_file)

    mob_ca, ref_ca, n_common = get_paired_ca_positions(mobile_protein, ref_protein)

    if n_common < min_common_ca:
        import warnings

        warnings.warn(f"{pdb_id}: only {n_common} common Cα — skipping alignment", stacklevel=2)
        return None

    rmsd_before = float(np.sqrt(np.mean(np.sum((mob_ca - ref_ca) ** 2, axis=1))))
    R, t = kabsch(mob_ca, ref_ca)
    mob_ca_aligned = (R @ mob_ca.T).T + t
    rmsd_after = float(np.sqrt(np.mean(np.sum((mob_ca_aligned - ref_ca) ** 2, axis=1))))

    if out_path is not None:
        write_transformed_cif(mobile_cif_file, R, t, out_path)

    return {
        "pdb_id": pdb_id,
        "n_common_ca": n_common,
        "rmsd_before": round(rmsd_before, 3),
        "rmsd_after": round(rmsd_after, 3),
        "R": R,
        "t": t,
    }
