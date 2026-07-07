from __future__ import annotations

import biotite.structure as struc
import gemmi
import numpy as np

from cw.io import edia_scores_in_order, normalize_ins_code, water_oxygen_mask


def best_sym_positions(
    water_pos_orth: np.ndarray,
    protein_pos_orth: np.ndarray,
    unitcell: gemmi.UnitCell,
    spacegroup: gemmi.SpaceGroup,
) -> tuple[np.ndarray, np.ndarray]:
    """For each water O, find the protein atom image (symop + lattice shift) that
    is nearest, then apply the inverse transform to place the water in the canonical
    protein ASU.  This matches the phenix.sort_hetatms convention.

    Ported from porting_reference/filter_waters_by_distance.py::best_sym_positions.

    Parameters
    ----------
    water_pos_orth   : (N, 3) orthogonal coordinates in Å
    protein_pos_orth : (M, 3) orthogonal coordinates in Å

    Returns
    -------
    best_pos   : (N, 3) canonical ASU coordinates for each water O
    best_min_d : (N,)   distance to nearest protein atom at best_pos
    """
    orth2frac = np.array(unitcell.frac.mat.tolist())
    frac2orth = np.array(unitcell.orth.mat.tolist())
    ops_list = list(spacegroup.operations())

    water_frac = water_pos_orth @ orth2frac.T
    best_pos = water_pos_orth.copy()
    best_min_d = np.full(len(water_pos_orth), np.inf)
    best_op_indices = np.zeros(len(water_pos_orth), dtype=int)
    best_shifts = np.zeros((len(water_pos_orth), 3))

    for i_w in range(len(water_pos_orth)):
        w_gemmi = gemmi.Position(*water_pos_orth[i_w].tolist())
        for p_orth in protein_pos_orth:
            p_gemmi = gemmi.Position(*p_orth.tolist())
            ni = unitcell.find_nearest_image(w_gemmi, p_gemmi, gemmi.Asu.Any)
            if ni.dist() < best_min_d[i_w]:
                best_min_d[i_w] = ni.dist()
                best_op_indices[i_w] = ni.sym_idx
                best_shifts[i_w] = np.array(ni.pbc_shift)

        op = ops_list[best_op_indices[i_w]]
        R = np.array(op.rot) / op.DEN
        T = np.array(op.tran) / op.DEN
        R_inv = np.linalg.inv(R)
        best_pos[i_w] = (R_inv @ (water_frac[i_w] - T - best_shifts[i_w])) @ frac2orth.T

    return best_pos, best_min_d


def keep_by_distance(
    atoms: struc.AtomArray,
    water_O_indices: np.ndarray,
    cell: gemmi.UnitCell,
    spacegroup: gemmi.SpaceGroup,
    cutoff: float,
) -> tuple[np.ndarray, int]:
    """Distance filter (relocating). Move each water O to the canonical ASU position
    nearest any protein heavy atom under any symmetry image — matching
    phenix.sort_hetatms — writing the new coords back into `atoms`, then keep the
    waters within `cutoff` Å.

    Returns
    -------
    keep    : boolean mask over `water_O_indices` (True = kept)
    n_moved : water O atoms actually repositioned (|shift| > 1e-4 Å)
    """
    protein_heavy_mask = (~atoms.hetero) & (atoms.element != "H")
    orig_pos = atoms.coord[water_O_indices].copy()
    best_pos, best_min_d = best_sym_positions(
        orig_pos, atoms.coord[protein_heavy_mask], cell, spacegroup
    )
    atoms.coord[water_O_indices] = best_pos
    n_moved = int(np.sum(np.linalg.norm(best_pos - orig_pos, axis=1) > 1e-4))
    return best_min_d <= cutoff, n_moved


def keep_by_edia(
    water_atoms: struc.AtomArray,
    edia_lists: dict[tuple[str, int, str], list[float]],
    cutoff: float,
) -> np.ndarray:
    """EDIA filter (coordinate-independent). Keep waters whose EDIAm >= `cutoff`; a
    missing score is dropped (NaN >= cutoff is False), so an empty `edia_lists` drops
    every water — the "structure has no EDIA JSON, drop it" policy.

    Altloc pairing follows cw.io.edia_scores_in_order (the EDIA JSON has no altloc
    field; the Nth water-O of a residue maps to the Nth score).

    Returns a boolean mask over `water_atoms`.
    """
    keys = [
        (
            str(water_atoms.chain_id[i]),
            int(water_atoms.res_id[i]),
            normalize_ins_code(water_atoms.ins_code[i]),
        )
        for i in range(len(water_atoms))
    ]
    return edia_scores_in_order(keys, edia_lists) >= cutoff


def filter_waters(
    atoms: struc.AtomArray,
    cell: gemmi.UnitCell,
    spacegroup: gemmi.SpaceGroup,
    distance_cutoff: float,
    edia_lists: dict[tuple[str, int, str], list[float]] | None = None,
    edia_cutoff: float | None = None,
) -> tuple[struc.AtomArray, dict[str, int], np.ndarray]:
    """Filter waters by composing per-water keep-masks, returning the filtered
    structure, a stats dict, and the CIF-row keep-mask.

    Each concern is a single-purpose keep_by_* filter returning a boolean mask over the
    water-O atoms; the masks are AND-ed in order so a removal is attributed to the first
    filter that dropped it. Distance runs first (it also relocates each water O to its
    canonical ASU position); EDIA runs on the distance survivors when both `edia_lists`
    and `edia_cutoff` are given. To add a future concern (e.g. a B-factor cutoff), write
    another keep_by_* filter and AND it in here — filter_waters is the only place that
    knows how the concerns combine.

    Parameters
    ----------
    atoms          : biotite AtomArray (loaded with altloc="all" to preserve every
                     water altloc; duplicate protein atoms are fine — min-distance is
                     taken so extra altloc variants only make the cutoff stricter)
    cell           : gemmi.UnitCell from the same structure
    spacegroup     : gemmi.SpaceGroup from the same structure
    distance_cutoff: water–protein distance cutoff in Å (typically 4.0)
    edia_lists     : per-residue EDIAm score lists from cw.io.load_edia_all_altlocs,
                     or None to skip EDIA filtering
    edia_cutoff    : minimum EDIAm to keep a water, or None to skip EDIA filtering

    Returns
    -------
    filtered_atoms : AtomArray of the whole structure (protein plus kept waters), with
                     removed waters dropped and survivors at their canonical ASU coords
    stats          : dict of counts — n_water, n_moved, n_removed_distance,
                     n_removed_edia (a removal counts once, against its dropping filter)
    keep_mask      : boolean mask (length = len(atoms)) for filtering CIF rows
    """
    is_water = (atoms.res_name == "HOH") & atoms.hetero
    water_O_mask = water_oxygen_mask(atoms)
    water_O_indices = np.where(water_O_mask)[0]
    n_water = len(water_O_indices)

    stats = {"n_water": n_water, "n_moved": 0, "n_removed_distance": 0, "n_removed_edia": 0}
    if n_water == 0:
        return atoms, stats, (~is_water).copy()

    keep, stats["n_moved"] = keep_by_distance(
        atoms, water_O_indices, cell, spacegroup, distance_cutoff
    )
    stats["n_removed_distance"] = int(np.sum(~keep))

    if edia_lists is not None and edia_cutoff is not None:
        edia_keep = keep_by_edia(atoms[water_O_mask], edia_lists, edia_cutoff)
        stats["n_removed_edia"] = int(np.sum(keep & ~edia_keep))
        keep &= edia_keep

    final_mask = (~is_water).copy()
    final_mask[water_O_indices[keep]] = True
    return atoms[final_mask], stats, final_mask
