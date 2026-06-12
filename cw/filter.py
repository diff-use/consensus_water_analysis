from __future__ import annotations

import biotite.structure as struc
import gemmi
import numpy as np

from cw.io import water_oxygen_mask


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


def filter_by_distance(
    atoms: struc.AtomArray,
    cell: gemmi.UnitCell,
    spacegroup: gemmi.SpaceGroup,
    cutoff: float,
) -> tuple[struc.AtomArray, int, int, int, np.ndarray]:
    """Keep only waters within `cutoff` Å of any protein heavy atom under any
    symmetry image.  Water O coordinates are updated to the canonical ASU position.

    Parameters
    ----------
    atoms     : biotite AtomArray (loaded with altloc="all" to preserve every
                water altloc; duplicate protein atoms are fine — min-distance
                is taken so extra altloc variants only make the cutoff stricter)
    cell      : gemmi.UnitCell from the same structure
    spacegroup: gemmi.SpaceGroup from the same structure
    cutoff    : distance cutoff in Å (typically 4.0)

    Returns
    -------
    filtered_atoms : AtomArray of whole structure including protein, with distant
                     waters removed.
    n_water        : total water O atoms before filtering
    n_removed      : number of water O atoms removed
    n_moved        : number of water O atoms repositioned by symmetry before
                     filtering (|shift| > 1e-4 Å)
    keep_mask      : boolean mask (length = len(atoms)) for filtering CIF rows
    """
    is_water = (atoms.res_name == "HOH") & atoms.hetero
    water_O_mask = water_oxygen_mask(atoms)
    protein_heavy_mask = (~atoms.hetero) & (atoms.element != "H")
    non_water_mask = ~is_water

    water_O_indices = np.where(water_O_mask)[0]
    n_water = len(water_O_indices)

    if n_water == 0:
        return atoms, 0, 0, 0, non_water_mask.copy()

    orig_pos = atoms.coord[water_O_mask].copy()
    best_pos, best_min_d = best_sym_positions(
        atoms.coord[water_O_mask],
        atoms.coord[protein_heavy_mask],
        cell,
        spacegroup,
    )

    n_moved = int(np.sum(np.linalg.norm(best_pos - orig_pos, axis=1) > 1e-4))

    # Move each water O to the canonical ASU position nearest to the protein.
    atoms.coord[water_O_indices] = best_pos

    keep_mask = best_min_d <= cutoff
    n_removed = int(np.sum(~keep_mask))

    kept_water_O_indices = water_O_indices[keep_mask]
    final_mask = non_water_mask.copy()
    final_mask[kept_water_O_indices] = True

    return atoms[final_mask], n_water, n_removed, n_moved, final_mask
