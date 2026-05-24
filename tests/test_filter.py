import biotite.structure.io.pdbx as pdbx
import gemmi
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from cw.filter import best_sym_positions, filter_by_distance


def _make_cell(a=100.0):
    return gemmi.UnitCell(a, a, a, 90, 90, 90)


def _p1():
    return gemmi.SpaceGroup("P 1")


# ── best_sym_positions unit tests ──────────────────────────────────────────────


def test_p1_positions_unchanged():
    """P 1 has only the identity op — no water should be moved."""
    cell = _make_cell()
    water = np.array([[5.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    protein = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    best_pos, _ = best_sym_positions(water, protein, cell, _p1())

    np.testing.assert_allclose(best_pos, water, atol=1e-6)


def test_periodic_image_closer():
    """Water near the cell boundary should be pulled to the image closer to the
    protein when the cell is small enough that a lattice shift matters."""
    # 10 Å cubic P1 cell.  Protein at origin; water at (9, 0, 0).
    # Naive distance = 9 Å.  Periodic image at (9-10, 0, 0) = (-1, 0, 0) → 1 Å.
    cell = gemmi.UnitCell(10, 10, 10, 90, 90, 90)
    water = np.array([[9.0, 0.0, 0.0]])
    protein = np.array([[0.0, 0.0, 0.0]])

    best_pos, best_dists = best_sym_positions(water, protein, cell, _p1())

    assert best_dists[0] == pytest.approx(1.0, abs=1e-5)


# ── filter_by_distance integration test (5f14) ────────────────────────────────

CUTOFF = 4.0


@pytest.fixture(scope="module")
def filtered_5f14(cif_path_5f14):
    st = gemmi.read_structure(str(cif_path_5f14))
    cell = st.cell
    sg = st.find_spacegroup() or gemmi.SpaceGroup("P 1")
    atoms = pdbx.get_structure(
        pdbx.CIFFile.read(str(cif_path_5f14)),
        model=1,
        altloc="all",
        extra_fields=["b_factor", "occupancy"],
    )
    filtered, n_water, n_removed, n_moved, keep_mask = filter_by_distance(atoms, cell, sg, CUTOFF)
    return atoms, filtered, n_removed


def test_filter_5f14_removes_some_waters(filtered_5f14):
    atoms, filtered, n_removed = filtered_5f14
    n_before = int(((atoms.res_name == "HOH") & atoms.hetero & (atoms.element == "O")).sum())
    assert n_removed > 0
    assert n_removed < n_before


def test_filter_5f14_remaining_waters_within_cutoff(filtered_5f14):
    """All kept water O atoms (at their updated canonical-ASU coordinates)
    must be within the cutoff of some protein heavy atom."""
    _, filtered, _ = filtered_5f14
    water_mask = (filtered.res_name == "HOH") & filtered.hetero & (filtered.element == "O")
    prot_mask = ~filtered.hetero & (filtered.element != "H")
    min_dists = cdist(filtered.coord[water_mask], filtered.coord[prot_mask]).min(axis=1)
    assert (min_dists <= CUTOFF + 1e-4).all()


def test_filter_5f14_protein_atoms_preserved(filtered_5f14):
    atoms, filtered, _ = filtered_5f14
    n_prot_before = int((~atoms.hetero).sum())
    n_prot_after = int((~filtered.hetero).sum())
    assert n_prot_after == n_prot_before
