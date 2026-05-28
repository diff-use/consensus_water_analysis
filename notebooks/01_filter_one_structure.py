import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # 01 — filter by distance

    Exercises `cw/filter.py::filter_by_distance` on a single CIF for sanity checks. We use 5F14 as an example because it has quite a few waters that relocated as well as filtered.

    **Goals**
    - Waters removed vs kept count looks plausible.
    - Distribution of water to protein min distance is improved after relocating and filtering
    """)
    return


@app.cell
def _(mo):
    test_PDB_ID = "5f14"
    try:
        import config

        _cif_default = f"{config.ALL_PDB_REDO_DIR}/{config.CIF_TEMPLATE.format(pdb_id=test_PDB_ID)}"
        _cutoff_default = str(config.WATER_PROT_DIST_CUTOFF)
    except ImportError:
        _cif_default = ""
        _cutoff_default = "4.0"

    cif_input = mo.ui.text(
        value=_cif_default,
        placeholder="/path/to/<id>/<id>_final.cif",
        label="CIF path",
        full_width=True,
    )
    cutoff_input = mo.ui.number(
        value=float(_cutoff_default),
        start=1.0,
        stop=10.0,
        step=0.5,
        label="Distance cutoff (Å)",
    )
    mo.vstack([cif_input, cutoff_input])
    return cif_input, cutoff_input


@app.cell
def _(cutoff_input):
    from pathlib import Path

    cutoff = float(cutoff_input.value)
    return Path, cutoff


@app.cell
def _(Path, cif_input, mo):
    cif_path = Path(cif_input.value.strip()) if cif_input.value.strip() else None

    if cif_path is None:
        mo.stop(True, mo.md("**Set a CIF path above to continue.**"))
    if not cif_path.exists():
        mo.stop(True, mo.md(f"CIF not found: `{cif_path}`"))
    return (cif_path,)


@app.cell
def _(cif_path):
    import biotite.structure.io.pdbx as pdbx
    import gemmi

    cif_file = pdbx.CIFFile.read(str(cif_path))
    atoms = pdbx.get_structure(
        cif_file,
        model=1,
        altloc="all",
        extra_fields=["b_factor", "occupancy"],
    )

    st = gemmi.read_structure(str(cif_path))
    cell = st.cell
    sg = st.find_spacegroup() or gemmi.SpaceGroup("P 1")

    n_water_before = int(((atoms.res_name == "HOH") & atoms.hetero & (atoms.element == "O")).sum())
    return atoms, cell, n_water_before, sg


@app.cell
def _(atoms, cell, cutoff, mo, n_water_before, sg):
    from cw.filter import filter_by_distance

    filtered, n_water_original, n_removed, n_moved, final_mask = filter_by_distance(
        atoms.copy(), cell, sg, cutoff
    )

    n_water_after = int(
        ((filtered.res_name == "HOH") & filtered.hetero & (filtered.element == "O")).sum()
    )

    assert n_water_original == n_water_before

    mo.md(
        f"## Filter result  (cutoff = {cutoff} Å)\n\n"
        f"- Waters before: **{n_water_before}**\n"
        f"- Waters moved: **{n_moved}**\n"
        f"- Waters removed: **{n_removed}**\n"
        f"- Waters kept: **{n_water_after}**\n"
        f"- Space group: `{cell}` / `{sg.hm if sg else 'P 1'}`"
    )
    return (n_moved,)


@app.cell
def _(atoms, cell, sg):
    import numpy as np
    from scipy.spatial.distance import cdist

    from cw.filter import best_sym_positions

    _water_O_mask = (atoms.res_name == "HOH") & atoms.hetero & (atoms.element == "O")
    _protein_heavy_mask = (~atoms.hetero) & (atoms.element != "H")

    _water_coords = atoms.coord[_water_O_mask]
    _protein_coords = atoms.coord[_protein_heavy_mask]

    naive_dists = cdist(_water_coords, _protein_coords).min(axis=1)
    _, dists = best_sym_positions(_water_coords, _protein_coords, cell, sg)
    return dists, naive_dists, np


@app.cell
def _(cutoff, dists, mo, n_moved, naive_dists, np):
    import matplotlib.pyplot as plt

    bins = np.linspace(0, max(naive_dists.max(), dists.max()), 60)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(naive_dists, bins=bins, alpha=0.5, color="gray", label="naive (no symop)")
    ax.hist(dists, bins=bins, alpha=0.7, color="steelblue", label="symmetry-aware")
    ax.axvline(cutoff, color="tomato", linewidth=1.5, label=f"cutoff = {cutoff} Å")
    ax.set_xlabel("Distance to nearest protein atom (Å)")
    ax.set_ylabel("Waters")
    ax.set_title("Water–protein distances: naive vs symmetry-aware")
    ax.legend()
    fig.tight_layout()
    mo.md(
        f"## Distance distribution\n\n"
        f"Waters whose position was moved to a symmetry image: "
        f"**{n_moved}** / {len(dists)}\n\n"
        f"|           | min | median | max |\n"
        f"|-----------|-----|--------|-----|\n"
        f"| naive     | {naive_dists.min():.2f} | {np.median(naive_dists):.2f} | {naive_dists.max():.2f} |\n"
        f"| sym-aware | {dists.min():.2f} | {np.median(dists):.2f} | {dists.max():.2f} |"
    )
    return (fig,)


@app.cell
def _(fig):
    fig
    return


if __name__ == "__main__":
    app.run()
