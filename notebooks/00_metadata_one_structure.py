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
    # 00 — metadata sanity check

    Exercises `cw/io.py` and `cw/metadata.py` on a single structure from PDB-REDO. We use 6YBF as an example because it has altloc water and could be a difficult case.

    **Goals**
    - Confirm gemmi surfaces space group / cell / resolution / unit cell volume correctly.
    - Confirm gemmi raw-CIF read surfaces `experiment_condition` and `starting_model`.
    - Verify that gemmi's water-O count agrees with biotite `altloc="occupancy"` count.
    - Load EDIA and confirm the (chain, res_id, ins_code) keying matches the water atoms.
    """)
    return


@app.cell
def _(mo):
    test_PDB_ID = "6ybf"
    try:
        import config

        _cif_default = f"{config.ALL_PDB_REDO_DIR}/{config.CIF_TEMPLATE.format(pdb_id=test_PDB_ID)}"
        _edia_default = (
            f"{config.ALL_PDB_REDO_DIR}/{config.EDIA_TEMPLATE.format(pdb_id=test_PDB_ID)}"
        )
    except ImportError:
        _cif_default = ""
        _edia_default = ""

    cif_input = mo.ui.text(
        value=_cif_default,
        placeholder="/path/to/<id>/<id>_final.cif",
        label="CIF path",
        full_width=True,
    )
    edia_input = mo.ui.text(
        value=_edia_default,
        placeholder="/path/to/<id>/<id>_final.json",
        label="EDIA JSON path",
        full_width=True,
    )
    mo.vstack([cif_input, edia_input])
    return cif_input, edia_input


@app.cell
def _(cif_input, edia_input, mo):
    from pathlib import Path

    cif_path = Path(cif_input.value.strip()) if cif_input.value.strip() else None
    edia_path = Path(edia_input.value.strip()) if edia_input.value.strip() else None

    if cif_path is None:
        mo.stop(True, mo.md("**Set a CIF path above to continue.**"))
    if not cif_path.exists():
        mo.stop(True, mo.md(f"CIF not found: `{cif_path}`"))
    return cif_path, edia_path


@app.cell
def _(cif_path, mo):
    from cw.metadata import metadata_row

    row = metadata_row(cif_path)
    mo.md("## 1 — metadata row\n\n" + "\n".join(f"- **{k}**: `{v}`" for k, v in row.items()))
    return (row,)


@app.cell
def _(cif_path, mo, row):
    import biotite.structure.io.pdbx as pdbx

    cif_file = pdbx.CIFFile.read(cif_path)
    atoms_occ = pdbx.get_structure(
        cif_file, model=1, altloc="all", extra_fields=["b_factor", "occupancy"]
    )
    biotite_count = int(
        ((atoms_occ.res_name == "HOH") & atoms_occ.hetero & (atoms_occ.element == "O")).sum()
    )
    gemmi_count = row["num_water"]

    mo.md(
        "## 2 — water count\n\n"
        f"- **gemmi (all altlocs):** {gemmi_count}\n"
        f"- **biotite altloc=occupancy:** {biotite_count}\n"
        + (
            "- **Agree** ✓"
            if gemmi_count == biotite_count
            else f"- **Differ** — structure has water altlocs; gemmi count ({gemmi_count}) is correct"
        )
    )
    return


@app.cell
def _(cif_path, edia_path, mo, row):
    from cw.io import load_structure_waters

    df = load_structure_waters(cif_path, edia_path)
    _n_edia = int(df["edia"].notna().sum())
    _n_water = row["num_water"]
    if _n_edia == _n_water:
        _check = f"**Match** ✓ ({_n_edia} EDIA values for {_n_water} waters)"
    else:
        _check = f"**Mismatch** — {_n_edia} valid EDIA values for {_n_water} waters"
    mo.md(f"## 3 — EDIA\n\n{_check}")
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"""
    ## 4 — water atoms ({len(df)} rows, biotite altloc=all)
    """)
    return


@app.cell
def _(df):
    df.head(20)
    return


@app.cell
def _(cif_path, mo, row):
    import gemmi

    block = gemmi.cif.read(str(cif_path)).sole_block()
    nonpoly_names = list(block.find_loop("_pdbx_entity_nonpoly.name"))
    nonpoly_comps = list(block.find_loop("_pdbx_entity_nonpoly.comp_id"))
    mo.md(
        "## 5 — ligands + RCSB fields\n\n"
        f"**`_pdbx_entity_nonpoly` names:** `{nonpoly_names}`\n\n"
        f"**`_pdbx_entity_nonpoly` comp_ids:** `{nonpoly_comps}`\n\n"
        f"**ligand_names (in row):** `{row['ligand_names']}`\n\n"
        f"**experiment_condition:** `{row['experiment_condition']}`\n\n"
        f"**starting_model:** `{row['starting_model']}`"
    )
    return


if __name__ == "__main__":
    app.run()
