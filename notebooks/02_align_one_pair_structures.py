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
    # 02 — align structures (pair)

    Tests `cw/align.py` on one reference + one mobile structure.
    Reports RMSD before/after and verifies the output CIF retains all atoms.
    """)
    return


@app.cell
def _(mo):
    mobile_pdb_id = "5fek"
    try:
        import config

        _pdb_store = config.ALL_PDB_REDO_DIR
        _cif_tmpl = config.CIF_TEMPLATE
        _ref_default = f"{_pdb_store}/{_cif_tmpl.format(pdb_id=config.REF_PDB_ID)}"
        _mobile_default = f"{_pdb_store}/{_cif_tmpl.format(pdb_id=mobile_pdb_id)}"
    except ImportError:
        _ref_default = ""

    ref_input = mo.ui.text(
        value=_ref_default,
        placeholder="/path/to/ref/<id>_final.cif",
        label="Reference CIF",
        full_width=True,
    )
    mob_input = mo.ui.text(
        value=_mobile_default,
        placeholder="/path/to/mobile/<id>_final.cif",
        label="Mobile CIF",
        full_width=True,
    )
    out_input = mo.ui.text(
        value="data/test_aligned.cif",
        label="Output path",
        full_width=True,
    )
    mo.vstack([ref_input, mob_input, out_input])
    return mob_input, out_input, ref_input


@app.cell
def _(mo, mob_input, out_input, ref_input):
    from pathlib import Path

    ref_path = Path(ref_input.value.strip()) if ref_input.value.strip() else None
    mob_path = Path(mob_input.value.strip()) if mob_input.value.strip() else None
    out_path = Path(out_input.value.strip() or "/tmp/test_aligned.cif")

    if not (ref_path and mob_path):
        mo.stop(True, mo.md("**Set both CIF paths above to continue.**"))
    for p, label in [(ref_path, "Reference"), (mob_path, "Mobile")]:
        if not p.exists():
            mo.stop(True, mo.md(f"**{label} not found:** `{p}`"))
    return mob_path, out_path, ref_path


@app.cell
def _(mo, mob_path, out_path, ref_path):
    from cw.align import align_to_reference, load_protein

    ref_protein, _ = load_protein(ref_path)
    report = align_to_reference(mob_path, ref_protein, out_path=out_path)

    if report is None:
        mo.stop(True, mo.md("**Alignment skipped** — fewer than 10 common Cα atoms."))

    mo.md(
        "## Alignment report\n\n"
        "| field | value |\n|---|---|\n"
        + "\n".join(f"| `{k}` | `{v}` |" for k, v in report.items())
    )
    return


@app.cell
def _(mo, mob_path, out_path):
    import biotite.structure.io.pdbx as pdbx

    def _atom_counts(cif_path):
        f = pdbx.CIFFile.read(str(cif_path))
        atoms = pdbx.get_structure(f, model=1, altloc="all")
        n_all = len(atoms)
        n_water = int(((atoms.res_name == "HOH") & atoms.hetero & (atoms.element == "O")).sum())
        return n_all, n_water

    n_before, w_before = _atom_counts(mob_path)
    n_after, w_after = _atom_counts(out_path)

    status = (
        "✓ match"
        if (n_before == n_after and w_before == w_after)
        else "✗ mismatch — check altloc write"
    )
    mo.md(
        "## Atom counts (mobile vs aligned output)\n\n"
        f"| | before | after |\n|---|---|---|\n"
        f"| total atoms (altloc=all) | {n_before} | {n_after} |\n"
        f"| water oxygens | {w_before} | {w_after} |\n\n"
        f"**{status}**"
    )
    return


if __name__ == "__main__":
    app.run()
