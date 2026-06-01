from __future__ import annotations

import time
from pathlib import Path

import gemmi
import requests

_RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
_WATER_COMPS = frozenset({"HOH", "WAT", "DOD"})


def _cif_first_float(block: gemmi.cif.Block, tag: str) -> float | str:
    """Return first parseable float for a loop column or scalar tag, else '<missing>'."""
    for v in block.find_loop(tag):
        s = gemmi.cif.as_string(v)
        if s and s not in (".", "?"):
            try:
                return float(s)
            except ValueError:
                pass
    val = block.find_value(tag)
    if val is not None:
        s = gemmi.cif.as_string(val)
        if s and s not in (".", "?"):
            try:
                return float(s)
            except ValueError:
                pass
    return "<missing>"


def _fetch_rcsb_entry(pdb_id: str) -> dict | None:
    """Fetch RCSB entry JSON. Returns None on 404 or persistent failure.

    Ported from porting_reference/get_starting_models.py::fetch_starting_model.
    """
    url = _RCSB_ENTRY_URL.format(pdb_id=pdb_id.lower())
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
        except requests.RequestException:
            if attempt == 2:
                return None
            time.sleep(2.0)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        if attempt == 2:
            return None
        time.sleep(2.0)
    return None


def metadata_row(cif_path: Path) -> dict:
    """Extract one metadata CSV row from a local mmCIF file + RCSB API.

    Library split:
      - gemmi.read_structure → space_group, unit_cell, resolution, num_water
      - gemmi.cif.read       → r_work, r_free (local re-refined CIF has these)
                               ligand_names (from _pdbx_entity_nonpoly.comp_id loop)
      - RCSB Data API        → experiment_condition, starting_model
                               (absent from re-refined local CIFs)
    """
    pdb_id = cif_path.stem.removesuffix("_final")

    # --- gemmi structure: space group, unit cell, resolution ---
    st = gemmi.read_structure(str(cif_path))
    sg = st.find_spacegroup()
    space_group = sg.hm if sg is not None else "<missing>"
    cell = st.cell
    resolution = st.resolution if st.resolution > 0.0 else "<missing>"

    # --- biotite: water oxygen count across all altlocs ---
    # altloc="all" returns a flat AtomArray with one entry per altloc variant,
    # so a water with altlocs A and B contributes two oxygen atoms.
    import biotite.structure.io.pdbx as pdbx

    _cif = pdbx.CIFFile.read(cif_path)
    _atoms = pdbx.get_structure(_cif, model=1, altloc="all")
    num_water = int(((_atoms.res_name == "HOH") & _atoms.hetero & (_atoms.element == "O")).sum())

    # --- gemmi raw CIF: R-work, R-free, ligand names ---
    block = gemmi.cif.read(str(cif_path)).sole_block()

    r_work = _cif_first_float(block, "_refine.ls_R_factor_R_work")
    r_free = _cif_first_float(block, "_refine.ls_R_factor_R_free")

    nonpoly_comp_ids = {
        gemmi.cif.as_string(v)
        for v in block.find_loop("_pdbx_entity_nonpoly.comp_id")
        if v not in (".", "?")
    } - _WATER_COMPS
    ligand_names = "|".join(sorted(nonpoly_comp_ids)) if nonpoly_comp_ids else ""

    # --- RCSB Data API: experiment condition and starting model ---
    entry = _fetch_rcsb_entry(pdb_id)
    if entry is not None:
        refine_blocks = entry.get("refine") or []
        sm_values = list(
            dict.fromkeys(
                v.strip()
                for r in refine_blocks
                for v in [r.get("pdbx_starting_model")]
                if v and v.strip()
            )
        )
        starting_model = " | ".join(v.lower() for v in sm_values) if sm_values else "<missing>"

        grow_blocks = entry.get("exptl_crystal_grow") or []
        grow_details = [g["pdbx_details"] for g in grow_blocks if g.get("pdbx_details")]
        experiment_condition = "; ".join(grow_details) if grow_details else "<missing>"
    else:
        starting_model = "<missing>"
        experiment_condition = "<missing>"

    return {
        "pdb_id": pdb_id,
        "space_group": space_group,
        "cell_a": cell.a,
        "cell_b": cell.b,
        "cell_c": cell.c,
        "cell_alpha": cell.alpha,
        "cell_beta": cell.beta,
        "cell_gamma": cell.gamma,
        "unit_cell_volume": cell.volume,
        "resolution": resolution,
        "r_work": r_work,
        "r_free": r_free,
        "num_water": num_water,
        "ligand_names": ligand_names,
        "experiment_condition": experiment_condition,
        "starting_model": starting_model,
    }
