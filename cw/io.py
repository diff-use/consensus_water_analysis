from __future__ import annotations

import json
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd

# ── utilities ─────────────────────────────────────────────────────────────────


def normalize_ins_code(ins_code) -> str:
    if ins_code is None:
        return ""
    value = str(ins_code).strip()
    return "" if value in {"", "?"} else value


def read_cohort(txt_path: Path) -> list[str]:
    """Read member IDs from a cohort .txt file.

    Blank lines and lines starting with '#' are ignored.
    Returns IDs in file order with whitespace stripped.
    """
    return [
        line.strip()
        for line in txt_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def cif_path_for(member_id: str, all_pdb_redo_dir: Path | str, cif_template: str) -> Path:
    """Resolve the CIF path for a member ID.

    member_id is the cohort-file entry (e.g. '5f14_final').
    pdb_code strips the '_final' suffix for the subdirectory name.
    """
    pdb_code = member_id.removesuffix("_final")
    return Path(all_pdb_redo_dir) / cif_template.format(pdb_id=pdb_code)


# ── CIF reading ───────────────────────────────────────────────────────────────


def load_edia(json_path: Path) -> dict[tuple[str, int, str], float] | None:
    """Load EDIAm scores from a per-structure JSON file.

    Deprecated for clustering use — prefer load_structure_waters, which adds
    altloc_id by pairing JSON entries with the CIF atom_site ordering.  This
    function silently overwrites duplicate keys for altloc waters.

    Keyed on (chain_id, res_id, ins_code) → EDIAm float.
    Returns None if the file is missing or unreadable.

    Ported from Vratin's cluster.py::load_edia (uses the stricter
    (chain, seqNum, insCode) triple rather than seqNum-only keying).
    """
    if not json_path.exists():
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    lookup: dict[tuple[str, int, str], float] = {}
    for entry in payload:
        if entry.get("compID") not in {"HOH", "WAT"}:
            continue
        pdb_info = entry.get("pdb", {})
        key = (
            str(pdb_info.get("strandID", "")),
            int(pdb_info.get("seqNum", 0)),
            normalize_ins_code(pdb_info.get("insCode", "")),
        )
        lookup[key] = float(entry.get("EDIAm", 0.0))
    return lookup


def load_protein(
    src: Path | pdbx.CIFFile, altloc: str = "occupancy"
) -> tuple[struc.AtomArray, int]:
    """Load protein-only atoms (highest-occupancy altloc) and return (array, res_id_offset).

    Ported from align_pdbs.py::get_protein_clean_array, adapted for CIF.
    The offset renumbers the first residue to 1 so BLOSUM62 alignment is
    robust across structures with different deposited numbering schemes.

    src may be a Path (CIF is read here) or an already-loaded CIFFile
    (caller's read is reused, avoiding a second disk hit).
    """
    cif_file = pdbx.CIFFile.read(src) if isinstance(src, Path) else src
    atoms = pdbx.get_structure(cif_file, model=1, altloc=altloc)

    protein_mask = struc.filter_amino_acids(atoms)
    protein = atoms[protein_mask]

    offset = int(protein.res_id[0]) - 1
    if offset > 0:
        protein.res_id = protein.res_id - offset

    return protein, offset


def load_structure_waters(cif_path: Path, json_path: Path | None = None) -> pd.DataFrame:
    """Load water O records from one CIF, attaching EDIA in a single CIF read.

    Reads the CIF once with altloc='all'. If json_path is supplied, parses the
    EDIA JSON and pairs each altloc variant with its EDIAm value by position
    within each (chain, res_id, ins_code) group — the same ordering assumption
    as load_edia_with_altlocs, but without a second CIF read.

    Columns: pdb_id, chain_id, res_id, altloc, x, y, z, b_factor, occupancy, edia
    """
    pdb_id = cif_path.stem.removesuffix("_final")
    cf = pdbx.CIFFile.read(cif_path)
    atoms = pdbx.get_structure(cf, model=1, altloc="all", extra_fields=["b_factor", "occupancy"])
    water_mask = (atoms.res_name == "HOH") & atoms.hetero & (atoms.element == "O")
    waters = atoms[water_mask]

    edia_lists: dict[tuple[str, int, str], list[float]] = {}
    if json_path is not None and json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                payload = json.load(f)
            for entry in payload:
                if entry.get("compID") not in {"HOH", "WAT"}:
                    continue
                pdb_info = entry.get("pdb", {})
                key = (
                    str(pdb_info.get("strandID", "")),
                    int(pdb_info.get("seqNum", 0)),
                    normalize_ins_code(pdb_info.get("insCode", "")),
                )
                edia_lists.setdefault(key, []).append(float(entry.get("EDIAm", 0.0)))
        except Exception:
            pass

    df = pd.DataFrame(
        {
            "pdb_id": pdb_id,
            "chain_id": waters.chain_id.astype(str),
            "res_id": waters.res_id.astype(int),
            "ins_code": [normalize_ins_code(ic) for ic in waters.ins_code],
            "altloc": waters.altloc_id.astype(str),
            "x": waters.coord[:, 0].astype(float),
            "y": waters.coord[:, 1].astype(float),
            "z": waters.coord[:, 2].astype(float),
            "b_factor": waters.b_factor.astype(float),
            "occupancy": waters.occupancy.astype(float),
        }
    )

    if edia_lists:
        df["_pos"] = df.groupby(["chain_id", "res_id", "ins_code"]).cumcount()
        edia_df = pd.DataFrame(
            [
                {
                    "chain_id": chain_id,
                    "res_id": res_id,
                    "ins_code": ins_code,
                    "_pos": pos,
                    "edia": val,
                }
                for (chain_id, res_id, ins_code), vals in edia_lists.items()
                for pos, val in enumerate(vals)
            ]
        )
        df = df.merge(edia_df, on=["chain_id", "res_id", "ins_code", "_pos"], how="left")
        df = df.drop(columns=["_pos"])
    else:
        df["edia"] = float("nan")

    return df.drop(columns=["ins_code"])


def collect_aligned_waters(
    cif_json_pairs: list[tuple[Path, Path | None]],
    *,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Concatenate water records from aligned CIFs into one DataFrame.

    Each element of cif_json_pairs is (aligned_cif, edia_json_path), where
    edia_json_path may be None if EDIA is unavailable for that structure.
    If out_path is given the result is also written to CSV before returning.
    """
    frames = [load_structure_waters(cif, json) for cif, json in cif_json_pairs]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


# ── CIF writing ───────────────────────────────────────────────────────────────


def write_filtered_cif(
    src: Path | pdbx.CIFFile,
    keep_mask: np.ndarray,
    filtered_atoms: struc.AtomArray,
    out_path: Path,
) -> None:
    """Write a filtered CIF preserving all atom_site columns (including label_alt_id).

    Drops atom_site rows where keep_mask is False, then writes Cartn_x/y/z
    from filtered_atoms.coord so relocated waters are captured.
    pdbx.set_structure is not used because it regenerates atom_site from the
    AtomArray, overwriting label_alt_id with biotite's internal altloc_id values,
    which do not round-trip the original CIF '.'/'A'/'B' notation exactly.

    src may be a Path (CIF is read here) or an already-loaded CIFFile.
    """
    cif_file = pdbx.CIFFile.read(src) if isinstance(src, Path) else src
    atom_site = cif_file.block["atom_site"]

    filtered_data = {key: atom_site[key].as_array(str)[keep_mask] for key in atom_site}
    filtered_data["Cartn_x"] = filtered_atoms.coord[:, 0]
    filtered_data["Cartn_y"] = filtered_atoms.coord[:, 1]
    filtered_data["Cartn_z"] = filtered_atoms.coord[:, 2]
    # A new CIFCategory is required here rather than mutating the existing one's columns
    # in-place: CIFCategory caches _row_count the first time row_count is accessed
    # (e.g. via get_structure), and __setitem__ never resets it, so serialize() would
    # raise SerializationError comparing the stale original length against the filtered columns.
    cif_file.block["atom_site"] = pdbx.CIFCategory(filtered_data, "atom_site")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cif_file.write(out_path)


def write_transformed_cif(
    src: Path | pdbx.CIFFile,
    R: np.ndarray,
    t: np.ndarray,
    out_path: Path,
) -> None:
    """Apply a rigid transform to all atoms in a CIF and write to out_path.

    Modifies Cartn_x/y/z directly so every other column — including
    label_alt_id — is preserved unchanged.

    src may be a Path (CIF is read here) or an already-loaded CIFFile.
    """
    cif_file = pdbx.CIFFile.read(src) if isinstance(src, Path) else src
    atom_site = cif_file.block["atom_site"]

    x = atom_site["Cartn_x"].as_array(float)
    y = atom_site["Cartn_y"].as_array(float)
    z = atom_site["Cartn_z"].as_array(float)
    coords = np.column_stack([x, y, z])
    coords = (R @ coords.T).T + t

    atom_site["Cartn_x"] = coords[:, 0]
    atom_site["Cartn_y"] = coords[:, 1]
    atom_site["Cartn_z"] = coords[:, 2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cif_file.write(out_path)
