from __future__ import annotations

import json
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd

# Altloc values PSEUDO treats as "no altloc" when generating MUSE scores.
MUSE_VALID_ALTLOCS: frozenset[str] = frozenset({"\x00", " ", "A", ""})

# ── utilities ─────────────────────────────────────────────────────────────────


def normalize_ins_code(ins_code) -> str:
    if ins_code is None:
        return ""
    value = str(ins_code).strip()
    return "" if value in {"", "?"} else value


def read_cohort(txt_path: Path) -> list[str]:
    """Read PDB IDs from a cohort .txt file.

    Blank lines and lines starting with '#' are ignored.
    Returns lowercase 4-char PDB IDs in file order.
    Each line is split on whitespace; only the first token is used (inline
    comments and extra fields are ignored). Any suffix after the first '_' is
    stripped (e.g. '5F14_final' → '5f14').
    """
    ids = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        ids.append(token.split("_")[0].lower())
    return ids


def cif_path_for(pdb_id: str, all_pdb_redo_dir: Path | str, cif_template: str) -> Path:
    """Resolve the CIF path for a PDB ID."""
    return Path(all_pdb_redo_dir) / cif_template.format(pdb_id=pdb_id)


def water_oxygen_mask(atoms: struc.AtomArray) -> np.ndarray:
    """Boolean mask selecting water oxygen atoms (HOH, hetero, element O)."""
    return (atoms.res_name == "HOH") & atoms.hetero & (atoms.element == "O")


def count_water_oxygens(atoms: struc.AtomArray) -> int:
    """Count water oxygens (each altloc counted separately)."""
    return int(water_oxygen_mask(atoms).sum())


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


def load_edia_all_altlocs(json_path: Path) -> dict[tuple[str, int, str], list[float]]:
    """Like load_edia, but accumulates a list per residue instead of overwriting on 
    duplicate keys. This is used to preserve waters with altlocs, but altloc id is not
    explicitly stored in the JSON file.

    Keyed on (chain_id, res_id, ins_code) → [EDIAm, ...] in JSON file order.
    Returns an empty dict on missing file or parse error.
    """
    if not json_path.exists():
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    scores: dict[tuple[str, int, str], list[float]] = {}
    for entry in payload:
        if entry.get("compID") not in {"HOH", "WAT"}:
            continue
        pdb_info = entry.get("pdb", {})
        key = (
            str(pdb_info.get("strandID", "")),
            int(pdb_info.get("seqNum", 0)),
            normalize_ins_code(pdb_info.get("insCode", "")),
        )
        scores.setdefault(key, []).append(float(entry.get("EDIAm", 0.0)))
    return scores


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


def _attach_edia(df: pd.DataFrame, edia_dict_of_lists: dict[tuple[str, int, str], list[float]]) -> pd.DataFrame:
    """Merge EDIAm scores into df using positional ordering within each residue group.

    The EDIA JSON has no altloc field — it emits one entry per altloc variant in
    file order. biotite with altloc='all' preserves the same ordering in atom_site.
    cumcount() assigns a 0-based position to each CIF row within a
    (chain, res_id, ins_code) group; _pos in the EDIA frame mirrors that index,
    so the merge pairs CIF row 0 → JSON entry 0, row 1 → entry 1, etc.

    df must still contain ins_code. Adds an 'edia' column.
    """
    if not edia_dict_of_lists:
        df["edia"] = float("nan")
        return df

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
            for (chain_id, res_id, ins_code), vals in edia_dict_of_lists.items()
            for pos, val in enumerate(vals)
        ]
    )
    df = df.merge(edia_df, on=["chain_id", "res_id", "ins_code", "_pos"], how="left")
    return df.drop(columns=["_pos"])


def _attach_muse(df: pd.DataFrame, muse_csv: Path) -> pd.DataFrame:
    """Merge MUSE scores into df from a per-structure CSV.

    MUSE scores are atom-level but keyed without altloc. The join is on
    (chain_id, res_id, ins_code), which fans the score to all altloc variants
    of a residue; muse_score is then nulled for altlocs outside MUSE_VALID_ALTLOCS.

    df must still contain ins_code. Adds a 'muse_score' column.
    """
    if not muse_csv.exists():
        df["muse_score"] = float("nan")
        return df

    muse = pd.read_csv(muse_csv)
    muse = muse[muse["is_water"] == True].copy()  # noqa: E712
    muse["ins_code"] = muse["insertion_code"].apply(normalize_ins_code)
    muse = muse.rename(columns={"residue_seq_id": "res_id"})[["chain_id", "res_id", "ins_code", "score"]]

    df = df.merge(muse, on=["chain_id", "res_id", "ins_code"], how="left")
    df = df.rename(columns={"score": "muse_score"})

    df.loc[~df["altloc"].isin(MUSE_VALID_ALTLOCS), "muse_score"] = float("nan")

    return df


def load_structure_waters(
    cif_path: Path,
    json_path: Path | None = None,
    muse_csv: Path | None = None,
) -> pd.DataFrame:
    """Load water O records from one CIF, optionally attaching EDIA and MUSE scores.

    Reads the CIF once with altloc='all'. EDIA and MUSE are joined while ins_code
    is still present, then ins_code is dropped before returning.

    Columns: pdb_id, chain_id, res_id, altloc, x, y, z, b_factor, occupancy, edia[, muse_score]
    """
    pdb_id = cif_path.stem.removesuffix("_final")
    cf = pdbx.CIFFile.read(cif_path)
    atoms = pdbx.get_structure(cf, model=1, altloc="all", extra_fields=["b_factor", "occupancy"])
    waters = atoms[water_oxygen_mask(atoms)]

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

    if json_path is not None:
        df = _attach_edia(df, load_edia_all_altlocs(json_path))
    else:
        df["edia"] = float("nan")

    if muse_csv is not None:
        df = _attach_muse(df, muse_csv)

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


def write_cluster_cif(
    out_path: Path,
    *,
    clusters_df: pd.DataFrame | None = None,
    members_df: pd.DataFrame | None = None,
    clusters_csv: Path | None = None,
    members_csv: Path | None = None,
    include_noise: bool = False,
) -> None:
    """Write cluster centers (and optionally noise points) as a CIF file.

    Cluster centers are written as HOH oxygens on chain 'A':
      res_id = cluster_id + 1, occupancy = cluster_occupancy,
      B-factor = RMS spread sqrt((std_x² + std_y² + std_z²) / 3).
    Noise waters (cluster_id == -1) are written on chain 'B' when include_noise is True.

    Pass DataFrames directly (from build_cluster_tables) or CSV paths to load from
    disk — useful when called independently after the pipeline has already run.
    """
    if clusters_df is None:
        if clusters_csv is None:
            raise ValueError("provide clusters_df or clusters_csv")
        clusters_df = pd.read_csv(clusters_csv)

    noise_df: pd.DataFrame | None = None
    noise_occupancy: float = 0.0
    if include_noise:
        if members_df is None:
            if members_csv is None:
                raise ValueError("include_noise=True requires members_df or members_csv")
            members_df = pd.read_csv(members_csv)
        n_structures = members_df["pdb_id"].nunique()
        noise_occupancy = 1.0 / n_structures if n_structures > 0 else 0.0
        noise_df = members_df[members_df["cluster_id"] == -1].reset_index(drop=True)

    centers = pd.DataFrame(
        {
            "chain_id": "A",
            "res_id": clusters_df["cluster_id"].to_numpy(dtype=int) + 1,
            "x": clusters_df["center_x"].to_numpy(dtype=float),
            "y": clusters_df["center_y"].to_numpy(dtype=float),
            "z": clusters_df["center_z"].to_numpy(dtype=float),
            "b_factor": np.sqrt(
                (
                    clusters_df["std_x"] ** 2
                    + clusters_df["std_y"] ** 2
                    + clusters_df["std_z"] ** 2
                )
                / 3
            ).to_numpy(dtype=float),
            "occupancy": clusters_df["cluster_occupancy"].to_numpy(dtype=float),
        }
    )

    frames = [centers]

    if noise_df is not None and not noise_df.empty:
        noise = pd.DataFrame(
            {
                "chain_id": "B",
                "res_id": np.arange(1, len(noise_df) + 1, dtype=int),
                "x": noise_df["x"].to_numpy(dtype=float),
                "y": noise_df["y"].to_numpy(dtype=float),
                "z": noise_df["z"].to_numpy(dtype=float),
                "b_factor": noise_df["b_factor"].fillna(0.0).to_numpy(dtype=float),
                "occupancy": np.full(len(noise_df), noise_occupancy),
            }
        )
        frames.append(noise)

    rows = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    n = len(rows)

    atoms = struc.AtomArray(n)
    atoms.coord = rows[["x", "y", "z"]].to_numpy(dtype=float)
    atoms.chain_id = rows["chain_id"].to_numpy().astype("U4")
    atoms.res_id = rows["res_id"].to_numpy(dtype=int)
    atoms.res_name[:] = "HOH"
    atoms.atom_name[:] = "O"
    atoms.element[:] = "O"
    atoms.hetero[:] = True
    atoms.add_annotation("b_factor", dtype=float)
    atoms.b_factor[:] = rows["b_factor"].to_numpy(dtype=float)
    atoms.add_annotation("occupancy", dtype=float)
    atoms.occupancy[:] = rows["occupancy"].to_numpy(dtype=float)

    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, atoms, data_block="clusters")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cif_file.write(out_path)


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
