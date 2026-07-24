"""How many analysis waters got their EDIA from the .get("EDIAm", 0.0) coercion?

cluster_members.csv preserves per-structure aligned-CIF atom order, so we can replay
the exact positional altloc-pairing the pipeline uses (edia_scores_in_order) with a
NaN default instead of 0.0. A row is coercion-affected iff its stored edia is 0.0 but
the NaN-default replay yields NaN — i.e. its JSON water entry was present but had no
EDIAm key. Genuine zeros (EDIAm literally 0.0) and missing residues (already NaN) are
unaffected. A 0.0-default replay is also run to validate key reconstruction against the
stored column (cluster_members drops ins_code; waters almost always have none).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from cw.io import edia_scores_in_order, normalize_ins_code

COHORTS = ["hewls_65", "endothiapepsin_000240_iso", "carbonicanhydrase_000562_iso"]

data_dir = Path(config.DATA_DIR)
edia_dir = Path(config.ALL_PDB_REDO_DIR)


def load_edia_lists(json_path, default):
    """load_edia_all_altlocs with a chosen default for a missing EDIAm key.
    Returns (scores, n_water_entries, n_missing_ediam)."""
    if not json_path.exists():
        return {}, 0, 0
    try:
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}, 0, 0
    scores, n_water, n_missing = {}, 0, 0
    for entry in payload:
        if entry.get("compID") not in {"HOH", "WAT"}:
            continue
        n_water += 1
        pdb_info = entry.get("pdb", {})
        key = (
            str(pdb_info.get("strandID", "")),
            int(pdb_info.get("seqNum", 0)),
            normalize_ins_code(pdb_info.get("insCode", "")),
        )
        if "EDIAm" not in entry or entry.get("EDIAm") is None:
            n_missing += 1
            scores.setdefault(key, []).append(default)
        else:
            scores.setdefault(key, []).append(float(entry["EDIAm"]))
    return scores, n_water, n_missing


for cohort in COHORTS:
    cm_path = data_dir / cohort / "cluster_members.csv"
    if not cm_path.exists():
        print(f"\n=== {cohort} === (no cluster_members.csv)")
        continue
    cm = pd.read_csv(cm_path)
    total_w = len(cm)
    stored = cm["edia"]
    n_zero = int((stored == 0.0).sum())
    n_nan = int(stored.isna().sum())

    affected = key_match = 0
    json_water = json_missing = n_struct_json = 0
    for pdb_id, group in cm.groupby("pdb_id", sort=False):
        json_path = edia_dir / config.EDIA_TEMPLATE.format(pdb_id=pdb_id)
        scores_nan, n_water, n_missing = load_edia_lists(json_path, np.nan)
        scores_zero, _, _ = load_edia_lists(json_path, 0.0)
        json_water += n_water
        json_missing += n_missing
        n_struct_json += int(json_path.exists())

        keys = [(str(c), int(r), "") for c, r in zip(group["chain_id"], group["res_id"])]
        edia_nan = edia_scores_in_order(keys, scores_nan)
        edia_zero = edia_scores_in_order(keys, scores_zero)
        stored_g = group["edia"].to_numpy(dtype=float)

        both_nan = np.isnan(edia_zero) & np.isnan(stored_g)
        key_match += int((both_nan | np.isclose(edia_zero, stored_g)).sum())
        affected += int(((stored_g == 0.0) & np.isnan(edia_nan)).sum())

    print(f"\n=== {cohort} ===")
    print(f"cluster_members waters:        {total_w:,}")
    print(f"  edia == 0.0:                 {n_zero:,} ({n_zero / total_w:.2%})")
    print(f"  edia == NaN:                 {n_nan:,} ({n_nan / total_w:.2%})")
    print(f"  key-recompute match / total: {key_match:,}/{total_w:,} ({key_match / total_w:.2%})")
    pct_all = affected / total_w if total_w else 0
    pct_zero = f"{affected / n_zero:.1%} of edia==0" if n_zero else "n/a"
    print(f"AFFECTED by .get('EDIAm',0.0): {affected:,} waters ({pct_all:.2%} of all; {pct_zero})")
    print(f"  JSON-level missing EDIAm:    {json_missing:,}/{json_water:,} water entries "
          f"({json_missing / json_water:.2%}) over {n_struct_json} structures with JSON")
