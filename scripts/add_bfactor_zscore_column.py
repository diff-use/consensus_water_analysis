"""Append a per-structure B-factor z-score column to a cluster_members.csv.

For each pdb_id in the table, the z-score standardises each water's B-factor
against that structure's full water-O population read from the original CIF —
the same reference `cw.filter.keep_by_bfactor(mode="zscore", population="water")`
uses. This means `b_factor_zscore < X` selects exactly the waters the B-factor
z-score filter at cutoff X would keep. A degenerate structure (population std 0)
gets z = 0.0, matching keep_by_bfactor's "keep every water" behaviour there.

Usage:
    uv run scripts/add_bfactor_zscore_column.py <cluster_members.csv> [-o OUT]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from cw.io import cif_path_for, load_structure_waters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("members_csv", type=Path, help="cluster_members.csv to augment")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: overwrite the input in place)",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    df = pd.read_csv(args.members_csv)

    zscore = pd.Series(float("nan"), index=df.index)
    for pdb_id in df["pdb_id"].unique():
        cif_path = cif_path_for(pdb_id, config.ALL_PDB_REDO_DIR, config.CIF_TEMPLATE)
        if not cif_path.exists():
            logger.warning(f"{pdb_id}: CIF not found ({cif_path}) — leaving z-score NaN")
            continue
        ref = load_structure_waters(cif_path, pdb_id=pdb_id)["b_factor"].to_numpy()
        std = ref.std()
        rows = df["pdb_id"] == pdb_id
        if std == 0:
            zscore.loc[rows] = 0.0
        else:
            zscore.loc[rows] = (df.loc[rows, "b_factor"] - ref.mean()) / std

    df["b_factor_zscore"] = zscore

    out_path = args.output or args.members_csv
    df.to_csv(out_path, index=False)
    n_missing = int(zscore.isna().sum())
    logger.info(
        f"Wrote {out_path} ({len(df)} rows)"
        + (f"; {n_missing} rows left NaN (missing CIF)" if n_missing else "")
    )


if __name__ == "__main__":
    main()
