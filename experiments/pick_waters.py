"""Pick example waters for figures/inspection.

No --structure: survey structures in the cohort, ranked by resolution, with
per-PDB water counts and edia/z-score ranges (helps choose which structure to
pull examples from).

With --structure PDB_ID: within that structure, show the nearest candidate
waters for a set of EDIA bands and B-factor z-score targets.
"""
import argparse
import pandas as pd

DEFAULT_BASE = "/mnt/diffuse-shared/dorismai/water_analysis/hewls_65"
EDIA_TARGETS = [0.4, 0.6, 0.8]
Z_TARGETS = [2.0, 1.5, 1.0, 0.0, -1.0]


def survey_structures(base):
    cm = pd.read_csv(f"{base}/cluster_members.csv")
    md = pd.read_csv(f"{base}/metadata.csv")[["pdb_id", "resolution"]]
    cm = cm.merge(md, on="pdb_id", how="left")
    w = cm.dropna(subset=["edia", "b_factor_zscore"]).copy()
    res = (
        w.groupby("pdb_id")
        .agg(
            resolution=("resolution", "first"),
            n=("edia", "size"),
            edia_min=("edia", "min"),
            edia_max=("edia", "max"),
            z_min=("b_factor_zscore", "min"),
            z_max=("b_factor_zscore", "max"),
        )
        .sort_values("resolution")
    )
    print("=== highest-resolution structures (n waters with edia+z) ===")
    print(res.head(12).to_string())


def pick_waters(base, structure, n):
    cm = pd.read_csv(f"{base}/cluster_members.csv")
    w = cm[cm.pdb_id == structure].dropna(subset=["edia", "b_factor_zscore"]).copy()

    def show(df):
        for _, r in df.head(n).iterrows():
            print(
                f"    res_id={int(r.res_id):>4d} altloc={str(r.altloc):>3s} "
                f"edia={r.edia:.3f} b={r.b_factor:6.2f} z={r.b_factor_zscore:+.2f} "
                f"occ={r.occupancy:.2f} cluster={int(r.cluster_id)}"
            )

    print(f"### {structure} EDIA targets (nearest {n} candidates each) ###")
    d = w[w.edia < 0.4]
    print("edia < 0.4 (near 0.25):")
    show(d.iloc[(d.edia - 0.25).abs().argsort()])
    for t in EDIA_TARGETS:
        d = w[w.edia > t]
        print(f"edia > ~{t}:")
        show(d.iloc[(d.edia - t).abs().argsort()])
    d = w[w.edia >= 1.0]
    print("edia >= 1.0:")
    show(d.iloc[(d.edia - 1.0).abs().argsort()])

    print(f"\n### {structure} B-factor z-score targets (nearest {n} candidates each) ###")
    for t in Z_TARGETS:
        print(f"z ~ {t:+.1f}:")
        show(w.iloc[(w.b_factor_zscore - t).abs().argsort()])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE, help="cohort directory")
    ap.add_argument("--structure", help="PDB ID to pick waters from; omit to survey structures")
    ap.add_argument("-n", type=int, default=5, help="candidates to show per target")
    args = ap.parse_args()

    if args.structure:
        pick_waters(args.base, args.structure, args.n)
    else:
        survey_structures(args.base)


if __name__ == "__main__":
    main()
