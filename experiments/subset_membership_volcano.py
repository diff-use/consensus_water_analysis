import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Subset-membership volcano — does the iso subset shift cluster occupancy?

    Fixes the clustering to the **all-cohort** result and asks, per cluster,
    whether structures *inside* the iso subset contribute a within-cutoff water at
    a different rate than structures *outside* it. Both variables are binary
    (member / not × in-subset / not), so each cluster is a 2×2 table tested with
    **Fisher's exact**; the effect is **Δoccupancy = occ(iso) − occ(non-iso)**,
    bounded −1..+1. Benjamini-Hochberg FDR, two-gate hit call (|Δ| **and** q).

    A hit is a cluster whose conservedness is carried disproportionately by one
    group — i.e. belonging to the iso subset does move its occupancy.
    """)
    return


@app.cell
def _():
    import pandas as _pd

    df = _pd.read_csv("/mnt/diffuse-shared/dorismai/water_analysis/hewls_65/cluster_members.csv")

    resids = [478, 471, 449, 515, 379, 342, 420, 343]

    w = df[(df.pdb_id == "7der") & (df.res_id.isin(resids))]
    print(w[["res_id", "altloc", "edia", "b_factor", "b_factor_zscore", "occupancy", "cluster_id"]]
          .sort_values("res_id")
          .to_string(index=False))
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import fisher_exact

    import config

    return Path, config, fisher_exact, np, pd, plt


@app.cell
def _(Path, config, mo):
    members_input = mo.ui.text(
        value=str(Path(config.DATA_DIR) / "C001033" / "cluster_members.csv"),
        label="all-cohort cluster_members.csv",
        full_width=True,
    )
    subset_input = mo.ui.text(
        value="data/C000836_iso.txt",
        label="subset cohort .txt (defines the in-subset group)",
        full_width=True,
    )
    mo.vstack([members_input, subset_input])
    return members_input, subset_input


@app.cell
def _(Path, members_input, mo, pd, subset_input):
    _mp = Path(members_input.value.strip())
    _sp = Path(subset_input.value.strip())
    mo.stop(not _mp.exists(), mo.md(f"**cluster_members.csv not found:** `{_mp}`"))
    mo.stop(not _sp.exists(), mo.md(f"**subset .txt not found:** `{_sp}`"))

    members = pd.read_csv(_mp)
    members["pdb_id"] = members["pdb_id"].astype(str).str.lower()

    subset_ids = {
        line.strip().lower() for line in _sp.read_text().splitlines() if line.strip()
    }
    universe = sorted(set(members["pdb_id"]))
    in_subset = [p for p in universe if p in subset_ids]
    missing = sorted(subset_ids - set(universe))

    mo.md(
        f"Clustering universe: **{len(universe)}** structures — "
        f"**{len(in_subset)}** in subset, **{len(universe) - len(in_subset)}** out. "
        + (f"⚠️ {len(missing)} subset IDs not in the clustering: {missing}" if missing else "")
    )
    return members, subset_ids, universe


@app.cell
def _(members, subset_ids, universe):
    # clusters × structures 0/1 membership over the fixed all-cohort universe.
    _kept = members[(members["within_cutoff"]) & (members["cluster_id"] >= 0)]
    member_matrix = (
        _kept.assign(_present=1)
        .pivot_table(index="cluster_id", columns="pdb_id", values="_present",
                     aggfunc="max", fill_value=0)
        .reindex(columns=universe, fill_value=0)
    )
    is_iso = member_matrix.columns.isin(subset_ids).astype(int)
    return is_iso, member_matrix


@app.cell
def _(fisher_exact, is_iso, member_matrix, np, pd):
    def bh(pvals):
        p = np.asarray(pvals, dtype=float)
        m = len(p)
        order = np.argsort(p)
        q = p[order] * m / np.arange(1, m + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.clip(q, 0, 1)
        return out

    _n_iso = int(is_iso.sum())
    _n_out = int((1 - is_iso).sum())
    _rows = []
    for _cid, _row in member_matrix.iterrows():
        _y = _row.to_numpy()
        _a = int((_y & is_iso).sum())          # iso, member
        _c = int((_y & (1 - is_iso)).sum())    # out, member
        _table = [[_a, _n_iso - _a], [_c, _n_out - _c]]
        _odds, _p = fisher_exact(_table)
        _rows.append({
            "cluster_id": int(_cid),
            "n_member": int(_y.sum()),
            "occ_all": _y.sum() / len(_y),
            "occ_iso": _a / _n_iso,
            "occ_out": _c / _n_out,
            "delta": _a / _n_iso - _c / _n_out,
            "odds_ratio": float(_odds),
            "p": float(_p),
        })

    stats = pd.DataFrame(_rows)
    stats["q"] = bh(stats["p"].to_numpy())
    return (stats,)


@app.cell
def _(mo):
    q_cut = mo.ui.number(value=0.05, start=1e-6, stop=1.0, step=0.005, label="q cutoff (FDR)")
    effect_cut = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.3, label="|Δoccupancy| cutoff", show_value=True
    )
    mo.vstack([q_cut, effect_cut])
    return effect_cut, q_cut


@app.cell
def _(effect_cut, np, q_cut, stats):
    scored = stats.copy()
    _sig = scored["q"] < float(q_cut.value)
    _big = scored["delta"].abs() >= float(effect_cut.value)
    scored["category"] = np.select([_sig & _big, _sig & ~_big], ["hit", "significant, small effect"], default="ns")
    scored["direction"] = np.where(scored["delta"] >= 0, "enriched_in_iso", "depleted_in_iso")
    return (scored,)


@app.cell
def _(effect_cut, mo, np, plt, q_cut, scored):
    # Fisher underflows to p=0 for fully group-separated clusters; floor q so those
    # points stay on-plot instead of flying to inf. Raw q is kept in the hit table.
    _Y_FLOOR = 1e-50
    _y = -np.log10(scored["q"].clip(lower=_Y_FLOOR))
    _fig, _ax = plt.subplots(figsize=(3.4, 3.2))
    _palette = {"ns": "lightgray", "significant, small effect": "steelblue", "hit": "orange"}
    for _cat, _color in _palette.items():
        _mask = scored["category"] == _cat
        _ax.scatter(scored.loc[_mask, "delta"], _y[_mask], s=20, alpha=0.6,
                    color=_color, edgecolor="k" if _cat == "hit" else "none")
    _ax.axhline(-np.log10(float(q_cut.value)), ls="--", color="tomato", lw=1)
    for _x in (float(effect_cut.value), -float(effect_cut.value)):
        _ax.axvline(_x, ls="--", color="0.5", lw=1)
    _ax.axvline(0, color="0.85", lw=0.6)
    _ax.set_xlim(-1, 1)
    _ax.set_xlabel("Δoccupancy  (occ_iso − occ_out)")
    _ax.set_ylabel("−log10(q)  (q floored at 1e-50)")
    _ax.set_title("iso-subset membership vs cluster occupancy")
    _fig.tight_layout()

    _n_hit = int((scored["category"] == "hit").sum())
    _n_small = int((scored["category"] == "significant, small effect").sum())
    mo.vstack([
        mo.md(f"**{_n_hit} hits** at q<{float(q_cut.value):g}, |Δ|≥{float(effect_cut.value):g} "
              f"— {_n_small} significant but sub-threshold."),
        _fig,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Hit clusters (sorted by q, then |Δ|)
    """)
    return


@app.cell
def _(mo, scored):
    _hits = scored[scored["category"] == "hit"].copy()
    _hits["abs_delta"] = _hits["delta"].abs()
    _hits = _hits.sort_values(["q", "abs_delta"], ascending=[True, False])
    _cols = ["cluster_id", "n_member", "occ_all", "occ_iso", "occ_out",
             "delta", "odds_ratio", "p", "q", "direction"]
    mo.stop(_hits.empty, mo.md("*No hits at the current cutoffs.*"))
    _hits[_cols].reset_index(drop=True)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-structure precision / recall, coloured by group

    Places every structure in the **all-cohort** clustering on the same P/R plane —
    recall = fraction of consensus centers this structure covers, precision =
    fraction of its waters that land on a center — then colours the points by a
    categorical / thresholded group instead of `num_water`. Reference centers are
    the clusters with `cluster_occupancy ≥ cutoff`; matching uses the
    `CLUSTER_MEMBER_RADIUS` cutoff. Ask whether being in the iso subset (or a given
    space group, or a low `rmsd_after`) separates the point cloud.
    """)
    return


@app.cell
def _():
    from cw.metrics import consensus_centers, per_structure_consensus_pr

    return consensus_centers, per_structure_consensus_pr


@app.cell
def _(Path, members_input, mo, pd):
    _base = Path(members_input.value.strip()).parent
    _cl = _base / "clusters.csv"
    _md = _base / "metadata.csv"
    mo.stop(not _cl.exists(), mo.md(f"**clusters.csv not found:** `{_cl}`"))
    mo.stop(not _md.exists(), mo.md(f"**metadata.csv not found:** `{_md}`"))

    clusters_all = pd.read_csv(_cl)
    meta_all = pd.read_csv(_md)
    meta_all["pdb_id"] = meta_all["pdb_id"].astype(str).str.lower()

    # rmsd_after lives in the alignment report next to the aligned CIFs, not metadata.
    _reports = sorted((_base / "aligned_pdbs").glob("alignment_report_*.csv"))
    if _reports:
        _rep = pd.read_csv(_reports[0])
        _rep["pdb_id"] = _rep["pdb_id"].astype(str).str.lower()
        meta_all = meta_all.merge(_rep[["pdb_id", "rmsd_after"]], on="pdb_id", how="left")
    else:
        meta_all["rmsd_after"] = float("nan")
    return clusters_all, meta_all


@app.cell
def _(mo):
    occ_cutoff = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.3,
        label="consensus cutoff for reference centers", show_value=True,
    )
    color_mode = mo.ui.dropdown(
        options=["in/out subset", "space_group", "rmsd_after threshold"],
        value="in/out subset", label="colour structures by",
    )
    rmsd_thr = mo.ui.number(
        value=1.0, start=0.0, stop=10.0, step=0.1, label="rmsd_after threshold (Å)"
    )
    mo.vstack([occ_cutoff, color_mode, rmsd_thr])
    return color_mode, occ_cutoff, rmsd_thr


@app.cell
def _(
    clusters_all,
    config,
    consensus_centers,
    members,
    meta_all,
    occ_cutoff,
    per_structure_consensus_pr,
    subset_ids,
):
    _centers = consensus_centers(clusters_all, float(occ_cutoff.value))
    pr_df = per_structure_consensus_pr(members, _centers, config.CLUSTER_MEMBER_RADIUS)
    pr_df = pr_df.merge(
        meta_all[["pdb_id", "space_group", "rmsd_after"]], on="pdb_id", how="left"
    )
    pr_df["in_subset"] = pr_df["pdb_id"].isin(subset_ids)
    return (pr_df,)


@app.cell
def _(color_mode, mo, np, occ_cutoff, pd, plt, pr_df, rmsd_thr):
    mo.stop(len(pr_df) == 0, mo.md("*No per-structure P/R rows.*"))
    _mode = color_mode.value
    if _mode == "space_group":
        _key = pr_df["space_group"].fillna("n/a").astype(str)
    elif _mode == "in/out subset":
        _key = np.where(pr_df["in_subset"], "in subset", "out")
    else:
        _t = float(rmsd_thr.value)
        _key = np.where(
            pr_df["rmsd_after"].isna(), "n/a",
            np.where(pr_df["rmsd_after"] <= _t, f"≤ {_t:g} Å", f"> {_t:g} Å"),
        )
    _key = pd.Series(_key, index=pr_df.index)

    _fig, _ax = plt.subplots(figsize=(4.4, 4))
    _cmap = plt.get_cmap("tab10")
    _labels = sorted(_key.unique(), key=lambda s: (s == "n/a", s))
    for _i, _lab in enumerate(_labels):
        _m = _key == _lab
        _ax.scatter(
            pr_df.loc[_m, "recall"], pr_df.loc[_m, "precision"],
            s=32, alpha=0.8, edgecolors="k", linewidths=0.4,
            color="0.7" if _lab == "n/a" else _cmap(_i % 10),
            label=f"{_lab} (n={int(_m.sum())})",
        )
    _ax.set_xlim(0, 1.01)
    _ax.set_ylim(0, 1.01)
    _ax.set_box_aspect(1)
    _ax.set_xlabel("recall")
    _ax.set_ylabel("precision")
    _ax.set_title(f"per-structure P/R vs consensus (cutoff={float(occ_cutoff.value):g})")
    _ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
