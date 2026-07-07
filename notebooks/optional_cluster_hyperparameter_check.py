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
    # optional — Cluster HDBSCAN hyperparameter check

    Re-clusters the already-pooled water oxygens from a cohort's
    `cluster_members.csv`, varying HDBSCAN parameters to pick a detection floor
    from data rather than the `0.3 × N` rule. Reuses `cw.cluster.run_hdbscan` +
    `build_cluster_tables`, so every number matches the real pipeline. Also
    renders the pre-computed grid search from
    `scripts/find_clustering_hyperparameters.py`, if present.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(".")))

    import config
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from cw.cluster import build_cluster_tables, run_hdbscan

    return Path, build_cluster_tables, config, np, pd, plt, run_hdbscan


@app.cell
def _(Path, config, pd):
    COHORT = "hewls_65" #"carbonicanhydrase_000562" #"hewls_65"
    DATA = Path(config.DATA_DIR) / COHORT
    SUBSET = Path(config.DATA_DIR) / Path(COHORT + "_iso")
    if not SUBSET.exists():
        SUBSET = DATA

    cluster_members = pd.read_csv(SUBSET / "cluster_members.csv")

    print(f"cluster_members rows: {len(cluster_members)}")
    return SUBSET, cluster_members


@app.cell
def _(mo):
    mo.md("""
    ## HDBSCAN parameter sweep

    - **Mode A** — vary `min_cluster_size` at a fixed `min_samples`.
    - **Mode B** — vary `min_samples` at a fixed `min_cluster_size`.
    """)
    return


@app.cell
def _(cluster_members, mo):
    # Recover the raw HDBSCAN input: drop the prior run's labels (build_cluster_tables
    # re-inserts cluster_id/within_cutoff and errors if they already exist).
    records = cluster_members.drop(columns=["cluster_id", "within_cutoff"], errors="ignore")
    coords = records[["x", "y", "z"]].to_numpy()
    n_structures = int(records["pdb_id"].nunique())

    mo.md(
        f"**{len(records):,}** pooled water oxygens across **{n_structures}** "
        f"structures — avg **{len(records) / n_structures:.2f}** waters/structure. "
        f"`min_cluster_size` is in *waters*, so ≈ this many structures agreeing."
    )
    return coords, n_structures, records


@app.cell
def _(build_cluster_tables, config, np, pd, plt, run_hdbscan):
    def run_sweep(coords, records, n_structures, param_pairs, xname):
        """param_pairs: list of (min_cluster_size, min_samples_or_None). Returns
        (summary_df, occ_by_x), occ_by_x mapping the varying value -> occupancy array."""
        cutoffs = (0.25, 0.5, 0.75)
        rows, occ_by_x = [], {}
        for mcs, ms in param_pairs:
            labels = run_hdbscan(coords, min_cluster_size=mcs, min_samples=ms)
            _, clusters = build_cluster_tables(
                records,
                labels,
                radius=config.CLUSTER_MEMBER_RADIUS,
                n_total_structures=n_structures,
            )
            occ = clusters["cluster_occupancy"]
            row = {
                "min_cluster_size": mcs,
                "min_samples": mcs if ms is None else ms,
                "n_clusters": len(clusters),
                "noise_frac": float((labels == -1).mean()),
                "median_occ": float(occ.median()) if len(occ) else float("nan"),
                "max_occ": float(occ.max()) if len(occ) else float("nan"),
            }
            for c in cutoffs:
                row[f"n_occ_ge_{c}"] = int((occ >= c).sum())
            rows.append(row)
            occ_by_x[row[xname]] = occ.to_numpy()
        return pd.DataFrame(rows).sort_values(xname).reset_index(drop=True), occ_by_x

    def plot_sweep(df, occ_by_x, xcol):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes[0, 0].plot(df[xcol], df["n_clusters"], "o-", color="steelblue")
        axes[0, 0].set(xlabel=xcol, ylabel="n clusters", title="Clusters detected")
        axes[0, 1].plot(df[xcol], df["noise_frac"], "o-", color="tomato")
        axes[0, 1].set(xlabel=xcol, ylabel="noise fraction", title="Waters left as noise")
        for col in [c for c in df.columns if c.startswith("n_occ_ge_")]:
            axes[1, 0].plot(df[xcol], df[col], "o-", label=col.replace("n_occ_ge_", "occ ≥ "))
        axes[1, 0].set(xlabel=xcol, ylabel="n clusters", title="Conserved sites above cutoff")
        axes[1, 0].legend()
        _bins = np.linspace(0, 1, 21)
        for x, occ in sorted(occ_by_x.items()):
            if len(occ):
                axes[1, 1].hist(occ, bins=_bins, histtype="step", lw=1.5, label=f"{xcol}={x}")
        axes[1, 1].set(
            xlabel="cluster_occupancy", ylabel="n clusters", title="Occupancy distribution"
        )
        axes[1, 1].legend()
        fig.tight_layout()
        return fig

    return plot_sweep, run_sweep


@app.cell
def _(SUBSET, mo, pd):
    # Grid-search results, if the user ran scripts/find_clustering_hyperparameters.py.
    # Read-only: this cell only renders the pre-computed scores; it does not re-search.
    _scores_path = SUBSET / "clustering_hyperparameters.csv"
    if not _scores_path.exists():
        grid_scores = None
        grid_view = mo.md(
            "*No grid search found. Run "
            "`uv run scripts/find_clustering_hyperparameters.py <cohort.txt>` "
            "to populate `clustering_hyperparameters.csv`, then re-run this cell.*"
        )
    else:
        grid_scores = pd.read_csv(_scores_path)
        _well_formed = grid_scores[grid_scores["dbcv"].notna()]
        _rec = grid_scores[grid_scores["recommended"]].iloc[0]
        _relaxed = bool(grid_scores["guard_relaxed"].iloc[0])

        if _relaxed:
            _note = "⚠️ no params kept clusters within the membership radius; guard relaxed."
        elif _rec["dbcv_rank"] == 1 and _rec["stab_rank"] == 1:
            _note = "best on both separation and reproducibility — criteria agree, low-risk."
        else:
            _note = "balance point of the DBCV (crisp) ↔ stability (coarse) tension; neither extreme."

        _n_occ = _well_formed["n_occ_ge_0_3"]
        _frac = _well_formed["frac_water_in_occ"]
        grid_view = mo.vstack(
            [
                mo.md(
                    f"### HDBSCAN grid search — recommended params\n\n"
                    f"**Recommended:** `min_cluster_size={int(_rec['min_cluster_size'])}`, "
                    f"`min_samples={int(_rec['min_samples'])}` "
                    f"(DBCV rank #{int(_rec['dbcv_rank'])}, stability rank #{int(_rec['stab_rank'])}; "
                    f"{int(_rec['n_clusters'])} clusters, {int(_rec['n_occ_ge_0_3'])} with consensus>0.3)  \n"
                    f"{_note}\n\n"
                    f"**Across {len(_well_formed)} well-formed candidates:** "
                    f"{int(_n_occ.min())}–{int(_n_occ.max())} conserved sites (consensus>0.3); "
                    f"{_frac.min():.0%}–{_frac.max():.0%} of pooled waters fall in them.\n\n"
                    f"To apply: set `HDBSCAN_MIN_CLUSTER_SIZE`/`HDBSCAN_MIN_SAMPLES` in `config.py` "
                    f"(or pass `--min-cluster-size`/`--min-samples`) and re-run `cluster_waters.py`."
                ),
                mo.ui.table(
                    grid_scores.round(3).sort_values("max_rank", na_position="last"),
                    selection=None,
                    pagination=False,
                ),
            ]
        )
    grid_view
    return


@app.cell
def _(mo):
    mo.md("""
    ### Mode A — vary `min_cluster_size`
    """)
    return


@app.cell
def _(mo):
    sweep_form = (
        mo.md(
            """
            **min_cluster_size values:** {mcs}

            **min_samples:** {ms} &nbsp; tie to min_cluster_size: {tie}
            """
        )
        .batch(
            mcs=mo.ui.text(value="2,5,10,20", full_width=True),
            ms=mo.ui.number(value=3, start=1, stop=20),
            tie=mo.ui.checkbox(value=False),
        )
        .form()
    )
    sweep_form
    return (sweep_form,)


@app.cell
def _(coords, mo, n_structures, records, run_sweep, sweep_form):
    mo.stop(sweep_form.value is None, mo.md("*Set parameters and submit to run Mode A.*"))
    _v = sweep_form.value
    _mcs_list = [int(x) for x in _v["mcs"].replace(" ", "").split(",") if x]
    _ms = None if _v["tie"] else int(_v["ms"])
    sweep_a, occ_a = run_sweep(
        coords, records, n_structures, [(m, _ms) for m in _mcs_list], "min_cluster_size"
    )
    sweep_a
    return occ_a, sweep_a


@app.cell
def _(occ_a, plot_sweep, sweep_a):
    plot_sweep(sweep_a, occ_a, "min_cluster_size")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Mode B — vary `min_samples`
    """)
    return


@app.cell
def _(mo):
    ms_form = (
        mo.md(
            """
            **min_samples values:** {ms}

            **fixed min_cluster_size:** {mcs}
            """
        )
        .batch(
            ms=mo.ui.text(value="3,5,10,20", full_width=True),
            mcs=mo.ui.number(value=5, start=2, stop=50),
        )
        .form()
    )
    ms_form
    return (ms_form,)


@app.cell
def _(coords, mo, ms_form, n_structures, records, run_sweep):
    mo.stop(ms_form.value is None, mo.md("*Set parameters and submit to run Mode B.*"))
    _v = ms_form.value
    _ms_list = [int(x) for x in _v["ms"].replace(" ", "").split(",") if x]
    _mcs = int(_v["mcs"])
    sweep_b, occ_b = run_sweep(
        coords, records, n_structures, [(_mcs, s) for s in _ms_list], "min_samples"
    )
    sweep_b
    return occ_b, sweep_b


@app.cell
def _(occ_b, plot_sweep, sweep_b):
    plot_sweep(sweep_b, occ_b, "min_samples")
    return


if __name__ == "__main__":
    app.run()
