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
    # 01 — Cluster analysis

    Reads pre-computed `clusters.csv`, `cluster_members.csv`, and `metadata.csv`
    for one cohort, then explores cluster occupancy, water metrics, per-structure
    agreement with the consensus, and HDBSCAN parameter sensitivity. Re-run
    `scripts/cluster_waters.py` to refresh the CSVs.
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
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import seaborn as sns

    from cw.cluster import build_cluster_tables, run_hdbscan
    from cw.metrics import (
        chamfer_distance,
        consensus_centers,
        pareto_front,
        per_structure_consensus_pr,
        precision_recall,
    )
    from cw.plots import plot_pr_scatter, quantile_boundaries

    return (
        Path,
        build_cluster_tables,
        chamfer_distance,
        config,
        consensus_centers,
        np,
        pareto_front,
        pd,
        per_structure_consensus_pr,
        plot_pr_scatter,
        plt,
        precision_recall,
        quantile_boundaries,
        run_hdbscan,
        sns,
    )


@app.cell
def _(Path, config, pd):
    COHORT = "hewls_65" #"carbonicanhydrase_000562" #"hewls_65"
    DATA = Path(config.DATA_DIR) / COHORT
    SUBSET = Path(config.DATA_DIR) / Path(COHORT + "_iso")
    if not SUBSET.exists():
        SUBSET = DATA

    clusters = pd.read_csv(SUBSET / "clusters.csv")
    cluster_members = pd.read_csv(SUBSET / "cluster_members.csv")
    metadata = pd.read_csv(DATA / "metadata.csv")

    print(f"clusters rows:        {len(clusters)}")
    print(f"cluster_members rows: {len(cluster_members)}")
    return SUBSET, cluster_members, clusters, metadata


@app.cell
def _(mo):
    mo.md("""
    ## Cluster occupancy distribution
    """)
    return


@app.cell
def _(mo):
    include_noise_toggle = mo.ui.checkbox(value=True, label="include noise waters")
    occ_font_size = mo.ui.slider(
        start=6, stop=20, step=1, value=16, label="font size", show_value=True,
    )
    mo.hstack([include_noise_toggle, occ_font_size], justify="start")
    return include_noise_toggle, occ_font_size


@app.cell
def _(
    cluster_members,
    clusters,
    include_noise_toggle,
    np,
    occ_font_size,
    pd,
    plt,
    sns,
):
    cluster_occupancy_cutoff = 0.3
    _fs = occ_font_size.value
    _tick_font_size = 14  # tick labels are fixed, independent of the slider

    if len(clusters) == 0:
        _fig, _ax = plt.subplots(figsize=(4, 3))
        _ax.text(0.5, 0.5, "No clusters", transform=_ax.transAxes, ha="center", fontsize=_fs)
        plt.tight_layout()

    elif not include_noise_toggle.value:
        _fig, _ax = plt.subplots(figsize=(4, 3))
        sns.histplot(clusters["cluster_occupancy"], bins=30, ax=_ax)
        _ax.set_xlabel("cluster occupancy (fraction of structures)", fontsize=_fs)
        _ax.set_ylabel("count", fontsize=_fs)
        # _ax.set_title("Cluster occupancy distribution", fontsize=_fs)
        _ax.tick_params(labelsize=_tick_font_size)
        _ax.axvline(cluster_occupancy_cutoff, color="red", linestyle="--", linewidth=1, label=str(cluster_occupancy_cutoff))
        _ax.legend(fontsize=_fs)
        plt.tight_layout()

    else:
        _n = cluster_members["pdb_id"].nunique()
        _noise_occ = 1.0 / _n if _n > 0 else 0.0
        _noise_n = int((cluster_members["cluster_id"] == -1).sum())
        _data = pd.concat([
            clusters[["cluster_occupancy"]].assign(source="cluster"),
            pd.DataFrame({"cluster_occupancy": [_noise_occ] * _noise_n, "source": "noise"}),
        ], ignore_index=True)

        _bin_edges = np.linspace(_data["cluster_occupancy"].min(),
                                 _data["cluster_occupancy"].max() + 1e-9, 31)
        _cluster_max = int(np.histogram(clusters["cluster_occupancy"], bins=_bin_edges)[0].max())
        _noise_max = _noise_n  # all noise lands in one bin

        _fig, (_ax_top, _ax_bot) = plt.subplots(
            2, 1, sharex=True, figsize=(4, 3),
            gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05},
        )
        for _ax in (_ax_top, _ax_bot):
            sns.histplot(_data, x="cluster_occupancy", hue="source",
                         bins=_bin_edges, ax=_ax)
            _ax.axvline(cluster_occupancy_cutoff, color="red", linestyle="--", linewidth=1)
            _ax.tick_params(labelsize=_tick_font_size)
            if _ax.get_legend():
                _ax.get_legend().remove()

        _ax_top.set_ylim(_noise_max * 0.8, _noise_max * 1.15)
        _ax_bot.set_ylim(0, _cluster_max * 1.25)

        _ax_top.spines["bottom"].set_visible(False)
        _ax_bot.spines["top"].set_visible(False)
        _ax_top.tick_params(bottom=False)

        _d = 0.015
        _kw = dict(color="k", clip_on=False, linewidth=1, transform=_ax_top.transAxes)
        _ax_top.plot((-_d, +_d), (-_d, +_d), **_kw)
        _ax_top.plot((1 - _d, 1 + _d), (-_d, +_d), **_kw)
        _kw["transform"] = _ax_bot.transAxes
        _ax_bot.plot((-_d, +_d), (1 - _d, 1 + _d), **_kw)
        _ax_bot.plot((1 - _d, 1 + _d), (1 - _d, 1 + _d), **_kw)

        _ax_top.set_ylabel("")
        _ax_top.set_xlabel("")
        # _ax_top.set_title("Cluster occupancy distribution", fontsize=_fs)
        _ax_bot.set_xlabel("cluster occupancy", fontsize=_fs)
        _ax_bot.set_ylabel("count", fontsize=_fs)

        _pal = sns.color_palette()
        # _ax_bot.legend(handles=[
        #     mpatches.Patch(color=_pal[0], label="cluster"),
        #     mpatches.Patch(color=_pal[1], label=f"noise"),
        #     mlines.Line2D([], [], color="red", linestyle="--", linewidth=1,
        #                   label=f"cutoff = {cluster_occupancy_cutoff}"),
        # ], fontsize=_tick_font_size)

        plt.tight_layout()

    _fig
    return (cluster_occupancy_cutoff,)


@app.cell
def _(mo):
    mo.md("""
    ### Cumulative water members vs cluster occupancy
    """)
    return


@app.cell
def _(cluster_members, cluster_occupancy_cutoff, clusters, plt):
    if len(clusters) == 0:
        _fig, _ax = plt.subplots(figsize=(5, 3))
        _ax.text(0.5, 0.5, "No clusters", transform=_ax.transAxes, ha="center")
    else:
        _member_counts = (
            cluster_members[cluster_members["within_cutoff"]]
            .groupby("cluster_id")
            .size()
            .rename("n_members")
        )
        _cum = (
            clusters[["cluster_id", "cluster_occupancy"]]
            .merge(_member_counts, on="cluster_id")
            .sort_values("cluster_occupancy")
        )
        _cum["cum_members"] = _cum["n_members"].cumsum()

        _fig, _ax = plt.subplots(figsize=(5, 3))
        _ax.step(_cum["cluster_occupancy"], _cum["cum_members"], where="post")
        _ax.axvline(
            cluster_occupancy_cutoff, color="red", linestyle="--", linewidth=1,
            label=f"cutoff = {cluster_occupancy_cutoff}",
        )
        _ax.set_xlabel("cluster occupancy (fraction of structures)")
        _ax.set_ylabel("cumulative water members")
        _ax.legend()
        plt.tight_layout()

    _fig
    return


@app.cell
def _(cluster_members):
    cluster_members
    return


@app.cell
def _(cluster_members, cluster_occupancy_cutoff, clusters):
    consensus_mask = clusters["cluster_occupancy"] > cluster_occupancy_cutoff
    nonconsensus_clusters = clusters[~consensus_mask]
    consensus_cluster_ids = set(clusters.loc[consensus_mask, "cluster_id"])

    n_structure = cluster_members["pdb_id"].nunique()
    n_noise = int((cluster_members["cluster_id"] == -1).sum())

    is_consensus_water = (
        cluster_members["within_cutoff"]
        & cluster_members["cluster_id"].isin(consensus_cluster_ids)
    )
    n_nonconsensus_water = int((~is_consensus_water).sum())
    n_total_water = len(cluster_members)

    print(
        f"{len(nonconsensus_clusters)} non-consensus clusters out of {len(clusters)}: "
        f"{len(nonconsensus_clusters) / len(clusters):.3f}"
    )
    print(f"{n_total_water} total waters from {n_structure} PDBs")
    print(f"{n_noise} noise waters; {n_nonconsensus_water} non-consensus waters (incl. noise)")
    print(f"{n_nonconsensus_water / n_total_water:.3f} of waters are non-consensus")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Water metrics
    """)
    return


@app.cell
def _(cluster_members, plt, sns):
    _assigned = cluster_members[cluster_members["within_cutoff"]].copy()

    _metrics = ["b_factor", "occupancy", "edia"]
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3))

    for _ax, _col in zip(_axes, _metrics):
        _vals = _assigned[_col].dropna()
        if len(_vals) == 0:
            _ax.set_title(f"{_col} (no data)")
            continue
        sns.histplot(_vals, bins=30, ax=_ax)
        _ax.set_xlabel(_col)
        _ax.set_ylabel("count")
        _ax.set_title(_col)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Metric correlations
    """)
    return


@app.cell
def _(cluster_members, mo):
    _candidates = ["b_factor", "edia", "muse_score", "occupancy"]
    _available = [
        c for c in _candidates
        if c in cluster_members.columns and cluster_members[c].notna().any()
    ]
    pairplot_metrics = mo.ui.multiselect(
        options=_available,
        value=[m for m in ["b_factor", "edia", "muse_score"] if m in _available],
        label="median metrics for pairplot (vs cluster_occupancy)",
    )
    pairplot_metrics
    return (pairplot_metrics,)


@app.cell
def _(cluster_members, clusters, pairplot_metrics, plt, sns):
    _metrics = list(pairplot_metrics.value)
    _assigned = cluster_members[cluster_members["within_cutoff"]]

    _medians = (
        _assigned.groupby("cluster_id")[_metrics]
        .median()
        .reset_index()
        .merge(clusters[["cluster_id", "cluster_occupancy"]], on="cluster_id")
        .drop(columns="cluster_id")
        .rename(columns={m: f"median_{m}" for m in _metrics})
    )

    _vars = ["cluster_occupancy"] + [f"median_{m}" for m in _metrics]
    _g = sns.pairplot(_medians.dropna(), vars=_vars, plot_kws={"s": 10, "alpha": 0.3, "edgecolor": "k"}, corner=True)
    plt.tight_layout()
    _g.figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Precision–recall per structure
    """)
    return


@app.cell
def _(mo):
    plot_style = mo.ui.dropdown(
        options=["scatter", "hexbin"], value="scatter", label="plot style"
    )
    axis_range = mo.ui.range_slider(
        start=0.0, stop=1.02, step=0.05, value=[0.0, 1.02],
        label="axis range (equal x/y)", show_value=True,
    )
    point_alpha = mo.ui.slider(
        start=0.1, stop=1.0, step=0.1, value=0.5, label="alpha", show_value=True,
    )
    font_size = mo.ui.slider(
        start=6, stop=20, step=1, value=14, label="font size", show_value=True,
    )
    mo.hstack([plot_style, axis_range, point_alpha, font_size], justify="start")
    return axis_range, font_size, plot_style, point_alpha


@app.cell
def _(
    cluster_members,
    cluster_occupancy_cutoff,
    clusters,
    config,
    consensus_centers,
    metadata,
    pd,
    per_structure_consensus_pr,
):
    # Per-structure precision/recall/f1 vs the consensus centers, plus num_water.
    # Shared by the scatter and the Pareto-front cells below.
    pr_df = per_structure_consensus_pr(
        cluster_members,
        consensus_centers(clusters, cluster_occupancy_cutoff),
        config.CLUSTER_MEMBER_RADIUS,
    ).merge(metadata[["pdb_id", "num_water"]], on="pdb_id", how="left")
    pr_df["num_water"] = pd.to_numeric(pr_df["num_water"], errors="coerce")
    pr_df = pr_df.dropna(subset=["num_water"])
    return (pr_df,)


@app.cell
def _(
    axis_range,
    cluster_occupancy_cutoff,
    font_size,
    plot_pr_scatter,
    plot_style,
    plt,
    point_alpha,
    pr_df,
):
    _fig, _ax = plot_pr_scatter(
        pr_df,
        color="num_water",
        n_color_bins=10,
        style=plot_style.value,
        alpha=point_alpha.value,
        lim=tuple(axis_range.value),
        fontsize=font_size.value,
        title=f"cluster occupancy cutoff = {cluster_occupancy_cutoff}",
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Pareto front & knee

    Precision and recall are both "higher is better," so the non-dominated
    upper-right envelope of the per-structure cloud is a genuine Pareto front,
    with `num_water` sliding you along it. The **knee** (maximum-F1 balance point)
    and its **distance to the ideal (1, 1) corner** give one-number descriptors of
    the front's shape and position — comparable across cohorts.
    """)
    return


@app.cell
def _(
    axis_range,
    font_size,
    np,
    pareto_front,
    pd,
    plot_pr_scatter,
    point_alpha,
    pr_df,
    quantile_boundaries,
):
    # knee = maximum-F1 balance point; distance to the ideal (1, 1) corner
    _knee = pr_df.loc[pr_df["f1"].idxmax()]
    _dist_to_ideal = float(np.hypot(1.0 - _knee["recall"], 1.0 - _knee["precision"]))

    # empirical Pareto front + num_water-binned curve. The curve reuses the exact
    # color-bin edges (10 quantile bins) so one marker sits per colorbar band; the
    # capped 1%/99% tails fold into the end bands via ±inf edges.
    _front = pareto_front(pr_df)
    _edges = quantile_boundaries(pr_df["num_water"], 10).astype(float)
    _edges[0], _edges[-1] = -np.inf, np.inf
    _bins = pd.cut(pr_df["num_water"], bins=_edges).rename("num_water_bin")
    _curve = (
        pr_df.groupby(_bins, observed=True)[["recall", "precision", "num_water"]]
        .mean()
        .sort_values("num_water")
    )

    _fig, _ax = plot_pr_scatter(
        pr_df, color="num_water", n_color_bins=10,
        alpha=point_alpha.value, lim=tuple(axis_range.value),
        fontsize=font_size.value, marker_size=30
        # title=f"cluster occupancy cutoff = {cluster_occupancy_cutoff}",
    )
    _ax.scatter([_knee["recall"]], [_knee["precision"]], marker="*", s=200,
                facecolors="none", edgecolors="r", zorder=5, label="max F1")
    _ax.plot(_front["recall"], _front["precision"], color="r", lw=1.5, label="Pareto front")
    _ax.plot(_curve["recall"], _curve["precision"], color="r", marker="o", ls='--',
             ms=2, lw=1.2, label="average")

    _ax.set_xticks(_ax.get_yticks())
    _ax.set_xlim(tuple(axis_range.value))
    # _ax.legend(fontsize=font_size.value, loc="lower left")

    print(
        f"knee (max F1): recall={_knee['recall']:.2f}, precision={_knee['precision']:.2f}, "
        f"F1max={_knee['f1']:.2f}, num_water*={int(round(_knee['num_water']))}, "
        f"dist_to_ideal={_dist_to_ideal:.2f}"
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Consensus agreement score
    """)
    return


@app.cell
def _(mo):
    score_metric = mo.ui.radio(
        options=["F1", "Chamfer distance"],
        value="F1",
        label="per-structure score vs consensus",
    )
    score_metric
    return (score_metric,)


@app.cell
def _(
    chamfer_distance,
    cluster_members,
    cluster_occupancy_cutoff,
    clusters,
    config,
    consensus_centers,
    np,
    plt,
    precision_recall,
    score_metric,
    sns,
):
    _centers = consensus_centers(clusters, cluster_occupancy_cutoff)

    _scores = []
    _opt_pdb_id = None
    if score_metric.value == "F1":
        _opt_score = 0
    else:
        _opt_score = np.inf
    for _pdb_id, _group in cluster_members.groupby("pdb_id"):
        _coords = _group[["x", "y", "z"]].to_numpy()
        if score_metric.value == "F1":
            _score = precision_recall(_centers, _coords, config.CLUSTER_MEMBER_RADIUS)["f1"]
            _scores.append(_score)
            if _score > _opt_score:
                _opt_pdb_id = _pdb_id
                _opt_score = _score
        else:
            _score = chamfer_distance(_centers, _coords)
            _scores.append(_score)
            if _score < _opt_score:
                _opt_pdb_id = _pdb_id
                _opt_score = _score

    print(f"best pdb ID {_opt_pdb_id} has score {_opt_score}")

    _label = "F1" if score_metric.value == "F1" else "Chamfer distance (Å)"

    _fig, _ax = plt.subplots(figsize=(6, 3))
    sns.histplot(_scores, bins=30, ax=_ax)
    _ax.set_xlabel(_label)
    _ax.set_ylabel("count")
    _ax.set_title(
        f"{_label} per structure vs consensus (occ ≥ {cluster_occupancy_cutoff})"
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## HDBSCAN parameter sweep

    Re-clusters the already-pooled water oxygens (incl. noise) from the loaded
    `cluster_members`, varying HDBSCAN parameters to pick a detection floor from
    data rather than the `0.3 × N` rule. Reuses `cw.cluster.run_hdbscan` +
    `build_cluster_tables`, so every number matches the real pipeline.

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
