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
    for one cohort, then explores cluster occupancy, water metrics, and
    per-structure agreement with the consensus. Re-run
    `scripts/cluster_waters.py` to refresh the CSVs. HDBSCAN parameter
    sensitivity lives in `optional_cluster_hyperparameter_check.py`.
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

    from cw.metrics import (
        consensus_centers,
        pareto_front,
        per_structure_consensus_chamfer,
        per_structure_consensus_pr,
    )
    from cw.plots import plot_pr_scatter, quantile_boundaries

    return (
        Path,
        config,
        consensus_centers,
        np,
        pareto_front,
        pd,
        per_structure_consensus_chamfer,
        per_structure_consensus_pr,
        plot_pr_scatter,
        plt,
        quantile_boundaries,
        sns,
    )


@app.cell
def _(Path, config, pd):
    COHORT = "hewls_65" #"carbonicanhydrase_000562" #"hewls_65"
    # Member-radius variant of the clustering CSVs to read. Set to e.g. "0.5" or
    # "1.0" to load the radius-sweep outputs (clusters_<r>.csv /
    # cluster_members_<r>.csv written by find_clustering_hyperparameters.py
    # --radius); None reads the default clusters.csv / cluster_members.csv.
    MEMBER_RADIUS = 0.5 #None
    DATA = Path(config.DATA_DIR) / COHORT
    SUBSET = Path(config.DATA_DIR) / Path(COHORT + "_iso")
    if not SUBSET.exists():
        SUBSET = DATA

    _suffix = f"_{MEMBER_RADIUS}" if MEMBER_RADIUS is not None else ""
    clusters = pd.read_csv(SUBSET / f"clusters{_suffix}.csv")
    cluster_members = pd.read_csv(SUBSET / f"cluster_members{_suffix}.csv")
    metadata = pd.read_csv(DATA / "metadata.csv")

    # Radius used downstream to match waters to consensus centers — follows the
    # loaded variant so the analysis is self-consistent; falls back to config.
    match_radius = float(MEMBER_RADIUS) if MEMBER_RADIUS is not None else config.CLUSTER_MEMBER_RADIUS

    PLOTS_DIR = Path("data") / "plots" / COHORT

    print(f"clusters rows:        {len(clusters)}")
    print(f"cluster_members rows: {len(cluster_members)}")
    return PLOTS_DIR, cluster_members, clusters, match_radius, metadata


@app.cell
def _(mo):
    mo.md("""
    ## Cluster occupancy distribution
    """)
    return


@app.cell
def _(PLOTS_DIR, mo):
    include_noise_toggle = mo.ui.checkbox(value=True, label="include noise waters")
    occ_font_size = mo.ui.slider(
        start=6, stop=20, step=1, value=16, label="font size", show_value=True,
    )
    occ_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "cluster_occupancy.png"),
        label="save path", full_width=True,
    )
    occ_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    occ_save_button = mo.ui.run_button(label="save figure")
    mo.vstack([
        mo.hstack([include_noise_toggle, occ_font_size], justify="start"),
        mo.hstack([occ_save_path, occ_save_dpi, occ_save_button], justify="start"),
    ])
    return (
        include_noise_toggle,
        occ_font_size,
        occ_save_button,
        occ_save_dpi,
        occ_save_path,
    )


@app.cell
def _(
    Path,
    cluster_members,
    clusters,
    include_noise_toggle,
    np,
    occ_font_size,
    occ_save_button,
    occ_save_dpi,
    occ_save_path,
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

    if occ_save_button.value:
        _out = Path(occ_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=int(occ_save_dpi.value), bbox_inches="tight")
        print(f"saved to {_out}")

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
def _(PLOTS_DIR, mo):
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
    pr_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "precision_recall.png"),
        label="save path", full_width=True,
    )
    pr_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    pr_save_button = mo.ui.run_button(label="save figure")
    mo.vstack([
        mo.hstack([plot_style, axis_range, point_alpha, font_size], justify="start"),
        mo.hstack([pr_save_path, pr_save_dpi, pr_save_button], justify="start"),
    ])
    return (
        axis_range,
        font_size,
        plot_style,
        point_alpha,
        pr_save_button,
        pr_save_dpi,
        pr_save_path,
    )


@app.cell
def _(
    cluster_members,
    cluster_occupancy_cutoff,
    clusters,
    consensus_centers,
    match_radius,
    metadata,
    pd,
    per_structure_consensus_pr,
):
    # Per-structure precision/recall/f1 vs the consensus centers, plus num_water.
    # Shared by the scatter and the Pareto-front cells below.
    pr_df = per_structure_consensus_pr(
        cluster_members,
        consensus_centers(clusters, cluster_occupancy_cutoff),
        match_radius,
    ).merge(metadata[["pdb_id", "num_water"]], on="pdb_id", how="left")
    pr_df["num_water"] = pd.to_numeric(pr_df["num_water"], errors="coerce")
    pr_df = pr_df.dropna(subset=["num_water"])
    return (pr_df,)


@app.cell
def _(
    Path,
    axis_range,
    cluster_occupancy_cutoff,
    font_size,
    plot_pr_scatter,
    plot_style,
    plt,
    point_alpha,
    pr_df,
    pr_save_button,
    pr_save_dpi,
    pr_save_path,
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

    if pr_save_button.value:
        _out = Path(pr_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=int(pr_save_dpi.value), bbox_inches="tight")
        print(f"saved to {_out}")

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
def _(PLOTS_DIR, mo):
    pareto_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "precision_recall_pareto.png"),
        label="save path", full_width=True,
    )
    pareto_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    pareto_save_button = mo.ui.run_button(label="save figure")
    mo.hstack([pareto_save_path, pareto_save_dpi, pareto_save_button], justify="start")
    return pareto_save_button, pareto_save_dpi, pareto_save_path


@app.cell
def _(
    Path,
    axis_range,
    font_size,
    np,
    pareto_front,
    pareto_save_button,
    pareto_save_dpi,
    pareto_save_path,
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

    if pareto_save_button.value:
        _out = Path(pareto_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=int(pareto_save_dpi.value), bbox_inches="tight")
        print(f"saved to {_out}")

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
    cluster_members,
    cluster_occupancy_cutoff,
    clusters,
    consensus_centers,
    match_radius,
    per_structure_consensus_chamfer,
    per_structure_consensus_pr,
    plt,
    score_metric,
    sns,
):
    _centers = consensus_centers(clusters, cluster_occupancy_cutoff)

    if score_metric.value == "F1":
        _col, _label = "f1", "F1"
        _scores = per_structure_consensus_pr(
            cluster_members, _centers, match_radius
        )
        _best = _scores.loc[_scores[_col].idxmax()]
    else:
        _col, _label = "chamfer", "Chamfer distance (Å)"
        _scores = per_structure_consensus_chamfer(cluster_members, _centers)
        _best = _scores.loc[_scores[_col].idxmin()]

    print(f"best pdb ID {_best['pdb_id']} has score {_best[_col]}")

    _fig, _ax = plt.subplots(figsize=(6, 3))
    sns.histplot(_scores[_col], bins=30, ax=_ax)
    _ax.set_xlabel(_label)
    _ax.set_ylabel("count")
    _ax.set_title(
        f"{_label} per structure vs consensus (occ ≥ {cluster_occupancy_cutoff})"
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
