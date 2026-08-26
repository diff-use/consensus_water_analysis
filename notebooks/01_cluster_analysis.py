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

    Reads pre-computed `clusters.csv` and `cluster_members.csv`
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
    from matplotlib.colors import LogNorm, Normalize
    import seaborn as sns

    from cw.io import find_cohort_metadata
    from cw.metrics import (
        consensus_centers,
        pareto_front,
        per_structure_consensus_pr,
    )
    from cw.plots import (
        broken_y_axis,
        noise_stats,
        plot_pr_scatter,
        quantile_boundaries,
    )

    return (
        Path,
        broken_y_axis,
        config,
        consensus_centers,
        find_cohort_metadata,
        noise_stats,
        np,
        pareto_front,
        pd,
        per_structure_consensus_pr,
        plot_pr_scatter,
        plt,
        quantile_boundaries,
        sns,
    )


@app.cell
def _(mo):
    # Cohort is a fill-in box, not a file constant, so switching cohorts is a
    # runtime action that never shows up as a git diff (marimo persists the empty
    # default `value=""`, not what you type here).
    cohort_input = mo.ui.text(
        value="hewls_65",
        placeholder="e.g. hewls_65 or carbonicanhydrase_000562_iso or endothiapepsin_000240_iso",
        label="cohort",
        full_width=True,
    )
    cluster_occupancy_label_input = mo.ui.text(
        value="consensus score",
        label="cluster occupancy label",
        full_width=True,
    )
    mo.vstack([cohort_input, cluster_occupancy_label_input])
    return cluster_occupancy_label_input, cohort_input


@app.cell
def _(Path, cohort_input, config, find_cohort_metadata, mo, pd):
    COHORT = cohort_input.value.strip()
    mo.stop(
        not COHORT,
        mo.md("**Enter a cohort in the box above to load its data** — "
              "e.g. `hewls_65` or `carbonicanhydrase_000562_iso`."),
    )
    # Member-radius variant of the clustering CSVs to read. Set to e.g. "0.5" or
    # "1.0" to load the radius-sweep outputs (clusters_<r>.csv /
    # cluster_members_<r>.csv written by find_clustering_hyperparameters.py
    # --radius); None reads the default clusters.csv / cluster_members.csv.
    MEMBER_RADIUS = None
    # The box above names the directory to read verbatim — water-level subsets
    # (`<cohort>_iso`, `<cohort>_bfactor_z2.0water`) are typed out in full rather
    # than derived here, so the notebook never loads a cohort you did not ask for.
    COHORT_DIR = Path(config.DATA_DIR) / COHORT

    _suffix = f"_{MEMBER_RADIUS}" if MEMBER_RADIUS is not None else ""
    clusters = pd.read_csv(COHORT_DIR / f"clusters{_suffix}.csv")
    cluster_members = pd.read_csv(COHORT_DIR / f"cluster_members{_suffix}.csv")

    # Per-structure deposited metadata (resolution, R-free, ...). Water-level
    # subsets do not re-deposit it, so find_cohort_metadata walks up the trailing
    # `_<token>` segments to the parent cohort that has it.
    _meta_path = find_cohort_metadata(config.DATA_DIR, COHORT)
    metadata = pd.read_csv(_meta_path) if _meta_path is not None else None

    # Occupancy above which a cluster counts as consensus — used by the occupancy
    # figure and every per-structure precision/recall computation downstream.
    cluster_occupancy_cutoff = 0.3
    # Radius used downstream to match waters to consensus centers — follows the
    # loaded variant so the analysis is self-consistent; falls back to config.
    match_radius = float(MEMBER_RADIUS) if MEMBER_RADIUS is not None else config.CLUSTER_MEMBER_RADIUS

    PLOTS_DIR = Path("data") / "plots" / COHORT

    print(f"clusters rows:        {len(clusters)}")
    print(f"cluster_members rows: {len(cluster_members)}")
    print(f"match radius:         {match_radius}")
    print(f"cohort directory:     {COHORT_DIR}")
    return (
        PLOTS_DIR,
        cluster_members,
        cluster_occupancy_cutoff,
        clusters,
        match_radius,
    )


@app.cell
def _(cluster_occupancy_label_input):
    cluster_occupancy_label = cluster_occupancy_label_input.value
    return (cluster_occupancy_label,)


@app.cell
def _(mo):
    mo.md("""
    ## Cluster occupancy distribution

    The bars are the raw count per occupancy bin: *clusters* (gray) plus *noise*
    waters (brown, a single pile at occupancy `1 / n_structures`). The right
    axis is **broken** — the top panel zooms the noise pile, the bottom panel
    the cluster-count detail — because noise dwarfs the per-bin cluster counts.
    The long tail of tiny low-occupancy clusters dominates, which is what makes
    non-consensus look prevalent. The black line (right axis) is member-weighted:
    the reverse-cumulative fraction of all *waters* at occupancy ≥ x (noise
    included). It starts at 100% and falls as the score rises, so its height at
    the cutoff is the consensus water fraction — showing those many small
    low-occupancy clusters actually hold only a minority of waters.
    """)
    return


@app.cell
def _(PLOTS_DIR, mo):
    water_occ_font_size = mo.ui.slider(
        start=6, stop=20, step=1, value=14, label="font size", show_value=True,
    )
    water_occ_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "cluster_occupancy_water_weighted.png"),
        label="save path", full_width=True,
    )
    water_occ_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    water_occ_save_button = mo.ui.run_button(label="save figure")
    mo.vstack([
        mo.hstack([water_occ_font_size], justify="start"),
        mo.hstack([water_occ_save_path, water_occ_save_dpi, water_occ_save_button], justify="start"),
    ])
    return (
        water_occ_font_size,
        water_occ_save_button,
        water_occ_save_dpi,
        water_occ_save_path,
    )


@app.cell
def _(
    Path,
    broken_y_axis,
    cluster_members,
    cluster_occupancy_cutoff,
    cluster_occupancy_label,
    clusters,
    noise_stats,
    np,
    plt,
    water_occ_font_size,
    water_occ_save_button,
    water_occ_save_dpi,
    water_occ_save_path,
):
    _fs = water_occ_font_size.value
    _tick_font_size = 14

    if len(clusters) == 0:
        _fig, _ax = plt.subplots(figsize=(3.5, 3))
        _ax.text(0.5, 0.5, "No clusters", transform=_ax.transAxes, ha="center", fontsize=_fs)
        plt.tight_layout()
    else:
        _members = (
            cluster_members[cluster_members["within_cutoff"]]
            .groupby("cluster_id").size().rename("n_members")
        )
        _occ = clusters[["cluster_id", "cluster_occupancy"]].merge(_members, on="cluster_id")

        _n_structures, _noise_occ, _noise_n = noise_stats(cluster_members)

        _edges = np.linspace(0, 1, 21)
        _width = 0.9 * (_edges[1] - _edges[0])
        _centers = (_edges[:-1] + _edges[1:]) / 2
        # left axis: raw number of clusters per occupancy bin
        _cluster_hist, _ = np.histogram(clusters["cluster_occupancy"], bins=_edges)

        _water_per_bin, _ = np.histogram(_occ["cluster_occupancy"], bins=_edges, weights=_occ["n_members"])
        _noise_per_bin, _ = np.histogram([_noise_occ], bins=_edges, weights=[_noise_n])

        # right axis: reverse cumulative over ALL waters — 100% at low score,
        # falling as the score rises. Denominator is every water oxygen
        # (len(cluster_members)); above the noise bin the curve counts only
        # clustered waters within the radius cutoff (n_members), so every
        # non-clustered water (HDBSCAN noise AND radius-rejected members) is
        # lumped into the lowest bin, where it only sets the 100% start and never
        # contributes at higher scores.
        _n_total = len(cluster_members)
        _n_clustered = int(cluster_members["within_cutoff"].sum())
        _rev_counts = _water_per_bin.astype(float).copy()
        _rev_counts[0] += _n_total - _n_clustered
        _cum_reverse = _rev_counts[::-1].cumsum()[::-1] / _n_total
        # dashed marker: clustered within-cutoff waters at or above the consensus
        # cutoff, over all waters — matches the curve's value at the cutoff.
        _frac_ge_cutoff = (
            _occ.loc[_occ["cluster_occupancy"] >= cluster_occupancy_cutoff, "n_members"].sum() / _n_total
        )

        _cluster_max = int(_cluster_hist.max())
        _noise_bin = int(_noise_per_bin.argmax())
        _noise_top = int(_cluster_hist[_noise_bin] + _noise_n)

        # Broken #clusters axis (right): a small top panel zooms the noise pile,
        # the larger bottom panel shows the cluster-count detail. A full-height
        # overlay carries the reverse-cumulative water line (right) so its 100%
        # ceiling lines up with the top of the noise pile.
        _fig, (_ax_top, _ax_bot) = plt.subplots(
            2, 1, sharex=True, figsize=(3.3, 3),
            gridspec_kw={"height_ratios": [1, 3], "hspace": 0.1},
        )
        _fig.subplots_adjust(left=0.17, right=0.83, bottom=0.2, top=0.9)

        for _axis in (_ax_top, _ax_bot):
            _axis.bar(_centers, _cluster_hist, width=_width, color="gray", alpha=0.5)
            _axis.bar(_centers, _noise_per_bin, width=_width, bottom=_cluster_hist,
                      color="lightgrey", alpha=0.7)
            _axis.tick_params(axis="y", colors="gray", labelsize=_tick_font_size)
            # the right spine is carried by the full-height cumulative overlay
            # below; hiding it here removes the hspace gap between the two panels
            _axis.spines["right"].set_visible(False)

        # top panel ends exactly at the noise-pile top so the curve's 100% meets it
        _ax_top.set_ylim(_noise_top * 0.85, _noise_top)
        _ax_bot.set_ylim(0, _cluster_max * 1.25)
        # the bottom panel is 3x taller (height_ratios [1, 3]) but the default
        # tick locator gives it a similar tick count to the top, so its ticks look
        # sparse — bump its tick count to match the top panel's visual density.
        _ax_bot.locator_params(axis="y", nbins=8)
        _ax_bot.set_ylabel("#cluster sites", fontsize=_fs, color="gray")
        _ax_bot.set_xlabel(f"{cluster_occupancy_label}", fontsize=_fs)
        _ax_bot.tick_params(axis="x", labelsize=_tick_font_size)

        # diagonal break marks between the two panels — left side only, so the
        # cumulative water% axis on the right stays continuous (unbroken)
        broken_y_axis(_ax_top, _ax_bot, right=False, d=0.025, linewidth=1.5, color="gray")

        # full-height overlay for the reverse-cumulative water fraction,
        # foregrounded; spans both panels so 100% sits at the noise-pile ceiling
        _pt = _ax_top.get_position()
        _pb = _ax_bot.get_position()
        _ax_cum = _fig.add_axes([_pb.x0, _pb.y0, _pb.width, _pt.y1 - _pb.y0])
        _ax_cum.set_xlim(_ax_bot.get_xlim())
        _ax_cum.set_ylim(0, 1)
        _ax_cum.plot(_centers, _cum_reverse, color="k", lw=2.5)
        _ax_cum.set_ylabel("water% above score", fontsize=_fs, color="k")
        _ax_cum.set_yticks(np.linspace(0, 1, 6))
        _ax_cum.set_yticklabels([f"{int(_t * 100)}" for _t in np.linspace(0, 1, 6)])
        _ax_cum.tick_params(axis="y", colors="k", labelsize=_tick_font_size)
        # dashed guides meeting at the curve point (cutoff, frac_ge_cutoff): the
        # vertical drops from the curve down to the x-axis, the horizontal runs
        # from the curve out to the right y-axis where the value is read.
        _x_right = _ax_cum.get_xlim()[1]
        _ax_cum.plot([cluster_occupancy_cutoff, cluster_occupancy_cutoff], [0, _frac_ge_cutoff],
                     color="C0", linestyle="--", linewidth=1)
        _ax_cum.plot([cluster_occupancy_cutoff, _x_right], [_frac_ge_cutoff, _frac_ge_cutoff],
                     color="k", linestyle="--", linewidth=1)
        _ax_cum.patch.set_visible(False)
        _ax_cum.xaxis.set_visible(False)
        _ax_cum.yaxis.tick_right()
        _ax_cum.yaxis.set_label_position("right")
        for _s in ("top", "left", "bottom"):
            _ax_cum.spines[_s].set_visible(False)

    if water_occ_save_button.value:
        _out = Path(water_occ_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=int(water_occ_save_dpi.value), bbox_inches="tight")
        print(f"saved to {_out}")

    _fig
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
    ## Metric correlations
    """)
    return


@app.cell
def _(cluster_members, mo):
    _candidates = ["b_factor", "edia", "occupancy", "b_factor_zscore"]
    _available = [
        c for c in _candidates
        if c in cluster_members.columns and cluster_members[c].notna().any()
    ]
    pairplot_metrics = mo.ui.multiselect(
        options=_available,
        value=[m for m in ["b_factor_zscore", "edia"] if m in _available],
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
    return axis_range, font_size, point_alpha


@app.cell
def _(
    cluster_members,
    cluster_occupancy_cutoff,
    clusters,
    consensus_centers,
    match_radius,
    per_structure_consensus_pr,
):
    # Per-structure precision/recall/f1 vs the consensus centers, plus num_water.
    # num_water comes straight from cluster_members (each structure's pooled water
    # count = its prediction-set size), so it matches the waters actually clustered
    # for this cohort — not a deposited metadata total — and needs no metadata.csv.
    # Shared by the scatter and the Pareto-front cells below.
    pr_df = per_structure_consensus_pr(
        cluster_members,
        consensus_centers(clusters, cluster_occupancy_cutoff),
        match_radius,
    )
    return (pr_df,)


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
        pr_df, color="num_water", color_label="#water", n_color_bins=10,
        alpha=point_alpha.value, lim=tuple(axis_range.value),
        fontsize=font_size.value, marker_size=30
        # title=f"cluster occupancy cutoff = {cluster_occupancy_cutoff}",
    )
    _ax.scatter([_knee["recall"]], [_knee["precision"]], marker="*", s=350,
                facecolors="r", edgecolors="white", lw=1,
                zorder=5, label="max F1")
    # _ax.plot(_front["recall"], _front["precision"], color="white", lw=2, label="Pareto front")
    _ax.plot(_front["recall"], _front["precision"], color="red", ls='-', lw=3, label="Pareto front")
    # _ax.plot(_curve["recall"], _curve["precision"], color="r", marker="o", ls='--',
    #          ms=2, lw=1.2, label="average")

    _ax.set_xticks(_ax.get_yticks())
    _ax.set_xlim(tuple(axis_range.value))
    # _ax.legend(fontsize=font_size.value, loc="lower left")

    print(
        f"knee (max F1): recall={_knee['recall']:.2f}, precision={_knee['precision']:.2f}, "
        f"F1max={_knee['f1']:.2f}, num_water*={int(round(_knee['num_water']))}, "
        f"dist_to_ideal={_dist_to_ideal:.2f}"
    )

    _lo, _hi = _curve.iloc[0], _curve.iloc[-1]
    print(
        f"binned curve low end  (num_water={int(round(_lo['num_water']))}): "
        f"recall={_lo['recall']:.2f}, precision={_lo['precision']:.2f}"
    )
    print(
        f"binned curve high end (num_water={int(round(_hi['num_water']))}): "
        f"recall={_hi['recall']:.2f}, precision={_hi['precision']:.2f}"
    )

    if pareto_save_button.value:
        _out = Path(pareto_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=int(pareto_save_dpi.value), bbox_inches="tight")
        print(f"saved to {_out}")

    _fig
    return


if __name__ == "__main__":
    app.run()
