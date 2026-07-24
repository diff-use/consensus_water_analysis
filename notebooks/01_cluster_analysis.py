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
        LogNorm,
        Normalize,
        Path,
        broken_y_axis,
        config,
        consensus_centers,
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
        placeholder="e.g. hewls_65 or carbonicanhydrase_000562 or endothiapepsin_000240",
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
def _(Path, cohort_input, config, mo, pd):
    COHORT = cohort_input.value.strip()
    mo.stop(
        not COHORT,
        mo.md("**Enter a cohort in the box above to load its data** — "
              "e.g. `hewls_65` or `carbonicanhydrase_000562`."),
    )
    # Member-radius variant of the clustering CSVs to read. Set to e.g. "0.5" or
    # "1.0" to load the radius-sweep outputs (clusters_<r>.csv /
    # cluster_members_<r>.csv written by find_clustering_hyperparameters.py
    # --radius); None reads the default clusters.csv / cluster_members.csv.
    MEMBER_RADIUS = None
    DATA = Path(config.DATA_DIR) / COHORT
    subset_suffix = "_iso" #"_bfactor_z2.0water" #"_iso"
    SUBSET = Path(config.DATA_DIR) / (COHORT + subset_suffix) #/ "min_cluster_size_5_min_samples_5"
    if not SUBSET.exists():
        SUBSET = DATA
    else: 
        COHORT += subset_suffix

    _suffix = f"_{MEMBER_RADIUS}" if MEMBER_RADIUS is not None else ""
    clusters = pd.read_csv(SUBSET / f"clusters{_suffix}.csv")
    cluster_members = pd.read_csv(SUBSET / f"cluster_members{_suffix}.csv")

    # Per-structure deposited metadata (resolution, R-free, ...). Water-level
    # subsets share the parent cohort's metadata.csv, so fall back to DATA.
    _meta_path = SUBSET / "metadata.csv"
    if not _meta_path.exists():
        _meta_path = DATA / "metadata.csv"
    metadata = pd.read_csv(_meta_path) if _meta_path.exists() else None

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
    print(f"SUBSET: {SUBSET}")
    return (
        PLOTS_DIR,
        cluster_members,
        cluster_occupancy_cutoff,
        clusters,
        match_radius,
        metadata,
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
    _candidates = ["b_factor", "edia", "muse_score", "occupancy", "b_factor_zscore"]
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
    ### All waters' metric vs cluster occupancy

    Every water is binned at the occupancy of the cluster it belongs to (noise
    excluded via `within_cutoff`) as a **hexbin** density map, showing the full
    within-cluster spread of each metric against occupancy without the
    overplotting a plain scatter suffers here. Enter comma-separated reference
    values per metric to draw horizontal dashed guide lines (e.g. `0.4, 0.6, 0.8`
    for EDIA); tune gridsize and the log color scale to read the density. Panels
    share a single count colorbar, and the y-axis can be clamped to each metric's
    1–99 percentile so extreme outliers don't dominate the range.
    """)
    return


@app.cell
def _(PLOTS_DIR, mo, pairplot_metrics):
    _default_refs = {"b_factor_zscore": "1.0, 1.5, 2.0", "edia": "0.4, 0.6, 0.8"}
    occ_metric_refs = mo.ui.dictionary({
        _m: mo.ui.text(value=_default_refs.get(_m, ""), label=f"{_m} reference values", full_width=True)
        for _m in pairplot_metrics.value
    })
    water_metric_gridsize = mo.ui.slider(
        start=10, stop=80, step=5, value=30, label="hexbin gridsize", show_value=True,
    )
    water_metric_log = mo.ui.checkbox(value=True, label="log color scale")
    water_metric_clamp = mo.ui.checkbox(value=True, label="clamp y to 0.1–99.9 pct")
    water_metric_font_size = mo.ui.slider(
        start=6, stop=24, step=1, value=14, label="font size", show_value=True,
    )
    _metric_slug = pairplot_metrics.value[0] if len(pairplot_metrics.value) == 1 else "metrics"
    water_metric_save_path = mo.ui.text(
        value=str(PLOTS_DIR / f"occupancy_vs_{_metric_slug}_allwaters.png"),
        label="save path", full_width=True,
    )
    water_metric_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    water_metric_save_button = mo.ui.run_button(label="save figure")
    mo.vstack([
        mo.hstack([water_metric_gridsize, water_metric_log, water_metric_clamp, water_metric_font_size], justify="start"),
        occ_metric_refs,
        mo.hstack([water_metric_save_path, water_metric_save_dpi, water_metric_save_button], justify="start"),
    ])
    return (
        occ_metric_refs,
        water_metric_clamp,
        water_metric_font_size,
        water_metric_gridsize,
        water_metric_log,
        water_metric_save_button,
        water_metric_save_dpi,
        water_metric_save_path,
    )


@app.cell
def _(
    LogNorm,
    Normalize,
    Path,
    cluster_members,
    cluster_occupancy_label,
    clusters,
    occ_metric_refs,
    pairplot_metrics,
    plt,
    water_metric_clamp,
    water_metric_font_size,
    water_metric_gridsize,
    water_metric_log,
    water_metric_save_button,
    water_metric_save_dpi,
    water_metric_save_path,
):
    _metrics = list(pairplot_metrics.value)
    metric2label = {_m: _m for _m in _metrics}
    metric2label["edia"] = "EDIA"
    metric2label["b_factor_zscore"] = "B-factor z-score"
    _fs = water_metric_font_size.value
    _waters = (
        cluster_members[cluster_members["within_cutoff"]]
        .merge(clusters[["cluster_id", "cluster_occupancy"]], on="cluster_id")
    )

    _n = max(len(_metrics), 1)
    _fig, _axes = plt.subplots(1, _n, figsize=(4 * _n, 3), squeeze=False, constrained_layout=True)
    _axes = _axes[0]

    # First pass: draw every hexbin (clamping the y-axis to the 1–99 percentile
    # so a few extreme values don't dominate the range), tracking the global max
    # count so all panels can share one normalization and one colorbar.
    _hexbins = []
    _vmax = 1
    for _ax, _metric in zip(_axes, _metrics):
        _sub = _waters[["cluster_occupancy", _metric]].dropna()
        if water_metric_clamp.value and len(_sub):
            _lo, _hi = _sub[_metric].quantile([0.001, 0.999])
            _sub = _sub[_sub[_metric].between(_lo, _hi)]
        _hb = _ax.hexbin(
            _sub["cluster_occupancy"], _sub[_metric],
            gridsize=int(water_metric_gridsize.value),
            mincnt=1, cmap="viridis",
            alpha=0.75
        )
        _hexbins.append(_hb)
        if _hb.get_array().size:
            _vmax = max(_vmax, float(_hb.get_array().max()))

        _ax.set_xlabel(f"{cluster_occupancy_label}", fontsize=_fs)
        _ax.set_ylabel(metric2label[_metric], fontsize=_fs)
        _ax.tick_params(labelsize=_fs)

        _yticks = {
            "b_factor_zscore": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            "edia": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        }.get(_metric)
        if _yticks is not None:
            _ax.set_yticks(_yticks)
            _ax.set_yticklabels([f"{_t:.1f}" for _t in _yticks])

        _refs = []
        for _tok in occ_metric_refs.value.get(_metric, "").replace(",", " ").split():
            try:
                _refs.append(float(_tok))
            except ValueError:
                pass
        for _rv in _refs:
            # _ax.axhline(_rv, color="r", linestyle="-", linewidth=3)
            _ax.axhline(_rv, color="k", linestyle="--", linewidth=3)

    # Shared normalization across panels, then a single colorbar for the figure.
    _norm = LogNorm(vmin=1, vmax=_vmax) if water_metric_log.value else Normalize(vmin=0, vmax=_vmax)
    for _hb in _hexbins:
        _hb.set_norm(_norm)
    _cb = _fig.colorbar(_hexbins[-1], ax=list(_axes), pad=0.005)
    _cb.set_label("count", fontsize=_fs)
    _cb.ax.tick_params(labelsize=_fs)

    if water_metric_save_button.value:
        _out = Path(water_metric_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=int(water_metric_save_dpi.value), bbox_inches="tight")
        print(f"saved to {_out}")

    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### Consensus vs non-consensus water quality

    The premise behind every filter, tested at the water level: are the waters that
    landed in a conserved site actually higher-quality than the ones that didn't?
    Each water is labelled **consensus** (a within-cutoff member of a cluster whose
    occupancy ≥ the consensus cutoff) or **non-consensus** (noise, radius-rejected,
    or a member of a low-occupancy cluster), then the per-metric distributions of the
    two groups are overlaid — each density normalized separately, since the groups
    differ hugely in size. Dashed lines mark the reference values entered above. The
    x-axis can be clamped to the 99.9 percentile so a few extreme values don't
    stretch the range.
    """)
    return


@app.cell
def _(PLOTS_DIR, mo):
    quality_clamp = mo.ui.checkbox(value=True, label="clamp x to 99.9 pct")
    quality_vertical = mo.ui.checkbox(value=True, label="vertical layout (uncheck for horizontal)")
    quality_font_size = mo.ui.slider(
        start=6, stop=24, step=1, value=14, label="font size", show_value=True,
    )
    quality_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "consensus_vs_nonconsensus_metric_distribution.png"),
        label="save path", full_width=True,
    )
    quality_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    quality_save_button = mo.ui.run_button(label="save figure")
    mo.vstack([
        mo.hstack([quality_clamp, quality_vertical, quality_font_size], justify="start"),
        mo.hstack([quality_save_path, quality_save_dpi, quality_save_button], justify="start"),
    ])
    return (
        quality_clamp,
        quality_font_size,
        quality_save_button,
        quality_save_dpi,
        quality_save_path,
        quality_vertical,
    )


@app.cell
def _(
    Path,
    cluster_members,
    cluster_occupancy_cutoff,
    clusters,
    occ_metric_refs,
    pairplot_metrics,
    plt,
    quality_clamp,
    quality_font_size,
    quality_save_button,
    quality_save_dpi,
    quality_save_path,
    quality_vertical,
    sns,
):
    _metrics = list(pairplot_metrics.value)
    _label = {"edia": "EDIA", "b_factor_zscore": "B-factor z-score"}

    # conserved = within-cutoff member of a cluster whose occupancy clears the
    # cutoff; everything else (noise, radius-rejected, low-occupancy members) is
    # not. Matches the labelling in the occupancy figure. The boolean column name
    # becomes the legend title ("conserved") with True/False entries.
    _conserved_ids = set(
        clusters.loc[clusters["cluster_occupancy"] >= cluster_occupancy_cutoff, "cluster_id"]
    )
    _labelled = cluster_members.assign(
        conserved=cluster_members["within_cutoff"]
        & cluster_members["cluster_id"].isin(_conserved_ids)
    )

    _fs = quality_font_size.value
    _n = max(len(_metrics), 1)

    # Pin the data rectangle in inches so it is identical across cohorts: each
    # panel is PANEL_W x PANEL_H, with fixed label gutters. tight_layout /
    # bbox_inches="tight" would instead resize the rectangle to fit each cohort's
    # tick labels, which is why the same figure came out at different sizes.
    _panel_w, _panel_h = 2.4, 2.2
    _left, _right, _top, _bottom, _gap = 0.75, 0.2, 0.2, 0.6, 0.6
    if quality_vertical.value:
        _fig_w = _left + _panel_w + _right
        _fig_h = _bottom + _n * _panel_h + (_n - 1) * _gap + _top
        _fig, _axes = plt.subplots(_n, 1, figsize=(_fig_w, _fig_h), squeeze=False)
        _axes = _axes[:, 0]
        _fig.subplots_adjust(
            left=_left / _fig_w, right=1 - _right / _fig_w,
            bottom=_bottom / _fig_h, top=1 - _top / _fig_h,
            hspace=_gap / _panel_h,
        )
    else:
        _fig_w = _left + _n * _panel_w + (_n - 1) * _gap + _right
        _fig_h = _bottom + _panel_h + _top
        _fig, _axes = plt.subplots(1, _n, figsize=(_fig_w, _fig_h), squeeze=False)
        _axes = _axes[0]
        _fig.subplots_adjust(
            left=_left / _fig_w, right=1 - _right / _fig_w,
            bottom=_bottom / _fig_h, top=1 - _top / _fig_h,
            wspace=_gap / _panel_w,
        )

    for _ax, _metric in zip(_axes, _metrics):
        _data = _labelled.dropna(subset=[_metric])
        if quality_clamp.value and len(_data):
            _hi = _data[_metric].quantile(0.999)
            _data = _data[_data[_metric] <= _hi]
        sns.histplot(
            data=_data, x=_metric, hue="conserved",
            stat="density", common_norm=False,
            element="step", fill=True, alpha=0.5, ax=_ax,
            hue_order=[True, False],
            # palette="Dark2",
            palette={True: "r", False: "grey"},
            legend=False,
        )
        _refs = []
        for _tok in occ_metric_refs.value.get(_metric, "").replace(",", " ").split():
            try:
                _refs.append(float(_tok))
            except ValueError:
                pass
        for _rv in _refs:
            _ax.axvline(_rv, color="k", ls="--", lw=1.5)
            # _ax.text(_rv, _ax.get_ylim()[1] * 0.98, f"{_rv:g}", rotation=90,
            #          va="top", ha="right", fontsize=_fs - 3, color="0.4")
        _ax.set_xlabel(_label.get(_metric, _metric), fontsize=_fs)
        _ax.set_ylabel("density", fontsize=_fs)
        _ax.tick_params(labelsize=_fs)

        # Stats on the full (unclamped) data so min/max reflect the real
        # distribution, not the display clamp. Grouped by consensus label.
        _summary = (
            _labelled.dropna(subset=[_metric])
            .groupby("conserved")[_metric]
            .agg(["min", "median", "mean", "std", "max", "count"])
        )
        print(f"[water] {_metric}:")
        for _grp, _row in _summary.iterrows():
            _name = "consensus" if _grp else "non-consensus"
            print(
                f"  {_name:>13}: min={_row['min']:.4g} median={_row['median']:.4g} "
                f"mean={_row['mean']:.4g} std={_row['std']:.4g} max={_row['max']:.4g} "
                f"n={int(_row['count'])}"
            )

        # Fraction of each group filtered OUT at each reference threshold (the
        # vertical dashed lines), matching cw.filter's inclusive defaults: a water
        # exactly on the cutoff is KEPT, so the drop test is strict. EDIA keeps
        # EDIAm >= cutoff, so it drops edia < cutoff; B-factor (z-score or raw)
        # keeps b <= cutoff, so it drops b > cutoff. Same full unclamped data as
        # the stats above. Metrics without a defined filter direction are skipped.
        _drop_below = {"edia"}
        _drop_above = {"b_factor_zscore", "b_factor"}
        _full = _labelled.dropna(subset=[_metric])
        for _rv in _refs:
            if _metric in _drop_below:
                _dropped = _full[_metric] < _rv
                _op = "<"
            elif _metric in _drop_above:
                _dropped = _full[_metric] > _rv
                _op = ">"
            else:
                continue
            print(f"  filtered out ({_metric} {_op} {_rv:g}):")
            for _grp, _idx in _full.groupby("conserved").groups.items():
                _name = "consensus" if _grp else "non-consensus"
                _g = _dropped.loc[_idx]
                print(f"    {_name:>13}: {_g.mean():.4f} (n={len(_g)})")

    if quality_save_button.value:
        _out = Path(quality_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=int(quality_save_dpi.value))
        print(f"saved to {_out}")

    _fig
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


@app.cell
def _(mo):
    mo.md("""
    ## Good vs poor structures — metadata distributions

    The mirror of the consensus/non-consensus water plot, but at the *structure*
    level. Each structure is labelled **good** or **poor** by whether a chosen
    per-structure metric (default `f1`, the structure's agreement with the
    consensus) is **at or above** a cutoff, then the deposited-metadata
    distributions of the two groups are overlaid (default `resolution` and
    `deposited_r_free`) — each density normalized separately, since the groups
    differ in size. This asks whether the structures that best reproduce the
    conserved sites are also the higher-quality depositions. The split metric can
    be any per-structure column (precision/recall/f1/num_water from the
    precision–recall analysis, or a metadata column); the x-axis can be clamped to
    the 0.1–99.9 percentile so a few extreme values don't stretch the range.
    """)
    return


@app.cell
def _(metadata, pd, pr_df):
    # One row per structure: precision/recall/f1/num_water joined to deposited
    # metadata. metadata's own num_water (deposited count) is kept as
    # num_water_deposited so pr_df's clustered count stays the plain `num_water`.
    # Metadata scalars can carry "<missing>" strings, so coerce non-id columns to
    # numeric; string columns (ligand_names, space_group, ...) become all-NaN and
    # drop out of the numeric-column menus below.
    if metadata is None:
        structure_metrics = pr_df.copy()
    else:
        _meta = metadata.copy()
        for _c in _meta.columns:
            if _c != "pdb_id":
                _meta[_c] = pd.to_numeric(_meta[_c], errors="coerce")
        structure_metrics = pr_df.merge(
            _meta, on="pdb_id", how="left", suffixes=("", "_deposited")
        )

    structure_numeric_cols = [
        _c for _c in structure_metrics.columns
        if _c != "pdb_id"
        and pd.api.types.is_numeric_dtype(structure_metrics[_c])
        and structure_metrics[_c].notna().any()
    ]
    return structure_metrics, structure_numeric_cols


@app.cell
def _(PLOTS_DIR, mo, structure_numeric_cols):
    good_poor_split_metric = mo.ui.dropdown(
        options=structure_numeric_cols,
        value="f1" if "f1" in structure_numeric_cols else structure_numeric_cols[0],
        label="split metric (good = ≥ cutoff)",
    )
    _dist_default = [
        _m for _m in ["resolution", "deposited_r_free"] if _m in structure_numeric_cols
    ]
    good_poor_dist_metrics = mo.ui.multiselect(
        options=structure_numeric_cols,
        value=_dist_default or structure_numeric_cols[:1],
        label="distribution metrics",
    )
    good_poor_clamp = mo.ui.checkbox(value=True, label="clamp x to 0.1–99.9 pct")
    good_poor_vertical = mo.ui.checkbox(value=True, label="vertical layout (uncheck for horizontal)")
    good_poor_font_size = mo.ui.slider(
        start=6, stop=24, step=1, value=14, label="font size", show_value=True,
    )
    good_poor_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "good_vs_poor_metadata_distribution.png"),
        label="save path", full_width=True,
    )
    good_poor_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    good_poor_save_button = mo.ui.run_button(label="save figure")
    mo.vstack([
        mo.hstack([good_poor_split_metric, good_poor_dist_metrics], justify="start"),
        mo.hstack([good_poor_clamp, good_poor_vertical, good_poor_font_size], justify="start"),
        mo.hstack([good_poor_save_path, good_poor_save_dpi, good_poor_save_button], justify="start"),
    ])
    return (
        good_poor_clamp,
        good_poor_dist_metrics,
        good_poor_font_size,
        good_poor_save_button,
        good_poor_save_dpi,
        good_poor_save_path,
        good_poor_split_metric,
        good_poor_vertical,
    )


@app.cell
def _(good_poor_split_metric, mo, structure_metrics):
    # Cutoff defaults to the median of the selected split metric; the stats line
    # (min / median / mean / max) is shown so a different cutoff can be picked by
    # hand. Defined in its own cell so it re-defaults to the median whenever the
    # split metric changes, without resetting the other widgets above.
    _split = good_poor_split_metric.value
    _vals = structure_metrics[_split].dropna()
    if len(_vals):
        good_poor_cutoff = mo.ui.number(value=round(float(_vals.median()), 4), step=0.01, label="cutoff")
        _stats = mo.md(
            f"**{_split}** — min `{_vals.min():.4g}` · median `{_vals.median():.4g}` · "
            f"mean `{_vals.mean():.4g}` · max `{_vals.max():.4g}`  (n = {len(_vals)})"
        )
    else:
        good_poor_cutoff = mo.ui.number(value=0.0, step=0.01, label="cutoff")
        _stats = mo.md(f"**{_split}** — no values")
    mo.vstack([_stats, good_poor_cutoff])
    return (good_poor_cutoff,)


@app.cell
def _(
    Path,
    good_poor_clamp,
    good_poor_cutoff,
    good_poor_dist_metrics,
    good_poor_font_size,
    good_poor_save_button,
    good_poor_save_dpi,
    good_poor_save_path,
    good_poor_split_metric,
    good_poor_vertical,
    np,
    plt,
    sns,
    structure_metrics,
):
    _label = {
        "resolution": "resolution (Å)",
        "deposited_r_free": "deposited R-free",
        "deposited_r_work": "deposited R-work",
        "r_free": "R-free",
        "r_work": "R-work",
        "num_water": "#water (clustered)",
        "num_water_deposited": "#water (deposited)",
        "unit_cell_volume": "unit-cell volume (Å³)",
        "f1": "F1",
        "precision": "precision",
        "recall": "recall",
    }
    _split = good_poor_split_metric.value
    _cutoff = float(good_poor_cutoff.value)
    _metrics = list(good_poor_dist_metrics.value)

    # good = split metric at or above the cutoff; poor = below. Structures whose
    # split metric is missing are dropped from both groups. The categorical column
    # name ("group") drives the legend, with good_order = [good, poor].
    _base = structure_metrics.dropna(subset=[_split])
    _labelled = _base.assign(
        group=np.where(_base[_split] >= _cutoff, "good", "poor")
    )
    _n_good = int((_labelled["group"] == "good").sum())
    _n_poor = int((_labelled["group"] == "poor").sum())
    print(f"split on {_split} @ {_cutoff}: {_n_good} good (≥), {_n_poor} poor (<)")

    _fs = good_poor_font_size.value
    _n = max(len(_metrics), 1)

    # Fixed data-rectangle layout so the figure is identical across cohorts —
    # matches the consensus/non-consensus quality plot above.
    _panel_w, _panel_h = 2.4, 2.2
    _left, _right, _top, _bottom, _gap = 0.75, 0.2, 0.4, 0.6, 0.6
    if good_poor_vertical.value:
        _fig_w = _left + _panel_w + _right
        _fig_h = _bottom + _n * _panel_h + (_n - 1) * _gap + _top
        _fig, _axes = plt.subplots(_n, 1, figsize=(_fig_w, _fig_h), squeeze=False)
        _axes = _axes[:, 0]
        _fig.subplots_adjust(
            left=_left / _fig_w, right=1 - _right / _fig_w,
            bottom=_bottom / _fig_h, top=1 - _top / _fig_h,
            hspace=_gap / _panel_h,
        )
    else:
        _fig_w = _left + _n * _panel_w + (_n - 1) * _gap + _right
        _fig_h = _bottom + _panel_h + _top
        _fig, _axes = plt.subplots(1, _n, figsize=(_fig_w, _fig_h), squeeze=False)
        _axes = _axes[0]
        _fig.subplots_adjust(
            left=_left / _fig_w, right=1 - _right / _fig_w,
            bottom=_bottom / _fig_h, top=1 - _top / _fig_h,
            wspace=_gap / _panel_w,
        )

    for _idx, (_ax, _metric) in enumerate(zip(_axes, _metrics)):
        _data = _labelled.dropna(subset=[_metric])
        if good_poor_clamp.value and len(_data):
            _lo, _hi = _data[_metric].quantile([0.001, 0.999])
            _data = _data[_data[_metric].between(_lo, _hi)]
        sns.histplot(
            data=_data, x=_metric, hue="group",
            stat="density", common_norm=False,
            element="step", fill=True, alpha=0.5, ax=_ax,
            hue_order=["good", "poor"],
            palette={"good": "r", "poor": "grey"},
            legend=False,
            # legend=(_idx == 0),
        )
        _ax.set_xlabel(_label.get(_metric, _metric), fontsize=_fs)
        _ax.set_ylabel("density", fontsize=_fs)
        _ax.tick_params(labelsize=_fs)
        if _idx == 0 and _ax.get_legend() is not None:
            _ax.get_legend().set_title(f"{_label.get(_split, _split)} ≥ {_cutoff:g}")
            for _txt in _ax.get_legend().get_texts():
                _txt.set_fontsize(_fs - 2)

        # Stats on the full (unclamped) data so min/max reflect the real
        # distribution, not the display clamp. Grouped by good/poor label.
        _summary = (
            _labelled.dropna(subset=[_metric])
            .groupby("group")[_metric]
            .agg(["min", "median", "mean", "std", "max", "count"])
        )
        print(f"[structure] {_metric}:")
        for _grp, _row in _summary.iterrows():
            print(
                f"  {_grp:>4}: min={_row['min']:.4g} median={_row['median']:.4g} "
                f"mean={_row['mean']:.4g} std={_row['std']:.4g} max={_row['max']:.4g} "
                f"n={int(_row['count'])}"
            )

    if good_poor_save_button.value:
        _out = Path(good_poor_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=int(good_poor_save_dpi.value))
        print(f"saved to {_out}")

    _fig
    return


if __name__ == "__main__":
    app.run()
