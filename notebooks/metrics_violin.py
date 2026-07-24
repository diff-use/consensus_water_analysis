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
    # Metric violins — cohorts side by side

    Condensed violin takes on two overlaid-histogram panels in
    `01_cluster_analysis.py`, both drawn by the same split-violin routine:

    - **Per-water** (`consensus vs non-consensus water quality`): one panel per
      water-level metric; within a panel the x-axis is the cohorts and each cohort
      is a split violin — left half consensus waters, right half non-consensus.
    - **Per-structure** (`good vs poor structures — metadata distributions`): the
      structure-level mirror. Each structure is labelled *good* or *poor* by whether
      a chosen per-structure metric (default `f1`, its agreement with the consensus)
      clears a cutoff; then one panel per deposited-metadata metric shows a split
      violin per cohort — left half good, right half poor.

    Each silhouette is a *raw binned density* (a mirrored histogram,
    `density=True`) — no kernel smoothing — so the shape is the actual
    distribution, not a KDE. Each half is normalized independently (matching the
    old `common_norm=False`) since the groups differ hugely in size.

    Reads the same pre-computed `clusters.csv` / `cluster_members.csv` /
    `metadata.csv` per cohort. A water is **consensus** iff it is a within-cutoff
    member of a cluster whose occupancy clears the consensus cutoff; everything
    else (noise, radius-rejected, low-occupancy members) is non-consensus.
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
    from matplotlib.patches import Patch

    from cw.metrics import consensus_centers, per_structure_consensus_pr

    return (
        Patch,
        Path,
        config,
        consensus_centers,
        np,
        pd,
        per_structure_consensus_pr,
        plt,
    )


@app.cell
def _(mo):
    # Comma-separated cohort directory names (each is a folder under DATA_DIR with
    # its own clusters.csv / cluster_members.csv / metadata.csv). Unlike
    # 01_cluster_analysis.py there is no subset-suffix guessing — spell out the
    # exact directory names.
    cohorts_input = mo.ui.text(
        value="hewls_65, endothiapepsin_000240_iso, carbonicanhydrase_000562_iso",
        label="cohorts (comma-separated)",
        full_width=True,
    )
    # Optional display labels aligned to the cohorts above; blank falls back to the
    # directory names. Handy because the raw names are long on the x-axis.
    cohort_labels_input = mo.ui.text(
        value="HEWL, EAP, CA",
        label="x-axis labels (comma-separated, optional)",
        full_width=True,
    )
    cutoff_input = mo.ui.number(
        value=0.3, start=0.0, stop=1.0, step=0.05, label="consensus cutoff",
    )
    mo.vstack([cohorts_input, cohort_labels_input, cutoff_input])
    return cohort_labels_input, cohorts_input, cutoff_input


@app.cell
def _(
    Path,
    cohort_labels_input,
    cohorts_input,
    config,
    consensus_centers,
    cutoff_input,
    mo,
    pd,
    per_structure_consensus_pr,
):
    COHORTS = [c.strip() for c in cohorts_input.value.split(",") if c.strip()]
    mo.stop(
        not COHORTS,
        mo.md("**Enter at least one cohort above to load its data.**"),
    )
    _raw_labels = [s.strip() for s in cohort_labels_input.value.split(",") if s.strip()]
    COHORT_LABELS = _raw_labels if len(_raw_labels) == len(COHORTS) else COHORTS

    # Member-radius variant of the clustering CSVs (see 01_cluster_analysis.py);
    # None reads the default clusters.csv / cluster_members.csv.
    MEMBER_RADIUS = None
    _suffix = f"_{MEMBER_RADIUS}" if MEMBER_RADIUS is not None else ""
    cluster_occupancy_cutoff = float(cutoff_input.value)
    match_radius = (
        float(MEMBER_RADIUS) if MEMBER_RADIUS is not None else config.CLUSTER_MEMBER_RADIUS
    )

    # One long per-water frame and one long per-structure frame, both tagged by
    # cohort with the split label already resolved (each cohort has its own cluster
    # ids / occupancies / consensus centers).
    _water_frames = []
    _structure_frames = []
    for _cohort in COHORTS:
        _dir = Path(config.DATA_DIR) / _cohort
        _clusters = pd.read_csv(_dir / f"clusters{_suffix}.csv")
        _members = pd.read_csv(_dir / f"cluster_members{_suffix}.csv")

        _conserved_ids = set(
            _clusters.loc[_clusters["cluster_occupancy"] >= cluster_occupancy_cutoff, "cluster_id"]
        )
        _labelled = _members.assign(
            conserved=_members["within_cutoff"] & _members["cluster_id"].isin(_conserved_ids),
            cohort=_cohort,
        )
        _water_frames.append(_labelled)

        # Per-structure precision/recall/f1/num_water vs the consensus centers,
        # joined to deposited metadata. metadata's own num_water becomes
        # num_water_deposited so pr_df's clustered count stays the plain num_water;
        # metadata scalars can carry "<missing>" strings, so coerce to numeric.
        _centers = consensus_centers(_clusters, cluster_occupancy_cutoff)
        _pr = per_structure_consensus_pr(_members, _centers, match_radius)
        # _iso / _bfactor / … are water-level subsets that don't re-deposit
        # per-structure metadata (resolution/R-free are per-PDB, shared across
        # water filters); it lives in the parent cohort dir. Walk up by stripping
        # trailing _<token> segments until a metadata.csv turns up — mirrors the
        # SUBSET→DATA fallback in 01_cluster_analysis.py.
        _meta_path = _dir / "metadata.csv"
        _probe = _cohort
        while not _meta_path.exists() and "_" in _probe:
            _probe = _probe.rsplit("_", 1)[0]
            _meta_path = Path(config.DATA_DIR) / _probe / "metadata.csv"
        if _meta_path.exists():
            _meta = pd.read_csv(_meta_path)
            for _column in _meta.columns:
                if _column != "pdb_id":
                    _meta[_column] = pd.to_numeric(_meta[_column], errors="coerce")
            _struct = _pr.merge(_meta, on="pdb_id", how="left", suffixes=("", "_deposited"))
            _matched = int(_pr["pdb_id"].isin(_meta["pdb_id"]).sum())
            _meta_note = f"metadata {_meta_path.parent.name}/ ({_matched}/{len(_pr)} matched)"
        else:
            _struct = _pr
            _meta_note = "no metadata.csv found"
        _structure_frames.append(_struct.assign(cohort=_cohort))

        _n_cons = int(_labelled["conserved"].sum())
        print(
            f"{_cohort}: {len(_members)} waters, "
            f"{_n_cons} consensus / {len(_members) - _n_cons} non-consensus, "
            f"{len(_clusters)} clusters, {len(_pr)} structures, {_meta_note}"
        )

    waters = pd.concat(_water_frames, ignore_index=True)
    structures = pd.concat(_structure_frames, ignore_index=True)
    print(
        f"total: {len(waters)} waters, {len(structures)} structures "
        f"across {len(COHORTS)} cohorts"
    )
    return COHORTS, COHORT_LABELS, structures, waters


@app.cell
def _(mo):
    # Styling shared by both figures so the two panels read as one system.
    violin_bins = mo.ui.slider(
        start=10, stop=80, step=5, value=30, label="density bins", show_value=True,
    )
    violin_clamp = mo.ui.checkbox(value=True, label="clamp value axis to 0–99.9 pct")
    violin_center = mo.ui.dropdown(
        options=["median", "mean", "both", "none"], value="median", label="center line",
    )
    violin_iqr = mo.ui.checkbox(value=True, label="show IQR (Q1–Q3)")
    violin_despine = mo.ui.checkbox(value=True, label="hide top/right spines")
    violin_vertical = mo.ui.checkbox(value=False, label="vertical layout (uncheck = horizontal)")
    violin_font_size = mo.ui.slider(
        start=6, stop=24, step=1, value=14, label="font size", show_value=True,
    )
    mo.hstack(
        [violin_bins, violin_clamp, violin_center, violin_iqr, violin_despine,
         violin_vertical, violin_font_size],
        justify="start",
    )
    return (
        violin_bins,
        violin_center,
        violin_clamp,
        violin_despine,
        violin_font_size,
        violin_iqr,
        violin_vertical,
    )


@app.cell
def _(Patch, np, plt):
    def make_violin_figure(
        df, metrics, cohorts, cohort_labels, metric_labels,
        split_col, left_value, right_value, left_label, right_label,
        n_bins, clamp, center_mode, show_iqr, despine, vertical, font_size,
        stats_tag,
        left_color="r", right_color="grey",
        left_line="darkred", right_line="black",
    ):
        # One panel per metric. Within a panel the x-axis is the cohorts and each
        # cohort is a split violin: left half = rows where df[split_col] ==
        # left_value, right half = right_value. Returns the figure and prints
        # per-cohort × per-group stats on the full (unclamped) values.
        _fs = font_size
        _metrics = list(metrics)
        _n = max(len(_metrics), 1)
        _n_cohorts = len(cohorts)
        _n_bins = int(n_bins)
        _halfwidth = 0.4

        def _half_density(values, edges):
            # Raw binned density (area = 1), no smoothing. Returns per-bin density
            # aligned to bin centers; empty/degenerate input gives all-zeros.
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return np.zeros(len(edges) - 1)
            dens, _ = np.histogram(values, bins=edges, density=True)
            return dens

        def _draw_stats(ax, values, x_center, side, color, dens, edges, scale):
            # Short horizontal marks on one half: solid = median, diamond = mean,
            # dotted = Q1/Q3, colored by group (dark shade of the fill). Each mark
            # spans only the violin's width at its own y (the scaled density of the
            # bin it lands in), so it never overshoots the silhouette. Stats use the
            # full (unclamped) values, so a mark can sit just outside the y-limits.
            v = np.asarray(values, dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                return

            def _edge_at(y):
                _b = int(np.clip(np.searchsorted(edges, y, side="right") - 1, 0, len(dens) - 1))
                return x_center + side * dens[_b] * scale

            if center_mode in ("median", "both"):
                _m = np.median(v)
                ax.plot([x_center, _edge_at(_m)], [_m, _m], color=color, lw=2,
                        solid_capstyle="butt", zorder=4)
            if center_mode in ("mean", "both"):
                _mu = v.mean()
                # diamond at the midpoint of the violin's width at the mean's height
                ax.plot([(x_center + _edge_at(_mu)) / 2], [_mu], marker="D",
                        ms=6, mfc="white", mec=color, mew=1.5, zorder=5)
            if show_iqr:
                for _q in np.percentile(v, [25, 75]):
                    ax.plot([x_center, _edge_at(_q)], [_q, _q], color=color, lw=2.0,
                            ls=":", zorder=4)

        # Fixed data-rectangle layout (matches the panels in 01_cluster_analysis.py)
        # so the figure is identical across runs. Gutters scale mildly with font
        # size so they hug the labels at the default size but don't clip when the
        # font is bumped. left = y-label + tick digits; bottom = x tick labels.
        _panel_w, _panel_h = 1.7 * _n_cohorts, 2.5
        _left, _right, _top, _gap = 0.55 + _fs * 0.02, 0.2, 0.3, 1.2
        _bottom = 0.30 + _fs * 0.02
        if vertical:
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
            _all_vals = df[_metric].dropna()
            if clamp and len(_all_vals):
                _lo, _hi = _all_vals.quantile([0, 0.999])
            elif len(_all_vals):
                _lo, _hi = float(_all_vals.min()), float(_all_vals.max())
            else:
                _lo, _hi = 0.0, 1.0
            if _hi <= _lo:
                _hi = _lo + 1.0
            # Shared bin edges across cohorts within a metric → violins are directly
            # comparable along the (shared) y-axis of this panel.
            _edges = np.linspace(_lo, _hi, _n_bins + 1)
            _centers = (_edges[:-1] + _edges[1:]) / 2

            for _i, _cohort in enumerate(cohorts):
                _sub = df[df["cohort"] == _cohort]
                _left_vals = _sub.loc[_sub[split_col] == left_value, _metric]
                _right_vals = _sub.loc[_sub[split_col] == right_value, _metric]
                _dl = _half_density(_left_vals, _edges)
                _dr = _half_density(_right_vals, _edges)
                # Each half normalized to its own area=1 (common_norm=False), then
                # both scaled by a single per-cohort factor so the taller peak just
                # fills the half-slot — relative peak height between halves is kept.
                _peak = max(_dl.max(), _dr.max())
                if _peak <= 0:
                    continue
                _scale = _halfwidth / _peak
                _ax.fill_betweenx(
                    _centers, _i, _i - _dl * _scale, step="mid",
                    color=left_color, alpha=0.5, edgecolor=left_color, linewidth=0.8,
                )
                _ax.fill_betweenx(
                    _centers, _i, _i + _dr * _scale, step="mid",
                    color=right_color, alpha=0.5, edgecolor="dimgrey", linewidth=0.8,
                )
                _draw_stats(_ax, _left_vals, _i, -1, left_line, _dl, _edges, _scale)
                _draw_stats(_ax, _right_vals, _i, +1, right_line, _dr, _edges, _scale)

            _ax.set_xticks(range(_n_cohorts))
            _ax.set_xticklabels(cohort_labels, fontsize=_fs, ha="center")
            _ax.set_xlim(-0.6, _n_cohorts - 0.4)
            _ax.set_ylim(_lo, _hi)
            _ax.set_ylabel(metric_labels.get(_metric, _metric), fontsize=_fs)
            _ax.tick_params(axis="y", labelsize=_fs)
            if despine:
                _ax.spines["top"].set_visible(False)
                _ax.spines["right"].set_visible(False)

            # Stats on the full (unclamped) data, grouped by cohort × split label.
            print(f"[{stats_tag}] {_metric}:")
            for _cohort in cohorts:
                _sub = df[df["cohort"] == _cohort]
                _summary = (
                    _sub.dropna(subset=[_metric]).groupby(split_col)[_metric]
                    .agg(["min", "median", "mean", "std", "max", "count"])
                )
                print(f"  {_cohort}:")
                for _grp, _row in _summary.iterrows():
                    _name = left_label if _grp == left_value else right_label
                    print(
                        f"    {_name:>14}: min={_row['min']:.4g} median={_row['median']:.4g} "
                        f"mean={_row['mean']:.4g} std={_row['std']:.4g} max={_row['max']:.4g} "
                        f"n={int(_row['count'])}"
                    )

        _axes[0].legend(
            handles=[
                Patch(facecolor=left_color, alpha=0.5, edgecolor=left_color, label=left_label),
                Patch(facecolor=right_color, alpha=0.5, edgecolor="dimgrey", label=right_label),
            ],
            fontsize=_fs - 2, loc="lower right", ncol=2, columnspacing=1.0,
            framealpha=0.9,
        )
        return _fig

    return (make_violin_figure,)


@app.cell
def _(mo):
    mo.md("""
    ## Per-water metric violins — consensus vs non-consensus
    """)
    return


@app.cell
def _(mo, waters):
    _candidates = ["b_factor_zscore", "edia", "b_factor", "occupancy", "muse_score"]
    _available = [
        c for c in _candidates
        if c in waters.columns and waters[c].notna().any()
    ]
    metrics_select = mo.ui.multiselect(
        options=_available,
        value=[m for m in ["b_factor_zscore", "edia"] if m in _available],
        label="water metrics (one panel each, left→right)",
    )
    metrics_select
    return (metrics_select,)


@app.cell
def _(mo):
    water_save_path = mo.ui.text(
        value="data/plots/per_water_metric_violins.png",
        label="save path", full_width=True,
    )
    water_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    water_save_button = mo.ui.run_button(label="save water figure")
    mo.hstack([water_save_path, water_save_dpi, water_save_button], justify="start")
    return water_save_button, water_save_dpi, water_save_path


@app.cell
def _(
    COHORTS,
    COHORT_LABELS,
    Path,
    make_violin_figure,
    metrics_select,
    violin_bins,
    violin_center,
    violin_clamp,
    violin_despine,
    violin_font_size,
    violin_iqr,
    violin_vertical,
    water_save_button,
    water_save_dpi,
    water_save_path,
    waters,
):
    _fig = make_violin_figure(
        waters,
        metrics=list(metrics_select.value),
        cohorts=COHORTS,
        cohort_labels=COHORT_LABELS,
        metric_labels={"edia": "EDIA", "b_factor_zscore": "B-factor z-score"},
        split_col="conserved",
        left_value=True,
        right_value=False,
        left_label="consensus",
        right_label="non-consensus",
        n_bins=violin_bins.value,
        clamp=violin_clamp.value,
        center_mode=violin_center.value,
        show_iqr=violin_iqr.value,
        despine=violin_despine.value,
        vertical=violin_vertical.value,
        font_size=violin_font_size.value,
        stats_tag="water",
    )

    if water_save_button.value:
        _out = Path(water_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=int(water_save_dpi.value))
        print(f"saved to {_out}")

    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-structure metric violins — good vs poor structures

    Each structure is labelled **good** or **poor** by whether a chosen
    per-structure metric (default `f1`, its agreement with the consensus) is **at
    or above that cohort's own cutoff** for the metric, then the deposited-metadata
    distributions of the two groups are shown as split violins per cohort. This
    asks whether the structures that best reproduce the conserved sites are also
    the higher-quality depositions. The cutoff is **per-cohort** — each cohort gets
    its own box, defaulting to that cohort's median (its scale differs between
    cohorts, so a shared cutoff wouldn't halve them comparably) and editable to
    move the boundary; the values used and the resulting good/poor counts are
    printed below. The split metric can be any per-structure column
    (precision/recall/f1/num_water or a metadata column); the panels can be any
    numeric metadata columns.
    """)
    return


@app.cell
def _(pd, structures):
    structure_numeric_cols = [
        _c for _c in structures.columns
        if _c not in ("pdb_id", "cohort")
        and pd.api.types.is_numeric_dtype(structures[_c])
        and structures[_c].notna().any()
    ]
    return (structure_numeric_cols,)


@app.cell
def _(mo, structure_numeric_cols):
    structure_split_metric = mo.ui.dropdown(
        options=structure_numeric_cols,
        value="f1" if "f1" in structure_numeric_cols else structure_numeric_cols[0],
        label="split metric (good = ≥ cohort median)",
    )
    _dist_default = [
        _m for _m in ["resolution", "deposited_r_free"] if _m in structure_numeric_cols
    ]
    structure_dist_metrics = mo.ui.multiselect(
        options=structure_numeric_cols,
        value=_dist_default or structure_numeric_cols[:1],
        label="metadata metrics (one panel each, left→right)",
    )
    mo.hstack([structure_split_metric, structure_dist_metrics], justify="start")
    return structure_dist_metrics, structure_split_metric


@app.cell
def _(COHORTS, mo, structure_split_metric, structures):
    # One cutoff per cohort, each defaulting to that cohort's own median of the
    # split metric (the metric's scale differs between cohorts, so a shared cutoff
    # wouldn't halve them comparably). Editable — change a box to move that cohort's
    # good/poor boundary. Rebuilt and re-defaulted whenever the split metric changes.
    _split = structure_split_metric.value
    _medians = structures.dropna(subset=[_split]).groupby("cohort")[_split].median()
    structure_cutoffs = mo.ui.dictionary({
        _cohort: mo.ui.number(
            value=round(float(_medians.get(_cohort, 0.0)), 4),
            step=0.01,
            label=f"{_cohort} (median {_split} = {_medians.get(_cohort, float('nan')):.4g})",
        )
        for _cohort in COHORTS
    })
    mo.vstack([
        mo.md(f"**good = {_split} ≥ cutoff** — one cutoff per cohort, defaulting to its median:"),
        structure_cutoffs,
    ])
    return (structure_cutoffs,)


@app.cell
def _(mo, structure_split_metric):
    structure_save_path = mo.ui.text(
        value=f"data/plots/{structure_split_metric.value}_good_vs_poor_metadata_violins.png",
        label="save path", full_width=True,
    )
    structure_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    structure_save_button = mo.ui.run_button(label="save structure figure")
    mo.hstack([structure_save_path, structure_save_dpi, structure_save_button], justify="start")
    return structure_save_button, structure_save_dpi, structure_save_path


@app.cell
def _(
    COHORTS,
    COHORT_LABELS,
    Path,
    make_violin_figure,
    np,
    structure_cutoffs,
    structure_dist_metrics,
    structure_save_button,
    structure_save_dpi,
    structure_save_path,
    structure_split_metric,
    structures,
    violin_bins,
    violin_center,
    violin_clamp,
    violin_despine,
    violin_font_size,
    violin_iqr,
    violin_vertical,
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
    _split = structure_split_metric.value
    _cutoffs = structure_cutoffs.value

    # good = split metric at or above that cohort's own cutoff; poor = below.
    # Structures whose split metric is missing are dropped from both groups.
    _base = structures.dropna(subset=[_split])
    _labelled = _base.assign(
        group=np.where(_base[_split] >= _base["cohort"].map(_cutoffs), "good", "poor")
    )
    print(f"split on {_split} at each cohort's own cutoff:")
    for _cohort in COHORTS:
        _rows = _labelled[_labelled["cohort"] == _cohort]
        _n_good = int((_rows["group"] == "good").sum())
        _n_poor = int((_rows["group"] == "poor").sum())
        print(
            f"  {_cohort}: cutoff {_cutoffs.get(_cohort):.4g} → "
            f"{_n_good} good (≥), {_n_poor} poor (<)"
        )

    _fig = make_violin_figure(
        _labelled,
        metrics=list(structure_dist_metrics.value),
        cohorts=COHORTS,
        cohort_labels=COHORT_LABELS,
        metric_labels=_label,
        split_col="group",
        left_value="good",
        right_value="poor",
        left_label=f"above median F1",
        right_label=f"below",
        # left_label=f"above median {structure_split_metric.value}",
        # right_label=f"below median {structure_split_metric.value}",
        n_bins=violin_bins.value,
        clamp=violin_clamp.value,
        center_mode=violin_center.value,
        show_iqr=violin_iqr.value,
        despine=violin_despine.value,
        vertical=violin_vertical.value,
        font_size=violin_font_size.value,
        stats_tag="structure",
    )

    if structure_save_button.value:
        _out = Path(structure_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _fig.savefig(_out, dpi=int(structure_save_dpi.value))
        print(f"saved to {_out}")

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
