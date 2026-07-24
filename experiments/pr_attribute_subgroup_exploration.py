import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Precision–recall — attribute & subgroup exploration

    Exploratory companion to `notebooks/01_cluster_analysis.py`. Reads the same
    pre-computed `clusters.csv` / `cluster_members.csv` (and optional
    `metadata.csv` / `pairwise_metrics_<r>.csv`) and reconstructs the per-structure
    precision/recall table, then colors and splits it by structure attributes:

    - **P–R colored by structure attributes** — metadata query, subset membership,
      or hierarchical-cluster label.
    - **Consensus agreement score** — F1 / Chamfer distribution per structure.
    - **F1 distribution by subgroup** — the same split as the attribute scatter,
      overlaid as step histograms.

    Set the cohort with the `COHORT` environment variable before launching, e.g.
    `COHORT=hewls_65 marimo edit experiments/pr_attribute_subgroup_exploration.py`.
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
    import seaborn as sns

    from cw.metrics import (
        consensus_centers,
        per_structure_consensus_chamfer,
        per_structure_consensus_pr,
    )
    from cw.plots import plot_pr_scatter, quantile_boundaries

    return (
        Path,
        config,
        consensus_centers,
        np,
        pd,
        per_structure_consensus_chamfer,
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
        placeholder="e.g. hewls_65 or carbonicanhydrase_000562",
        label="cohort",
        full_width=True,
    )
    cohort_input
    return (cohort_input,)


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
    SUBSET = Path(config.DATA_DIR) / (COHORT + subset_suffix)
    if not SUBSET.exists():
        SUBSET = DATA

    _suffix = f"_{MEMBER_RADIUS}" if MEMBER_RADIUS is not None else ""
    clusters = pd.read_csv(SUBSET / f"clusters{_suffix}.csv")
    cluster_members = pd.read_csv(SUBSET / f"cluster_members{_suffix}.csv")

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
        DATA,
        PLOTS_DIR,
        SUBSET,
        cluster_members,
        cluster_occupancy_cutoff,
        clusters,
        match_radius,
    )


@app.cell
def _(mo):
    # Shared axis / font controls for the attribute scatter and the F1 histogram.
    axis_range = mo.ui.range_slider(
        start=0.0, stop=1.02, step=0.05, value=[0.0, 1.02],
        label="axis range (equal x/y)", show_value=True,
    )
    font_size = mo.ui.slider(
        start=6, stop=20, step=1, value=14, label="font size", show_value=True,
    )
    mo.hstack([axis_range, font_size], justify="start")
    return axis_range, font_size


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
    pr_df = per_structure_consensus_pr(
        cluster_members,
        consensus_centers(clusters, cluster_occupancy_cutoff),
        match_radius,
    )
    return (pr_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Precision–recall colored by a metadata column

    The per-structure precision/recall cloud, colored by any numeric column of the
    cohort `metadata.csv` (e.g. `resolution`, the deposited `deposited_r_free`, or
    the pdb-redo `r_free`). Clamp the color to a percentile window so a few extreme
    structures don't wash out the scale, and pick either a **continuous** colorbar
    or **quantile bins** (equal-count bands with capped tails, exactly as
    `num_water` is colored in `notebooks/01_cluster_analysis.py`).

    Reuses `cw.plots.plot_pr_scatter` and the axis-range / font-size controls above.
    """)
    return


@app.cell
def _(PLOTS_DIR, meta_df, mo):
    _meta_cols = [c for c in meta_df.columns if c != "pdb_id"]
    _default_col = (
        "resolution" if "resolution" in _meta_cols
        else (_meta_cols[0] if _meta_cols else None)
    )
    metacolor_column = mo.ui.dropdown(
        options=_meta_cols, value=_default_col, label="color by metadata column",
    )
    metacolor_scale = mo.ui.radio(
        options=["quantile bins", "continuous"], value="quantile bins",
        label="color scale",
    )
    metacolor_pct = mo.ui.range_slider(
        start=0, stop=100, step=1, value=[1, 99],
        label="color clamp percentile", show_value=True,
    )
    metacolor_bins = mo.ui.slider(
        start=2, stop=10, step=1, value=10, label="quantile bins", show_value=True,
    )
    metacolor_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "precision_recall_by_metadata.png"),
        label="save path", full_width=True,
    )
    metacolor_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    metacolor_save_button = mo.ui.run_button(label="save figure")
    mo.vstack([
        mo.hstack([metacolor_column, metacolor_scale], justify="start"),
        mo.hstack([metacolor_pct, metacolor_bins], justify="start"),
        mo.hstack([metacolor_save_path, metacolor_save_dpi, metacolor_save_button], justify="start"),
    ])
    return (
        metacolor_bins,
        metacolor_column,
        metacolor_pct,
        metacolor_save_button,
        metacolor_save_dpi,
        metacolor_save_path,
        metacolor_scale,
    )


@app.cell
def _(
    Path,
    axis_range,
    font_size,
    meta_df,
    metacolor_bins,
    metacolor_column,
    metacolor_pct,
    metacolor_save_button,
    metacolor_save_dpi,
    metacolor_save_path,
    metacolor_scale,
    mo,
    np,
    pd,
    plot_pr_scatter,
    plt,
    pr_df,
    quantile_boundaries,
):
    _col = metacolor_column.value
    if meta_df.empty or _col is None:
        _out = mo.md("*No `metadata.csv` columns available — cannot color by metadata.*")
    else:
        # Attach the chosen metadata column to the per-structure P/R table and keep
        # only structures that carry a numeric value for it.
        _merged = pr_df.merge(meta_df[["pdb_id", _col]], on="pdb_id", how="left")
        _merged[_col] = pd.to_numeric(_merged[_col], errors="coerce")
        _plot_df = _merged[_merged[_col].notna()].copy()
        if _plot_df.empty:
            _out = mo.md(f"*Column `{_col}` has no numeric values for this cohort.*")
        else:
            _lo_pct, _hi_pct = metacolor_pct.value
            _n_missing = len(_merged) - len(_plot_df)
            _title = _col if not _n_missing else f"{_col} ({_n_missing} without value dropped)"
            if metacolor_scale.value == "continuous":
                # Clamp the color values to the percentile window so extreme
                # structures saturate at the ends instead of stretching the scale.
                _lo, _hi = np.percentile(_plot_df[_col], [_lo_pct, _hi_pct])
                _color = np.clip(_plot_df[_col].to_numpy(), _lo, _hi)
                _fig, _ax = plot_pr_scatter(
                    _plot_df, color=_color, color_label=_col, n_color_bins=None,
                    alpha=0.5, cmap="viridis_r",
                    lim=tuple(axis_range.value), fontsize=font_size.value, title=_title,
                )
            else:
                # Quantile bins: plot_pr_scatter caps the tails at the percentile
                # window itself (via color_quantile_range) — same path as num_water.
                _fig, _ax = plot_pr_scatter(
                    _plot_df, color=_col, color_label=_col,
                    n_color_bins=int(metacolor_bins.value),
                    color_quantile_range=(_lo_pct / 100, _hi_pct / 100),
                    alpha=0.5, cmap="tab10",
                    lim=tuple(axis_range.value), fontsize=font_size.value, title=_title,
                )
                # Report the actual per-band structure counts. quantile_boundaries
                # is the same function plot_pr_scatter bins on, so this is exactly
                # the color scale drawn. Ties in the column (many identical values)
                # collapse edges via np.unique, so the realised band count can be
                # far below the requested one and the counts wildly uneven — the
                # usual cause of a "bins look off" scatter.
                _edges = quantile_boundaries(
                    _plot_df[_col].to_numpy(), int(metacolor_bins.value),
                    (_lo_pct / 100, _hi_pct / 100),
                )
                # ±inf outer edges fold the below/above-window structures into the
                # end bands, matching BoundaryNorm(extend="both") in plot_pr_scatter.
                _fold = _edges.astype(float).copy()
                _fold[0], _fold[-1] = -np.inf, np.inf
                _counts = (
                    pd.cut(_plot_df[_col], bins=_fold, include_lowest=True)
                    .value_counts().sort_index()
                )
                print(
                    f"{_col}: {int(metacolor_bins.value)} bins requested, "
                    f"{len(_edges) - 1} realised (edges: "
                    f"{', '.join(f'{_e:.3g}' for _e in _edges)})"
                )
                for _interval, _n in _counts.items():
                    print(f"  {_interval}: {int(_n)}")
            plt.tight_layout()
            if metacolor_save_button.value:
                _p = Path(metacolor_save_path.value)
                _p.parent.mkdir(parents=True, exist_ok=True)
                _fig.savefig(_p, dpi=int(metacolor_save_dpi.value), bbox_inches="tight")
                print(f"saved to {_p}")
            _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Free scatter — any metadata column vs any other

    A general two-column scatter over the per-structure table (every
    `metadata.csv` column joined to the P/R metrics `precision`, `recall`, `f1`,
    `num_water`). Pick any numeric column for each axis, optionally **reverse**
    either one (e.g. deposited R-free against reversed resolution), and color by a
    third column with the same clamp/binning controls as the P–R scatter above —
    percentile clamp plus **continuous** or **quantile bins**.

    Reuses `cw.plots.plot_pr_scatter` (axes repointed at the chosen columns, equal
    x/y range disabled) and the shared font-size control.
    """)
    return


@app.cell
def _(meta_df, pd, pr_df):
    # One per-structure table for the free scatter: P/R metrics (precision, recall,
    # f1, num_water) joined to every metadata column, all coerced to numeric so any
    # pair can drive the axes or the color scale. Non-numeric metadata columns
    # (space group, etc.) become all-NaN and drop out of the selectable list.
    xy_df = pr_df.merge(meta_df, on="pdb_id", how="left")
    for _c in xy_df.columns:
        if _c != "pdb_id":
            xy_df[_c] = pd.to_numeric(xy_df[_c], errors="coerce")
    xy_numeric_cols = [
        _c for _c in xy_df.columns if _c != "pdb_id" and xy_df[_c].notna().any()
    ]
    return xy_df, xy_numeric_cols


@app.cell
def _(PLOTS_DIR, mo, xy_numeric_cols):
    _cols = xy_numeric_cols
    _default_x = "resolution" if "resolution" in _cols else (_cols[0] if _cols else None)
    _default_y = next(
        (c for c in ("deposited_r_free", "r_free") if c in _cols),
        (_cols[1] if len(_cols) > 1 else None),
    )
    _default_color = "num_water" if "num_water" in _cols else "(none)"
    xy_x_column = mo.ui.dropdown(options=_cols, value=_default_x, label="x axis")
    xy_reverse_x = mo.ui.checkbox(value=False, label="reverse x")
    xy_y_column = mo.ui.dropdown(options=_cols, value=_default_y, label="y axis")
    xy_reverse_y = mo.ui.checkbox(value=False, label="reverse y")
    xy_color_column = mo.ui.dropdown(
        options=["(none)"] + _cols, value=_default_color, label="color by",
    )
    xy_color_scale = mo.ui.radio(
        options=["quantile bins", "continuous"], value="continuous", label="color scale",
    )
    xy_color_pct = mo.ui.range_slider(
        start=0, stop=100, step=1, value=[1, 99],
        label="color clamp percentile", show_value=True,
    )
    xy_color_bins = mo.ui.slider(
        start=2, stop=10, step=1, value=10, label="quantile bins", show_value=True,
    )
    # Optional axis limits — leave blank for autoscale on that end, so e.g. "0" in
    # xy_x_min with an empty xy_x_max pins the x axis to start at 0 and end auto.
    xy_x_min = mo.ui.text(value="", placeholder="auto", label="x min")
    xy_x_max = mo.ui.text(value="", placeholder="auto", label="x max")
    xy_y_min = mo.ui.text(value="", placeholder="auto", label="y min")
    xy_y_max = mo.ui.text(value="", placeholder="auto", label="y max")
    xy_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "metadata_scatter.png"), label="save path", full_width=True,
    )
    xy_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    xy_save_button = mo.ui.run_button(label="save figure")
    mo.vstack([
        mo.hstack([xy_x_column, xy_reverse_x, xy_y_column, xy_reverse_y], justify="start"),
        mo.hstack([xy_color_column, xy_color_scale], justify="start"),
        mo.hstack([xy_color_pct, xy_color_bins], justify="start"),
        mo.hstack([xy_x_min, xy_x_max, xy_y_min, xy_y_max], justify="start"),
        mo.hstack([xy_save_path, xy_save_dpi, xy_save_button], justify="start"),
    ])
    return (
        xy_color_bins,
        xy_color_column,
        xy_color_pct,
        xy_color_scale,
        xy_reverse_x,
        xy_reverse_y,
        xy_save_button,
        xy_save_dpi,
        xy_save_path,
        xy_x_column,
        xy_x_max,
        xy_x_min,
        xy_y_column,
        xy_y_max,
        xy_y_min,
    )


@app.cell
def _(
    Path,
    font_size,
    mo,
    np,
    pd,
    plot_pr_scatter,
    plt,
    quantile_boundaries,
    xy_color_bins,
    xy_color_column,
    xy_color_pct,
    xy_color_scale,
    xy_df,
    xy_reverse_x,
    xy_reverse_y,
    xy_save_button,
    xy_save_dpi,
    xy_save_path,
    xy_x_column,
    xy_x_max,
    xy_x_min,
    xy_y_column,
    xy_y_max,
    xy_y_min,
):
    def _limit(_box):
        _s = _box.value.strip()
        return float(_s) if _s else None

    _x, _y = xy_x_column.value, xy_y_column.value
    _color_col = xy_color_column.value if xy_color_column.value != "(none)" else None
    if _x is None or _y is None:
        _out = mo.md("*No numeric columns available for the free scatter.*")
    else:
        # Keep only structures that carry a value on both axes (and the color column
        # when one is chosen) so the scatter and its color scale align row-for-row.
        _need = [_x, _y] + ([_color_col] if _color_col else [])
        _plot_df = xy_df.dropna(subset=_need).copy()
        if _plot_df.empty:
            _out = mo.md(f"*No structures have values for `{'`, `'.join(_need)}`.*")
        else:
            _lo_pct, _hi_pct = xy_color_pct.value
            if _color_col is None:
                _fig, _ax = plot_pr_scatter(
                    _plot_df, precision_col=_y, recall_col=_x,
                    alpha=0.5, lim=None, fontsize=font_size.value,
                )
            elif xy_color_scale.value == "continuous":
                # Clamp the color values to the percentile window so extreme
                # structures saturate at the ends instead of stretching the scale.
                _lo, _hi = np.percentile(_plot_df[_color_col], [_lo_pct, _hi_pct])
                _color = np.clip(_plot_df[_color_col].to_numpy(), _lo, _hi)
                _fig, _ax = plot_pr_scatter(
                    _plot_df, precision_col=_y, recall_col=_x,
                    color=_color, color_label=_color_col, n_color_bins=None,
                    alpha=0.5, cmap="viridis", lim=None, fontsize=font_size.value,
                )
            else:
                # Quantile bins: plot_pr_scatter caps the tails at the percentile
                # window itself (via color_quantile_range) — same path as the P–R
                # scatter above.
                _fig, _ax = plot_pr_scatter(
                    _plot_df, precision_col=_y, recall_col=_x,
                    color=_color_col, color_label=_color_col,
                    n_color_bins=int(xy_color_bins.value),
                    color_quantile_range=(_lo_pct / 100, _hi_pct / 100),
                    alpha=0.5, cmap="viridis", lim=None, fontsize=font_size.value,
                )
                # Report the realised per-band structure counts, exactly as the P–R
                # metadata scatter does: np.unique collapses tied edges, so the
                # realised band count can fall short of the requested one.
                _edges = quantile_boundaries(
                    _plot_df[_color_col].to_numpy(), int(xy_color_bins.value),
                    (_lo_pct / 100, _hi_pct / 100),
                )
                _fold = _edges.astype(float).copy()
                _fold[0], _fold[-1] = -np.inf, np.inf
                _counts = (
                    pd.cut(_plot_df[_color_col], bins=_fold, include_lowest=True)
                    .value_counts().sort_index()
                )
                print(
                    f"{_color_col}: {int(xy_color_bins.value)} bins requested, "
                    f"{len(_edges) - 1} realised (edges: "
                    f"{', '.join(f'{_e:.3g}' for _e in _edges)})"
                )
                for _interval, _n in _counts.items():
                    print(f"  {_interval}: {int(_n)}")

            # plot_pr_scatter hard-codes recall/precision labels — repoint them at
            # the chosen columns, then honor the reverse toggles.
            _ax.set_xlabel(_x, fontsize=font_size.value)
            _ax.set_ylabel(_y, fontsize=font_size.value)
            # Optional manual limits: override only the specified end, leaving the
            # autoscaled value on the other. Apply in ascending order first, then
            # let the reverse toggles flip the axis.
            _xlo, _xhi = _limit(xy_x_min), _limit(xy_x_max)
            if _xlo is not None or _xhi is not None:
                _cur = sorted(_ax.get_xlim())
                _ax.set_xlim(_cur[0] if _xlo is None else _xlo,
                             _cur[1] if _xhi is None else _xhi)
            _ylo, _yhi = _limit(xy_y_min), _limit(xy_y_max)
            if _ylo is not None or _yhi is not None:
                _cur = sorted(_ax.get_ylim())
                _ax.set_ylim(_cur[0] if _ylo is None else _ylo,
                             _cur[1] if _yhi is None else _yhi)
            if xy_reverse_x.value:
                _ax.invert_xaxis()
            if xy_reverse_y.value:
                _ax.invert_yaxis()
            plt.tight_layout()
            if xy_save_button.value:
                _p = Path(xy_save_path.value)
                _p.parent.mkdir(parents=True, exist_ok=True)
                _fig.savefig(_p, dpi=int(xy_save_dpi.value), bbox_inches="tight")
                print(f"saved to {_p}")
            _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## num_water by metadata quantile bin

    Box plots of each structure's water count (`num_water`, its pooled prediction-set
    size) split into **equal-count quantile bins** of a chosen `metadata.csv` column —
    resolution and the deposited / pdb-redo R-free are the ones of interest. The bin
    count is user-set (default 5); ties in the column (common for resolution) collapse
    via `pd.qcut(duplicates="drop")`, so the realised bin count can be lower and is
    printed below along with each bin's structure count. Optionally clamp the column to
    a percentile window first to drop extreme structures before binning. Reuses the
    shared font-size control above.
    """)
    return


@app.cell
def _(PLOTS_DIR, meta_df, mo):
    _meta_cols = [c for c in meta_df.columns if c != "pdb_id"]
    _default_col = next(
        (c for c in ("resolution", "deposited_r_free", "r_free") if c in _meta_cols),
        (_meta_cols[0] if _meta_cols else None),
    )
    numwater_column = mo.ui.dropdown(
        options=_meta_cols, value=_default_col, label="metadata column to bin by",
    )
    numwater_bins = mo.ui.slider(
        start=2, stop=10, step=1, value=5, label="quantile bins", show_value=True,
    )
    numwater_pct = mo.ui.range_slider(
        start=0, stop=100, step=1, value=[1, 99],
        label="clamp column to percentile window", show_value=True,
    )
    numwater_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "num_water_by_metadata_bin.png"),
        label="save path", full_width=True,
    )
    numwater_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    numwater_save_button = mo.ui.run_button(label="save figure")
    mo.vstack([
        mo.hstack([numwater_column, numwater_bins, numwater_pct], justify="start"),
        mo.hstack([numwater_save_path, numwater_save_dpi, numwater_save_button], justify="start"),
    ])
    return (
        numwater_bins,
        numwater_column,
        numwater_pct,
        numwater_save_button,
        numwater_save_dpi,
        numwater_save_path,
    )


@app.cell
def _(
    Path,
    font_size,
    meta_df,
    mo,
    np,
    numwater_bins,
    numwater_column,
    numwater_pct,
    numwater_save_button,
    numwater_save_dpi,
    numwater_save_path,
    pd,
    plt,
    pr_df,
    sns,
):
    _col = numwater_column.value
    if meta_df.empty or _col is None:
        _out = mo.md("*No `metadata.csv` columns available — cannot bin num_water.*")
    else:
        _merged = pr_df.merge(meta_df[["pdb_id", _col]], on="pdb_id", how="left")
        _merged[_col] = pd.to_numeric(_merged[_col], errors="coerce")
        _data = _merged.dropna(subset=[_col, "num_water"]).copy()
        # Optional clamp: drop structures outside the percentile window of the
        # binned column so a few extreme values don't skew the bin edges.
        _lo_pct, _hi_pct = numwater_pct.value
        _n_pre_clamp = len(_data)
        if len(_data) and (_lo_pct > 0 or _hi_pct < 100):
            _lo, _hi = np.percentile(_data[_col], [_lo_pct, _hi_pct])
            _data = _data[_data[_col].between(_lo, _hi)]
        if _data.empty:
            _out = mo.md(f"*Column `{_col}` has no numeric values for this cohort.*")
        else:
            _fs = font_size.value
            _n_req = int(numwater_bins.value)
            # Equal-count quantile bins. duplicates="drop" folds tied edges (many
            # identical resolutions collapse cut points), so qcut doesn't raise —
            # at the cost of fewer realised bins than requested.
            _data = _data.assign(_bin=pd.qcut(_data[_col], q=_n_req, duplicates="drop"))
            _counts = _data["_bin"].value_counts().sort_index()
            _n_real = int(_data["_bin"].cat.categories.size)
            print(
                f"{_col}: {_n_req} bins requested, {_n_real} realised; "
                f"{_n_pre_clamp - len(_data)} structures dropped by percentile clamp"
            )
            for _interval, _n in _counts.items():
                print(f"  {_interval}: {int(_n)}")

            _fig, _ax = plt.subplots(figsize=(1.2 * _n_real + 2, 3.5))
            sns.boxplot(data=_data, x="_bin", y="num_water", ax=_ax, color="steelblue")
            _ax.set_xlabel(_col, fontsize=_fs)
            _ax.set_ylabel("num_water", fontsize=_fs)
            _ax.tick_params(labelsize=_fs)
            # Round the quantile-interval edges to 3 decimals so labels read
            # "(1.05, 1.1]" instead of "(1.05, 1.0990000000000002]".
            _labels = [
                f"({round(_iv.left, 3):g}, {round(_iv.right, 3):g}]"
                for _iv in _data["_bin"].cat.categories
            ]
            _ax.set_xticks(range(len(_labels)))
            _ax.set_xticklabels(_labels, rotation=30, ha="right")
            plt.tight_layout()
            if numwater_save_button.value:
                _p = Path(numwater_save_path.value)
                _p.parent.mkdir(parents=True, exist_ok=True)
                _fig.savefig(_p, dpi=int(numwater_save_dpi.value), bbox_inches="tight")
                print(f"saved to {_p}")
            _out = _fig
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Precision–recall colored by structure attributes

    The per-structure precision/recall cloud, with each structure given a
    two-color (or per-cluster) label instead of a continuous scale:

    - **metadata query** — a boolean `pandas.query` over the cohort `metadata.csv`
      (e.g. `resolution <= 2.0`, `r_free <= 0.25 and resolution <= 2.0`); points
      split into **match** / **other**.
    - **subset membership** — provide a `.txt` of PDB IDs (one per line); points
      are colored by whether the structure is in that subset.
    - **hierarchical cluster** — cut the precomputed pairwise matrix
      (`pairwise_metrics_<cutoff>.csv`) with complete linkage on either Cα RMSD or
      max cell diff at a chosen threshold, and color by the resulting cluster
      (same pairwise partition as `optional_find_isomorphous_subset_and_align_ref.py` §6).
      Only clusters with more than 5 members get a unique color; smaller ones are
      pooled into a single grey "other" bucket.

    Reuses the axis-range / font-size controls above.
    """)
    return


@app.cell
def _(DATA, SUBSET, mo, pd):
    _meta_path = SUBSET / "metadata.csv"
    if not _meta_path.exists():
        _meta_path = DATA / "metadata.csv"
    if _meta_path.exists():
        meta_df = pd.read_csv(_meta_path)
        for _col in ("resolution", "r_work", "r_free", "deposited_r_work", "deposited_r_free"):
            if _col in meta_df.columns:
                meta_df[_col] = pd.to_numeric(meta_df[_col], errors="coerce")
        _msg = f"Loaded metadata for **{len(meta_df)}** structures from `{_meta_path}`."
    else:
        meta_df = pd.DataFrame(columns=["pdb_id"])
        _msg = (
            f"**No metadata.csv found** (looked in `{SUBSET}` and `{DATA}`) — "
            "the metadata-column coloring mode is unavailable."
        )
    mo.md(_msg)
    return (meta_df,)


@app.cell
def _(PLOTS_DIR, meta_df, mo):
    _meta_cols = [c for c in meta_df.columns if c != "pdb_id"]

    color_mode = mo.ui.dropdown(
        options=["metadata query", "subset membership", "hierarchical cluster"],
        value="metadata query",
        label="color by",
    )
    metadata_query = mo.ui.text(
        value="resolution <= 2.0",
        placeholder="resolution <= 2.0",
        label="metadata query (splits structures into match / other)",
        full_width=True,
    )
    subset_txt = mo.ui.text(
        value="",
        placeholder="/path/to/subset.txt (one PDB ID per line)",
        label="subset .txt (subset-membership mode)",
        full_width=True,
    )
    hier_metric = mo.ui.dropdown(
        options={"Cα RMSD (Å)": "rmsd_after", "max cell diff (%)": "max_cell_diff"},
        value="max cell diff (%)",
        label="hierarchical metric",
    )
    hier_cutoff = mo.ui.number(
        start=0.0, stop=50.0, step=0.05, value=2.0,
        label="hierarchical cut (max pairwise value within a cluster)",
    )
    attr_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "precision_recall_by_attribute.png"),
        label="save path", full_width=True,
    )
    attr_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    attr_save_button = mo.ui.run_button(label="save figure")
    mo.vstack([
        mo.hstack([color_mode, hier_metric, hier_cutoff], justify="start"),
        metadata_query,
        subset_txt,
        mo.md(f"**metadata columns:** `{'`, `'.join(_meta_cols)}`" if _meta_cols else "*no metadata columns*"),
        mo.hstack([attr_save_path, attr_save_dpi, attr_save_button], justify="start"),
    ])
    return (
        attr_save_button,
        attr_save_dpi,
        attr_save_path,
        color_mode,
        hier_cutoff,
        hier_metric,
        metadata_query,
        subset_txt,
    )


@app.cell
def _(
    Path,
    SUBSET,
    color_mode,
    hier_cutoff,
    hier_metric,
    match_radius,
    meta_df,
    metadata_query,
    mo,
    np,
    pd,
    pr_df,
    sns,
    subset_txt,
):
    # Shared subgroup labels for BOTH the scatter and the F1 histogram below, so
    # the two views always reflect the same split (same colors, same groups).
    # Produces a per-structure label Series aligned to pr_df, an ordered level
    # list, and a matching {level: color} map; attr_message carries any
    # unavailable/invalid-input feedback and attr_labels is None in that case.
    _na = "n/a"
    _small_bucket = None  # set by the hierarchical branch to pool tiny clusters
    attr_labels = None
    attr_levels = None
    attr_title = ""
    attr_message = None

    if color_mode.value == "metadata query":
        _expr = metadata_query.value.strip()
        if meta_df.empty:
            attr_message = mo.md("*No metadata available — cannot use a metadata query.*")
        elif not _expr:
            attr_message = mo.md("*Enter a query, e.g. `resolution <= 2.0`.*")
        else:
            try:
                _match_ids = set(meta_df.query(_expr)["pdb_id"])
            except Exception as _exc:
                attr_message = mo.md(f"**Invalid query:** `{type(_exc).__name__}: {_exc}`")
            else:
                attr_labels = pd.Series(
                    np.where(pr_df["pdb_id"].isin(_match_ids), "match", "other"),
                    index=pr_df.index,
                )
                attr_levels = ["match", "other"]
                attr_title = _expr

    elif color_mode.value == "subset membership":
        _path = Path(subset_txt.value.strip()) if subset_txt.value.strip() else None
        if _path is None or not _path.exists():
            attr_message = mo.md(f"*Provide an existing subset `.txt` (got `{subset_txt.value}`).*")
        else:
            _ids = {_l.strip() for _l in _path.read_text().splitlines() if _l.strip()}
            attr_labels = pd.Series(
                np.where(pr_df["pdb_id"].isin(_ids), "in subset", "not in subset"),
                index=pr_df.index,
            )
            attr_levels = ["in subset", "not in subset"]
            attr_title = _path.name

    else:  # hierarchical cluster
        _pw_path = SUBSET / f"pairwise_metrics_{match_radius}.csv"
        if not _pw_path.exists():
            attr_message = mo.md(
                f"**pairwise_metrics.csv not found:** `{_pw_path}`\n\n"
                "Generate it first:\n\n"
                "```\n"
                "uv run scripts/pairwise_water_metrics.py data/<cohort>.txt\n"
                "```"
            )
        else:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform

            _metric_labels = {"rmsd_after": "Cα RMSD (Å)", "max_cell_diff": "max cell diff (%)"}
            _pw = pd.read_csv(_pw_path)
            _mat = _pw.pivot(index="structure_ref", columns="structure_mobile", values=hier_metric.value)
            _mat = _mat.reindex(index=_mat.index, columns=_mat.index)
            _ids = list(_mat.index)
            # complete linkage on the symmetrized pairwise matrix, cut at the
            # threshold — every pair within a cluster is ≤ cutoff (matches §6 of
            # optional_find_isomorphous_subset_and_align_ref.py).
            _d = _mat.to_numpy(dtype=float).copy()
            _d[np.isnan(_d)] = np.nanmax(_d)
            _d = (_d + _d.T) / 2.0
            np.fill_diagonal(_d, 0.0)
            _lab = fcluster(
                linkage(squareform(_d, checks=False), method="complete"),
                t=float(hier_cutoff.value), criterion="distance",
            )
            _id_to_cluster = {_i: f"C{_c}" for _i, _c in zip(_ids, _lab)}
            attr_labels = pr_df["pdb_id"].map(_id_to_cluster).fillna(_na)
            # Only clusters with more than _min_members get a unique color; smaller
            # ones are pooled into a single grey bucket so a long tail of tiny
            # clusters doesn't exhaust the palette.
            _min_members = 5
            _counts = attr_labels.value_counts()
            _big = [_l for _l in _counts.index if _l != _na and _counts[_l] > _min_members]
            _small_bucket = f"other (≤{_min_members})"
            attr_labels = attr_labels.where(
                attr_labels.isin(_big) | (attr_labels == _na), _small_bucket
            )
            # big clusters largest-first, then the pooled small bucket, then n/a
            attr_levels = list(_big)
            if (attr_labels == _small_bucket).any():
                attr_levels = attr_levels + [_small_bucket]
            if (attr_labels == _na).any():
                attr_levels = attr_levels + [_na]
            attr_title = (
                f"{_metric_labels[hier_metric.value]} ≤ {float(hier_cutoff.value):g} "
                f"({len(_big)} clusters >{_min_members})"
            )

    if attr_labels is not None:
        attr_labels = attr_labels.astype(str)
        # grey the special buckets (unmatched / pooled tiny clusters); give the
        # rest unique colors so the palette isn't spent on singletons.
        _grey = {}
        if _na in attr_levels:
            _grey[_na] = (0.6, 0.6, 0.6)
        if _small_bucket is not None and _small_bucket in attr_levels:
            _grey[_small_bucket] = (0.8, 0.8, 0.8)
        _colored = [_l for _l in attr_levels if _l not in _grey]
        attr_palette = dict(
            zip(_colored, sns.color_palette("Dark2", len(_colored)))
        )
        attr_palette.update(_grey)
    else:
        attr_palette = None

    attr_message if attr_labels is None else None
    return attr_labels, attr_levels, attr_message, attr_palette, attr_title


@app.cell
def _(
    Path,
    attr_labels,
    attr_levels,
    attr_message,
    attr_palette,
    attr_save_button,
    attr_save_dpi,
    attr_save_path,
    attr_title,
    axis_range,
    font_size,
    plt,
    pr_df,
):
    _lim = tuple(axis_range.value)
    _fs = font_size.value

    if attr_labels is None:
        _out = attr_message
    else:
        _fig, _ax = plt.subplots(figsize=(4, 3))
        for _lvl in attr_levels:
            _m = attr_labels == _lvl
            _ax.scatter(
                pr_df.loc[_m, "recall"], pr_df.loc[_m, "precision"],
                color=attr_palette[_lvl], alpha=0.5, edgecolors="k",
                s=30, label=f"{_lvl} ({int(_m.sum())})",
            )
        _ax.set_xlabel("recall", fontsize=_fs)
        _ax.set_ylabel("precision", fontsize=_fs)
        _ax.tick_params(labelsize=_fs)
        if _lim is not None:
            _ax.set_xlim(*_lim)
            _ax.set_ylim(*_lim)
            _ax.set_yticks(_ax.get_xticks())
            _ax.set_ylim(*_lim)
        _ax.set_box_aspect(1)
        _ax.legend(title=attr_title, fontsize=_fs, title_fontsize=8, loc="lower left", frameon=False)
        plt.tight_layout()
        if attr_save_button.value:
            _p = Path(attr_save_path.value)
            _p.parent.mkdir(parents=True, exist_ok=True)
            _fig.savefig(_p, dpi=int(attr_save_dpi.value), bbox_inches="tight")
            print(f"saved to {_p}")
        _out = _fig
    _out
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
def _(mo):
    mo.md(r"""
    ## F1 distribution by subgroup

    Splits the per-structure F1 (vs consensus) by the **same subgroup selected in
    the "colored by structure attributes" controls above** — metadata query,
    subset membership, or hierarchical cluster — and overlays one step histogram
    per subgroup so the distributions are directly comparable. Toggle raw counts
    vs density (each group normalized independently, so sizes don't distort the
    shape) and an optional KDE overlay.
    """)
    return


@app.cell
def _(PLOTS_DIR, mo):
    f1_hist_stat = mo.ui.dropdown(
        options=["count", "density"], value="density", label="y-axis",
    )
    f1_hist_kde = mo.ui.checkbox(value=False, label="kernel smoothing (KDE)")
    f1_hist_bins = mo.ui.slider(
        start=5, stop=60, step=1, value=20, label="bins", show_value=True,
    )
    f1_hist_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "f1_by_subgroup.png"), label="save path", full_width=True,
    )
    f1_hist_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    f1_hist_save_button = mo.ui.run_button(label="save figure")
    mo.vstack([
        mo.hstack([f1_hist_stat, f1_hist_kde, f1_hist_bins], justify="start"),
        mo.hstack([f1_hist_save_path, f1_hist_save_dpi, f1_hist_save_button], justify="start"),
    ])
    return (
        f1_hist_bins,
        f1_hist_kde,
        f1_hist_save_button,
        f1_hist_save_dpi,
        f1_hist_save_path,
        f1_hist_stat,
    )


@app.cell
def _(
    Path,
    attr_labels,
    attr_levels,
    attr_message,
    attr_palette,
    attr_title,
    f1_hist_bins,
    f1_hist_kde,
    f1_hist_save_button,
    f1_hist_save_dpi,
    f1_hist_save_path,
    f1_hist_stat,
    font_size,
    plt,
    pr_df,
    sns,
):
    _fs = font_size.value

    if attr_labels is None:
        _out = attr_message
    else:
        _df = pr_df.assign(_group=attr_labels)
        _fig, _ax = plt.subplots(figsize=(6, 3))
        sns.histplot(
            data=_df, x="f1", hue="_group", hue_order=attr_levels, palette=attr_palette,
            element="step", stat=f1_hist_stat.value, common_norm=False,
            bins=int(f1_hist_bins.value), kde=f1_hist_kde.value, ax=_ax,
        )
        _ax.set_xlabel("F1 vs consensus", fontsize=_fs)
        _ax.set_ylabel(f1_hist_stat.value, fontsize=_fs)
        _ax.tick_params(labelsize=_fs)
        if _ax.get_legend() is not None:
            _ax.get_legend().set_title(attr_title)
        plt.tight_layout()
        if f1_hist_save_button.value:
            _p = Path(f1_hist_save_path.value)
            _p.parent.mkdir(parents=True, exist_ok=True)
            _fig.savefig(_p, dpi=int(f1_hist_save_dpi.value), bbox_inches="tight")
            print(f"saved to {_p}")
        _out = _fig
    _out
    return


if __name__ == "__main__":
    app.run()
