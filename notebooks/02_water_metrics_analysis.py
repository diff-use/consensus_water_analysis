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
    # Metric violins, Q-Q plots and effect sizes — cohorts side by side

    Two levels — **per-water** (consensus vs non-consensus waters) and
    **per-structure** (structures that reproduce the consensus well vs poorly, read
    against their deposited metadata) — each with three independent sections:
    *violins* (mirrored raw histograms, no KDE, each half normalized to its own
    area), *Q-Q* (the same two halves quantile by quantile, unclamped), and
    *statistics* (median / IQR, effect size with bootstrap CI, exportable as CSV).
    The sections are separate cells and the statistics are opt-in, so plotting a
    violin never waits on a bootstrap. Reads the pre-computed `clusters.csv` /
    `cluster_members.csv` / `metadata.csv` of each cohort.
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
    from scipy import stats

    from cw.io import find_cohort_metadata
    from cw.metrics import (
        consensus_centers,
        consensus_water_mask,
        per_structure_consensus_pr,
    )

    return (
        Patch,
        Path,
        config,
        consensus_centers,
        consensus_water_mask,
        find_cohort_metadata,
        np,
        pd,
        per_structure_consensus_pr,
        plt,
        stats,
    )


@app.cell
def _():
    METRIC_LABELS = {
        "edia": "EDIA",
        "b_factor_zscore": "B-factor z-score",
        "b_factor": "B-factor",
        "occupancy": "occupancy",
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
    # Deposited as fractions, reported as percentages in the statistics tables.
    PERCENT_METRICS = frozenset({"r_free", "r_work", "deposited_r_free", "deposited_r_work"})
    # Okabe-Ito, the published colourblind-safe qualitative set. Assigned to cohorts
    # in fixed positional order and never cycled, so a cohort keeps its colour when
    # the cohort list is reordered or shortened.
    COHORT_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    return COHORT_COLORS, METRIC_LABELS, PERCENT_METRICS


@app.cell
def _(mo):
    mo.md("""
    ## Cohorts
    """)
    return


@app.cell
def _(mo):
    # Comma-separated cohort directory names (each a folder under DATA_DIR with its
    # own clusters.csv / cluster_members.csv / metadata.csv). Unlike
    # 01_cluster_analysis.py there is no subset-suffix guessing — spell them out.
    cohorts_input = mo.ui.text(
        value="hewls_65, endothiapepsin_000240_iso, carbonicanhydrase_000562_iso",
        label="cohorts (comma-separated)",
        full_width=True,
    )
    # Optional display labels aligned to the cohorts above; blank falls back to the
    # directory names, which are long on an x-axis.
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
    config,
    consensus_centers,
    consensus_water_mask,
    find_cohort_metadata,
    np,
    pd,
    per_structure_consensus_pr,
):
    def load_cohort(cohort, cutoff, match_radius, suffix=""):
        """(per-water frame, per-structure frame, one-line summary) for one cohort.

        Both frames carry a `cohort` column and a `group` column holding the split
        label, resolved here because each cohort has its own cluster ids,
        occupancies and consensus centers. A water is consensus iff it is a
        within-cutoff member of a cluster whose occupancy clears `cutoff`.
        """
        directory = Path(config.DATA_DIR) / cohort
        clusters = pd.read_csv(directory / f"clusters{suffix}.csv")
        members = pd.read_csv(directory / f"cluster_members{suffix}.csv")

        is_consensus = consensus_water_mask(members, clusters, cutoff)
        waters = members.assign(
            group=np.where(is_consensus, "consensus", "nonconsensus"), cohort=cohort,
        )

        # Per-structure precision/recall/f1/num_water against the consensus centers,
        # joined to deposited metadata. The merge renames metadata's own num_water to
        # num_water_deposited so the clustered count stays the plain num_water;
        # metadata scalars can carry "<missing>" strings, so coerce to numeric.
        centers = consensus_centers(clusters, cutoff)
        pr = per_structure_consensus_pr(members, centers, match_radius)
        meta_path = find_cohort_metadata(config.DATA_DIR, cohort)
        if meta_path is None:
            structures, note = pr, "no metadata.csv found"
        else:
            meta = pd.read_csv(meta_path)
            scalars = [c for c in meta.columns if c != "pdb_id"]
            meta[scalars] = meta[scalars].apply(pd.to_numeric, errors="coerce")
            structures = pr.merge(meta, on="pdb_id", how="left", suffixes=("", "_deposited"))
            matched = int(pr["pdb_id"].isin(meta["pdb_id"]).sum())
            note = f"metadata {meta_path.parent.name}/ ({matched}/{len(pr)} matched)"

        n_consensus = int((waters["group"] == "consensus").sum())
        summary = (
            f"{cohort}: {len(members)} waters, {n_consensus} consensus / "
            f"{len(members) - n_consensus} non-consensus, {len(clusters)} clusters, "
            f"{len(pr)} structures, {note}"
        )
        return waters, structures.assign(cohort=cohort), summary

    return (load_cohort,)


@app.cell
def _(
    cohort_labels_input,
    cohorts_input,
    config,
    cutoff_input,
    load_cohort,
    mo,
    pd,
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
    _match_radius = (
        float(MEMBER_RADIUS) if MEMBER_RADIUS is not None else config.CLUSTER_MEMBER_RADIUS
    )

    _loaded = [
        load_cohort(_cohort, float(cutoff_input.value), _match_radius, _suffix)
        for _cohort in COHORTS
    ]
    for *_, _summary in _loaded:
        print(_summary)

    waters = pd.concat([_frames[0] for _frames in _loaded], ignore_index=True)
    structures = pd.concat([_frames[1] for _frames in _loaded], ignore_index=True)
    print(
        f"total: {len(waters)} waters, {len(structures)} structures "
        f"across {len(COHORTS)} cohorts"
    )
    return COHORTS, COHORT_LABELS, structures, waters


@app.cell
def _(mo):
    mo.md("""
    ## Shared display controls

    Styling for every figure below, so the violin and Q-Q families read as one
    system. Nothing here triggers a statistic.
    """)
    return


@app.cell
def _(mo):
    violin_bins = mo.ui.slider(
        start=10, stop=80, step=5, value=30, label="density bins", show_value=True,
    )
    violin_clamp = mo.ui.checkbox(value=True, label="clamp value axis to 0–99.9 pct")
    violin_center = mo.ui.dropdown(
        options=["median", "mean", "both", "none"], value="median", label="center line",
    )
    violin_iqr = mo.ui.checkbox(value=True, label="show IQR (Q1–Q3)")
    qq_n_quantiles = mo.ui.slider(
        start=25, stop=499, step=2, value=199, label="quantile points", show_value=True,
    )
    qq_show_fit = mo.ui.checkbox(value=True, label="show least-squares fit")
    qq_marker_size = mo.ui.slider(
        start=1, stop=12, step=1, value=4, label="Q-Q marker size", show_value=True,
    )
    shared_despine = mo.ui.checkbox(value=True, label="hide top/right spines")
    shared_vertical = mo.ui.checkbox(
        value=False, label="vertical layout (uncheck = horizontal)",
    )
    shared_font_size = mo.ui.slider(
        start=6, stop=24, step=1, value=14, label="font size", show_value=True,
    )
    mo.vstack([
        mo.hstack([violin_bins, violin_clamp, violin_center, violin_iqr],
                  justify="start"),
        mo.hstack([qq_n_quantiles, qq_show_fit, qq_marker_size], justify="start"),
        mo.hstack([shared_despine, shared_vertical, shared_font_size], justify="start"),
    ])
    return (
        qq_marker_size,
        qq_n_quantiles,
        qq_show_fit,
        shared_despine,
        shared_font_size,
        shared_vertical,
        violin_bins,
        violin_center,
        violin_clamp,
        violin_iqr,
    )


@app.cell
def _(
    qq_marker_size,
    qq_n_quantiles,
    qq_show_fit,
    shared_despine,
    shared_font_size,
    shared_vertical,
    violin_bins,
    violin_center,
    violin_clamp,
    violin_iqr,
):
    # The control values as one kwargs bundle per figure family, so each figure
    # cell is a call and not a wall of `.value`s.
    _shared = dict(
        despine=shared_despine.value,
        vertical=shared_vertical.value,
        font_size=shared_font_size.value,
    )
    VIOLIN_STYLE = dict(
        n_bins=violin_bins.value,
        clamp=violin_clamp.value,
        center_mode=violin_center.value,
        show_iqr=violin_iqr.value,
        **_shared,
    )
    QQ_STYLE = dict(
        n_quantiles=qq_n_quantiles.value,
        show_fit=qq_show_fit.value,
        marker_size=qq_marker_size.value,
        **_shared,
    )
    return QQ_STYLE, VIOLIN_STYLE


@app.cell
def _(Path, mo):
    def save_controls(default_path, button_label, with_dpi=True):
        """`{path, dpi, save}` UI trio for one figure or table. Display the returned
        dict with `show_controls`; hand it to `save_figure` / `save_table`."""
        ui = {"path": mo.ui.text(value=default_path, label="save path", full_width=True)}
        if with_dpi:
            ui["dpi"] = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
        ui["save"] = mo.ui.run_button(label=button_label)
        return ui


    def show_controls(*uis):
        return mo.hstack([e for ui in uis for e in ui.values()], justify="start")


    def _target(ui):
        if not ui["save"].value:
            return None
        out = Path(ui["path"].value)
        out.parent.mkdir(parents=True, exist_ok=True)
        return out


    def save_figure(fig, ui):
        out = _target(ui)
        if out is not None:
            fig.savefig(out, dpi=int(ui["dpi"].value))
            print(f"saved to {out}")


    def save_table(df, ui):
        out = _target(ui)
        if out is not None:
            df.to_csv(out, index=False)
            print(f"saved to {out}")

    return save_controls, save_figure, save_table, show_controls


@app.cell
def _(np, plt):
    def finite(values):
        """`values` as a float array with NaN/inf dropped."""
        values = np.asarray(values, dtype=float)
        return values[np.isfinite(values)]


    def binned_density(values, edges):
        """Raw binned density (area = 1, no kernel smoothing) aligned to bin
        centers, so a violin silhouette is the actual distribution rather than a
        KDE. All-zeros for empty input."""
        values = finite(values)
        if values.size == 0:
            return np.zeros(len(edges) - 1)
        return np.histogram(values, bins=edges, density=True)[0]


    def metric_limits(values, clamp):
        """Non-degenerate (lo, hi) display range: the full range, or 0–99.9 pct
        when clamped."""
        values = finite(values)
        if values.size == 0:
            return 0.0, 1.0
        lo, hi = (
            (float(np.quantile(values, 0.0)), float(np.quantile(values, 0.999)))
            if clamp else (float(values.min()), float(values.max()))
        )
        return lo, hi if hi > lo else lo + 1.0


    def panel_grid(n, panel_w, panel_h, gutters, vertical):
        """`n` panels of a fixed data-rectangle size in one column (`vertical`) or
        one row, so a figure is identical across runs. `gutters` is
        (left, right, top, bottom, gap) in inches. Returns (fig, axes)."""
        left, right, top, bottom, gap = gutters
        if vertical:
            fig_w = left + panel_w + right
            fig_h = bottom + n * panel_h + (n - 1) * gap + top
            spacing = {"hspace": gap / panel_h}
        else:
            fig_w = left + n * panel_w + (n - 1) * gap + right
            fig_h = bottom + panel_h + top
            spacing = {"wspace": gap / panel_w}
        fig, axes = plt.subplots(
            *((n, 1) if vertical else (1, n)), figsize=(fig_w, fig_h), squeeze=False,
        )
        fig.subplots_adjust(
            left=left / fig_w, right=1 - right / fig_w,
            bottom=bottom / fig_h, top=1 - top / fig_h, **spacing,
        )
        return fig, (axes[:, 0] if vertical else axes[0])


    def despine_axis(ax, despine):
        if despine:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    return binned_density, despine_axis, finite, metric_limits, panel_grid


@app.cell
def _(mo):
    mo.md("""
    ## Figure builders

    A **split spec** — `{col, left, right}` with each side carrying its `value` in
    that column plus its label and colours — defines a comparison once and is then
    shared by that level's violin, Q-Q and statistics sections, so the three cannot
    drift apart. Both builders draw only; every statistic lives in a statistics
    section.
    """)
    return


@app.cell
def _(
    Patch,
    binned_density,
    despine_axis,
    finite,
    metric_limits,
    np,
    panel_grid,
):
    def make_violin_figure(df, metrics, cohorts, cohort_labels, split, metric_labels,
                           n_bins, clamp, center_mode, show_iqr, despine, vertical,
                           font_size, legend_loc="upper left"):
        """One panel per metric; within a panel the x-axis is the cohorts and each
        cohort is a split violin, left half = the split's left side.

        Each half is normalized to its own area (the groups differ hugely in size),
        then both are scaled by a single per-cohort factor so the taller peak just
        fills the half-slot — relative peak height between the halves is kept.
        """
        metrics = list(metrics)
        fs = font_size
        halfwidth = 0.4
        fig, axes = panel_grid(
            max(len(metrics), 1), panel_w=1.7 * len(cohorts), panel_h=2.5,
            # left = y-label + tick digits, bottom = x tick labels; both scale mildly
            # with the font so they hug the labels but don't clip when it is bumped.
            gutters=(0.55 + fs * 0.02, 0.2, 0.3, 0.30 + fs * 0.02, 1.2),
            vertical=vertical,
        )

        def draw_marks(ax, values, x_center, sign, color, density, edges, scale):
            # Short horizontal marks on one half: solid = median, diamond = mean,
            # dotted = Q1/Q3, coloured by group. Each spans only the violin's width
            # at its own y (the scaled density of the bin it lands in), so it never
            # overshoots the silhouette. Marks use unclamped values, so one can sit
            # just outside the y-limits.
            values = finite(values)
            if values.size == 0:
                return

            def edge_at(y):
                b = np.searchsorted(edges, y, side="right") - 1
                return x_center + sign * density[int(np.clip(b, 0, len(density) - 1))] * scale

            if center_mode in ("median", "both"):
                m = np.median(values)
                ax.plot([x_center, edge_at(m)], [m, m], color=color, lw=2,
                        solid_capstyle="butt", zorder=4)
            if center_mode in ("mean", "both"):
                mu = values.mean()
                # diamond at the midpoint of the violin's width at the mean's height
                ax.plot([(x_center + edge_at(mu)) / 2], [mu], marker="D", ms=6,
                        mfc="white", mec=color, mew=1.5, zorder=5)
            if show_iqr:
                for q in np.percentile(values, [25, 75]):
                    ax.plot([x_center, edge_at(q)], [q, q], color=color, lw=2.0,
                            ls=":", zorder=4)

        for ax, metric in zip(axes, metrics):
            lo, hi = metric_limits(df[metric], clamp)
            # Shared bin edges across cohorts within a metric → the violins are
            # directly comparable along this panel's y-axis.
            edges = np.linspace(lo, hi, int(n_bins) + 1)
            centers = (edges[:-1] + edges[1:]) / 2

            for i, cohort in enumerate(cohorts):
                sub = df[df["cohort"] == cohort]
                halves = []
                for side, sign in (("left", -1), ("right", +1)):
                    spec = split[side]
                    values = sub.loc[sub[split["col"]] == spec["value"], metric]
                    halves.append((spec, sign, values, binned_density(values, edges)))
                peak = max(density.max() for *_, density in halves)
                if peak <= 0:
                    continue
                scale = halfwidth / peak
                for spec, sign, values, density in halves:
                    ax.fill_betweenx(
                        centers, i, i + sign * density * scale, step="mid",
                        color=spec["fill"], alpha=0.5, edgecolor=spec["edge"],
                        linewidth=0.8,
                    )
                    draw_marks(ax, values, i, sign, spec["mark"], density, edges, scale)

            ax.set_xticks(range(len(cohorts)))
            ax.set_xticklabels(cohort_labels, fontsize=fs, ha="center")
            ax.set_xlim(-0.6, len(cohorts) - 0.4)
            ax.set_ylim(lo, hi)
            ax.set_ylabel(metric_labels.get(metric, metric), fontsize=fs)
            ax.tick_params(axis="y", labelsize=fs)
            despine_axis(ax, despine)

        axes[0].legend(
            handles=[
                Patch(facecolor=split[side]["fill"], alpha=0.5,
                      edgecolor=split[side]["edge"], label=split[side]["label"])
                for side in ("left", "right")
            ],
            fontsize=fs - 2, loc=legend_loc, ncol=2, columnspacing=1.0, framealpha=0.9,
        )
        return fig

    return (make_violin_figure,)


@app.cell
def _(COHORT_COLORS, despine_axis, finite, np, panel_grid, pd, stats):
    def make_qq_figure(df, metrics, cohorts, cohort_labels, split, metric_labels,
                       n_quantiles, despine, vertical, font_size, show_fit=True,
                       marker_size=4, legend_loc="upper left"):
        """One panel per metric; within a panel each cohort is one series of paired
        quantiles — x is the right half's value at probability *p*, y the left
        half's at the same *p*, both read off a shared probability grid so unequal
        group sizes are handled by interpolation.

        The dashed 45° line is "the two halves are one distribution": a constant
        offset is a pure location shift (the reading Cliff's delta and Mann-Whitney
        assume), a slope away from 1 is a scale difference, and departure at one end
        only is a tail-only effect that a single pooled effect size misreports.
        Deliberately unclamped, with both halves on one shared axis pair rather than
        each normalized to its own area — the tails a violin hides are the point.

        Returns (figure, per-cohort least-squares fits) — the fits are a plain
        regression of the paired quantiles, not an inferential test.
        """
        metrics = list(metrics)
        fs = font_size
        # Mid-bin probabilities: avoids asking for the 0th and 100th percentile,
        # where a single outlier would set the whole panel's limits.
        count = int(n_quantiles)
        probs = (np.arange(count) + 0.5) / count
        left_spec, right_spec = split["left"], split["right"]
        # Square panels so the reference line sits at a true 45°; wider gutters than
        # the violins because both axes carry a label here.
        fig, axes = panel_grid(
            max(len(metrics), 1), panel_w=2.6, panel_h=2.6,
            gutters=(0.75 + fs * 0.03, 0.25, 0.3 + fs * 0.025, 0.55 + fs * 0.03, 1.15),
            vertical=vertical,
        )
        fits = []

        for ax, metric in zip(axes, metrics):
            low, high = np.inf, -np.inf
            for index, cohort in enumerate(cohorts):
                sub = df[df["cohort"] == cohort]
                sides = {
                    side: finite(sub.loc[sub[split["col"]] == split[side]["value"], metric])
                    for side in ("left", "right")
                }
                row = {
                    "metric": metric_labels.get(metric, metric),
                    "cohort": cohort_labels[index],
                    f"n_{left_spec['value']}": sides["left"].size,
                    f"n_{right_spec['value']}": sides["right"].size,
                }
                if min(sides["left"].size, sides["right"].size) < 2:
                    fits.append({**row, "slope": np.nan, "intercept": np.nan, "r2": np.nan})
                    continue
                y = np.quantile(sides["left"], probs)
                x = np.quantile(sides["right"], probs)
                color = COHORT_COLORS[index % len(COHORT_COLORS)]
                ax.plot(x, y, ls="none", marker="o", ms=marker_size, mfc=color,
                        mec="none", alpha=0.8, label=cohort_labels[index], zorder=3)
                fit = stats.linregress(x, y)
                if show_fit:
                    span = np.array([x.min(), x.max()])
                    ax.plot(span, fit.intercept + fit.slope * span, color=color,
                            lw=1.4, alpha=0.9, zorder=2)
                fits.append({
                    **row,
                    "slope": round(float(fit.slope), 3),
                    "intercept": round(float(fit.intercept), 3),
                    "r2": round(float(fit.rvalue ** 2), 4),
                })
                low = min(low, y.min(), x.min())
                high = max(high, y.max(), x.max())

            if not np.isfinite(low):
                low, high = 0.0, 1.0
            span = (high - low) or 1.0
            low, high = low - 0.04 * span, high + 0.04 * span
            ax.plot([low, high], [low, high], color="black", lw=1.0, ls="--",
                    alpha=0.55, zorder=1)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal", adjustable="box")
            # Metric on the title, group names on the axes: repeating the metric on
            # both axes reads as redundant and, in the vertical layout, a long metric
            # name on the x-axis overruns the figure edge.
            ax.set_title(metric_labels.get(metric, metric), fontsize=fs, pad=6)
            ax.set_xlabel(right_spec["label"], fontsize=fs)
            ax.set_ylabel(left_spec["label"], fontsize=fs)
            ax.tick_params(labelsize=fs)
            despine_axis(ax, despine)

        if len(cohorts) > 1:
            axes[0].legend(fontsize=fs - 2, loc=legend_loc, framealpha=0.9,
                           handletextpad=0.4, borderpad=0.5)
        return fig, pd.DataFrame(fits)

    return (make_qq_figure,)


@app.cell
def _(mo):
    mo.md("""
    ## Statistics builders

    `comparison_table` is the only place a statistic is computed, and both
    statistics sections below call it. Nothing here is drawn on a figure.
    """)
    return


@app.cell
def _(PERCENT_METRICS, np, pd, stats):
    def effect_size(a, b, test):
        """Signed effect size matching `test`; positive = `a` sits higher than `b`.

        mannwhitney  Cliff's delta = P(a>b) - P(a<b), the Mann-Whitney U rescaled
                     to [-1, 1] (and, for two independent samples, identical to the
                     rank-biserial correlation). 0 = the halves overlap completely,
                     ±1 = they separate completely. By searchsorted rather than via
                     scipy, so the bootstrap can call it thousands of times.
        ks           the KS D statistic (unsigned, in [0, 1])
        welch        Cohen's d
        """
        if test == "ks":
            return float(stats.ks_2samp(a, b).statistic)
        if test == "welch":
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")
        b_sorted = np.sort(b)
        n_pairs = a.size * b.size
        b_below_a = int(np.searchsorted(b_sorted, a, "left").sum())
        b_above_a = n_pairs - int(np.searchsorted(b_sorted, a, "right").sum())
        return (b_below_a - b_above_a) / n_pairs


    def group_values(values, units):
        """Split `values` into one array per unit label."""
        order = np.argsort(units, kind="stable")
        values, units = values[order], units[order]
        edges = np.flatnonzero(np.r_[True, units[1:] != units[:-1], True])
        return {units[i]: values[i:j] for i, j in zip(edges[:-1], edges[1:])}


    def compare_halves(left_values, right_values, test="mannwhitney", n_boot=2000,
                       seed=0, left_units=None, right_units=None):
        """Two-sample comparison of a split's two halves, NaNs dropped.

        Returns p / effect / ci_low / ci_high / n_left / n_right. The effect size is
        the headline number and the p-value secondary: at large n a difference far
        too small to matter still clears p < 0.001.

        `n_boot` percentile-bootstrap resamples (0 to skip) give the 95% CI on the
        effect size. Pass `*_units` when rows are not independent (waters sharing a
        pdb_id): the bootstrap then resamples whole units, one shared draw per
        iteration so the within-unit split stays paired, and p is inverted from that
        distribution rather than from a test that would count correlated rows as
        independent — which floors it at 1 / n_boot.
        """
        a, b = np.asarray(left_values, dtype=float), np.asarray(right_values, dtype=float)
        keep_a, keep_b = np.isfinite(a), np.isfinite(b)
        a, b = a[keep_a], b[keep_b]
        if a.size < 2 or b.size < 2:
            return dict(p=float("nan"), effect=float("nan"), ci_low=float("nan"),
                        ci_high=float("nan"), n_left=a.size, n_right=b.size)
        clustered = left_units is not None

        p = float("nan")
        if not clustered:
            if test == "mannwhitney":
                p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            elif test == "ks":
                p = stats.ks_2samp(a, b).pvalue
            else:
                p = stats.ttest_ind(a, b, equal_var=False).pvalue

        ci_low = ci_high = float("nan")
        if n_boot:
            rng = np.random.default_rng(seed)
            if clustered:
                by_unit_a = group_values(a, np.asarray(left_units)[keep_a])
                by_unit_b = group_values(b, np.asarray(right_units)[keep_b])
                names = np.array(sorted(set(by_unit_a) | set(by_unit_b)))
                empty = np.empty(0)

                def resample():
                    drawn = rng.choice(names, names.size)
                    return (np.concatenate([by_unit_a.get(u, empty) for u in drawn]),
                            np.concatenate([by_unit_b.get(u, empty) for u in drawn]))
            else:
                def resample():
                    return rng.choice(a, a.size), rng.choice(b, b.size)

            boot = []
            for _ in range(int(n_boot)):
                draw_a, draw_b = resample()
                boot.append(
                    effect_size(draw_a, draw_b, test)
                    if draw_a.size and draw_b.size else np.nan
                )
            boot = np.array(boot)
            ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
            if clustered:
                side = min(np.nanmean(boot <= 0), np.nanmean(boot >= 0))
                p = float(np.clip(2 * side, 1 / int(n_boot), 1.0))
        return dict(p=float(p), effect=effect_size(a, b, test),
                    ci_low=float(ci_low), ci_high=float(ci_high),
                    n_left=a.size, n_right=b.size)


    def p_stars(p):
        if not np.isfinite(p):
            return ""
        for threshold, mark in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")):
            if p < threshold:
                return mark
        return "ns"


    def mantissa(p):
        """`p` keeping 2 decimals of its mantissa (1.43e-22) — rounding to 2
        decimals outright would render every p as 0.00."""
        return float(f"{p:.2e}")


    def spearman(df, x_col, y_col):
        """Rank correlation of two columns, NaNs dropped; (nan, nan) when undefined
        (the same column twice, or fewer than 3 complete pairs)."""
        if x_col == y_col:
            return float("nan"), float("nan")
        pair = df[[x_col, y_col]].dropna()
        if len(pair) < 3:
            return float("nan"), float("nan")
        rho, p = stats.spearmanr(pair[x_col], pair[y_col])
        return float(rho), float(p)


    def comparison_table(df, metrics, cohorts, cohort_labels, split, metric_labels,
                         test="mannwhitney", n_boot=2000, unit_col=None,
                         continuous_split_col=None, percent_metrics=PERCENT_METRICS):
        """One row per metric × cohort: both halves' median / IQR / n, the effect
        size with its bootstrap CI, and the p-value — the whole statistical readout
        of a split-violin figure, tidy enough to go straight into a manuscript
        table. This is the expensive cell in each level: `n_boot` drives it.

        Metric-major, so the table reads down one panel before moving to the next,
        and cohorts keep the order they were entered in. `percent_metrics` are
        deposited as fractions and reported as percentages — only the medians and
        IQRs move, the rank-based columns being scale-invariant. `unit_col`
        resamples whole units instead of rows; `continuous_split_col` adds the rank
        correlation of the *undichotomized* split metric against the panel metric,
        a check on whether a result is an artifact of where the cut was made.
        p-values are per metric × cohort and **unadjusted** for multiplicity.
        """
        labels = dict(zip(cohorts, cohort_labels))
        rows = []
        for metric in metrics:
            scale = 100.0 if metric in percent_metrics else 1.0
            name = metric_labels.get(metric, metric)
            for cohort in cohorts:
                sub = df[df["cohort"] == cohort]
                halves = {
                    side: sub[sub[split["col"]] == split[side]["value"]]
                    for side in ("left", "right")
                }
                result = compare_halves(
                    halves["left"][metric], halves["right"][metric], test, n_boot,
                    left_units=None if unit_col is None else halves["left"][unit_col],
                    right_units=None if unit_col is None else halves["right"][unit_col],
                )
                row = {
                    "metric": f"{name} (%)" if scale != 1.0 else name,
                    "cohort": labels.get(cohort, cohort),
                }
                for side in ("left", "right"):
                    key = split[side]["value"]
                    values = halves[side][metric].dropna() * scale
                    row[f"{key}_median"] = round(float(values.median()), 2)
                    row[f"{key}_iqr"] = (round(float(values.quantile(0.25)), 2),
                                         round(float(values.quantile(0.75)), 2))
                row[f"n_{split['left']['value']}"] = result["n_left"]
                row[f"n_{split['right']['value']}"] = result["n_right"]
                row["r"] = round(result["effect"], 2)
                row["CI"] = (round(result["ci_low"], 2), round(result["ci_high"], 2))
                row["p"] = mantissa(result["p"])
                row["sig"] = p_stars(result["p"])
                if continuous_split_col:
                    rho, rho_p = spearman(sub, continuous_split_col, metric)
                    row["split_rho"] = round(rho, 2)
                    row["split_rho_p"] = mantissa(rho_p)
                rows.append(row)
        return pd.DataFrame(rows)

    return (comparison_table,)


@app.cell
def _(mo):
    mo.md("""
    # Per-water — consensus vs non-consensus

    A water is **consensus** iff it is a within-cutoff member of a cluster whose
    occupancy clears the cutoff set at the top; everything else — noise,
    radius-rejected, low-occupancy members — is non-consensus. The two halves are
    the same throughout this level's three sections.
    """)
    return


@app.cell
def _(mo, waters):
    _candidates = ["b_factor_zscore", "edia", "b_factor", "occupancy", "muse_score"]
    _available = [
        c for c in _candidates if c in waters.columns and waters[c].notna().any()
    ]
    water_metrics = mo.ui.multiselect(
        options=_available,
        value=[m for m in ["b_factor_zscore", "edia"] if m in _available],
        label="water metrics (one panel each, left→right)",
    )
    water_metrics
    return (water_metrics,)


@app.cell
def _():
    WATER_SPLIT = dict(
        col="group",
        left=dict(value="consensus", label="consensus",
                  fill="r", edge="r", mark="darkred"),
        right=dict(value="nonconsensus", label="non-consensus",
                   fill="grey", edge="dimgrey", mark="black"),
    )
    return (WATER_SPLIT,)


@app.cell
def _(mo):
    mo.md("""
    ## Per-water violins
    """)
    return


@app.cell
def _(save_controls, show_controls):
    water_violin_save = save_controls(
        "data/plots/per_water_metric_violins.png", "save water violins",
    )
    show_controls(water_violin_save)
    return (water_violin_save,)


@app.cell
def _(
    COHORTS,
    COHORT_LABELS,
    METRIC_LABELS,
    VIOLIN_STYLE,
    WATER_SPLIT,
    make_violin_figure,
    save_figure,
    water_metrics,
    water_violin_save,
    waters,
):
    _fig = make_violin_figure(
        waters, list(water_metrics.value), COHORTS, COHORT_LABELS,
        WATER_SPLIT, METRIC_LABELS, legend_loc="upper left", **VIOLIN_STYLE,
    )
    save_figure(_fig, water_violin_save)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-water Q-Q

    `x` is a non-consensus water's value at probability *p*, `y` a consensus
    water's at the same *p*, dashed line `y = x`. This answers what the violins
    cannot — are consensus waters better *everywhere*, or is the whole signal a
    pile-up at the top of the `edia` range? — because the violins' 0–99.9 pct clamp
    and per-half area normalization both hide exactly that. `edia` is bounded with a
    lot of mass at its ceiling, so a stair-stepped upper right is ties, not a
    plotting artifact. Waters not being independent within a structure is no
    objection here: it invalidates *inference*, not a description of the marginals.
    """)
    return


@app.cell
def _(save_controls, show_controls):
    water_qq_save = save_controls(
        "data/plots/per_water_metric_qq.png", "save water Q-Q",
    )
    show_controls(water_qq_save)
    return (water_qq_save,)


@app.cell
def _(
    COHORTS,
    COHORT_LABELS,
    METRIC_LABELS,
    QQ_STYLE,
    WATER_SPLIT,
    make_qq_figure,
    save_figure,
    water_metrics,
    water_qq_save,
    waters,
):
    _fig, water_qq_fits = make_qq_figure(
        waters, list(water_metrics.value), COHORTS, COHORT_LABELS,
        WATER_SPLIT, METRIC_LABELS, legend_loc="upper left", **QQ_STYLE,
    )
    save_figure(_fig, water_qq_save)
    _fig
    return (water_qq_fits,)


@app.cell
def _(mo, water_qq_fits):
    # slope 1 + intercept 0 = one distribution; the fit decomposes the difference
    # into scale (slope) and location (intercept).
    mo.ui.table(water_qq_fits, selection=None, page_size=30)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-water statistics

    `r` is Cliff's delta / the rank-biserial correlation, negative meaning the
    consensus half sits lower, and its **95% CI** carries the precision — a small
    cohort earns a wide CI rather than a hidden one. Waters within a structure share
    a crystal, a resolution and a refinement protocol, so the bootstrap resamples
    whole `pdb_id`s; the p-value inverted from that distribution floors at
    `1 / n_boot` and would pin every row to the floor, so it is dropped here rather
    than reported as a measurement. Each resample rebuilds every water in the
    cohort, making this far heavier than its per-structure counterpart — hence the
    switch, off by default.
    """)
    return


@app.cell
def _(mo, save_controls, show_controls):
    water_stats_run = mo.ui.checkbox(value=False, label="compute per-water statistics")
    water_stats_test = mo.ui.dropdown(
        options=["mannwhitney", "ks", "welch"], value="mannwhitney", label="test",
    )
    water_stats_n_boot = mo.ui.number(
        start=0, stop=5000, step=100, value=500, label="pdb_id bootstrap resamples",
    )
    water_table_save = save_controls(
        "data/consensus_vs_nonconsensus_comparison_table.csv",
        "save water table", with_dpi=False,
    )
    mo.vstack([
        mo.hstack([water_stats_run, water_stats_test, water_stats_n_boot],
                  justify="start"),
        show_controls(water_table_save),
    ])
    return (
        water_stats_n_boot,
        water_stats_run,
        water_stats_test,
        water_table_save,
    )


@app.cell
def _(
    COHORTS,
    COHORT_LABELS,
    METRIC_LABELS,
    WATER_SPLIT,
    comparison_table,
    mo,
    save_table,
    water_metrics,
    water_stats_n_boot,
    water_stats_run,
    water_stats_test,
    water_table_save,
    waters,
):
    mo.stop(
        not water_stats_run.value,
        mo.md("*Tick **compute per-water statistics** above to run the bootstrap.*"),
    )
    water_comparison_table = comparison_table(
        waters, list(water_metrics.value), COHORTS, COHORT_LABELS,
        WATER_SPLIT, METRIC_LABELS,
        test=water_stats_test.value, n_boot=int(water_stats_n_boot.value),
        unit_col="pdb_id",
    ).drop(columns=["p", "sig"])
    save_table(water_comparison_table, water_table_save)
    mo.ui.table(water_comparison_table, selection=None, page_size=30)
    return


@app.cell
def _(mo):
    mo.md("""
    # Per-structure — good vs poor structures

    Each structure is **good** or **poor** by whether a chosen per-structure metric
    (default `f1`, its agreement with the consensus) is at or above its cohort's
    cutoff, and the sections below then compare the two groups' deposited metadata:
    do the structures that best reproduce the conserved sites also come from the
    higher-quality depositions? The cutoff is **per-cohort** — the metric's scale
    differs between cohorts, so a shared cutoff wouldn't halve them comparably —
    defaulting to that cohort's median and editable below. The split metric can be
    any per-structure column (precision / recall / f1 / num_water, or a metadata
    column); the panels can be any numeric metadata columns.
    """)
    return


@app.cell
def _(pd, structures):
    structure_numeric_cols = [
        c for c in structures.columns
        if c not in ("pdb_id", "cohort", "group")
        and pd.api.types.is_numeric_dtype(structures[c])
        and structures[c].notna().any()
    ]
    return (structure_numeric_cols,)


@app.cell
def _(mo, structure_numeric_cols):
    structure_split_metric = mo.ui.dropdown(
        options=structure_numeric_cols,
        value="f1" if "f1" in structure_numeric_cols else structure_numeric_cols[0],
        label="split metric (good = ≥ cohort cutoff)",
    )
    _dist_default = [
        m for m in ["resolution", "deposited_r_free"] if m in structure_numeric_cols
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
    # Rebuilt and re-defaulted to each cohort's own median whenever the split
    # metric changes; edit a box to move that cohort's good/poor boundary.
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
        mo.md(
            f"**good = {_split} ≥ cutoff** — one cutoff per cohort, "
            "defaulting to its median:"
        ),
        structure_cutoffs,
    ])
    return (structure_cutoffs,)


@app.cell
def _(
    COHORTS,
    METRIC_LABELS,
    np,
    structure_cutoffs,
    structure_split_metric,
    structures,
):
    # Labelled once here and shared by all three per-structure sections. Structures
    # whose split metric is missing are dropped from both groups.
    split_metric = structure_split_metric.value
    split_cutoffs = structure_cutoffs.value
    _base = structures.dropna(subset=[split_metric])
    structures_labelled = _base.assign(
        group=np.where(
            _base[split_metric] >= _base["cohort"].map(split_cutoffs), "good", "poor",
        )
    )
    # Blue accent instead of the per-water figure's red, so the two levels are never
    # confused; red stays reserved for consensus waters.
    STRUCTURE_SPLIT = dict(
        col="group",
        left=dict(value="good",
                  label=f"≥ {METRIC_LABELS.get(split_metric, split_metric)} cutoff",
                  fill="mediumblue", edge="mediumblue", mark="darkblue"),
        right=dict(value="poor", label="below cutoff",
                   fill="saddlebrown", edge="dimgrey", mark="saddlebrown"),
    )

    print(f"split on {split_metric} at each cohort's own cutoff:")
    for _cohort in COHORTS:
        _rows = structures_labelled[structures_labelled["cohort"] == _cohort]
        print(
            f"  {_cohort}: cutoff {split_cutoffs.get(_cohort):.4g} → "
            f"{int((_rows['group'] == 'good').sum())} good (≥), "
            f"{int((_rows['group'] == 'poor').sum())} poor (<)"
        )
    return STRUCTURE_SPLIT, split_cutoffs, split_metric, structures_labelled


@app.cell
def _(mo):
    mo.md("""
    ## Per-structure violins
    """)
    return


@app.cell
def _(save_controls, show_controls, split_metric):
    structure_violin_save = save_controls(
        f"data/plots/{split_metric}_good_vs_poor_metadata_violins.png",
        "save structure violins",
    )
    show_controls(structure_violin_save)
    return (structure_violin_save,)


@app.cell
def _(
    COHORTS,
    COHORT_LABELS,
    METRIC_LABELS,
    STRUCTURE_SPLIT,
    VIOLIN_STYLE,
    make_violin_figure,
    save_figure,
    structure_dist_metrics,
    structure_violin_save,
    structures_labelled,
):
    _fig = make_violin_figure(
        structures_labelled, list(structure_dist_metrics.value), COHORTS, COHORT_LABELS,
        STRUCTURE_SPLIT, METRIC_LABELS, legend_loc="upper left", **VIOLIN_STYLE,
    )
    save_figure(_fig, structure_violin_save)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-structure Q-Q

    `x` is a poor structure's value at probability *p*, `y` a good structure's at
    the same *p* — read it like the per-water Q-Q: offset is location, slope is
    scale, one-sided departure is a tail-only effect. This is the panel that checks
    the premise behind the numbers the statistics section prints: Mann-Whitney is
    interpretable as a location shift only if the two halves have the same shape,
    and Cliff's delta collapses to one scalar that a uniform small shift and a large
    tail-only difference can both produce. Points parallel to `y = x` make that
    reading safe; if they fan or bend, the effect size is describing something other
    than "good structures are better by a constant".
    """)
    return


@app.cell
def _(save_controls, show_controls, split_metric):
    structure_qq_save = save_controls(
        f"data/plots/{split_metric}_good_vs_poor_metadata_qq.png",
        "save structure Q-Q",
    )
    show_controls(structure_qq_save)
    return (structure_qq_save,)


@app.cell
def _(
    COHORTS,
    COHORT_LABELS,
    METRIC_LABELS,
    QQ_STYLE,
    STRUCTURE_SPLIT,
    make_qq_figure,
    save_figure,
    structure_dist_metrics,
    structure_qq_save,
    structures_labelled,
):
    _fig, structure_qq_fits = make_qq_figure(
        structures_labelled, list(structure_dist_metrics.value), COHORTS, COHORT_LABELS,
        STRUCTURE_SPLIT, METRIC_LABELS, legend_loc="upper left", **QQ_STYLE,
    )
    save_figure(_fig, structure_qq_save)
    _fig
    return (structure_qq_fits,)


@app.cell
def _(mo, structure_qq_fits):
    mo.ui.table(structure_qq_fits, selection=None, page_size=30)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-structure statistics

    One row per metric × cohort, ready for a manuscript table: both halves' median
    / IQR / n, then `r` (Cliff's delta, the rank-biserial correlation) with its
    bootstrap **95% CI**, which is the headline pair — `p` is secondary, since at
    several hundred structures per half a difference far too small to matter still
    clears `p < 0.001`, and it is unadjusted for the metric × cohort multiplicity.
    `split_rho` is the rank correlation of the *undichotomized* split metric against
    the panel metric across both halves at once: a robustness check on where the cut
    was made, not a second result. R-factor panels are reported in percent.
    """)
    return


@app.cell
def _(mo, save_controls, show_controls):
    structure_stats_run = mo.ui.checkbox(
        value=False, label="compute per-structure statistics",
    )
    structure_stats_test = mo.ui.dropdown(
        options=["mannwhitney", "ks", "welch"], value="mannwhitney", label="test",
    )
    structure_stats_n_boot = mo.ui.number(
        start=0, stop=20000, step=500, value=2000, label="bootstrap resamples",
    )
    structure_table_save = save_controls(
        "data/good_vs_poor_comparison_table.csv", "save structure table",
        with_dpi=False,
    )
    mo.vstack([
        mo.hstack([structure_stats_run, structure_stats_test, structure_stats_n_boot],
                  justify="start"),
        show_controls(structure_table_save),
    ])
    return (
        structure_stats_n_boot,
        structure_stats_run,
        structure_stats_test,
        structure_table_save,
    )


@app.cell
def _(
    COHORTS,
    COHORT_LABELS,
    METRIC_LABELS,
    STRUCTURE_SPLIT,
    comparison_table,
    mo,
    save_table,
    split_cutoffs,
    split_metric,
    structure_dist_metrics,
    structure_stats_n_boot,
    structure_stats_run,
    structure_stats_test,
    structure_table_save,
    structures_labelled,
):
    mo.stop(
        not structure_stats_run.value,
        mo.md("*Tick **compute per-structure statistics** above to run the bootstrap.*"),
    )
    structure_comparison_table = comparison_table(
        structures_labelled, list(structure_dist_metrics.value), COHORTS, COHORT_LABELS,
        STRUCTURE_SPLIT, METRIC_LABELS,
        test=structure_stats_test.value, n_boot=int(structure_stats_n_boot.value),
        continuous_split_col=split_metric,
    )
    # The cutoff actually used, so a saved table documents its own split.
    _by_label = {
        label: split_cutoffs.get(cohort) for cohort, label in zip(COHORTS, COHORT_LABELS)
    }
    structure_comparison_table.insert(
        2, f"cutoff_{split_metric}",
        structure_comparison_table["cohort"].map(_by_label).round(2),
    )
    save_table(structure_comparison_table, structure_table_save)
    mo.ui.table(structure_comparison_table, selection=None, page_size=30)
    return


if __name__ == "__main__":
    app.run()
