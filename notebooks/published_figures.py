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
    # Published figures

    Redraws the main-text figures: the per-structure precision/recall Pareto front
    (one panel per cohort), the per-water metric violins and the per-structure
    metadata violins. Every figure is shown inline; nothing is written to disk —
    export from the notebook if you need a file.

    Every setting is a constant in the next cell and there are no controls, so a run
    is reproducible. Reads only the pre-computed `clusters.csv`,
    `cluster_members.csv` and `metadata.csv` of each cohort — no CIF parsing. The
    exploratory versions of these figures, with controls, live in
    `01_cluster_analysis.py` and `02_water_metrics_analysis.py`; all three notebooks
    draw with the same `cw.plots` code, so they cannot drift apart.
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

    from cw.io import load_cohort_frames
    from cw.metrics import pareto_front
    from cw.plots import (
        METRIC_LABELS,
        WATER_SPLIT,
        make_violin_figure,
        plot_pr_scatter,
        structure_split_spec,
    )

    return (
        METRIC_LABELS,
        WATER_SPLIT,
        config,
        load_cohort_frames,
        make_violin_figure,
        np,
        pareto_front,
        pd,
        plot_pr_scatter,
        structure_split_spec,
    )


@app.cell
def _(config):
    COHORTS = ["hewls_65", "endothiapepsin_000240_iso", "carbonicanhydrase_000562_iso"]
    # Directory names are long; these title the Pareto panels and label the violin
    # x-axes.
    COHORT_LABELS = ["HEWL", "EAP", "CA"]

    CONSENSUS_CUTOFF = 0.3
    MATCH_RADIUS = config.CLUSTER_MEMBER_RADIUS

    WATER_METRICS = ["b_factor_zscore", "edia"]
    STRUCTURE_METRICS = ["resolution", "r_free"]
    # Per-structure good/poor split, cut at each cohort's own median.
    SPLIT_METRIC = "f1"

    VIOLIN_STYLE = dict(
        n_bins=30, clamp=True, center_mode="median", show_iqr=True,
        despine=True, vertical=False, font_size=14,
    )
    PARETO_STYLE = dict(alpha=0.5, lim=(0.0, 1.02), fontsize=14, marker_size=30)
    return (
        COHORTS,
        COHORT_LABELS,
        CONSENSUS_CUTOFF,
        MATCH_RADIUS,
        PARETO_STYLE,
        SPLIT_METRIC,
        STRUCTURE_METRICS,
        VIOLIN_STYLE,
        WATER_METRICS,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Load

    One row per water in `waters`, one row per structure in `structures`, both
    carrying a `cohort` column and the consensus/non-consensus `group` label.
    """)
    return


@app.cell
def _(COHORTS, CONSENSUS_CUTOFF, MATCH_RADIUS, config, load_cohort_frames, pd):
    _loaded = [
        load_cohort_frames(config.DATA_DIR, _cohort, CONSENSUS_CUTOFF, MATCH_RADIUS)
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
    return structures, waters


@app.cell
def _(mo):
    mo.md("""
    ## Precision/recall Pareto front

    One panel per cohort, shown inline: every structure's waters scored against that
    cohort's consensus centers, colored by how many waters the structure contributes.
    The red line is the non-dominated upper-right envelope and the star is the max-F1
    knee.
    """)
    return


@app.cell
def _(
    COHORTS,
    COHORT_LABELS,
    PARETO_STYLE,
    mo,
    pareto_front,
    plot_pr_scatter,
    structures,
):
    _figs = []
    for _cohort, _label in zip(COHORTS, COHORT_LABELS):
        _pr = structures[structures["cohort"] == _cohort]
        _knee = _pr.loc[_pr["f1"].idxmax()]
        _front = pareto_front(_pr)

        _fig, _ax = plot_pr_scatter(
            _pr, color="num_water", color_label="#water", n_color_bins=10,
            title=_label, **PARETO_STYLE,
        )
        _ax.scatter([_knee["recall"]], [_knee["precision"]], marker="*", s=350,
                    facecolors="r", edgecolors="white", lw=1, zorder=5)
        _ax.plot(_front["recall"], _front["precision"], color="red", lw=3)
        _ax.set_xticks(_ax.get_yticks())
        _ax.set_xlim(*PARETO_STYLE["lim"])
        _figs.append(_fig)

        print(
            f"{_cohort}: knee precision={_knee['precision']:.3f} "
            f"recall={_knee['recall']:.3f} F1={_knee['f1']:.3f} "
            f"({len(_front)} front points)"
        )
    mo.hstack(_figs, widths="equal")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-water violins

    Consensus vs non-consensus waters, one panel per metric, cohorts along the x-axis.
    Each half is a mirrored raw histogram normalized to its own area, so the two
    groups are comparable despite differing hugely in size.
    """)
    return


@app.cell
def _(
    COHORTS,
    COHORT_LABELS,
    METRIC_LABELS,
    VIOLIN_STYLE,
    WATER_METRICS,
    WATER_SPLIT,
    make_violin_figure,
    waters,
):
    make_violin_figure(
        waters, WATER_METRICS, COHORTS, COHORT_LABELS,
        WATER_SPLIT, METRIC_LABELS, legend_loc="upper left", **VIOLIN_STYLE,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-structure violins

    Structures that reproduce their cohort's consensus well vs poorly — split on F1 at
    each cohort's own median — read against their deposited metadata.
    """)
    return


@app.cell
def _(
    COHORTS,
    COHORT_LABELS,
    METRIC_LABELS,
    SPLIT_METRIC,
    STRUCTURE_METRICS,
    VIOLIN_STYLE,
    make_violin_figure,
    np,
    structure_split_spec,
    structures,
):
    # Structures with no split metric are dropped from both groups.
    _base = structures.dropna(subset=[SPLIT_METRIC])
    # Rounded to 4 dp to match 02_water_metrics_analysis.py's cutoff control, whose
    # default is the rounded median. Structures sitting exactly on the unrounded
    # median fall on the other side of the rounded value, so dropping the round()
    # here would move a few structures between the groups.
    _cutoffs = _base.groupby("cohort")[SPLIT_METRIC].median().round(4)
    _labeled = _base.assign(
        group=np.where(
            _base[SPLIT_METRIC] >= _base["cohort"].map(_cutoffs), "good", "poor",
        )
    )
    for _cohort in COHORTS:
        _rows = _labeled[_labeled["cohort"] == _cohort]
        print(
            f"{_cohort}: {SPLIT_METRIC} cutoff {_cutoffs.get(_cohort):.4g} → "
            f"{int((_rows['group'] == 'good').sum())} good, "
            f"{int((_rows['group'] == 'poor').sum())} poor"
        )

    make_violin_figure(
        _labeled, STRUCTURE_METRICS, COHORTS, COHORT_LABELS,
        structure_split_spec(SPLIT_METRIC, METRIC_LABELS), METRIC_LABELS,
        legend_loc="upper left", **VIOLIN_STYLE,
    )
    return


if __name__ == "__main__":
    app.run()
