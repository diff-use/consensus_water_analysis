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
    # experimental — frozen-reference filtering

    Tests one question: **if you fix the consensus reference from the unfiltered
    clustering and then filter each structure's waters, does per-structure
    precision/recall improve?**

    Unlike `experimental_clustering_panel_grid.py`, this notebook does **not**
    re-cluster each filter variant. There is one clustering — the unfiltered run —
    and it is frozen:

    - **P/R reference is frozen.** The consensus centers come from
      `consensus_centers(unfiltered clusters, cutoff)` computed once, and are reused
      for every filter row. Filtering only shrinks each structure's *prediction*
      set, so recall can only hold or fall — the direction each structure's point
      moves is a direct test of the "filtered waters are mostly non-consensus noise"
      premise (straight up = premise true; left = premise false).
    - **Consensus columns are recomputed at fixed cluster geometry.** Each surviving
      water keeps its unfiltered `cluster_id`; occupancy is recomputed from the
      surviving within-cutoff members. This isolates the *occupancy headwind*
      (removing waters, never structures, lowers every cluster's occupancy) from the
      HDBSCAN re-partitioning that confounds the re-clustered panel-grid summary.

    Filters are applied in-memory to the unfiltered `cluster_members.csv`:
    `edia >= X` and `b_factor_zscore <= X` select exactly the waters the real
    EDIA / B-factor z-score filters keep, so no CIFs are re-read. Requires the
    `b_factor_zscore` column (add it with `scripts/add_bfactor_zscore_column.py`).
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(".")))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import config
    from cw.metrics import (
        consensus_centers,
        pareto_front,
        per_structure_consensus_pr,
    )
    from cw.plots import plot_pr_scatter

    return (
        Path,
        config,
        consensus_centers,
        np,
        pareto_front,
        pd,
        per_structure_consensus_pr,
        plot_pr_scatter,
        plt,
        sns,
    )


@app.cell
def _(mo):
    # Cohort is a fill-in box (persists as empty default, so switching cohorts
    # never shows up as a git diff), same rationale as the panel-grid notebook.
    cohort_input = mo.ui.text(
        value="",
        placeholder="e.g. hewls_65 or carbonicanhydrase_000562_iso or endothiapepsin_000240_iso",
        label="cohort",
        full_width=True,
    )
    cohort_input
    return (cohort_input,)


@app.cell
def _(Path, cohort_input, config, pd):
    # Auto-infer the clustering params to freeze from the cohort's grid search:
    # the recommended row of clustering_hyperparameters.csv. Returns None when the
    # cohort box is empty or the cohort has no grid-search file, in which case the
    # boxes below fall back to the hand-set 15 / 5 default.
    def _recommended_params(cohort_dir):
        hyperparameters_path = cohort_dir / "clustering_hyperparameters.csv"
        if not hyperparameters_path.exists():
            return None
        hyperparameters = pd.read_csv(hyperparameters_path)
        recommended = hyperparameters[hyperparameters["recommended"]].iloc[0]
        return int(recommended["min_cluster_size"]), int(recommended["min_samples"])

    _cohort = cohort_input.value.strip()
    recommended_params = (
        _recommended_params(Path(config.DATA_DIR) / _cohort) if _cohort else None
    )
    return (recommended_params,)


@app.cell
def _(cohort_input, mo, recommended_params):
    # Params of the single unfiltered clustering to freeze, plus the consensus
    # cutoff. These only locate the one clustering dir to read; nothing is
    # re-clustered here. min_cluster_size / min_samples default to the cohort's
    # recommended grid-search params (auto-filled once you enter the cohort);
    # edit either box to freeze a different clustering.
    _mcs_default, _ms_default = recommended_params or (15, 5)
    mcs_input = mo.ui.text(value=str(_mcs_default), label="min_cluster_size")
    ms_input = mo.ui.text(value=str(_ms_default), label="min_samples")
    cutoff_input = mo.ui.text(value="0.3", label="consensus cutoff")
    _source = (
        "auto-filled from this cohort's `clustering_hyperparameters.csv` "
        "(recommended row)"
        if recommended_params
        else f"no `clustering_hyperparameters.csv` for "
        f"`{cohort_input.value.strip() or '—'}` — using fallback **15 / 5**"
    )
    mo.vstack([
        mo.hstack([mcs_input, ms_input, cutoff_input], justify="start"),
        mo.md(f"*min_cluster_size / min_samples {_source}.*"),
    ])
    return cutoff_input, mcs_input, ms_input


@app.cell
def _(Path, cohort_input, config, cutoff_input, mcs_input, mo, ms_input):
    COHORT = cohort_input.value.strip()
    mo.stop(
        not COHORT,
        mo.md("**Enter a cohort in the box above to load its data.**"),
    )
    mo.stop(
        not (mcs_input.value and ms_input.value and cutoff_input.value),
        mo.md("**Set min_cluster_size / min_samples / consensus cutoff above.**"),
    )

    # ══ USER CONFIG ══════════════════════════════════════════════════════════════
    # One row per filter. (label, column, op, threshold); op "ge" keeps col >=
    # threshold, "le" keeps col <= threshold. None column is the unfiltered base.
    # NaN in the filter column fails the comparison and is dropped, matching the
    # real keep_by_edia / keep_by_bfactor semantics.
    FILTERS = [
        ("default", None, None, None),
        ("EDIA ≥ 0.4", "edia", "ge", 0.4),
        ("EDIA ≥ 0.6", "edia", "ge", 0.6),
        ("EDIA ≥ 0.8", "edia", "ge", 0.8),
        ("B-factor z ≤ 2.0", "b_factor_zscore", "le", 2.0),
        ("B-factor z ≤ 1.5", "b_factor_zscore", "le", 1.5),
        ("B-factor z ≤ 1.0", "b_factor_zscore", "le", 1.0),
    ]
    # ═════════════════════════════════════════════════════════════════════════════
    MIN_CLUSTER_SIZE = int(mcs_input.value)
    MIN_SAMPLES = int(ms_input.value)
    CLUSTER_OCCUPANCY_CUTOFF = float(cutoff_input.value)

    DATA_DIR = Path(config.DATA_DIR)
    MATCH_RADIUS = config.CLUSTER_MEMBER_RADIUS
    PLOTS_DIR = Path("data") / "plots" / COHORT

    _cohort_dir = DATA_DIR / COHORT
    _subfolder = _cohort_dir / f"min_cluster_size_{MIN_CLUSTER_SIZE}_min_samples_{MIN_SAMPLES}"
    UNFILTERED_DIR = _subfolder if _subfolder.exists() else _cohort_dir
    return (
        CLUSTER_OCCUPANCY_CUTOFF,
        FILTERS,
        MATCH_RADIUS,
        PLOTS_DIR,
        UNFILTERED_DIR,
    )


@app.cell
def _(CLUSTER_OCCUPANCY_CUTOFF, UNFILTERED_DIR, consensus_centers, mo, pd):
    unfiltered_clusters = pd.read_csv(UNFILTERED_DIR / "clusters.csv")
    unfiltered_members = pd.read_csv(UNFILTERED_DIR / "cluster_members.csv")

    mo.stop(
        "b_factor_zscore" not in unfiltered_members.columns,
        mo.md(
            "**`cluster_members.csv` has no `b_factor_zscore` column.** Run "
            "`scripts/add_bfactor_zscore_column.py` on this cohort's "
            f"`cluster_members.csv` first (looked in `{UNFILTERED_DIR}`)."
        ),
    )

    N_STRUCTURES = int(unfiltered_members["pdb_id"].nunique())
    FROZEN_CENTERS = consensus_centers(unfiltered_clusters, CLUSTER_OCCUPANCY_CUTOFF)
    return (
        FROZEN_CENTERS,
        N_STRUCTURES,
        unfiltered_clusters,
        unfiltered_members,
    )


@app.cell
def _(FROZEN_CENTERS, MATCH_RADIUS, N_STRUCTURES, per_structure_consensus_pr):
    def apply_filter(members, column, op, threshold):
        """Keep the rows a real EDIA / B-factor z-score filter would keep. NaN in
        the filter column fails the comparison and is dropped, matching keep_by_*."""
        if column is None:
            return members
        col = members[column]
        keep = col >= threshold if op == "ge" else col <= threshold
        return members[keep]

    def frozen_reference_summary(filtered_members, cutoff):
        """Metrics for one filtered members table against the FROZEN unfiltered
        consensus centers. Cluster geometry/assignment is fixed; occupancy is
        recomputed from the surviving within-cutoff members so the consensus columns
        reflect the filter without any HDBSCAN re-partitioning."""
        num_water = len(filtered_members)
        n_noise = int((filtered_members["cluster_id"] == -1).sum())
        present = filtered_members[filtered_members["cluster_id"] >= 0]
        n_clusters = int(present["cluster_id"].nunique())

        within = filtered_members[filtered_members["within_cutoff"]]
        occ = within.groupby("cluster_id")["pdb_id"].nunique() / N_STRUCTURES
        conserved_ids = set(occ.index[occ >= cutoff])
        num_conserved = len(conserved_ids)
        in_conserved = within["cluster_id"].isin(conserved_ids)

        pr_df = per_structure_consensus_pr(
            filtered_members, FROZEN_CENTERS, MATCH_RADIUS
        )
        knee = pr_df.loc[pr_df["f1"].idxmax()]

        return {
            "pr_df": pr_df,
            "num_water": num_water,
            "num_noise_waters": n_noise,
            "num_clusters": n_clusters,
            "conserved_water_frac": (
                int(in_conserved.sum()) / num_water if num_water else None
            ),
            "num_conserved_clusters": num_conserved,
            "conserved_clusters_frac": (
                num_conserved / n_clusters if n_clusters else None
            ),
            "knee_precision": float(knee["precision"]),
            "knee_recall": float(knee["recall"]),
            "knee_f1": float(knee["f1"]),
        }

    return apply_filter, frozen_reference_summary


@app.cell
def _(CLUSTER_OCCUPANCY_CUTOFF, FROZEN_CENTERS, MATCH_RADIUS, mo):
    mo.md(f"""
    ## summary table

    One row per filter. Every row is scored against the **same** frozen reference —
    the {len(FROZEN_CENTERS)} consensus centers of the unfiltered clustering
    (occupancy ≥ {CLUSTER_OCCUPANCY_CUTOFF}). Same columns as the panel-grid summary
    plus **num noise waters** and **num clusters**.

    - **num_water** — surviving waters after the filter (prediction set size).
    - **num noise waters** — surviving waters labelled noise (`cluster_id == -1`).
    - **num clusters** — distinct non-noise clusters that still retain a water.
    - **conserved_water%** — surviving within-{MATCH_RADIUS} Å members of a
      (recomputed-)conserved cluster over all surviving waters.
    - **num_conserved_clusters / conserved_clusters%** — clusters whose
      *recomputed* occupancy ≥ {CLUSTER_OCCUPANCY_CUTOFF}: count, and fraction of
      surviving clusters.
    - **Pareto knee precision/recall/F1** — the max-F1 per-structure point, scored
      against the frozen reference. Recall can only hold or fall vs the default row.
    """)
    return


@app.cell
def _(
    CLUSTER_OCCUPANCY_CUTOFF,
    FILTERS,
    apply_filter,
    frozen_reference_summary,
    pd,
    unfiltered_members,
):
    _FIELDS = [
        ("num_water", "num_water", "int"),
        ("num_noise_waters", "num noise waters", "int"),
        ("num_clusters", "num clusters", "int"),
        ("conserved_water_frac", "conserved_water%", "pct"),
        ("num_conserved_clusters", "num_conserved_clusters", "int"),
        ("conserved_clusters_frac", "conserved_clusters%", "pct"),
        ("knee_precision", "Pareto knee precision", "float"),
        ("knee_recall", "Pareto knee recall", "float"),
        ("knee_f1", "Pareto knee F1", "float"),
    ]

    def _fmt(value, kind):
        if value is None:
            return "—"
        if kind == "int":
            return f"{int(value):,}"
        return f"{value:.1%}" if kind == "pct" else f"{value:.2f}"

    summaries = {}
    _rows = []
    for _label, _col, _op, _thr in FILTERS:
        _fm = apply_filter(unfiltered_members, _col, _op, _thr)
        _s = frozen_reference_summary(_fm, CLUSTER_OCCUPANCY_CUTOFF)
        summaries[_label] = _s
        _row = {"variant": _label}
        for _field, _header, _kind in _FIELDS:
            _row[_header] = _fmt(_s[_field], _kind)
        _rows.append(_row)

    summary_table = pd.DataFrame(_rows).set_index("variant")
    summary_table
    return summaries, summary_table


@app.cell
def _(PLOTS_DIR, mo):
    table_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "frozen_reference_summary_table.csv"),
        label="table save path", full_width=True,
    )
    table_save_button = mo.ui.run_button(label="save table")
    mo.hstack([table_save_path, table_save_button], justify="start")
    return table_save_button, table_save_path


@app.cell
def _(Path, summary_table, table_save_button, table_save_path):
    if table_save_button.value:
        _out = Path(table_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        summary_table.to_csv(_out)
        print(f"saved table to {_out}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## per-structure P/R vs the frozen reference

    One panel per filter: each structure's waters scored against the frozen
    consensus centers. Red = that filter's own Pareto front and max-F1 knee (star);
    grey dashed = the **unfiltered** front, drawn on every panel so a rising or
    falling frontier is directly comparable. If filtering removed only off-center
    noise the cloud should move straight up (front rises); if it also removed
    on-center waters the cloud slides left (front holds or drops).
    """)
    return


@app.cell
def _(FILTERS, pareto_front, plot_pr_scatter, plt, summaries):
    def build_pr_grid(filters, summaries, *, panel=3.2, fontsize=11):
        default_front = pareto_front(summaries["default"]["pr_df"])
        n = len(filters)
        fig, axes = plt.subplots(1, n, figsize=(panel * n, panel), squeeze=False)
        for ax, (label, *_rest) in zip(axes[0], filters):
            pr_df = summaries[label]["pr_df"]
            plot_pr_scatter(
                pr_df, color="num_water", color_label="#water", n_color_bins=8,
                ax=ax, alpha=0.55, fontsize=fontsize, marker_size=22,
            )
            front = pareto_front(pr_df)
            knee = pr_df.loc[pr_df["f1"].idxmax()]
            ax.plot(default_front["recall"], default_front["precision"],
                    color="gray", ls="--", lw=1.3, zorder=3)
            ax.plot(front["recall"], front["precision"], color="r", lw=1.5, zorder=4)
            ax.scatter([knee["recall"]], [knee["precision"]], marker="*", s=180,
                       facecolors="none", edgecolors="r", zorder=5)
            ax.set_title(f"{label}\nknee F1 = {knee['f1']:.2f}", fontsize=fontsize)
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
        fig.tight_layout()
        return fig

    pr_grid_fig = build_pr_grid(FILTERS, summaries)
    pr_grid_fig
    return (pr_grid_fig,)


@app.cell
def _(PLOTS_DIR, mo):
    grid_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "frozen_reference_pr_grid.png"),
        label="save path", full_width=True,
    )
    grid_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    grid_save_button = mo.ui.run_button(label="save figure")
    mo.hstack([grid_save_path, grid_save_dpi, grid_save_button], justify="start")
    return grid_save_button, grid_save_dpi, grid_save_path


@app.cell
def _(Path, grid_save_button, grid_save_dpi, grid_save_path, pr_grid_fig):
    if grid_save_button.value:
        _out = Path(grid_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        pr_grid_fig.savefig(_out, dpi=int(grid_save_dpi.value), bbox_inches="tight")
        print(f"saved to {_out}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## per-structure motion (default → filtered)

    Each arrow is one structure moving from its **unfiltered** (recall, precision)
    to its **filtered** point, both against the frozen reference. This is the direct
    premise test: arrows pointing up = filtering removed off-center noise (precision
    up, recall held); arrows pointing left = filtering removed on-center waters
    (recall lost). Pick a filter below.
    """)
    return


@app.cell
def _(FILTERS, mo):
    _choices = [label for label, *_ in FILTERS if label != "default"]
    filter_dropdown = mo.ui.dropdown(
        options=_choices, value=_choices[0], label="filter to compare vs default",
    )
    filter_dropdown
    return (filter_dropdown,)


@app.cell
def _(filter_dropdown, np, pareto_front, plt, summaries):
    def build_arrow_plot(label, summaries, *, fontsize=11):
        base = summaries["default"]["pr_df"][["pdb_id", "recall", "precision"]]
        filt = summaries[label]["pr_df"][["pdb_id", "recall", "precision"]]
        merged = base.merge(filt, on="pdb_id", suffixes=("_0", "_1"))

        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        d_precision = merged["precision_1"] - merged["precision_0"]
        colors = np.where(d_precision >= 0, "C0", "C3")
        ax.quiver(
            merged["recall_0"], merged["precision_0"],
            merged["recall_1"] - merged["recall_0"], d_precision,
            angles="xy", scale_units="xy", scale=1, width=0.003,
            color=colors, alpha=0.6,
        )
        for name, color in (("default", "gray"), (label, "r")):
            key = "default" if name == "default" else label
            front = pareto_front(summaries[key]["pr_df"])
            ax.plot(front["recall"], front["precision"], color=color,
                    ls="--" if name == "default" else "-", lw=1.6,
                    label=f"{name} front")

        mean_dr = (merged["recall_1"] - merged["recall_0"]).mean()
        mean_dp = d_precision.mean()
        ax.set_title(
            f"default → {label}\nmean Δrecall = {mean_dr:+.3f}, "
            f"mean Δprecision = {mean_dp:+.3f}",
            fontsize=fontsize,
        )
        ax.set_xlabel("recall", fontsize=fontsize)
        ax.set_ylabel("precision", fontsize=fontsize)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.legend(fontsize=fontsize - 1, loc="lower left")
        fig.tight_layout()
        return fig

    arrow_fig = build_arrow_plot(filter_dropdown.value, summaries)
    arrow_fig
    return (arrow_fig,)


@app.cell
def _(PLOTS_DIR, filter_dropdown, mo):
    _default = f"frozen_reference_arrows_{filter_dropdown.value}.png"
    arrow_save_path = mo.ui.text(
        value=str(PLOTS_DIR / _default.replace(" ", "").replace("≤", "le").replace("≥", "ge")),
        label="save path", full_width=True,
    )
    arrow_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    arrow_save_button = mo.ui.run_button(label="save figure")
    mo.hstack([arrow_save_path, arrow_save_dpi, arrow_save_button], justify="start")
    return arrow_save_button, arrow_save_dpi, arrow_save_path


@app.cell
def _(Path, arrow_fig, arrow_save_button, arrow_save_dpi, arrow_save_path):
    if arrow_save_button.value:
        _out = Path(arrow_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        arrow_fig.savefig(_out, dpi=int(arrow_save_dpi.value), bbox_inches="tight")
        print(f"saved to {_out}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## quality of consensus vs non-consensus waters (frozen split)

    The premise behind every filter, tested at the water level: are the waters that
    landed in a conserved site actually higher-quality than the ones that didn't?
    Using the **unfiltered** clustering, each water is labelled **consensus** (a
    within-cutoff member of a cluster with occupancy ≥ cutoff) or **non-consensus**
    (noise, radius-rejected, or a member of a non-conserved cluster), then the EDIA
    and B-factor z-score distributions of the two groups are overlaid (each density
    normalized separately, since the groups differ hugely in size). Dashed lines mark
    the candidate filter cutoffs. A filter is selective exactly when its line sits
    where it removes mostly non-consensus waters — EDIA keeps ≥ its line, B-factor z
    keeps ≤ its line — so the win case is a non-consensus mass that extends past the
    line into the region a filter discards while the consensus mass stays clear of it.
    """)
    return


@app.cell
def _(
    CLUSTER_OCCUPANCY_CUTOFF,
    FILTERS,
    plt,
    sns,
    unfiltered_clusters,
    unfiltered_members,
):
    def build_quality_split_fig(members, clusters, cutoff, filters, *, fontsize=11):
        conserved_ids = set(
            clusters.loc[clusters["cluster_occupancy"] >= cutoff, "cluster_id"]
        )
        is_consensus = members["within_cutoff"] & members["cluster_id"].isin(
            conserved_ids
        )
        labelled = members.assign(
            group=is_consensus.map({True: "consensus", False: "non-consensus"})
        )

        specs = [
            ("edia", "EDIA", [t for _l, c, _o, t in filters if c == "edia"]),
            (
                "b_factor_zscore",
                "B-factor z-score",
                [t for _l, c, _o, t in filters if c == "b_factor_zscore"],
            ),
        ]
        fig, axes = plt.subplots(1, len(specs), figsize=(6.2 * len(specs), 4.6))
        for ax, (col, label, cuts) in zip(axes, specs):
            data = labelled.dropna(subset=[col])
            sns.histplot(
                data=data, x=col, hue="group", 
                stat="density", common_norm=False,
                element="step", fill=True, alpha=0.35, ax=ax,
                hue_order=["consensus", "non-consensus"],
                palette={"consensus": "C0", "non-consensus": "C3"},
            )
            # sns.kdeplot(
            #     data=data, x=col, hue="group", common_norm=False, fill=True,
            #     alpha=0.35, ax=ax, hue_order=["consensus", "non-consensus"],
            #     palette={"consensus": "C0", "non-consensus": "C3"},
            # )
            for cut in cuts:
                ax.axvline(cut, color="0.4", ls="--", lw=1.0)
                ax.text(cut, ax.get_ylim()[1] * 0.98, f"{cut:g}", rotation=90,
                        va="top", ha="right", fontsize=fontsize - 3, color="0.4")
            n_c = int((data["group"] == "consensus").sum())
            n_n = int((data["group"] == "non-consensus").sum())
            ax.set_title(
                f"{label}\nconsensus n={n_c:,} · non-consensus n={n_n:,}",
                fontsize=fontsize,
            )
            ax.set_xlabel(label, fontsize=fontsize)
        fig.tight_layout()
        return fig

    quality_split_fig = build_quality_split_fig(
        unfiltered_members, unfiltered_clusters, CLUSTER_OCCUPANCY_CUTOFF, FILTERS
    )
    quality_split_fig
    return (quality_split_fig,)


@app.cell
def _(PLOTS_DIR, mo):
    quality_save_path = mo.ui.text(
        value=str(PLOTS_DIR / "frozen_reference_quality_split.png"),
        label="save path", full_width=True,
    )
    quality_save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    quality_save_button = mo.ui.run_button(label="save figure")
    mo.hstack([quality_save_path, quality_save_dpi, quality_save_button], justify="start")
    return quality_save_button, quality_save_dpi, quality_save_path


@app.cell
def _(
    Path,
    quality_save_button,
    quality_save_dpi,
    quality_save_path,
    quality_split_fig,
):
    if quality_save_button.value:
        _out = Path(quality_save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        quality_split_fig.savefig(
            _out, dpi=int(quality_save_dpi.value), bbox_inches="tight"
        )
        print(f"saved to {_out}")
    return


if __name__ == "__main__":
    app.run()
