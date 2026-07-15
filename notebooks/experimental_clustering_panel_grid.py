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
    # experimental — clustering panel grid

    A panel grid comparing clustering across filter variants of one cohort:
    **one row per variant** (cohort + filter list set in the config cell), four
    columns per row.

    | col | plot | clustering |
    |-----|------|------------|
    | 1 | cluster-occupancy histogram + cumulative water% | fixed params |
    | 2 | per-structure precision–recall + Pareto front | fixed params |
    | 3–4 | same as 1–2 | per-cohort **best** params |

    **Fixed** columns read the `min_cluster_size_<mcs>_min_samples_<ms>/`
    subfolder (falling back to the cohort top-level when absent — correct only if
    best == fixed). **Best** columns read the cohort top-level CSVs, using the
    `recommended=True` row of `clustering_hyperparameters.csv`; when best == fixed
    (typically the base cohort) they are redundant and left blank.

    Plotting logic is ported from `01_cluster_analysis.py`. `num_water` (P/R
    point color) comes from `cluster_members`, so no `metadata.csv` is needed.
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
    from cw.plots import plot_pr_scatter, quantile_boundaries

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
        quantile_boundaries,
    )


@app.cell
def _(Path, config):
    import re

    # ══ USER CONFIG — everything you might tweak lives in this block ══════════════
    COHORT = "carbonicanhydrase_000562_iso"
    # One row per filter subset, top → bottom. "" is the unfiltered base cohort;
    # every other entry is the folder suffix appended to COHORT (e.g. "edia0.4" →
    # "<COHORT>_edia0.4"). The figure sizes itself to len(FILTER_SUBSETS), so add
    # or drop entries to change the rows. The base cohort should define the fixed
    # baseline params (its best == fixed) so its best columns come out blank.
    FILTER_SUBSETS = [
        "",
        "edia0.4",
        "bfactor_z2.0water",
        "edia0.4_bfactor_z2.0water",
    ]
    # Fixed clustering params shared across all rows (the left two columns), read
    # from each cohort's "min_cluster_size_<mcs>_min_samples_<ms>" subfolder.
    FIXED_MCS = 15
    FIXED_MS = 5
    CLUSTER_OCCUPANCY_CUTOFF = 0.3
    # ═════════════════════════════════════════════════════════════════════════════

    def _row_label(suffix):
        """Human label for a filter suffix: parse the known edia / bfactor_z
        tokens (any order/combination), falling back to the raw suffix for
        anything unrecognized so a new filter type still labels sensibly."""
        if suffix == "":
            return f"{COHORT}\n(unfiltered)"
        clauses = []
        edia = re.search(r"edia([\d.]+)", suffix)
        if edia:
            clauses.append(f"EDIA ≥ {edia.group(1)}")
        bfactor = re.search(r"bfactor_z([\d.]+)", suffix)
        if bfactor:
            clauses.append(f"B-factor z ≤ {bfactor.group(1)}")
        return " &\n".join(clauses) if clauses else suffix

    # ── Derived ──────────────────────────────────────────────────────────────────
    FIXED_SUBFOLDER = f"min_cluster_size_{FIXED_MCS}_min_samples_{FIXED_MS}"
    COHORT_ROWS = [COHORT if s == "" else f"{COHORT}_{s}" for s in FILTER_SUBSETS]
    ROW_LABELS = {row: _row_label(s) for row, s in zip(COHORT_ROWS, FILTER_SUBSETS)}
    COL_HEADERS = [
        f"cluster hist — same params ({FIXED_MCS}/{FIXED_MS})",
        f"p/r scatter — same params ({FIXED_MCS}/{FIXED_MS})",
        "cluster hist — best params",
        "p/r scatter — best params",
    ]
    DATA_DIR = Path(config.DATA_DIR)
    MATCH_RADIUS = config.CLUSTER_MEMBER_RADIUS
    PLOTS_DIR = Path("data") / "plots" / COHORT_ROWS[0]
    return (
        CLUSTER_OCCUPANCY_CUTOFF,
        COHORT_ROWS,
        COL_HEADERS,
        DATA_DIR,
        FIXED_SUBFOLDER,
        MATCH_RADIUS,
        PLOTS_DIR,
        ROW_LABELS,
    )


@app.cell
def _(
    DATA_DIR,
    FIXED_SUBFOLDER,
    consensus_centers,
    np,
    pareto_front,
    pd,
    per_structure_consensus_pr,
    plot_pr_scatter,
    plt,
    quantile_boundaries,
):
    # ── Panel helpers (ported from 01_cluster_analysis.py) ───────────────────────

    def parse_params(subfolder_name):
        """(min_cluster_size, min_samples) from a 'min_cluster_size_A_min_samples_B' name."""
        nums = [int(t) for t in subfolder_name.split("_") if t.isdigit()]
        return (nums[0], nums[1])

    def load_pair(dirpath):
        return (
            pd.read_csv(dirpath / "clusters.csv"),
            pd.read_csv(dirpath / "cluster_members.csv"),
        )

    def best_params(cohort_dir):
        hp = pd.read_csv(cohort_dir / "clustering_hyperparameters.csv")
        rec = hp[hp["recommended"]].iloc[0]
        return (int(rec["min_cluster_size"]), int(rec["min_samples"]))

    def resolve_cohort(cohort):
        """Locate the fixed-params and best-params clustering dirs for one cohort.

        fixed_dir is the shared FIXED_SUBFOLDER when present, else the cohort's
        top-level (correct only when best == fixed). best_dir is always the
        cohort top-level. redundant flags best == fixed → best columns blank.
        """
        cohort_dir = DATA_DIR / cohort
        fixed_params = parse_params(FIXED_SUBFOLDER)
        bp = best_params(cohort_dir)
        sub = cohort_dir / FIXED_SUBFOLDER
        fixed_dir = sub if sub.exists() else cohort_dir
        return {
            "cohort_dir": cohort_dir,
            "fixed_dir": fixed_dir,
            "best_dir": cohort_dir,
            "best_params": bp,
            "fixed_params": fixed_params,
            "redundant": bp == fixed_params,
        }

    def draw_cumulative_hist(
        fig, subspec, clusters, cluster_members, cutoff,
        *, fontsize=12, tick_fontsize=10, ylabel_left=True, ylabel_right=True,
    ):
        """Broken-axis cluster-count histogram with a full-height cumulative
        water% overlay, drawn into `subspec` (a top-level GridSpec cell). Returns
        (ax_top, ax_bot, ax_cum) for label placement by the caller."""
        inner = subspec.subgridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

        if len(clusters) == 0:
            ax_top.axis("off")
            ax_bot.text(0.5, 0.5, "No clusters", ha="center", va="center",
                        transform=ax_bot.transAxes, fontsize=fontsize)
            return ax_top, ax_bot, None

        members = (
            cluster_members[cluster_members["within_cutoff"]]
            .groupby("cluster_id").size().rename("n_members")
        )
        occ = clusters[["cluster_id", "cluster_occupancy"]].merge(members, on="cluster_id")

        n_structures = cluster_members["pdb_id"].nunique()
        noise_occ = 1.0 / n_structures if n_structures else 0.0
        noise_n = int((cluster_members["cluster_id"] == -1).sum())

        edges = np.linspace(0, 1, 21)
        width = 0.9 * (edges[1] - edges[0])
        centers = (edges[:-1] + edges[1:]) / 2
        cluster_hist, _ = np.histogram(clusters["cluster_occupancy"], bins=edges)
        water_per_bin, _ = np.histogram(
            occ["cluster_occupancy"], bins=edges, weights=occ["n_members"]
        )
        noise_per_bin, _ = np.histogram([noise_occ], bins=edges, weights=[noise_n])
        water_counts = water_per_bin + noise_per_bin
        total = water_counts.sum()
        cum_frac = water_counts.cumsum() / total
        frac = occ.loc[occ["cluster_occupancy"] >= cutoff, "n_members"].sum() / total
        cum_at_cutoff = 1 - frac

        cluster_max = int(cluster_hist.max())
        noise_bin = int(noise_per_bin.argmax())
        noise_top = int(cluster_hist[noise_bin] + noise_n)

        for bax in (ax_top, ax_bot):
            bax.bar(centers, cluster_hist, width=width, color="gray", alpha=0.5)
            bax.bar(centers, noise_per_bin, width=width, bottom=cluster_hist,
                    color="lightgrey", alpha=0.7)
            bax.tick_params(axis="y", colors="gray", labelsize=tick_fontsize)

        ax_top.set_ylim(noise_top * 0.85, noise_top)
        ax_bot.set_ylim(0, cluster_max * 1.25)
        if ylabel_left:
            ax_bot.set_ylabel("#clusters", fontsize=fontsize, color="gray")
        ax_bot.set_xlabel("cluster occupancy", fontsize=fontsize)
        ax_bot.tick_params(axis="x", labelsize=tick_fontsize)

        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(bottom=False)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        d = 0.015
        kw = dict(color="k", clip_on=False, linewidth=1, transform=ax_top.transAxes)
        ax_top.plot((-d, +d), (-d, +d), **kw)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kw)
        kw["transform"] = ax_bot.transAxes
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kw)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

        # full-height overlay carrying the cumulative water fraction; spans both
        # panels so its 100% ceiling lines up with the top of the noise pile.
        pt = ax_top.get_position()
        pb = ax_bot.get_position()
        ax_cum = fig.add_axes([pb.x0, pb.y0, pb.width, pt.y1 - pb.y0])
        ax_cum.set_xlim(ax_bot.get_xlim())
        ax_cum.set_ylim(0, cum_frac.max())
        ax_cum.plot(centers, cum_frac, color="k", lw=2.5)
        if ylabel_right:
            ax_cum.set_ylabel("cumulative water%", fontsize=fontsize, color="k")
        ax_cum.tick_params(axis="y", colors="k", labelsize=tick_fontsize)
        ax_cum.axvline(cutoff, color="C0", linestyle="--", linewidth=1)
        ax_cum.axhline(cum_at_cutoff, color="k", linestyle="--", linewidth=1)
        ax_cum.patch.set_visible(False)
        ax_cum.xaxis.set_visible(False)
        ax_cum.yaxis.tick_right()
        ax_cum.yaxis.set_label_position("right")
        for s in ("top", "right", "bottom"):
            ax_cum.spines[s].set_visible(False)
        return ax_top, ax_bot, ax_cum

    def draw_pr_panel(
        fig, ax, clusters, cluster_members, cutoff, radius, *, fontsize=12, alpha=0.6,
    ):
        """Per-structure precision–recall scatter (colored by pooled #water) with
        the empirical Pareto front, the max-F1 knee, and the num_water-binned
        average curve, drawn onto `ax`."""
        centers = consensus_centers(clusters, cutoff)
        if len(centers) == 0 or cluster_members.empty:
            ax.text(0.5, 0.5, "No consensus", ha="center", va="center",
                    transform=ax.transAxes, fontsize=fontsize)
            return

        pr_df = per_structure_consensus_pr(cluster_members, centers, radius)

        plot_pr_scatter(
            pr_df, color="num_water", color_label="#water", n_color_bins=10,
            ax=ax, alpha=alpha, fontsize=fontsize, marker_size=30,
        )
        knee = pr_df.loc[pr_df["f1"].idxmax()]
        front = pareto_front(pr_df)
        ax.scatter([knee["recall"]], [knee["precision"]], marker="*", s=200,
                   facecolors="none", edgecolors="r", zorder=5)
        ax.plot(front["recall"], front["precision"], color="r", lw=1.5)

        edges = quantile_boundaries(pr_df["num_water"], 10).astype(float)
        if len(edges) >= 2:
            edges[0], edges[-1] = -np.inf, np.inf
            bins = pd.cut(pr_df["num_water"], bins=edges).rename("num_water_bin")
            curve = (
                pr_df.groupby(bins, observed=True)[["recall", "precision", "num_water"]]
                .mean().sort_values("num_water")
            )
            ax.plot(curve["recall"], curve["precision"], color="r", marker="o",
                    ls="--", ms=2, lw=1.2)

        ax.set_xticks(ax.get_yticks())
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)

    return draw_cumulative_hist, draw_pr_panel, load_pair, resolve_cohort


@app.cell
def _(
    CLUSTER_OCCUPANCY_CUTOFF,
    COHORT_ROWS,
    COL_HEADERS,
    MATCH_RADIUS,
    ROW_LABELS,
    draw_cumulative_hist,
    draw_pr_panel,
    load_pair,
    plt,
    resolve_cohort,
):
    def build_panel_grid(
        cohort_rows, row_labels, col_headers, cutoff, radius,
        *, panel_w=4.2, panel_h=3.3, fontsize=12, tick_fontsize=10,
    ):
        """Assemble the 4×N grid: cols 0/2 cumulative-water histograms, cols 1/3
        precision–recall, fixed-params (0/1) vs best-params (2/3) per cohort row.
        Redundant best columns (best == fixed) are left blank."""
        n_rows = len(cohort_rows)
        fig = plt.figure(figsize=(panel_w * 4, panel_h * n_rows))
        gs = fig.add_gridspec(
            n_rows, 4, wspace=0.55, hspace=0.45,
            left=0.09, right=0.97, top=0.93, bottom=0.06,
        )

        col_axes = {}      # representative plotting axis per column, for headers
        row_bounds = {}    # (y_top, y_bottom) of each row's left histogram

        for r, cohort in enumerate(cohort_rows):
            info = resolve_cohort(cohort)
            fixed_clusters, fixed_members = load_pair(info["fixed_dir"])

            ax_top, ax_bot, _ = draw_cumulative_hist(
                fig, gs[r, 0], fixed_clusters, fixed_members, cutoff,
                fontsize=fontsize, tick_fontsize=tick_fontsize,
            )
            col_axes.setdefault(0, ax_bot)
            row_bounds[r] = (ax_top.get_position().y1, ax_bot.get_position().y0)

            ax_pr = fig.add_subplot(gs[r, 1])
            draw_pr_panel(fig, ax_pr, fixed_clusters, fixed_members, cutoff, radius,
                          fontsize=fontsize)
            col_axes.setdefault(1, ax_pr)

            if not info["redundant"]:
                best_clusters, best_members = load_pair(info["best_dir"])
                _, ax_bot_b, _ = draw_cumulative_hist(
                    fig, gs[r, 2], best_clusters, best_members, cutoff,
                    fontsize=fontsize, tick_fontsize=tick_fontsize,
                )
                col_axes.setdefault(2, ax_bot_b)
                ax_pr_b = fig.add_subplot(gs[r, 3])
                draw_pr_panel(fig, ax_pr_b, best_clusters, best_members, cutoff,
                              radius, fontsize=fontsize)
                col_axes.setdefault(3, ax_pr_b)

        # column headers, centered over each column's plotting axis
        for c, header in enumerate(col_headers):
            ax = col_axes.get(c)
            if ax is None:
                continue
            pos = ax.get_position()
            fig.text((pos.x0 + pos.x1) / 2, 0.965, header,
                     ha="center", va="bottom", fontsize=fontsize + 1, weight="bold")

        # left-hand row labels
        for r, cohort in enumerate(cohort_rows):
            y_top, y_bottom = row_bounds[r]
            fig.text(0.02, (y_top + y_bottom) / 2, row_labels.get(cohort, cohort),
                     ha="center", va="center", rotation=90,
                     fontsize=fontsize + 1, weight="bold")

        return fig

    fig = build_panel_grid(
        COHORT_ROWS, ROW_LABELS, COL_HEADERS, CLUSTER_OCCUPANCY_CUTOFF, MATCH_RADIUS,
    )
    fig
    return (fig,)


@app.cell
def _(PLOTS_DIR, mo):
    save_path = mo.ui.text(
        value=str(PLOTS_DIR / "clustering_panel_grid.png"),
        label="save path", full_width=True,
    )
    save_dpi = mo.ui.number(start=72, stop=1200, step=1, value=300, label="dpi")
    save_button = mo.ui.run_button(label="save figure")
    mo.hstack([save_path, save_dpi, save_button], justify="start")
    return save_button, save_dpi, save_path


@app.cell
def _(Path, fig, save_button, save_dpi, save_path):
    if save_button.value:
        _out = Path(save_path.value)
        _out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(_out, dpi=int(save_dpi.value), bbox_inches="tight")
        print(f"saved to {_out}")
    return


if __name__ == "__main__":
    app.run()
