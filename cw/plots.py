from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def quantile_boundaries(values, n_bins, quantile_range=(0.01, 0.99)):
    """Unique quantile bin edges over the inner quantile_range of values (default
    1–99%, so the extreme tails are capped). Integer edges when values are
    integer-valued. plot_pr_scatter builds its color bins from these; callers can
    reuse the same edges to bin a related series (e.g. a num_water tradeoff curve)
    onto bands identical to the colorbar."""
    values = np.asarray(values)
    q_lo, q_hi = quantile_range
    edges = np.unique(np.quantile(values, np.linspace(q_lo, q_hi, n_bins + 1)))
    if np.allclose(values, np.round(values)):
        edges = np.unique(np.rint(edges))
    return edges


def noise_stats(cluster_members):
    """(n_structures, noise_occupancy, noise_count) for a cluster_members table.
    noise_occupancy is the occupancy a single-structure water would carry
    (1 / n_structures); noise_count is the number of unclustered (cluster_id ==
    -1) waters. Used to place the noise pile in occupancy histograms."""
    n_structures = cluster_members["pdb_id"].nunique()
    noise_occupancy = 1.0 / n_structures if n_structures else 0.0
    noise_count = int((cluster_members["cluster_id"] == -1).sum())
    return n_structures, noise_occupancy, noise_count


def broken_y_axis(ax_top, ax_bot, *, d=0.015, left=True, right=True, linewidth=1, color="k"):
    """Turn two vertically stacked axes into a broken y-axis: hide the facing
    spines/ticks and draw the diagonal break marks across the gap. Assumes
    ax_top sits above ax_bot and both share the x-axis. left/right select which
    corners get break marks — set right=False when a continuous overlay axis
    (e.g. a cumulative curve) sits on the right and must not look broken.
    linewidth and color set the thickness and color of the diagonal break marks."""
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False)
    break_kw = dict(color=color, clip_on=False, linewidth=linewidth, transform=ax_top.transAxes)
    if left:
        ax_top.plot((-d, +d), (-d, +d), **break_kw)
    if right:
        ax_top.plot((1 - d, 1 + d), (-d, +d), **break_kw)
    break_kw["transform"] = ax_bot.transAxes
    if left:
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **break_kw)
    if right:
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **break_kw)


def plot_pr_scatter(
    pr_df: pd.DataFrame,
    *,
    color=None,
    color_label: str | None = None,
    precision_col: str = "precision",
    recall_col: str = "recall",
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    style: str = "scatter",
    marker_size: float = 20,
    alpha: float = 0.8,
    lim: tuple[float, float] | None = (0.0, 1.01),
    n_color_bins: int | None = None,
    color_quantile_range: tuple[float, float] = (0.01, 0.99),
    fontsize: float | None = None,
    cbar_kwargs: dict | None = None,
):
    """Precision vs recall for per-structure rows, on an equal-aspect square.

    color selects point coloring: a column name in pr_df, an array aligned to its
    rows, or None for a flat color. color_label labels the colorbar and defaults to
    the column name.

    n_color_bins, when set, discretizes the color scale into that many quantile bins
    over color_quantile_range (default inner 1–99%) — each bin holds ~equal counts and
    the extreme tails are capped, so the colorbar grows up/down triangles for the
    capped values. cmap is resampled to one distinct color per bin (with the two
    triangle colors taken from its ends), so qualitative palettes (e.g. "tab10")
    render as cleanly separated bands rather than collapsing onto one color — as long
    as the palette carries at least n_color_bins + 2 colors. Integer-valued colors get
    integer bin edges. Leave None for a plain continuous scale.

    style is "scatter" (alpha honored) or "hexbin" (hexes colored by the mean color
    value per cell). lim sets both axes to the same range (equal x/y); None autoscales.
    fontsize, when set, sizes every text element (axis labels, title, tick labels, and
    the colorbar label/ticks) so the caller can match it to a legend/annotation drawn
    on the returned ax; None keeps matplotlib defaults.

    cbar_kwargs is merged into the fig.colorbar call (over the {"pad": 0.02} default),
    so a caller can shrink the bar (e.g. {"fraction": 0.046}) when the square-aspect
    scatter leaves it towering over a small panel. Returns (fig, ax).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure

    color_values = None
    if color is not None:
        if isinstance(color, str):
            color_values = pr_df[color].to_numpy()
            color_label = color_label or color
        else:
            color_values = np.asarray(color)

    norm, boundaries, plot_cmap = None, None, cmap
    if color_values is not None and n_color_bins:
        edges = quantile_boundaries(color_values, n_color_bins, color_quantile_range)
        if len(edges) >= 2:
            boundaries = edges
            n_bands = len(edges) - 1
            # One distinct color per band. Sample n_bands + 2 colors and build a
            # ListedColormap of exactly n_bands entries with the outer two as the
            # under/over (extend-triangle) colors, then size the norm to that band
            # count. Handing the norm the raw cmap with ncolors=256 instead lets a
            # small qualitative colormap (e.g. Dark2's 8 entries) overflow every
            # band past the first onto its last color — the "all grey" bug.
            picks = plt.get_cmap(cmap)(np.linspace(0, 1, n_bands + 2))
            plot_cmap = mcolors.ListedColormap(picks[1:-1]).with_extremes(
                under=picks[0], over=picks[-1]
            )
            norm = mcolors.BoundaryNorm(boundaries, ncolors=plot_cmap.N)

    if style == "hexbin":
        sc = ax.hexbin(
            pr_df[recall_col],
            pr_df[precision_col],
            C=color_values,
            reduce_C_function=np.mean,
            gridsize=20,
            cmap=plot_cmap,
            norm=norm,
            mincnt=1,
        )
    elif color_values is None:
        ax.scatter(
            pr_df[recall_col],
            pr_df[precision_col],
            alpha=alpha,
            edgecolors="k",
            s=marker_size,
            color="steelblue",
        )
        sc = None
    else:
        sc = ax.scatter(
            pr_df[recall_col],
            pr_df[precision_col],
            c=color_values,
            cmap=plot_cmap,
            norm=norm,
            alpha=alpha,
            edgecolors="k",
            s=marker_size,
        )

    if sc is not None and color_values is not None:
        cbar_opts = {"pad": 0.02, **(cbar_kwargs or {})}
        cbar = (
            fig.colorbar(
                sc,
                ax=ax,
                extend="both",
                spacing="uniform",
                ticks=boundaries,
                format="%.3g",
                **cbar_opts,
            )
            if boundaries is not None
            else fig.colorbar(sc, ax=ax, **cbar_opts)
        )
        if color_label:
            cbar.set_label(color_label, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    ax.set_xlabel("recall", fontsize=fontsize)
    ax.set_ylabel("precision", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    if lim is not None:
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        # equal range → force identical ticks on both axes (set_yticks can widen
        # the view to fit out-of-range ticks, so re-pin ylim after)
        ax.set_yticks(ax.get_xticks())
        ax.set_ylim(*lim)
    ax.set_box_aspect(1)
    if title:
        ax.set_title(title, fontsize=fontsize)
    return fig, ax
