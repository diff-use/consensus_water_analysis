from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
):
    """Precision vs recall for per-structure rows, on an equal-aspect square.

    color selects point coloring: a column name in pr_df, an array aligned to its
    rows, or None for a flat color. color_label labels the colorbar and defaults to
    the column name.

    n_color_bins, when set, discretizes the color scale into that many quantile bins
    over color_quantile_range (default inner 1–99%) — each bin holds ~equal counts and
    the extreme tails are capped, so the colorbar grows up/down triangles for the
    capped values. Integer-valued colors get integer bin edges. Leave None for a plain
    continuous scale.

    style is "scatter" (alpha honored) or "hexbin" (hexes colored by the mean color
    value per cell). lim sets both axes to the same range (equal x/y); None autoscales.
    fontsize, when set, sizes every text element (axis labels, title, tick labels, and
    the colorbar label/ticks) so the caller can match it to a legend/annotation drawn
    on the returned ax; None keeps matplotlib defaults. Returns (fig, ax).
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

    norm, boundaries = None, None
    if color_values is not None and n_color_bins:
        edges = quantile_boundaries(color_values, n_color_bins, color_quantile_range)
        if len(edges) >= 2:
            boundaries = edges
            norm = mcolors.BoundaryNorm(boundaries, ncolors=256, extend="both")

    if style == "hexbin":
        sc = ax.hexbin(
            pr_df[recall_col], pr_df[precision_col],
            C=color_values, reduce_C_function=np.mean,
            gridsize=20, cmap=cmap, norm=norm, mincnt=1,
        )
    elif color_values is None:
        ax.scatter(
            pr_df[recall_col], pr_df[precision_col],
            alpha=alpha, edgecolors="k", s=marker_size, color="steelblue",
        )
        sc = None
    else:
        sc = ax.scatter(
            pr_df[recall_col], pr_df[precision_col],
            c=color_values, cmap=cmap, norm=norm,
            alpha=alpha, edgecolors="k", s=marker_size,
        )

    if sc is not None and color_values is not None:
        cbar = (
            fig.colorbar(sc, ax=ax, extend="both", spacing="uniform",
                         ticks=boundaries, format="%.3g")
            if boundaries is not None
            else fig.colorbar(sc, ax=ax)
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


# ── Heatmap toolkit ─────────────────────────────────────────────────────────
# Shared by the refinement-matrix notebooks (scalar metrics like r_free / n_water
# and pairwise water-set agreement). Both draw rows-of-panels of square matrices
# whose axes are pdb ids sorted by water count, so the helpers below are kept
# generic: build a {label: matrix} dict however you like and hand it to
# make_panels. Notebooks 05 and 06 keep their own inline copies; new notebooks
# should import from here instead of redefining.


def order_by_count(ids, counts: pd.Series) -> list:
    """Order ids by a count Series (ascending; ids missing from counts sort last)."""
    return list(counts.reindex(list(ids)).sort_values().index)


def to_matrix(df, value_col, row_order, col_order=None, *, index, columns):
    """Pivot a long df to an `index` x `columns` matrix of `value_col`, reindexed
    to the given order(s). col_order defaults to row_order (square matrix).

    Duplicate (index, columns) pairs are an error (e.g. a re-refinement leaving two
    CIFs in one dir): pivot with aggfunc="first" so a stray duplicate fails loud here
    rather than being silently averaged into the cell.
    """
    dup = df.duplicated(subset=[index, columns]).any()
    if dup:
        raise ValueError(f"to_matrix: duplicate ({index}, {columns}) pairs for {value_col!r}")
    m = df.pivot_table(index=index, columns=columns, values=value_col, aggfunc="first")
    return m.reindex(index=row_order, columns=col_order if col_order is not None else row_order)


def diagonal_matrix(series, order) -> pd.DataFrame:
    """Square `order` x `order` matrix, NaN everywhere but the leading diagonal,
    which is filled from `series` reindexed to `order`. The off-diagonal NaNs
    render blank under seaborn, so a heatmap of this shows only self-comparisons."""
    order = list(order)
    arr = np.full((len(order), len(order)), np.nan)
    np.fill_diagonal(arr, pd.Series(series).reindex(order).to_numpy(dtype=float))
    return pd.DataFrame(arr, index=order, columns=order)


def shared_range(matrices, center=None):
    """Common (vmin, vmax) across a {label: matrix} dict; symmetric about
    `center` when given."""
    if center is not None:
        radius = max((m - center).abs().max().max() for m in matrices.values())
        return center - radius, center + radius
    return (
        min(m.min().min() for m in matrices.values()),
        max(m.max().max() for m in matrices.values()),
    )


def draw_heatmap(matrix, ax, *, title="", cbar_label="", cmap="viridis", center=None,
                 annot=True, fmt=".2f", vmin=None, vmax=None, cbar=True,
                 mask_diagonal=False, xlabel="", ylabel=""):
    """Draw one annotated, square seaborn heatmap onto `ax`.

    mask_diagonal blanks the leading diagonal (self-comparisons) → NaN.
    """
    mask = np.eye(len(matrix), dtype=bool) if mask_diagonal else None
    sns.heatmap(
        matrix, ax=ax, cmap=cmap, center=center, annot=annot, fmt=fmt,
        vmin=vmin, vmax=vmax, cbar=cbar, mask=mask,
        square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"label": cbar_label, "shrink": 0.6},
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def make_panels(matrices, *, specs=None, cbar_label="", cmap="viridis", center=None,
                annot=True, fmt=".2f", panel_size=4.0, shared_cbar=False,
                mask_diagonal=False, xlabel="", ylabel=""):
    """Render a {label: matrix} dict as a row of heatmap panels.

    Two styling modes:

    - `specs` given — a {label: {label, cmap, vmin, vmax}} registry drives each
      panel's title/colour map/range individually (per-metric panels). `shared_cbar`
      is ignored; each panel keeps its own colorbar.
    - `specs` None — every panel uses the uniform `cmap`/`center`/`fmt` kwargs.
      `shared_cbar=True` puts them all on one scale (symmetric about `center` when
      given) with a single figure-level colorbar.

    Returns the figure.
    """
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(panel_size * n, panel_size), squeeze=False)
    axs = axes[0]

    if specs is not None:
        for ax, (key, m) in zip(axs, matrices.items()):
            spec = specs[key]
            draw_heatmap(
                m, ax, title=spec["label"], cbar_label=spec.get("cbar_label", spec["label"]),
                cmap=spec["cmap"], vmin=spec.get("vmin"), vmax=spec.get("vmax"),
                annot=annot, fmt=fmt, mask_diagonal=mask_diagonal,
                xlabel=xlabel, ylabel=ylabel,
            )
        fig.tight_layout()
        return fig

    vmin = vmax = None
    if shared_cbar:
        vmin, vmax = shared_range(matrices, center)
    for ax, (label, m) in zip(axs, matrices.items()):
        draw_heatmap(
            m, ax, title=label, cbar_label=cbar_label, cmap=cmap, center=center,
            annot=annot, fmt=fmt, vmin=vmin, vmax=vmax, cbar=not shared_cbar,
            mask_diagonal=mask_diagonal, xlabel=xlabel, ylabel=ylabel,
        )
    if shared_cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, ax=axs.tolist(), label=cbar_label, shrink=0.6)
        return fig
    fig.tight_layout()
    return fig
