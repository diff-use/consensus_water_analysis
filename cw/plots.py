from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
):
    """Scatter of per-structure precision vs recall on a fixed [0, 1] square.

    color selects the point coloring: a column name in pr_df, an array-like of
    values aligned to pr_df's rows, or None for a single flat color. A colorbar is
    drawn whenever color is given; color_label labels it and defaults to the
    column name when color is a string. Returns (fig, ax).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    if color is None:
        ax.scatter(
            pr_df[recall_col], pr_df[precision_col],
            alpha=0.8, edgecolors="k", s=20, color="steelblue",
        )
    else:
        if isinstance(color, str):
            color_values = pr_df[color].to_numpy()
            color_label = color_label or color
        else:
            color_values = np.asarray(color)
        sc = ax.scatter(
            pr_df[recall_col], pr_df[precision_col],
            c=color_values, cmap=cmap, alpha=0.8, edgecolors="k", s=20,
        )
        cbar = fig.colorbar(sc, ax=ax)
        if color_label:
            cbar.set_label(color_label)

    ax.set_xlabel("recall  (% clusters covered)")
    ax.set_ylabel("precision  (% waters near cluster)")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_box_aspect(1)
    if title:
        ax.set_title(title)
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
    to the given order(s). col_order defaults to row_order (square matrix)."""
    m = df.pivot_table(index=index, columns=columns, values=value_col)
    return m.reindex(index=row_order, columns=col_order if col_order is not None else row_order)


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
                m, ax, title=spec["label"], cbar_label=spec["label"],
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
