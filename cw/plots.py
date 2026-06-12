from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
