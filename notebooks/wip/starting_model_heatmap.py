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
    # Starting-model bias — the cross-refinement matrix

    The same cross-refinements as `starting_model_dumbbell.py` and
    `starting_model_partition_recall.py`, laid out as a **matrix over the ordered pairs**: one row
    per **starting model (A)**, one column per **mtz / data donor (B)**, so cell (A, B) is the
    cross-refinement `B_refined_by_A` — B's diffraction data, A's model.

    The **diagonal** holds the self-refinements, and whether it means anything depends on the
    grounding. Under **self-refined** it is a refinement scored against itself, so it is masked.
    Under **PDB-REDO** it is `X_refined_by_X` against X's *deposited* waters — a real measurement,
    and the one the cross-refinements in its row and column should be read against — so it is
    filled from the diagonal of `phenix_pairwise_metrics_<variant>_1.0.csv` and ringed to mark it
    as the different kind of measurement it is. The A-only panel keeps its diagonal masked either
    way: a template has no private sites against itself.

    Five matrices, all on the same 20 pairs. B's waters split into the half A also has and the half
    only B has, so all-B is the two recombined and can be opened up into them:

    | matrix | what it counts | denominator |
    | --- | --- | --- |
    | **recall — all-B** | B's own waters the cross-refinement placed, shared and B-unique together | B's water count |
    | **recall — A∩B shared** | the sites A and B agree on that it placed — the half of B the template already carries | shared count |
    | **recall — B-unique** | B's private sites (>cutoff from any A water) it placed — the half only the data can supply | B-unique count |
    | **recall — A-unique** | A's private sites (>cutoff from any B water) it placed — the bias signal | A-unique count |
    | **orphan fraction** | its own waters near neither A nor B | the cross-refinement's own count |

    Reading the matrix: the recalls against B are signal, the A-unique matrix is bias, and the
    orphan fraction is what it invented on top of both. Splitting B says where the signal came
    from — shared recall is the half a biased refinement gets for free, B-unique the half it can
    only get from the data. A **row** that runs dark in the A-unique matrix is a template that
    imposes its private sites on whatever data it meets; a **column** that runs dark there is data
    that accepts whatever template it is given. The B panels say what that cost.

    The picker opens on **all-B** and **A-unique**; the two halves of B and the orphan fraction are
    there to switch on.

    *Grounding.* Under **self-refined** the A and B water sets are themselves protocol-specific —
    they come from `<X>_refined_by_<X>` under the same protocol as the cell — so flipping the
    stripped/kept toggle moves both the predictor and the target. **PDB-REDO** holds A and B fixed
    across the toggle, which isolates the predictor effect.

    *Colour.* Each matrix has its own scale, fixed over **both** protocols at the selected
    grounding, so flipping stripped ↔ kept moves the colours only by as much as the numbers
    actually moved. Recall against B is the exception: it runs to a full 100%, so its saturation
    reads as a fraction of B rather than as a fraction of the best cell on the grid.
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

    return Path, config, np, pd, plt, sns


@app.cell
def _():
    # column -> (panel title, table name, colormap). Each matrix keeps its own scale — the three run
    # over different ranges and one shared window would flatten two of them. Its own cell, so the
    # panel picker below can offer these without waiting on the CSVs to load.
    METRICS = {
        "recall_b": ("recall — all-B", "recall B", "viridis"),
        "recall_shared": ("recall — A∩B shared", "recall shared", "viridis"),
        "recall_b_only": ("recall — B-unique", "recall B-unique", "viridis"),
        "recall_a_only": ("recall — A-unique", "recall A-unique", "viridis"),
        "orphan": ("orphan fraction", "orphan", "viridis"),
    }
    return (METRICS,)


@app.cell
def _(METRICS, mo):
    # all-B against A-unique is the pair that gets read together — signal against bias. The two
    # halves of B and the orphan fraction are the follow-up question, so they start off.
    _DEFAULT_PANELS = ("recall_b", "recall_a_only")
    panels = mo.ui.multiselect(
        options={_name: _column for _column, (_, _name, _) in METRICS.items()},
        value=[_name for _column, (_, _name, _) in METRICS.items()
               if _column in _DEFAULT_PANELS],
        label="Panels to plot",
    )
    grounding = mo.ui.radio(
        options={"self-refined — <X>_refined_by_<X>": "", "deposited — PDB-REDO": "_deposited"},
        value="self-refined — <X>_refined_by_<X>",
        label="Ground truth for the A and B water sets",
        inline=True,
    )
    variant = mo.ui.radio(
        options={"stripped — solvent re-placed de novo": "stripped",
                 "kept — starting waters retained": "auto"},
        value="stripped — solvent re-placed de novo",
        label="Cross-refinement protocol",
        inline=True,
    )
    layout = mo.ui.radio(options=["horizontal", "vertical"], value="horizontal",
                         label="Panel layout", inline=True)
    mo.vstack([panels, grounding, variant, layout])
    return grounding, layout, panels, variant


@app.cell
def _(Path, config, grounding, mo, pd):
    _dir = Path(f"{config.DATA_DIR}/hewls_65_subsampled_similar_default_phenix")
    VARIANTS = ("stripped", "auto")

    def _derive(_df):
        """The matrix values per pair.

        recall_b recombines the two disjoint halves of B (shared + b_only, which partition B
        exactly) into a whole-set recall, weighting each half by its own count; the halves are
        carried through beside it so either can be plotted on its own. The orphan fraction is the
        only one measured on the cross-refinement's side — an orphan has no ground-truth
        counterpart by definition.
        """
        _out = _df[["reference", "donor"]].copy()
        _out["recall_b"] = (_df["recall_shared"] * _df["n_shared"] + _df["recall_b_only"] * _df["n_b_only"]) / (
            _df["n_shared"] + _df["n_b_only"]
        )
        _out["recall_shared"] = _df["recall_shared"]
        _out["recall_b_only"] = _df["recall_b_only"]
        _out["recall_a_only"] = _df["recall_a_only"]
        _out["orphan"] = _df["n_pred_orphan"] / _df["n_pred"]
        _out["n_a_only"] = _df["n_a_only"]
        _out["n_b"] = _df["n_shared"] + _df["n_b_only"]
        _out["n_pred"] = _df["n_pred"]
        return _out.set_index(["reference", "donor"])

    GROUNDING_LABEL = "PDB-REDO" if grounding.value else "self-refined"
    heat, _missing = {}, []
    for _v in VARIANTS:
        _path = _dir / f"starting_model_partition{grounding.value}_{_v}_1.0.csv"
        if _path.exists():
            heat[_v] = _derive(pd.read_csv(_path))
        else:
            _missing.append(_path.name)

    # Under the deposited grounding the diagonal is a real measurement rather than a tautology:
    # X_refined_by_X against X's *PDB-REDO* waters, which is the diagonal of
    # phenix_pairwise_metrics_<variant>_1.0.csv — the same filtered_pdbs reference the deposited
    # partition uses. recall is recall_b with B = X, and 1 − precision is the orphan fraction, since
    # A and B are the same set there. A-only has no diagonal by construction — a template has no
    # private sites against itself — so that panel stays masked. Under the self-refined grounding
    # the whole diagonal would be a refinement scored against itself, so it is left out entirely.
    diagonal = {}
    if grounding.value:
        for _v in VARIANTS:
            _path = _dir / f"phenix_pairwise_metrics_{_v}_1.0.csv"
            if not _path.exists():
                _missing.append(_path.name)
                continue
            _self = pd.read_csv(_path)
            _self = _self[_self["reference"] == _self["predictor"]].set_index("reference")
            diagonal[_v] = pd.DataFrame({"recall_b": _self["recall"],
                                         "orphan": 1 - _self["precision"]})
    mo.stop(
        bool(_missing),
        mo.md("Missing: " + ", ".join(f"`{_m}`" for _m in _missing)
              + " — run `scripts/pairwise_water_metrics.py` with `--partition --cutoff 1.0`"
              + (" and with `--phenix`, both under `--ref-dir <cohort>/filtered_pdbs`."
                 if grounding.value else ".")),
    )

    pairs = heat["stripped"].index.intersection(heat["auto"].index)
    IDS = sorted({_i for _k in pairs for _i in _k})
    return GROUNDING_LABEL, IDS, VARIANTS, diagonal, heat, pairs


@app.cell
def _(IDS, diagonal, heat, np):
    def matrix_for(variant, column):
        """The (starting model × mtz) grid for one metric, in whatever unit the column carries.

        Cross-refinements come from the partition CSV, which has no self-pairs, so the diagonal
        arrives as NaN and is filled only where it has a meaning — the deposited grounding's
        recall_b and orphan. Everything still NaN afterwards is masked when drawn.
        """
        # copy=True: unstack can hand back a read-only view, which fill_diagonal cannot write to.
        values = (heat[variant][column].unstack("donor")
                  .reindex(index=IDS, columns=IDS).to_numpy(dtype=float, copy=True))
        if diagonal and column in diagonal[variant]:
            np.fill_diagonal(values, diagonal[variant][column].reindex(IDS).to_numpy(dtype=float))
        return values

    return (matrix_for,)


@app.cell
def _(IDS, np, plt):
    def frame_matrix(ax, ring_diagonal=False):
        """Ticks and the diagonal ring, shared by both matrix figures.

        `ring_diagonal` outlines the diagonal wherever it carries a self-refinement, which is a
        different kind of measurement from the cross-refinements around it.
        """
        # Masked cells read as blank, not as a light shade of the map.
        ax.set_facecolor("white")
        if ring_diagonal:
            for i in range(len(IDS)):
                ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="k", lw=2,
                                           zorder=20))
        ax.set_xticks(np.arange(len(IDS)) + 0.5)
        ax.set_yticks(np.arange(len(IDS)) + 0.5)
        ax.set_xticklabels(IDS, fontsize=11, rotation=0)
        ax.set_yticklabels(IDS, fontsize=11, rotation=0)
        ax.tick_params(length=0)
        ax.set_xlabel("mtz (B)", fontsize=14)
        ax.set_ylabel("starting model (A)", fontsize=14)

    return (frame_matrix,)


@app.cell
def _(layout, plt):
    # One panel's footprint, width × height, title and colorbar included. The two differ because the
    # colorbar sits under the panel in a row and beside it in a column, so it is spending the
    # figure's height in one case and its width in the other.
    PANEL_SIZE = {"horizontal": (3, 4.25), "vertical": (3.7, 3.05)}

    def panel_axes(n_panels):
        """A row (horizontal) or a column (vertical) of panels, with the axes handed back flat.

        squeeze=False keeps the axes array 2-D when only one panel is left; ravel() then reads the
        same in both layouts.
        """
        _vertical = layout.value == "vertical"
        _width, _height = PANEL_SIZE[layout.value]
        _shape = (n_panels, 1) if _vertical else (1, n_panels)
        _figsize = (_width, _height * n_panels) if _vertical else (_width * n_panels, _height)
        fig, axes = plt.subplots(*_shape, figsize=_figsize, squeeze=False)
        return fig, axes.ravel()

    def cbar_style(**extra):
        """Colorbar geometry — under the panel in a row, up its right-hand side in a column."""
        if layout.value == "vertical":
            return dict(orientation="vertical", pad=0.03, shrink=0.9, aspect=18, **extra)
        # pad is measured from the axes edge, so it has to clear the tick labels and the x label as
        # well as the matrix itself.
        return dict(orientation="horizontal", pad=0.185, shrink=0.82, aspect=28, **extra)

    def pack_panels(fig, n_panels):
        """tight_layout in whichever direction the panels run.

        Horizontal: dropping a panel narrows the figure, so the room the colorbars need grows as a
        fraction of it — hence the panel count in the bottom margin. Vertical holds its width, and
        with the colorbars off to the side nothing has to sit between the rows but the titles.
        """
        if layout.value == "vertical":
            fig.tight_layout(rect=(0, 0.01, 1, 0.99), h_pad=0.6)
        else:
            fig.tight_layout(rect=(0, 0.06 + 0.15 / n_panels, 1, 0.93), w_pad=2.0)

    return cbar_style, pack_panels, panel_axes


@app.cell
def _(
    METRICS,
    VARIANTS,
    cbar_style,
    diagonal,
    frame_matrix,
    matrix_for,
    mo,
    np,
    pack_panels,
    panel_axes,
    panels,
    sns,
    variant,
):
    # One matrix per selected metric, rows = starting model (A), columns = mtz (B). Iterating
    # METRICS and filtering, rather than iterating the selection, keeps the panels in their
    # canonical order however they were picked.
    _columns = [_c for _c in METRICS if _c in set(panels.value)]
    mo.stop(not _columns, mo.md("_Select at least one panel._"))

    # square=True fixes the cell aspect, so the panels are sized by whichever of the figure's two
    # dimensions binds first, and the panel footprint is the same in either layout.
    fig_heat, _axes = panel_axes(len(_columns))
    for _ax, _column in zip(_axes, _columns):
        _title, _, _cmap = METRICS[_column]
        _values = matrix_for(variant.value, _column)
        # Scale over both protocols, not just the selected one, so the toggle changes the colours
        # only by as much as the numbers moved.
        _both = np.concatenate([matrix_for(_v, _column).ravel() for _v in VARIANTS])
        # recall against B runs to the full scale rather than to its own maximum: it is the panel
        # read as "how much of B came back", and a stretched window would overstate that.
        _vmin = np.nanmin(_both)
        _vmax = 1.0 if _column == "recall_b" else np.nanmax(_both)
        sns.heatmap(_values, mask=np.isnan(_values), ax=_ax, cmap=_cmap, vmin=_vmin, vmax=_vmax,
                    square=True, linewidths=1.4, linecolor="white",
                    cbar_kws=cbar_style(format=lambda _t, _p: f"{_t:.0%}"))
        frame_matrix(_ax, ring_diagonal=bool(diagonal) and _column in diagonal[variant.value])
        _ax.set_title(_title, fontsize=14, pad=10)

    _protocol = "stripped" if variant.value == "stripped" else "kept"
    # fig_heat.suptitle(f"Cross-refinement by (starting model × mtz) — {_protocol}   —   ground "
    #                   f"truth: {GROUNDING_LABEL} (1.0 Å, n = {len(pairs)} pairs)", fontsize=13)
    pack_panels(fig_heat, len(_columns))
    fig_heat
    return (fig_heat,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## What the protocol changes — the difference matrix

    The same three matrices as one difference between the two protocols, in percentage points, in
    whichever direction the toggle below selects. The default, **kept − stripped**, matches the
    stripped → kept arrows in `starting_model_partition_recall.py`, so a positive cell means
    keeping the starting model's waters raised that metric; flip it to read the difference as what
    stripping buys instead.

    **Red is up and blue is down in every panel — but up is not the same as better.** On recall
    against B, a positive cell is more of B's own waters recovered, which is the protocol doing
    better. On A-only recall and the orphan fraction it is more of the template's private sites
    recovered and more waters invented — the protocol doing worse. A protocol that only slides
    down the first matrix while climbing the other two is trading signal for bias.

    Each panel is scaled symmetrically about zero on its own largest change, so white is exactly no
    change everywhere and the three panels' saturations are not comparable to each other. The
    diagonal follows the same rule as above — under PDB-REDO it carries the self-refinement's own
    response to the protocol, which is the baseline the cross-refinements' response sits against.

    This figure ignores the protocol toggle — it holds both — but it does follow the **grounding**
    toggle, and that choice is what the difference means. Under **PDB-REDO** the ground truth is
    fixed across the two protocols, so a cell is a clean predictor effect. Under **self-refined**
    the A and B sets move with the protocol too, so a cell is a whole-protocol comparison and a
    change can come from the target rather than the prediction.
    """)
    return


@app.cell
def _(mo):
    # label -> (minuend, subtrahend), as protocol keys. The label is reused verbatim on the figure,
    # so the two never drift apart.
    DELTA_DIRECTIONS = {"kept − stripped": ("auto", "stripped"),
                        "stripped − kept": ("stripped", "auto")}
    delta_direction = mo.ui.radio(options=list(DELTA_DIRECTIONS), value="kept − stripped",
                                  label="Difference direction", inline=True)
    delta_direction
    return DELTA_DIRECTIONS, delta_direction


@app.cell
def _(
    DELTA_DIRECTIONS,
    METRICS,
    cbar_style,
    delta_direction,
    diagonal,
    frame_matrix,
    matrix_for,
    mo,
    np,
    pack_panels,
    panel_axes,
    panels,
    sns,
):
    # The difference between the two protocols, in percentage points, on the same (A × B) grid —
    # diagonal included wherever the value matrices show one, so the self-refinement's own response
    # to the protocol sits beside the cross-refinements'. One diverging map for all three panels
    # rather than each metric's own hue: the sign is the whole point here, and a shared red-is-up
    # reading is worth more than keeping the panels colour-coded to the figure above.
    _minuend, _subtrahend = DELTA_DIRECTIONS[delta_direction.value]
    _raised_by = "the starting model's waters were kept" if _minuend == "auto" else \
                 "the solvent was re-placed de novo"
    _columns = [_c for _c in METRICS if _c in set(panels.value)]
    mo.stop(not _columns, mo.md("_Select at least one panel._"))

    fig_delta, _axes = panel_axes(len(_columns))
    for _ax, _column in zip(_axes, _columns):
        _title = METRICS[_column][0]
        _values = (matrix_for(_minuend, _column) - matrix_for(_subtrahend, _column)) * 100
        # Symmetric about zero so white is no change; per panel, since a 1 pp move in the orphan
        # fraction and a 1 pp move in recall are not the same size of effect.
        _limit = np.nanmax(np.abs(_values))
        sns.heatmap(_values, mask=np.isnan(_values), ax=_ax, cmap="RdBu_r", vmin=-_limit,
                    vmax=_limit, square=True, linewidths=1.4, linecolor="white",
                    # label="change (percentage points)"
                    cbar_kws=cbar_style(format=lambda _t, _p: f"{_t:+.0f}%"))
        frame_matrix(_ax, ring_diagonal=bool(diagonal) and _column in diagonal[_minuend])
        _ax.set_title(f"Δ {_title}", fontsize=14, pad=10)

    # fig_delta.suptitle(f"{delta_direction.value.capitalize()}, by (starting model × mtz)   —   "
    #                    f"ground truth: {GROUNDING_LABEL} (1.0 Å, n = {len(pairs)} pairs)",
    #                    fontsize=13)
    pack_panels(fig_delta, len(_columns))
    fig_delta
    return (fig_delta,)


@app.cell
def _(
    DELTA_DIRECTIONS,
    GROUNDING_LABEL,
    METRICS,
    VARIANTS,
    delta_direction,
    heat,
    mo,
    pairs,
    panels,
    pd,
    variant,
):
    # The selected protocol's cells as a long table, with the other protocol and the change beside
    # each — the matrix shows one protocol at a time, and the per-pair delta is the thing the
    # figure cannot show without a second grid. Δ follows the direction toggle rather than the
    # protocol one, so the table and the difference matrix always carry the same sign.
    _other = next(_v for _v in VARIANTS if _v != variant.value)
    _minuend, _subtrahend = DELTA_DIRECTIONS[delta_direction.value]
    _selected = heat[variant.value].loc[pairs]
    _alt = heat[_other].loc[pairs]
    _table = pd.DataFrame(index=pairs)
    _table["A-only waters"] = _selected["n_a_only"].round(0)
    _table["B waters"] = _selected["n_b"].round(0)
    _table["waters placed"] = _selected["n_pred"].round(0)
    for _column in [_c for _c in METRICS if _c in set(panels.value)]:
        _name = METRICS[_column][1]
        _table[f"{_name}: {variant.value}"] = (_selected[_column] * 100).round(1)
        _table[f"{_name}: {_other}"] = (_alt[_column] * 100).round(1)
        _table[f"{_name}: Δ"] = (
            (heat[_minuend].loc[pairs, _column] - heat[_subtrahend].loc[pairs, _column]) * 100
        ).round(1)
    _table = _table.rename_axis(index={"reference": "starting model (A)", "donor": "mtz (B)"})
    mo.vstack([
        mo.md(f"All {len(pairs)} cells, against the **{GROUNDING_LABEL}** ground truth. Metrics in "
              f"percent; Δ is **{_minuend} − {_subtrahend}**, in percentage points."),
        _table.reset_index(),
    ])
    return


@app.cell
def _(mo):
    # label -> (extension, dpi). PDF keeps the text and the cell edges vector; the raster options
    # differ only in dpi, so their filenames carry it to keep both on disk at once.
    SAVE_FORMATS = {"png — 300 dpi (screen)": ("png", 300),
                    "pdf — vector": ("pdf", None)}
    save_format = mo.ui.dropdown(options=list(SAVE_FORMATS), value="png — 300 dpi (screen)",
                                 label="Format")
    save_ui = mo.ui.checkbox(label="Save figures to data/plots/")
    mo.vstack([save_format, save_ui])
    return SAVE_FORMATS, save_format, save_ui


@app.cell
def _(
    DELTA_DIRECTIONS,
    Path,
    SAVE_FORMATS,
    delta_direction,
    fig_delta,
    fig_heat,
    grounding,
    mo,
    save_format,
    save_ui,
    variant,
):
    _out = Path("data/plots")
    _extension, _dpi = SAVE_FORMATS[save_format.value]
    if save_ui.value:
        _out.mkdir(parents=True, exist_ok=True)
        _saved = []
        # The difference figure holds both protocols, so it takes the direction in its name rather
        # than the selected variant. Only a non-default dpi lands in the name, so the screen-res
        # file keeps the plain name it has always had.
        _minuend, _subtrahend = DELTA_DIRECTIONS[delta_direction.value]
        _suffix = "" if _dpi in (None, 200) else f"_{_dpi}dpi"
        for _f, _name in ((fig_heat, f"partition_heatmap{grounding.value}_{variant.value}"),
                          (fig_delta, f"partition_heatmap{grounding.value}_delta_"
                                      f"{_minuend}_minus_{_subtrahend}")):
            _p = _out / f"{_name}_1.0{_suffix}.{_extension}"
            _f.savefig(_p, dpi=_dpi or 200, bbox_inches="tight")
            _saved.append(_p)
        _msg = mo.md("Saved → " + ", ".join(f"`{_p}`" for _p in _saved))
    else:
        _msg = mo.md(f"_Tick the box to write the figures to `{_out}/` as "
                     f"**{save_format.value}**._")
    _msg
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
