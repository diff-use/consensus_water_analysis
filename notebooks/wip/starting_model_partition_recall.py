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
    # Starting-model bias — is the water network set by the data or the template?

    For each ordered pair (**A** = starting model, **B** = data donor), the cross-refinement
    `B_refined_by_A` uses B's diffraction data but A's model. Score it, after Cα alignment into
    B's frame, against **two ground truths**:

    - **self-refined B** — every water of `B_refined_by_B`. What the data alone supports.
    - **template-only** — the waters of `A_refined_by_A` that sit >cutoff from *any* B water.
      A's private sites. Recovering these is bias, not signal.

    The two sets are disjoint by construction, so neither is inflated by the conserved core they
    would otherwise share. Precision is over the cross-refinement's own waters in both cases —
    the same denominator — so the two precisions are directly comparable fractions of one output.

    If the data drives the result: high P/R against B, low against template-only. The question
    for the protocol is whether **stripping** the starting model's waters moves a pair toward B
    and away from the template.

    *Caveat:* the ground truths are themselves variant-specific (`B_refined_by_B_auto` vs
    `..._stripped`), so an auto → stripped arrow is a whole-protocol comparison, not a pure
    predictor effect with the target held fixed.

    Reads `starting_model_partition_<variant>_1.0.csv` (from
    `scripts/pairwise_water_metrics.py --partition`); all four metrics are derived from its
    columns, nothing is re-parsed.
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

    return Path, config, np, pd, plt


@app.cell
def _(Path, config, mo, pd, plt):
    _dir = Path(f"{config.DATA_DIR}/hewls_65_subsampled_similar_default_phenix")
    KEY = ["reference", "donor"]
    # (axis label stem, precision column, recall column) per ground truth.
    GROUND_TRUTHS = {
        "b": ("self-refined B — every water", "precision_b", "recall_b"),
        "template": ("template-only — A's waters absent from B", "precision_template", "recall_template"),
    }

    def _derive(_df):
        """The four metrics, from the partition CSV's group counts.

        recall_b recombines the two disjoint halves of B (shared + b_only, which partition B
        exactly) into a whole-set recall. precision_template uses n_pred_a_only — predictor
        waters near A and not near B — as the count landing on template-only sites; it is a
        close proxy rather than an exact match for "within cutoff of an a_only water", since a
        predictor water can sit within cutoff of both an a_only and a B water (two ground-truth
        waters more than a cutoff apart can each be in range of the same point).
        """
        _out = _df[KEY].copy()
        _out["precision_b"] = _df["n_pred_b"] / _df["n_pred"]
        _out["recall_b"] = (_df["recall_shared"] * _df["n_shared"] + _df["recall_b_only"] * _df["n_b_only"]) / (
            _df["n_shared"] + _df["n_b_only"]
        )
        _out["precision_template"] = _df["n_pred_a_only"] / _df["n_pred"]
        _out["recall_template"] = _df["recall_a_only"]
        return _out.set_index(KEY)

    metrics, _missing = {}, []
    for _v in ("auto", "stripped"):
        _p = _dir / f"starting_model_partition_{_v}_1.0.csv"
        if _p.exists():
            metrics[_v] = _derive(pd.read_csv(_p))
        else:
            _missing.append(_p.name)
    mo.stop(
        bool(_missing),
        mo.md("Missing: " + ", ".join(f"`{_m}`" for _m in _missing) + " — run `scripts/pairwise_water_metrics.py --partition`."),
    )

    pairs = metrics["auto"].index.intersection(metrics["stripped"].index)
    # One colour per structure, shared by both roles: a marker's fill is its mtz (donor B),
    # its edge is its starting model (A), so a self-consistent pair reads as one hue.
    _ids = sorted({_i for _k in pairs for _i in _k})
    STRUCT_COLORS = dict(zip(_ids, plt.get_cmap("Set1").colors))
    return GROUND_TRUTHS, STRUCT_COLORS, metrics, pairs


@app.cell
def _(GROUND_TRUTHS, STRUCT_COLORS, metrics, np, pairs, plt):
    # One P–R plane per (ground truth × protocol), conventional orientation (recall x, precision y).
    # Axes are shared within a row and zoomed to that row's data, so auto and stripped are directly
    # comparable but the ranges differ between rows — the row separation is in the tick labels, not
    # the geometry. Reading P and R jointly matters because both move with the water count, which is
    # exactly what stripping changes — a per-metric strip chart would hide whether a precision gain
    # came from dropping junk or from dropping waters. The per-pair auto → stripped change is
    # quantified in the next figure rather than drawn as arrows here, which at n = 20 crossed into
    # illegibility.
    # alpha lives in the RGBA fill rather than in scatter(alpha=), which would fade the edge too —
    # the edge is the starting model and has to stay readable against the fill it sits on.
    _fills = [(*STRUCT_COLORS[_k[1]], 0.7) for _k in pairs]
    _edges = [STRUCT_COLORS[_k[0]] for _k in pairs]
    _VARIANTS = ("auto", "stripped")

    def _xy(_gt, _v):
        _, _pcol, _rcol = GROUND_TRUTHS[_gt]
        _df = metrics[_v].loc[pairs]
        return _df[_rcol].to_numpy(), _df[_pcol].to_numpy()

    def _row_limits(_gt, _step=0.05, _pad=0.02):
        """One square window per row, covering both metrics and both protocols.

        Precision and recall get the same range so the aspect stays 1:1 and the diagonal keeps
        meaning; rounding outward to `_step` puts the ticks on a round grid, and the small pad
        keeps a marker sitting exactly on a bound (template-only has a pair at 0 / 0) off the
        spine. Here this yields 65–100% for the B row and 0–60% for the template-only row.
        """
        _values = np.concatenate([_a for _v in _VARIANTS for _a in _xy(_gt, _v)])
        _lo = np.floor(_values.min() / _step) * _step
        _hi = np.ceil(_values.max() / _step) * _step
        _slack = (_hi - _lo) * _pad
        return _lo - _slack, _hi + _slack

    fig_pr, _axes = plt.subplots(len(GROUND_TRUTHS), len(_VARIANTS), figsize=(10.6, 10.4),
                                 sharex="row", sharey="row")
    for _row, _gt in zip(_axes, GROUND_TRUTHS):
        _lo, _hi = _row_limits(_gt)
        for _ax, _v in zip(_row, _VARIANTS):
            _x, _y = _xy(_gt, _v)
            _ax.plot([0, 1], [0, 1], color="0.92", lw=1, zorder=0)
            _ax.scatter(_x, _y, s=50, facecolor=_fills, edgecolor=_edges, lw=2.2, zorder=3)
            _ax.annotate(f"mean   R {_x.mean():.1%}   P {_y.mean():.1%}", (0.03, 0.97),
                         xycoords="axes fraction", ha="left", va="top", fontsize=10, color="0.25")
            _ax.set_xlim(_lo, _hi)
            _ax.set_ylim(_lo, _hi)
            _ax.set_aspect("equal")
            _ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
            _ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
            _ax.xaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
            _ax.yaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
            for _side in ("top", "right"):
                _ax.spines[_side].set_visible(False)

    for _ax, _v in zip(_axes[0], _VARIANTS):
        _ax.set_title(_v, fontsize=13)
    for _row, _gt in zip(_axes, GROUND_TRUTHS):
        _row[0].set_ylabel(f"{GROUND_TRUTHS[_gt][0]}\n\nprecision", fontsize=11)
    for _ax in _axes.ravel():
        _ax.set_xlabel("recall", fontsize=11)

    fig_pr.text(0.5, 0.125, "ground truth by row, protocol by column   ·   axes shared within a "
                "row and zoomed to it — the ranges differ between rows", ha="center", fontsize=10,
                color="0.3")
    fig_pr.text(0.5, 0.095, "precision = the cross-refinement's own waters lying on that ground "
                "truth   ·   recall = that ground truth recovered", ha="center", fontsize=9.5,
                color="0.45")
    fig_pr.legend(handles=[plt.Line2D([], [], marker="o", ls="none", markersize=11, markeredgewidth=2.2,
                                      markerfacecolor=(*_c, 0.7), markeredgecolor=_c, label=_i)
                           for _i, _c in STRUCT_COLORS.items()],
                  title="structure — fill = mtz (B), edge = starting model (A)",
                  loc="lower center", bbox_to_anchor=(0.5, 0.02), fontsize=10, title_fontsize=10,
                  frameon=False, ncol=len(STRUCT_COLORS))
    fig_pr.suptitle(f"Precision / recall against each ground truth (1.0 Å, n = {len(pairs)} pairs)",
                    fontsize=13)
    fig_pr.tight_layout(rect=(0, 0.15, 1, 0.97))
    fig_pr
    return (fig_pr,)


@app.cell
def _(GROUND_TRUTHS, STRUCT_COLORS, metrics, np, pairs, plt):
    # Everything on one plane: both ground truths, both protocols, all 20 pairs. Marker shape takes
    # over the ground truth (circle = supported by B, diamond = template-only), the marker sits at
    # stripped and the arrow leaving it lands on that pair's auto result, fill is the mtz donor (B)
    # and arrow colour the starting model (A). Overlaying costs the per-row zoom — the window has to
    # span both clouds, so it is back to the full square — but the separation between the two clouds
    # becomes a distance you can read off one pair of axes instead of a comparison across panels.
    _VARIANTS = ("stripped", "auto")
    _SHAPES = {"b": ("o", 50, "supported by B"), "template": ("D", 45, "template-only")}

    def _xy(_gt, _v):
        _, _pcol, _rcol = GROUND_TRUTHS[_gt]
        _df = metrics[_v].loc[pairs]
        return _df[_rcol].to_numpy(), _df[_pcol].to_numpy()

    _values = np.concatenate([_a for _gt in GROUND_TRUTHS for _v in _VARIANTS for _a in _xy(_gt, _v)])
    _lo = np.floor(_values.min() / 0.05) * 0.05
    _hi = np.ceil(_values.max() / 0.05) * 0.05
    _slack = (_hi - _lo) * 0.03

    fig_arrow, ax_arrow = plt.subplots(figsize=(6.0, 6.0))
    ax_arrow.plot([0, 1], [0, 1], color="0.92", lw=1, zorder=0)
    for _gt, (_marker, _size, _label) in _SHAPES.items():
        _stripped, _auto = (_xy(_gt, _v) for _v in _VARIANTS)
        for _k, _start, _end in zip(pairs, zip(*_stripped), zip(*_auto)):
            ax_arrow.annotate("", xy=_end, xytext=_start, zorder=2,
                              arrowprops=dict(arrowstyle="-|>", color=STRUCT_COLORS[_k[0]], lw=1.5,
                                              shrinkA=6, shrinkB=1))
        ax_arrow.scatter(*_stripped, marker=_marker, s=_size, edgecolor=[(*STRUCT_COLORS[_k[1]], 0.9) for _k in pairs],
                         lw=2, zorder=3, facecolor="none")
        # ax_arrow.scatter(*_stripped, marker=_marker, s=_size, edgecolor="k", lw=0.9, zorder=3,
        #                  facecolor=[(*STRUCT_COLORS[_k[1]], 0.7) for _k in pairs])
        ax_arrow.annotate(_label, (_stripped[0].mean(), _stripped[1].mean()),
                          textcoords="offset points", xytext=(-30, 34), ha="right", fontsize=12,
                          color="0.25")
    ax_arrow.set_xlim(_lo - _slack, _hi + _slack)
    ax_arrow.set_ylim(_lo - _slack, _hi + _slack)
    ax_arrow.set_aspect("equal")
    ax_arrow.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_arrow.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_arrow.xaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
    ax_arrow.yaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
    ax_arrow.set_xlabel("recall", fontsize=12)
    ax_arrow.set_ylabel("precision", fontsize=12)
    for _side in ("top", "right"):
        ax_arrow.spines[_side].set_visible(False)

    fig_arrow.text(0.5, 0.115, "marker = stripped, arrowhead = auto   ·   fill = mtz (B), arrow "
                   "colour = starting model (A)", ha="center", fontsize=10, color="0.3")
    _shape_legend = fig_arrow.legend(
        handles=[plt.Line2D([], [], marker=_m, ls="none", markersize=10, markerfacecolor="0.55",
                            markeredgecolor="white", label=_l) for _m, _s, _l in _SHAPES.values()],
        title="ground truth", loc="lower left", bbox_to_anchor=(0.09, 0.005), fontsize=10,
        title_fontsize=10, frameon=False)
    fig_arrow.add_artist(_shape_legend)
    fig_arrow.legend(handles=[plt.Line2D([], [], marker="o", ls="none", markersize=11,
                                         markerfacecolor=(*_c, 0.7), markeredgecolor="white",
                                         label=_i) for _i, _c in STRUCT_COLORS.items()],
                     title="structure", loc="lower right", bbox_to_anchor=(0.95, 0.005), fontsize=10,
                     title_fontsize=10, frameon=False, ncol=len(STRUCT_COLORS))
    fig_arrow.suptitle(f"Stripped → auto per pair (1.0 Å, n = {len(pairs)} pairs)", fontsize=13)
    fig_arrow.tight_layout(rect=(0, 0.16, 1, 0.97))
    fig_arrow
    return (fig_arrow,)


@app.cell
def _(GROUND_TRUTHS, STRUCT_COLORS, metrics, np, pairs, plt):
    # The overlay above stacks all 20 ordered pairs, so the reciprocal of any pair is somewhere in
    # the pile. One panel per unordered couple {X, Y} instead, holding both directions — X's data
    # refined from Y's model, and Y's data refined from X's — which is the comparison that isolates
    # the direction of the bias from the two structures' intrinsic difficulty. Four objects per
    # panel: two directions × two ground truths. Fill = mtz (B), arrow = starting model (A), so the
    # two directions of a couple carry the same two colours with the roles swapped. Axes are shared
    # across all ten panels.
    _VARIANTS = ("stripped", "auto")
    _SHAPES = {"b": ("o", 70, "supported by B"), "template": ("D", 55, "template-only")}
    _ids = sorted({_i for _k in pairs for _i in _k})
    _couples = [(_x, _y) for _i, _x in enumerate(_ids) for _y in _ids[_i + 1:]]

    def _point(_gt, _v, _k):
        _, _pcol, _rcol = GROUND_TRUTHS[_gt]
        _row = metrics[_v].loc[_k]
        return _row[_rcol], _row[_pcol]

    _values = np.concatenate(
        [metrics[_v][[_c for _, _pc, _rc in GROUND_TRUTHS.values() for _c in (_pc, _rc)]].to_numpy().ravel()
         for _v in _VARIANTS]
    )
    _lo = np.floor(np.nanmin(_values) / 0.05) * 0.05 - 0.03
    _hi = np.ceil(np.nanmax(_values) / 0.05) * 0.05 + 0.03

    # Each slot has to come out square or set_aspect("equal") shrinks the axes inside it and the
    # lower row's titles float up into the upper row's panels: 5 × 2 slots ≈ 3.1 in each, plus
    # room for the tick labels, the caption and the two legends.
    fig_facets, _axes = plt.subplots(2, 5, figsize=(16.5, 9.8), sharex=True, sharey=True)
    for _ax, _couple in zip(_axes.ravel(), _couples):
        _ax.plot([0, 1], [0, 1], color="0.92", lw=1, zorder=0)
        for _k in [_couple, _couple[::-1]]:
            if _k not in pairs:
                continue
            for _gt, (_marker, _size, _) in _SHAPES.items():
                _start, _end = (_point(_gt, _v, _k) for _v in _VARIANTS)
                _ax.annotate("", xy=_end, xytext=_start, zorder=2,
                             arrowprops=dict(arrowstyle="-|>", color=STRUCT_COLORS[_k[0]], lw=1.5,
                                             shrinkA=5, shrinkB=1))
                _ax.scatter(*_start, marker=_marker, s=_size, edgecolor="white", lw=0.8, zorder=3,
                            facecolor=[(*STRUCT_COLORS[_k[1]], 0.7)])
        _ax.set_xlim(_lo, _hi)
        _ax.set_ylim(_lo, _hi)
        _ax.set_aspect("equal")
        _ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
        _ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
        _ax.xaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
        _ax.yaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
        _ax.set_title(" ↔ ".join(_couple), fontsize=12)
        for _side in ("top", "right"):
            _ax.spines[_side].set_visible(False)
    for _ax in _axes[-1]:
        _ax.set_xlabel("recall", fontsize=11)
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("precision", fontsize=11)

    fig_facets.text(0.5, 0.115, "one panel per unordered couple, both directions in each   ·   "
                    "marker = stripped, arrowhead = auto   ·   fill = mtz (B), arrow colour = "
                    "starting model (A)", ha="center", fontsize=10, color="0.3")
    _shape_legend = fig_facets.legend(
        handles=[plt.Line2D([], [], marker=_m, ls="none", markersize=10, markerfacecolor="0.55",
                            markeredgecolor="white", label=_l) for _m, _s, _l in _SHAPES.values()],
        title="ground truth", loc="lower left", bbox_to_anchor=(0.10, 0.01), fontsize=10,
        title_fontsize=10, frameon=False, ncol=2)
    fig_facets.add_artist(_shape_legend)
    fig_facets.legend(handles=[plt.Line2D([], [], marker="o", ls="none", markersize=11,
                                          markerfacecolor=(*_c, 0.7), markeredgecolor="white",
                                          label=_i) for _i, _c in STRUCT_COLORS.items()],
                      title="structure", loc="lower right", bbox_to_anchor=(0.90, 0.01), fontsize=10,
                      title_fontsize=10, frameon=False, ncol=len(STRUCT_COLORS))
    fig_facets.suptitle(f"Stripped → auto, by reciprocal couple (1.0 Å, {len(_couples)} couples)",
                        fontsize=13)
    fig_facets.tight_layout(rect=(0, 0.15, 1, 0.96), h_pad=3.0)
    fig_facets
    return (fig_facets,)


@app.cell
def _(mo):
    facet_by = mo.ui.dropdown(
        options={"mtz — data donor (B)": "donor", "starting model — template (A)": "reference"},
        value="starting model — template (A)",
        label="One panel per",
    )
    facet_by
    return (facet_by,)


@app.cell
def _(GROUND_TRUTHS, STRUCT_COLORS, facet_by, metrics, np, pairs, plt):
    # Five panels, one per structure, holding the four pairs in which it plays the selected role.
    # Whichever role is faceted, that channel is constant inside a panel and the other varies: fixing
    # the mtz leaves the arrow colour (the starting model) as the free variable, and asks whether one
    # structure's data resists whatever template it is given; fixing the starting model leaves the
    # fill (the mtz) free, and asks whether one template imposes itself on whatever data it meets.
    _VARIANTS = ("stripped", "auto")
    _SHAPES = {"b": ("o", 50, "re-refined waters in B"), "template": ("D", 50, "re-refined waters unique to A")}
    _role = facet_by.value
    _slot = pairs.names.index(_role)
    _ids = sorted({_k[_slot] for _k in pairs})

    def _point(_gt, _v, _k):
        _, _pcol, _rcol = GROUND_TRUTHS[_gt]
        _row = metrics[_v].loc[_k]
        return _row[_rcol], _row[_pcol]

    _values = np.concatenate(
        [metrics[_v][[_c for _, _pc, _rc in GROUND_TRUTHS.values() for _c in (_pc, _rc)]].to_numpy().ravel()
         for _v in _VARIANTS]
    )
    _lo = np.floor(np.nanmin(_values) / 0.05) * 0.05 - 0.03
    _hi = np.ceil(np.nanmax(_values) / 0.05) * 0.05 + 0.03

    fig_role, _axes = plt.subplots(1, len(_ids), figsize=(16.0, 4), sharex=True, sharey=True)
    for _ax, _structure in zip(_axes, _ids):
        _ax.plot([0, 1], [0, 1], color="0.92", lw=1, zorder=0)
        for _k in [_k for _k in pairs if _k[_slot] == _structure]:
            for _gt, (_marker, _size, _) in _SHAPES.items():
                _start, _end = (_point(_gt, _v, _k) for _v in _VARIANTS)
                _ax.annotate("", xy=_end, xytext=_start, zorder=2,
                             arrowprops=dict(arrowstyle="-|>", color=STRUCT_COLORS[_k[0]], lw=2,
                                             shrinkA=5, shrinkB=1))
                # _ax.annotate("", xy=_end, xytext=_start, zorder=2,
                #              arrowprops=dict(arrowstyle="-|>", color="k", lw=2,
                #                              shrinkA=5, shrinkB=1))
                # _ax.scatter(*_start, marker=_marker, s=_size, edgecolor=[(*STRUCT_COLORS[_k[1]], 0.7)], lw=2, zorder=3,
                #             facecolor="none")
                _ax.scatter(*_start, marker=_marker, s=_size, edgecolor="none", lw=0.8, zorder=3,
                            facecolor=[(*STRUCT_COLORS[_k[1]], 0.7)])
        _ax.set_xlim(_lo, _hi)
        _ax.set_ylim(_lo, _hi)
        _ax.set_aspect("equal")
        _ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
        _ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
        _ax.xaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
        _ax.yaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
        _ax.set_xlabel("recall", fontsize=12)
        _ax.set_title(_structure, fontsize=13)
        for _side in ("top", "right"):
            _ax.spines[_side].set_visible(False)
    _axes[0].set_ylabel("precision", fontsize=12)

    _held, _free = (("mtz (B)", "arrow colour = starting model (A)") if _role == "donor"
                    else ("starting model (A)", "fill = mtz (B)"))
    # fig_role.text(0.5, 0.135, f"each panel holds one {_held} fixed, {_free}"
    #               "   ·   marker = stripped, arrowhead = kept", ha="center", fontsize=10, color="0.3")
    _shape_legend = fig_role.legend(
        handles=[plt.Line2D([], [], marker=_m, ls="none", markersize=10, markerfacecolor="0.55",
                            markeredgecolor="white", label=_l) for _m, _s, _l in _SHAPES.values()],
        title="ground truth", loc="lower left", bbox_to_anchor=(0.10, 0.005), fontsize=10,
        title_fontsize=10, frameon=False, ncol=2)
    fig_role.add_artist(_shape_legend)
    # fig_role.legend(handles=[plt.Line2D([], [], marker="o", ls="none", markersize=11,
    #                                     markerfacecolor=(*_c, 0.7), markeredgecolor="white",
    #                                     label=_i) for _i, _c in STRUCT_COLORS.items()],
    #                 title="partner structure", loc="lower right", bbox_to_anchor=(0.90, 0.005),
    #                 fontsize=10, title_fontsize=10, frameon=False, ncol=len(STRUCT_COLORS))
    fig_role.suptitle(f"cross-refinement stripped → kept", fontsize=14)
    fig_role.tight_layout(rect=(0, 0.22, 1, 0.95))
    fig_role
    return (fig_role,)


@app.cell
def _(mo):
    grounding = mo.ui.radio(
        options={"self-refined — <X>_refined_by_<X>": "", "deposited — PDB-REDO": "_deposited"},
        value="self-refined — <X>_refined_by_<X>",
        label="Ground truth for the A and B water sets",
        inline=True,
    )
    grounding
    return (grounding,)


@app.cell
def _(Path, config, grounding, mo, pd):
    # Same faceting as the cell above, but split B's waters into the two halves the partition
    # already distinguishes, giving three disjoint ground truths whose union is A ∪ B:
    #   AB-shared — B's waters that also sit on A. Conserved; any refinement should find these.
    #   B-only    — B's waters absent from A. The data's own contribution.
    #   A-only    — A's waters absent from B. The template's private sites; recovering them is bias.
    # Each gets its own precision numerator (n_pred_near_<group>, added to run_partition for this)
    # over the shared denominator n_pred. Those three numerators do not sum to n_pred: a predictor
    # water within cutoff of two ground-truth waters more than a cutoff apart counts for both, and
    # orphans count for none. So read each row as its own P–R plane, not as a composition.
    _dir = Path(f"{config.DATA_DIR}/hewls_65_subsampled_similar_default_phenix")
    GROUPS_3 = {
        "shared": "AB-shared — B waters also on A",
        "b_only": "B-only — B waters absent from A",
        "a_only": "A-only — A waters absent from B",
    }
    # The deposited grounding holds the three groups fixed across protocols (PDB-REDO does not
    # change when the refinement does), so a stripped → kept arrow is a pure predictor effect;
    # under the self-refined grounding both the predictor and the target move.
    GROUNDING_LABEL = "PDB-REDO" if grounding.value else "self-refined"
    group_metrics, _missing = {}, []
    for _v in ("auto", "stripped"):
        _p = _dir / f"starting_model_partition{grounding.value}_{_v}_1.0.csv"
        if not _p.exists():
            _missing.append(_p.name)
            continue
        _df = pd.read_csv(_p)
        if "n_pred_near_shared" not in _df.columns:
            _missing.append(f"{_p.name} (no n_pred_near_* columns — re-run --partition)")
            continue
        _out = _df[["reference", "donor"]].copy()
        for _g in GROUPS_3:
            _out[f"precision_{_g}"] = _df[f"n_pred_near_{_g}"] / _df["n_pred"]
            _out[f"recall_{_g}"] = _df[f"recall_{_g}"]
        group_metrics[_v] = _out.set_index(["reference", "donor"])
    mo.stop(
        bool(_missing),
        mo.md("Missing: " + ", ".join(f"`{_m}`" for _m in _missing)
              + " — run `scripts/pairwise_water_metrics.py --partition --cutoff 1.0`"
              + (" --ref-dir <cohort>/filtered_pdbs`." if grounding.value else "`.")),
    )
    return GROUNDING_LABEL, GROUPS_3, group_metrics


@app.cell
def _(
    GROUNDING_LABEL,
    GROUPS_3,
    STRUCT_COLORS,
    facet_by,
    group_metrics,
    np,
    pairs,
    plt,
):
    # Three rows (ground truth) × five columns (the faceted structure), reusing the dropdown above
    # so both breakdowns switch together. Axes are shared and squared per row — the three groups
    # differ by an order of magnitude in precision, so one common window would flatten two of them
    # into the corner. Arrows only, tail at stripped and head at kept: the column already fixes one
    # role, so colour is free to carry the partner — whichever of (starting model, mtz) is not the
    # faceted one, which keeps the encoding correct under either dropdown setting.
    _VARIANTS = ("stripped", "auto")
    _role = facet_by.value
    _slot = pairs.names.index(_role)
    _partner_slot = 1 - _slot
    _ids = sorted({_k[_slot] for _k in pairs})

    def _point(_g, _v, _k):
        _row = group_metrics[_v].loc[_k]
        return _row[f"recall_{_g}"], _row[f"precision_{_g}"]

    def _row_limits(_g, _step=0.05, _pad=0.04):
        _values = np.concatenate([group_metrics[_v][[f"recall_{_g}", f"precision_{_g}"]].to_numpy().ravel()
                                  for _v in _VARIANTS])
        _lo = np.floor(np.nanmin(_values) / _step) * _step
        _hi = np.ceil(np.nanmax(_values) / _step) * _step
        _slack = (_hi - _lo) * _pad
        return _lo - _slack, _hi + _slack

    fig_groups, _axes = plt.subplots(len(GROUPS_3), len(_ids), figsize=(16.5, 11.4),
                                     sharex="row", sharey="row")
    for _row, _g in zip(_axes, GROUPS_3):
        _lo, _hi = _row_limits(_g)
        # Tick interval scales with the row's span: B-only recall runs the full 0–100%, and 10%
        # ticks there collide into an unreadable smear across a panel this narrow.
        _tick = 0.25 if _hi - _lo > 0.7 else 0.1 if _hi - _lo > 0.3 else 0.05
        for _ax, _structure in zip(_row, _ids):
            _ax.plot([0, 1], [0, 1], color="0.92", lw=1, zorder=0)
            for _k in [_k for _k in pairs if _k[_slot] == _structure]:
                _start, _end = (_point(_g, _v, _k) for _v in _VARIANTS)
                _ax.annotate("", xy=_end, xytext=_start, zorder=2,
                             arrowprops=dict(arrowstyle="-|>", color=STRUCT_COLORS[_k[_partner_slot]],
                                             lw=1.8, shrinkA=0, shrinkB=0))
            _ax.set_xlim(_lo, _hi)
            _ax.set_ylim(_lo, _hi)
            _ax.set_aspect("equal")
            _ax.xaxis.set_major_locator(plt.MultipleLocator(_tick))
            _ax.yaxis.set_major_locator(plt.MultipleLocator(_tick))
            _ax.xaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
            _ax.yaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
            for _side in ("top", "right"):
                _ax.spines[_side].set_visible(False)
    for _ax, _structure in zip(_axes[0], _ids):
        _ax.set_title(_structure, fontsize=13)
    for _row, _g in zip(_axes, GROUPS_3):
        _row[0].set_ylabel(f"{GROUPS_3[_g]}\n\nprecision", fontsize=10.5)
    for _ax in _axes[-1]:
        _ax.set_xlabel("recall", fontsize=11)

    _held, _partner = (("mtz (B)", "starting model (A)") if _role == "donor"
                       else ("starting model (A)", "mtz (B)"))
    fig_groups.text(0.5, 0.075, f"column = the {_held} held fixed, row = ground truth   ·   arrow "
                    f"tail = stripped, head = kept   ·   arrow colour = the {_partner}   ·   axes "
                    "shared and squared per row", ha="center", fontsize=10, color="0.3")
    fig_groups.legend(handles=[plt.Line2D([], [], color=_c, lw=2.6, label=_i)
                               for _i, _c in STRUCT_COLORS.items()],
                      title=f"partner — the {_partner}", loc="lower center", bbox_to_anchor=(0.5, 0.005),
                      fontsize=10, title_fontsize=10, frameon=False, ncol=len(STRUCT_COLORS))
    fig_groups.suptitle(f"Cross-refinement stripped → kept, by water group and {_held}"
                        f"   —   ground truth: {GROUNDING_LABEL}", fontsize=14)
    fig_groups.tight_layout(rect=(0, 0.11, 1, 0.96), h_pad=2.5)
    fig_groups
    return (fig_groups,)


@app.cell
def _(mo):
    # Query the grid above: which (starting model, mtz) pairs move, in which group, and by how much.
    # Δ is oriented stripped → kept to match the arrows, so positive = the arrowhead is higher.
    GROUP_METRIC_COLUMNS = {
        "precision vs AB-shared": "precision_shared",
        "recall vs AB-shared": "recall_shared",
        "precision vs B-only": "precision_b_only",
        "recall vs B-only": "recall_b_only",
        "precision vs A-only": "precision_a_only",
        "recall vs A-only": "recall_a_only",
    }
    group_query_metrics = mo.ui.multiselect(options=list(GROUP_METRIC_COLUMNS),
                                            value=["recall vs A-only"], label="Metric")
    group_query_direction = mo.ui.dropdown(options={"increased": 1, "decreased": -1},
                                           value="increased",
                                           label="which, from stripped to kept,")
    group_query_threshold = mo.ui.slider(start=0, stop=30, step=1, value=0,
                                         label="by more than (percentage points)", show_value=True)
    group_query_match = mo.ui.radio(options={"all of them": "all", "any of them": "any"},
                                    value="all of them",
                                    label="When several are selected, match", inline=True)
    mo.vstack([group_query_metrics, group_query_direction, group_query_threshold, group_query_match])
    return (
        GROUP_METRIC_COLUMNS,
        group_query_direction,
        group_query_match,
        group_query_metrics,
        group_query_threshold,
    )


@app.cell
def _(
    GROUNDING_LABEL,
    GROUP_METRIC_COLUMNS,
    group_metrics,
    group_query_direction,
    group_query_match,
    group_query_metrics,
    group_query_threshold,
    mo,
    np,
    pairs,
    pd,
):
    # Reads group_metrics, so the table always matches whichever grounding the toggle has selected.
    # Each selected metric contributes its stripped value, its kept value and the change; the
    # unselected ones stay out of the table rather than padding it.
    _selected = [GROUP_METRIC_COLUMNS[_label] for _label in group_query_metrics.value]
    _sign = group_query_direction.value
    _threshold = group_query_threshold.value / 100

    _stripped = group_metrics["stripped"].loc[pairs]
    _kept = group_metrics["auto"].loc[pairs]
    _change = _kept - _stripped

    if not _selected:
        _hits = pairs[np.zeros(len(pairs), dtype=bool)]
    else:
        _tests = np.column_stack([(_change[_col] * _sign > _threshold).to_numpy() for _col in _selected])
        _keep = _tests.all(axis=1) if group_query_match.value == "all" else _tests.any(axis=1)
        _hits = pairs[_keep]

    _table = pd.DataFrame(index=_hits)
    for _label in group_query_metrics.value:
        _col = GROUP_METRIC_COLUMNS[_label]
        _table[f"{_label}: stripped"] = (_stripped.loc[_hits, _col] * 100).round(1)
        _table[f"{_label}: kept"] = (_kept.loc[_hits, _col] * 100).round(1)
        _table[f"{_label}: Δ"] = (_change.loc[_hits, _col] * 100).round(1)
    _table = _table.rename_axis(index={"reference": "starting model (A)", "donor": "mtz (B)"})
    if _selected:
        _table = _table.sort_values(f"{group_query_metrics.value[0]}: Δ", ascending=_sign < 0)

    _verb = "increased" if _sign > 0 else "decreased"
    _joined = f" {'and' if group_query_match.value == 'all' else 'or'} ".join(group_query_metrics.value)
    _by = f" by more than {group_query_threshold.value} pp" if group_query_threshold.value else ""
    _summary = (f"**{len(_hits)} / {len(pairs)} pairs** where {_joined or '(nothing selected)'} "
                f"{_verb}{_by} from stripped to kept, against the **{GROUNDING_LABEL}** ground "
                f"truth. Values in percentage points.")
    mo.vstack([mo.md(_summary), _table.reset_index()])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    # Self-refinement against PDB-REDO — does keeping the deposited waters help?

    Everything above compares refinements to each other. This grounds them on the deposited waters
    instead: for each structure X, the self-refinement `X_refined_by_X` scored against X's
    PDB-REDO waters, under both protocols. `stripped` throws the PDB-REDO waters away and re-places
    solvent de novo; `auto` starts from them and refines. The arrow runs stripped → auto, so it
    measures what keeping the deposited waters buys.

    Unlike the cross-refinement cells, **the ground truth is fixed across the two protocols** —
    PDB-REDO does not change when the refinement does — so an arrow here is a clean predictor
    effect, not a whole-protocol comparison.

    Reads the diagonal rows (`reference == predictor`) of
    `phenix_pairwise_metrics_<variant>_1.0.csv`, whose reference is `<cohort>/filtered_pdbs/<id>.cif`
    — the distance-filtered PDB-REDO structure.
    """)
    return


@app.cell
def _(Path, config, mo, pd):
    _dir = Path(f"{config.DATA_DIR}/hewls_65_subsampled_similar_default_phenix")
    selfref, _missing = {}, []
    for _v in ("auto", "stripped"):
        _p = _dir / f"phenix_pairwise_metrics_{_v}_1.0.csv"
        if _p.exists():
            _df = pd.read_csv(_p)
            selfref[_v] = _df[_df["reference"] == _df["predictor"]].set_index("reference")
        else:
            _missing.append(_p.name)
    mo.stop(
        bool(_missing),
        mo.md("Missing: " + ", ".join(f"`{_m}`" for _m in _missing)
              + " — run `scripts/pairwise_water_metrics.py --phenix --ref-dir <cohort>/filtered_pdbs`."),
    )
    return (selfref,)


@app.cell
def _(STRUCT_COLORS, np, plt, selfref):
    # Five structures, one arrow each: the circle is stripped (PDB-REDO waters discarded, solvent
    # re-placed de novo), the arrowhead is auto (PDB-REDO waters kept as the starting point). Same
    # square-window and colour conventions as the cells above, so this panel can be read next to
    # them; there is no shape channel here because there is only one ground truth.
    _VARIANTS = ("stripped", "auto")
    _ids = sorted(selfref["stripped"].index.intersection(selfref["auto"].index))

    def _point(_v, _structure):
        _row = selfref[_v].loc[_structure]
        return _row["recall"], _row["precision"]

    _values = np.array([_point(_v, _s) for _v in _VARIANTS for _s in _ids])
    _lo = np.floor(_values.min() / 0.05) * 0.05 - 0.05
    _hi = np.ceil(_values.max() / 0.05) * 0.05 + 0.05

    fig_selfref, ax_selfref = plt.subplots(figsize=(3, 3))
    ax_selfref.plot([0, 1], [0, 1], color="0.92", lw=1, zorder=0)
    for _structure in _ids:
        _start, _end = (_point(_v, _structure) for _v in _VARIANTS)
        # Arrow only: tail = stripped, head = kept. shrinkA drops to 0 so the tail sits exactly on
        # the stripped value now that no marker covers it, and the label carries the identity the
        # marker used to.
        ax_selfref.annotate("", xy=_end, xytext=_start, zorder=2,
                            arrowprops=dict(arrowstyle="-|>", color=STRUCT_COLORS[_structure],
                                            lw=2, shrinkA=0, shrinkB=0))
        # ax_selfref.annotate(_structure, _start, textcoords="offset points", xytext=(-1, -1),
        #                     ha="right", va="center", fontsize=11, color=STRUCT_COLORS[_structure])
    ax_selfref.set_xlim(_lo, _hi)
    ax_selfref.set_ylim(_lo, _hi)
    ax_selfref.set_aspect("equal")
    ax_selfref.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_selfref.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_selfref.xaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
    ax_selfref.yaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
    ax_selfref.set_xlabel("recall", fontsize=14)
    ax_selfref.set_ylabel("precision", fontsize=14)
    for _side in ("top", "right"):
        ax_selfref.spines[_side].set_visible(False)
    ax_selfref.set_title("re-refinement (stripped → kept)",fontsize=14)
    fig_selfref.tight_layout()
    fig_selfref
    return (fig_selfref,)


@app.cell
def _(mo, pd, selfref):
    # _rows = []
    for _structure in sorted(selfref["stripped"].index.intersection(selfref["auto"].index)):
        _row = {"structure": _structure, "PDB-REDO waters": int(selfref["auto"].loc[_structure, "n_water_ref"])}
        for _what in ("precision", "recall"):
            _s = selfref["stripped"].loc[_structure, _what]
            _a = selfref["auto"].loc[_structure, _what]
            _row[f"{_what}: stripped"] = round(_s * 100, 1)
            _row[f"{_what}: auto"] = round(_a * 100, 1)
            _row[f"{_what}: Δ"] = round((_a - _s) * 100, 1)
        _row["waters: stripped"] = int(selfref["stripped"].loc[_structure, "n_water_pred"])
        _row["waters: auto"] = int(selfref["auto"].loc[_structure, "n_water_pred"])
        _rows.append(_row)
    mo.vstack([mo.md("Δ is auto − stripped, in percentage points."), pd.DataFrame(_rows)])
    return


@app.cell
def _(mo):
    save_ui = mo.ui.checkbox(label="Save figures to data/plots/")
    save_ui
    return (save_ui,)


@app.cell
def _(
    Path,
    facet_by,
    fig_arrow,
    fig_facets,
    fig_groups,
    fig_pr,
    fig_role,
    fig_selfref,
    grounding,
    mo,
    save_ui,
):
    _out = Path("data/plots")
    if save_ui.value:
        _out.mkdir(parents=True, exist_ok=True)
        _saved = []
        for _f, _name in ((fig_pr, "partition_precision_recall"),
                          (fig_arrow, "partition_pr_stripped_to_auto"),
                          (fig_facets, "partition_pr_by_couple"),
                          (fig_role, f"partition_pr_by_{facet_by.value}"),
                          (fig_groups, f"partition_pr_groups{grounding.value}_by_{facet_by.value}"),
                          (fig_selfref, "selfref_vs_pdbredo")):
            _p = _out / f"{_name}_1.0.png"
            _f.savefig(_p, dpi=200, bbox_inches="tight")
            _saved.append(_p)
        _msg = mo.md("Saved → " + ", ".join(f"`{_p}`" for _p in _saved))
    else:
        _msg = mo.md(f"_Tick the box to write the figures to `{_out}/`._")
    _msg
    return


if __name__ == "__main__":
    app.run()
