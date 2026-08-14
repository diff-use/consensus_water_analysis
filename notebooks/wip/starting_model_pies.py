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
    # Water-set composition — the same partitions as circles

    The P–R planes in `starting_model_partition_recall.py` put each water group on its own axes,
    which keeps the metrics exact but loses the sizes: a 50% recall of A-only reads the same whether
    A-only is 9 waters or 42. Here every group is drawn with **angle proportional to water count**
    instead, so a recall is read as the saturated share of a wedge whose size is itself meaningful.

    Two figures, both fed by the artifacts the other notebook already reads:

    - **cross-refinement, couple by couple** — one column per unordered couple {X, Y}, holding the
      direction-free partition of the two water sets and both cross-refinements of that couple.
    - **self-refinement against PDB-REDO** — one circle per structure, `X_refined_by_X` scored
      against X's deposited waters.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys
    import textwrap

    sys.path.insert(0, str(Path(".")))

    import config
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    from matplotlib.legend_handler import HandlerTuple

    return HandlerTuple, Path, config, np, pd, plt, textwrap


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
def _(Path, config, grounding, mo, pd, plt):
    # Raw counts, not the derived fractions: the circles size their wedges in waters. The deposited
    # grounding holds the three groups fixed across protocols (PDB-REDO does not change when the
    # refinement does); under the self-refined grounding both the predictor and the target move.
    _dir = Path(f"{config.DATA_DIR}/hewls_65_subsampled_similar_default_phenix")
    GROUNDING_LABEL = "PDB-REDO" if grounding.value else "self-refined"
    partition_raw, _missing = {}, []
    for _v in ("auto", "stripped"):
        _p = _dir / f"starting_model_partition{grounding.value}_{_v}_1.0.csv"
        if _p.exists():
            partition_raw[_v] = pd.read_csv(_p).set_index(["reference", "donor"])
        else:
            _missing.append(_p.name)
    mo.stop(
        bool(_missing),
        mo.md("Missing: " + ", ".join(f"`{_m}`" for _m in _missing)
              + " — run `scripts/pairwise_water_metrics.py --partition --cutoff 1.0`"
              + (" --ref-dir <cohort>/filtered_pdbs`." if grounding.value else "`.")),
    )

    pairs = partition_raw["auto"].index.intersection(partition_raw["stripped"].index)
    # One colour per structure, shared by both roles and by both figures below.
    _ids = sorted({_i for _k in pairs for _i in _k})
    STRUCT_COLORS = dict(zip(_ids, plt.get_cmap("Set1").colors))
    return GROUNDING_LABEL, STRUCT_COLORS, pairs, partition_raw


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Cross-refinement, couple by couple

    Each unordered couple {X, Y} gets a column of three circles:

    - **top** — the two water sets partitioned: X-only, shared, Y-only. Direction-free, so it is
      the same reference for both cross-refinements below it.
    - **middle** — the cross-refinement with **X as starting model, Y as mtz**, against two
      ground truths rather than three. The mtz's wedge is **all of B** — its private waters and
      the core it shares with A — since that whole set is what B's own data supports. The starting
      model contributes only its **private** waters, **ringed**: the shared core is not the
      template's to claim. The saturated part of each wedge is what the cross-refinement
      recovered, the pale part what it missed, and a **white** wedge holds the orphans — its own
      waters sitting near neither X nor Y.
    - **bottom** — the same couple with the roles swapped.

    A structure keeps the same side of the circle down the column, so the middle-vs-bottom
    comparison is the bias question directly: *is that structure's private set recovered because
    it was the template, or because the data supports it?* If it fills in only when its structure
    is the starting model, that is template bias. Note the side's **contents** change with its
    role — whole set as mtz, private waters only as starting model — so it is the ringed wedge,
    not the wedge's size, that is compared across the two rows.

    Note the mixed unit: the group wedges count **ground-truth** waters, the orphan wedge counts
    **cross-refinement** waters, since an orphan has no ground-truth counterpart by definition.
    The circle is "what was there to find, plus what was invented" — not one set. The shared count
    is measured on the mtz's side, so it moves by a water or two between the two directions; the
    top circle averages them.
    """)
    return


@app.cell
def _(mo):
    pie_variant = mo.ui.radio(
        options={"kept — starting waters retained": "auto",
                 "stripped — solvent re-placed de novo": "stripped"},
        value="kept — starting waters retained",
        label="Protocol",
        inline=True,
    )
    pie_variant
    return (pie_variant,)


@app.cell
def _(mo, pairs):
    # Which couples to draw. Keyed by label so the figure can keep the canonical alphabetical
    # column order whatever order they were clicked in.
    _ids = sorted({_i for _k in pairs for _i in _k})
    COUPLE_LABELS = {f"{_x} ↔ {_y}": (_x, _y)
                     for _i, _x in enumerate(_ids) for _y in _ids[_i + 1:]}
    couple_pick = mo.ui.multiselect(options=list(COUPLE_LABELS), value=list(COUPLE_LABELS),
                                    label="Couples")
    couple_pick
    return COUPLE_LABELS, couple_pick


@app.cell
def _(
    COUPLE_LABELS,
    GROUNDING_LABEL,
    STRUCT_COLORS,
    couple_pick,
    mo,
    np,
    pairs,
    partition_raw,
    pie_variant,
    plt,
    textwrap,
):
    # Three circles per couple. The X side comes first and the Y side second in every row, X being
    # the alphabetically first of the couple, so a structure holds the same side of the circle in
    # all three rows regardless of the role that row gives it — that fixed position is what makes
    # the template-vs-data comparison a single glance down a column. What the side contains does
    # change with the role: as mtz it is that structure's whole water set, shared core included; as
    # starting model it is only its private waters, the ringed ones. Each group wedge is split
    # angularly, not radially: covered and missed sit side by side in the same hue so the covered
    # share is read as an angle like everything else in the circle, rather than as a radius the eye
    # would have to square.
    _counts = partition_raw[pie_variant.value]
    _ids = sorted({_i for _k in pairs for _i in _k})
    _available = [_c for _c in COUPLE_LABELS.values()
                  if _c in _counts.index and _c[::-1] in _counts.index]
    _picked = set(couple_pick.value)
    _couples = [_c for _label, _c in COUPLE_LABELS.items()
                if _label in _picked and _c in _available]
    mo.stop(not _couples, mo.md("_Pick at least one couple._"))
    SHARED_GREY = (0.45, 0.45, 0.45)
    ORPHAN_WHITE = (1.0, 1.0, 1.0)

    def _tint(_c, _towards_white=0.82):
        return tuple(np.asarray(_c) * (1 - _towards_white) + _towards_white)

    def _sections(_couple, _direction):
        """[(count, recovered fraction, colour)] in wedge order, for one circle.

        `_direction` is (starting model, mtz), or None for the reference partition — which has no
        recovery to show, so its groups come back fully covered. The private sets are
        direction-free (A-only of one direction is B-only of the other, same waters), so only
        n_shared differs between the two directions and the reference circle averages it.
        """
        _x, _y = _couple
        if _direction is None:
            _forward, _back = _counts.loc[(_x, _y)], _counts.loc[(_y, _x)]
            return [(_forward["n_a_only"], 1.0, STRUCT_COLORS[_x]),
                    ((_forward["n_shared"] + _back["n_shared"]) / 2, 1.0, SHARED_GREY),
                    (_forward["n_b_only"], 1.0, STRUCT_COLORS[_y])]
        _row = _counts.loc[_direction]
        _a, _b = _direction
        # B is taken whole — its private waters *and* the core it shares with A — so the mtz's
        # wedge is the entire set its own data supports, and its recall is the two halves' recalls
        # weighted by their counts. A keeps only its private waters: the shared core is not the
        # template's to claim, and A-only is the set that answers the bias question.
        _b_total = _row["n_shared"] + _row["n_b_only"]
        _b_recall = np.nansum([_row["recall_shared"] * _row["n_shared"],
                               _row["recall_b_only"] * _row["n_b_only"]]) / _b_total
        _wedge = {_a: (_row["n_a_only"], _row["recall_a_only"], STRUCT_COLORS[_a]),
                  _b: (_b_total, _b_recall, STRUCT_COLORS[_b])}
        return [_wedge[_x], _wedge[_y], (_row["n_pred_orphan"], 1.0, ORPHAN_WHITE)]

    # Radius carries the total, so a couple with more waters draws a bigger circle and the orphan
    # wedge adds area instead of squeezing the groups; area ∝ count, hence the square root. The
    # scale runs over every available couple, not just the picked ones, so narrowing the selection
    # re-crops the figure without silently resizing the circles that stay in it.
    _totals = {_panel: sum(_s[0] for _s in _sections(*_panel))
               for _couple in _available
               for _panel in ((_couple, None), (_couple, _couple), (_couple, _couple[::-1]))}
    _biggest = max(_totals.values())

    def _draw(_ax, _panel):
        _fractions, _colors = [], []
        for _n, _recall, _color in _sections(*_panel):
            _covered = 0.0 if np.isnan(_recall) else _n * _recall
            _fractions += [_covered, _n - _covered]
            _colors += [_color, _tint(_color)]
        _radius = 0.97 * np.sqrt(_totals[_panel] / _biggest)
        _wedges, _ = _ax.pie(_fractions, colors=_colors, radius=_radius, startangle=90,
                             counterclock=False, frame=True,
                             wedgeprops=dict(edgecolor="white", linewidth=0.9))
        # In the two cross-refinement rows, ring the starting model's private waters — the whole
        # section, covered and missed together, since the question that section answers is how big
        # the template's private set is and how much of it came back, not either half alone.
        _highlight = None if _panel[1] is None else STRUCT_COLORS[_panel[1][0]]
        _labels = []
        for _i, (_n, _recall, _color) in enumerate(_sections(*_panel)):
            if _color == ORPHAN_WHITE:
                for _w in _wedges[2 * _i:2 * _i + 2]:
                    _w.set_edgecolor("0.7")
            if _n == 0:
                continue
            # if _color == _highlight:
            #     # counterclock=False, so the group runs from the missed half's theta1 round to the
            #     # covered half's theta2 — the ring spans both.
            #     _ax.add_patch(Wedge((0, 0), _radius * 1.04, _wedges[2 * _i + 1].theta1,
            #                         _wedges[2 * _i].theta2, facecolor="none", edgecolor="0.2",
            #                         lw=1.6, zorder=5))
            # Label at the mid-angle of the whole group, not of its covered half, so the labels
            # keep the same spacing whatever the recall is.
            _labels.append(((_wedges[2 * _i].theta1 + _wedges[2 * _i + 1].theta2) / 2,
                            f"{_n:.0f}" if _panel[1] is None or _color == ORPHAN_WHITE else f"{_recall:.0%}",
                            "0.45" if _color == ORPHAN_WHITE else _color))
        # A thin group next to a fat one puts their two mid-angles almost on top of each other.
        # Slide the later of such a pair along the arc, away from its neighbour, and out a little:
        # sliding beats pushing straight out, which at 3 or 9 o'clock only parks the two labels
        # side by side on the same line.
        _placed = []
        for _degrees, _text, _color in sorted(_labels):
            _nearest = min(_placed, key=lambda _other: abs(_degrees - _other), default=None)
            _clash = _nearest is not None and abs(_degrees - _nearest) < 26
            _placed.append(_degrees)
            if _clash:
                _degrees += 16 if _degrees >= _nearest else -16
            _factor = 1.22 if _clash else 1.1
            _theta = np.radians(_degrees)
            _ax.text(_factor * _radius * np.cos(_theta), _factor * _radius * np.sin(_theta), _text,
                     fontsize=8, color=_color,
                     ha="left" if np.cos(_theta) > 0.2 else "right" if np.cos(_theta) < -0.2 else "center",
                     va="bottom" if np.sin(_theta) > 0.2 else "top" if np.sin(_theta) < -0.2 else "center")
        _ax.set_xlim(-1.62, 1.62)
        _ax.set_ylim(-1.62, 1.62)
        _ax.set_aspect("equal")
        _ax.set_xticks([])
        _ax.set_yticks([])
        for _side in _ax.spines.values():
            _side.set_visible(False)

    # A one- or two-couple selection would leave the figure narrower than the caption and the two
    # legends need, so the width has a floor and everything below the panels re-flows: the caption
    # wraps to the width, and the legends stack centred instead of sitting left and right.
    _width = max(7.4, 1.72 * len(_couples))
    _narrow = _width < 12
    _shown = [_i for _i in _ids if any(_i in _couple for _couple in _couples)]

    fig_pies, _axes = plt.subplots(3, len(_couples), figsize=(_width, 6.9), squeeze=False)
    for _column, _couple in zip(_axes.T, _couples):
        for _ax, _direction in zip(_column, (None, _couple, _couple[::-1])):
            _draw(_ax, (_couple, _direction))
            if _direction is None:
                _ax.set_title(" ↔ ".join(_couple), fontsize=11.5, pad=6)
            else:
                _ax.set_title(f"start {_direction[0]} → mtz {_direction[1]}", fontsize=8.5,
                              color="0.3", pad=6)
    for _ax, _label in zip(_axes[:, 0], ("the two\nwater sets", "cross-\nrefinement",
                                         "roles\nswapped")):
        _ax.set_ylabel(_label, fontsize=10, color="0.25", labelpad=8)

    _caption = ("angle ∝ waters   ·   top row labelled in waters, lower rows in % of the group "
                "recovered   ·   lower rows hold two groups: the mtz's whole set, shared core "
                "included, and the starting model's private waters (ringed)   ·   saturated = "
                "recovered, pale = missed   ·   circle area ∝ total waters")
    fig_pies.text(0.5, 0.145 if _narrow else 0.10, textwrap.fill(_caption, int(_width * 15)),
                  ha="center", va="bottom", fontsize=10, color="0.3", linespacing=1.5)
    _fill_legend = fig_pies.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor=SHARED_GREY, edgecolor="white", label="shared by both (top row)"),
                 plt.Rectangle((0, 0), 1, 1, facecolor=_tint(SHARED_GREY), edgecolor="white", label="missed"),
                 plt.Rectangle((0, 0), 1, 1, facecolor=ORPHAN_WHITE, edgecolor="0.7", label="orphan"),
                 plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="0.2", linewidth=1.6,
                               label="ringed = the starting model's own waters")],
        loc="lower center" if _narrow else "lower left",
        bbox_to_anchor=(0.5, 0.065) if _narrow else (0.10, 0.01),
        fontsize=10, frameon=False, ncol=2 if _narrow else 4)
    fig_pies.add_artist(_fill_legend)
    # Only the structures on screen: pick one couple and the legend is that couple's two colours.
    fig_pies.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=STRUCT_COLORS[_i],
                                           edgecolor="white", label=_i) for _i in _shown],
                    title="waters of",
                    loc="lower center" if _narrow else "lower right",
                    bbox_to_anchor=(0.5, 0.005) if _narrow else (0.90, 0.01),
                    fontsize=10, title_fontsize=10, frameon=False, ncol=len(_shown))
    fig_pies.suptitle(f"Water-set composition and what the cross-refinement recovers "
                      f"({pie_variant.value}, 1.0 Å)   —   ground truth: {GROUNDING_LABEL}",
                      fontsize=13)
    # Explicit spacing, not tight_layout: with 30 equal-aspect axes tight_layout resolves the
    # aspect constraint by collapsing whole rows to zero height.
    # When the width floor kicks in, the columns are centred at their normal 1.72 in rather than
    # stretched to fill it — a picked couple then draws exactly the circle it draws in the full
    # figure, with the slack going to the margins instead of into the pies.
    _block = 1.72 * len(_couples) / _width
    _left, _right = ((1 - _block) / 2, (1 + _block) / 2) if _narrow else (0.035, 0.995)
    fig_pies.subplots_adjust(left=_left, right=_right, top=0.885,
                             bottom=0.25 if _narrow else 0.155, wspace=0.06, hspace=0.28)
    fig_pies
    return (fig_pies,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Self-refinement against PDB-REDO

    One circle per structure: `X_refined_by_X` scored against X's PDB-REDO waters. `stripped`
    throws those waters away and re-places solvent de novo; `auto` starts from them and refines.
    The ground truth is fixed across the two protocols — PDB-REDO does not change when the
    refinement does — so the toggle is a clean predictor comparison.

    There is one ground truth here and so no partition to make: the deposited set is the whole
    circle bar the coloured wedge, which holds the refinement's own waters that sit near none of
    them. Colour is spent on that wedge alone, since it is the only part that differs in kind
    between structures.

    Reads the diagonal rows (`reference == predictor`) of `phenix_pairwise_metrics_<variant>_1.0.csv`.
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
def _(mo):
    selfref_pie_variant = mo.ui.radio(
        options={"kept — deposited waters retained": "auto",
                 "stripped — solvent re-placed de novo": "stripped"},
        value="kept — deposited waters retained",
        label="Protocol",
        inline=True,
    )
    selfref_pie_sizing = mo.ui.radio(
        options={"area ∝ total waters": "scaled", "all circles the same size": "equal"},
        value="area ∝ total waters",
        label="Circle size",
        inline=True,
    )
    mo.vstack([selfref_pie_variant, selfref_pie_sizing])
    return selfref_pie_sizing, selfref_pie_variant


@app.cell
def _(
    HandlerTuple,
    STRUCT_COLORS,
    np,
    plt,
    selfref,
    selfref_pie_sizing,
    selfref_pie_variant,
):
    # Angle ∝ waters, as in the couple pies, but the two greys are common to all five structures:
    # recovered and missed mean the same thing in every panel, so the structure's colour is spent
    # only on the wedge that is its own — the waters phenix put down that no deposited water
    # accounts for. Every wedge carries its count, so no panel needs a caption of totals.
    _VARIANTS = ("auto", "stripped")
    _ids = sorted(selfref["stripped"].index.intersection(selfref["auto"].index))
    _variant = selfref_pie_variant.value
    RECOVERED_GREY = (0.38, 0.38, 0.38)
    MISSED_GREY = (0.86, 0.86, 0.86)

    def _circle(_v, _structure):
        """(recovered, missed, unique-to-phenix), in waters."""
        _row = selfref[_v].loc[_structure]
        _recovered = _row["n_water_ref"] * _row["recall"]
        return _recovered, _row["n_water_ref"] - _recovered, _row["n_water_pred"] * (1 - _row["precision"])

    # Under "scaled", area ∝ the circle's total (deposited + phenix-unique), so 6ybf's 107 waters
    # draw a visibly bigger circle than 9lmk's 74 and the wedge counts stay comparable in area from
    # panel to panel. The normalisation is over both protocols rather than the selected one, so
    # flipping Protocol moves a circle only by as much as its water count actually moved. Under
    # "equal" every circle is drawn at full radius: the totals are then only in the labels, and
    # each panel is read purely as a composition.
    _biggest = max(sum(_circle(_v, _s)) for _v in _VARIANTS for _s in _ids)

    fig_selfref_pies, _axes = plt.subplots(1, len(_ids), figsize=(11.0, 3))
    for _ax, _structure in zip(_axes, _ids):
        _color = STRUCT_COLORS[_structure]
        # (count, wedge colour, label colour inside, label colour outside)
        _sections = list(zip(_circle(_variant, _structure),
                             (RECOVERED_GREY, MISSED_GREY, _color),
                             ("white", "0.35", "white"),
                             ("0.35", "0.45", _color)))
        _radius = 0.97 if selfref_pie_sizing.value == "equal" else \
            0.97 * np.sqrt(sum(_n for _n, *_ in _sections) / _biggest)
        _wedges, _ = _ax.pie([_n for _n, *_ in _sections], colors=[_c for _, _c, *_ in _sections],
                             radius=_radius, startangle=90, counterclock=False, frame=True,
                             wedgeprops=dict(edgecolor="white", linewidth=0.9))
        for _wedge, (_n, _, _inside_color, _outside_color) in zip(_wedges, _sections):
            if _n == 0:
                continue
            # A wedge thinner than ~25° cannot hold its own count at 0.6 radius, so that one label
            # goes outside instead, in the wedge's colour rather than against it.
            _inside = _wedge.theta2 - _wedge.theta1 >= 25
            _theta = np.radians((_wedge.theta1 + _wedge.theta2) / 2)
            _factor = 0.62 if _inside else 1.12
            _ax.text(_factor * _radius * np.cos(_theta), _factor * _radius * np.sin(_theta),
                     f"{_n:.0f}", fontsize=14, color=_inside_color if _inside else _outside_color,
                     ha="center" if _inside else
                        "left" if np.cos(_theta) > 0.2 else "right" if np.cos(_theta) < -0.2 else "center",
                     va="center" if _inside else
                        "bottom" if np.sin(_theta) > 0.2 else "top" if np.sin(_theta) < -0.2 else "center")
        # These two set how much empty room each circle carries inside its own axes — the radius
        # tops out at 0.97, so the limit is the padding: raise it for looser spacing between
        # structures, lower it (toward 1.0) to close the row up. The outside labels sit at 1.12 r
        # and are not clipped, so they spill into the gap rather than being cut off.
        _ax.set_xlim(-1.12, 1.12)
        _ax.set_ylim(-1.12, 1.12)
        _ax.set_aspect("equal")
        _ax.set_xticks([])
        _ax.set_yticks([])
        for _side in _ax.spines.values():
            _side.set_visible(False)
        _ax.set_title(_structure, fontsize=14, color=_color, pad=3)

    # The unique-to-phenix swatch is striped in all five structure colours, since that wedge is the
    # only coloured one and its hue is the structure's.
    fig_selfref_pies.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor=RECOVERED_GREY, edgecolor="white"),
                 plt.Rectangle((0, 0), 1, 1, facecolor=MISSED_GREY, edgecolor="white"),
                 tuple(plt.Rectangle((0, 0), 1, 1, facecolor=_c, edgecolor="white")
                       for _c in STRUCT_COLORS.values())],
        labels=["common", "waters unique to PDB-REDO", "waters unique to Phenix"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc="lower center", bbox_to_anchor=(0.5, 0.05), fontsize=14, frameon=False, ncol=3)
    # fig_selfref_pies.suptitle(f"Self-refinement against PDB-REDO ({_variant}, 1.0 Å)", fontsize=13)
    fig_selfref_pies.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.17, wspace=0.04)
    fig_selfref_pies
    return (fig_selfref_pies,)


@app.cell
def _(STRUCT_COLORS, np, plt, selfref):
    # The circles above say what each protocol found; this says how the two score against the one
    # fixed target. One arrow per structure, tail at stripped (deposited waters discarded, solvent
    # re-placed de novo) and head at kept (deposited waters refined from), in the structure's
    # colour — so both protocols are on screen at once, unlike the pies, which show one at a time.
    # There is no shape channel because there is only one ground truth.
    _VARIANTS = ("stripped", "auto")
    _ids = sorted(selfref["stripped"].index.intersection(selfref["auto"].index))

    def _point(_v, _structure):
        _row = selfref[_v].loc[_structure]
        return _row["recall"], _row["precision"]

    _values = np.array([_point(_v, _s) for _v in _VARIANTS for _s in _ids])
    _lo = np.floor(_values.min() / 0.05) * 0.05
    _hi = np.ceil(_values.max() / 0.05) * 0.05
    fig_selfref, ax_selfref = plt.subplots(figsize=(3.0, 3.0))
    ax_selfref.plot([0, 1], [0, 1], color="0.92", lw=1, zorder=0)
    for _structure in _ids:
        _start, _end = (_point(_v, _structure) for _v in _VARIANTS)
        # shrinkA is 0 so the tail sits exactly on the stripped value — no marker covers it, and
        # the label at the tail carries the identity a marker would have.
        ax_selfref.annotate("", xy=_end, xytext=_start, zorder=2,
                            arrowprops=dict(arrowstyle="-|>", color=STRUCT_COLORS[_structure],
                                            lw=2, shrinkA=0, shrinkB=0))
        # ax_selfref.annotate(_structure, _start, textcoords="offset points", xytext=(-2, -1),
        #                     ha="right", va="center", fontsize=14, color=STRUCT_COLORS[_structure])
    ax_selfref.set_xlim(_lo, _hi)
    ax_selfref.set_ylim(_lo, _hi)
    ax_selfref.set_aspect("equal")
    ax_selfref.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_selfref.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_selfref.xaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
    ax_selfref.yaxis.set_major_formatter(lambda _t, _p: f"{_t:.0%}")
    ax_selfref.tick_params(labelsize=12)
    ax_selfref.set_xlabel("recall", fontsize=14)
    ax_selfref.set_ylabel("precision", fontsize=14)
    for _side in ("top", "right"):
        ax_selfref.spines[_side].set_visible(False)
    ax_selfref.set_title("re-refinement (stripped → kept)", fontsize=13)
    fig_selfref.tight_layout()
    fig_selfref
    return (fig_selfref,)


@app.cell
def _(mo, pd, selfref):
    # The arrows above as numbers: Δ = kept − stripped in percentage points, with the water counts
    # that produced each pair of scores.
    _rows = []
    for _structure in sorted(selfref["stripped"].index.intersection(selfref["auto"].index)):
        _row = {"structure": _structure,
                "PDB-REDO waters": int(selfref["auto"].loc[_structure, "n_water_ref"])}
        for _what in ("precision", "recall"):
            _stripped = selfref["stripped"].loc[_structure, _what]
            _kept = selfref["auto"].loc[_structure, _what]
            _row[f"{_what}: stripped"] = round(_stripped * 100, 1)
            _row[f"{_what}: kept"] = round(_kept * 100, 1)
            _row[f"{_what}: Δ"] = round((_kept - _stripped) * 100, 1)
        _row["waters: stripped"] = int(selfref["stripped"].loc[_structure, "n_water_pred"])
        _row["waters: kept"] = int(selfref["auto"].loc[_structure, "n_water_pred"])
        _rows.append(_row)
    mo.vstack([mo.md("Δ is kept − stripped, in percentage points."), pd.DataFrame(_rows)])
    return


@app.cell
def _(mo):
    FIGURE_LABELS = ("couple pies", "self-refinement pies", "self-refinement P–R")
    save_which = mo.ui.multiselect(options=list(FIGURE_LABELS), value=list(FIGURE_LABELS),
                                   label="Figures")
    save_ui = mo.ui.checkbox(label="Save to data/plots/")
    mo.vstack([save_which, save_ui])
    return save_ui, save_which


@app.cell
def _(
    Path,
    fig_pies,
    fig_selfref,
    fig_selfref_pies,
    grounding,
    mo,
    pie_variant,
    save_ui,
    save_which,
    selfref_pie_variant,
):
    # Only the selected figures are written, so re-exporting one panel does not also rewrite the
    # others at whatever the toggles happen to be set to.
    _out = Path("data/plots")
    _figures = {
        "couple pies": (fig_pies, f"partition_pies{grounding.value}_{pie_variant.value}"),
        "self-refinement pies": (fig_selfref_pies, f"selfref_pies_{selfref_pie_variant.value}"),
        "self-refinement P–R": (fig_selfref, "selfref_vs_pdbredo"),
    }
    if save_ui.value and save_which.value:
        _out.mkdir(parents=True, exist_ok=True)
        _saved = []
        for _label in save_which.value:
            _f, _name = _figures[_label]
            _p = _out / f"{_name}_1.0.png"
            _f.savefig(_p, dpi=300, bbox_inches="tight")
            _saved.append(_p)
        _msg = mo.md("Saved → " + ", ".join(f"`{_p}`" for _p in _saved))
    elif save_ui.value:
        _msg = mo.md("_Nothing selected — pick at least one figure._")
    else:
        _msg = mo.md(f"_Tick the box to write the selected figures to `{_out}/`._")
    _msg
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
