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
    # Starting-model bias — one bar per cross-refinement

    Same cross-refinements as `starting_model_partition_recall.py`, laid out as one diverging
    stacked bar per ordered pair (**A** = starting model, **B** = mtz), with the two protocols side
    by side on shared axes. Five blocks, left to right:

    | block | waters |
    | --- | --- |
    | **A-only missed** | A's private waters the cross-refinement did not place |
    | **A-only recovered** | A's private waters it did place — the bias signal |
    | **orphan** | its own waters near neither A nor B, centred on zero |
    | **B recovered** | B's waters it placed, shared and B-only together |
    | **B missed** | B's waters it did not place |

    Two readings of the same bar:

    - the **middle three** blocks are what the cross-refinement produced — everything it placed is
      either on A's private sites, on B, or on neither;
    - the **outer two on each side** are the ground truth — the left pair is exactly A-only, the
      right pair is exactly B, and the two together are exactly A ∪ B.

    So the bias question is the balance across the centre: a pair whose left saturated block is long
    relative to A-only is recovering the template's private sites, and the orphan block is what it
    invented on top. Reading a row across the two panels answers what stripping buys.

    *Caveat on the comparison.* Under the **self-refined** grounding the ground truth is itself
    protocol-specific — the A and B sets in the left panel come from the stripped self-refinements,
    those in the right panel from the auto ones — so a left-right difference is a whole-protocol
    comparison, and the outer blocks change length for that reason alone. The **deposited** grounding
    holds A and B fixed across both panels, which isolates the predictor effect.

    *Units.* Every block length is waters on one axis shared by both panels — nothing is normalised,
    so any two blocks anywhere in the figure are comparable by length. The percentages each block
    implies have three different denominators (A-only, B, and the cross-refinement's own count) and
    cannot share an axis; they are in the table instead. One consequence: the recovered blocks
    are counted on the **ground-truth** side (A-only and B waters that were matched), while the
    orphan block is counted on the **cross-refinement** side, since an orphan has no ground-truth
    counterpart. The middle three therefore sum to the cross-refinement's water count only where the
    matching is one-to-one; two of its waters landing on one ground-truth site would break the
    identity. In practice it is exact for all 20 pairs in the stripped / self-refined view and holds
    to within a few waters of ~77 elsewhere — the table reports the residual per row.
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
def _(mo):
    grouping = mo.ui.radio(
        options={"starting model (A)": "reference", "mtz — data donor (B)": "donor"},
        value="starting model (A)",
        label="Group the rows by",
        inline=True,
    )
    grounding = mo.ui.radio(
        options={"self-refined — <X>_refined_by_<X>": "", "deposited — PDB-REDO": "_deposited"},
        value="self-refined — <X>_refined_by_<X>",
        label="Ground truth for the A and B water sets",
        inline=True,
    )
    mo.vstack([grouping, grounding])
    return grounding, grouping


@app.cell
def _(Path, config, grounding, mo, pd, plt):
    _dir = Path(f"{config.DATA_DIR}/hewls_65_subsampled_similar_default_phenix")
    # Panel order, left to right, and the title each one carries.
    VARIANTS = {"stripped": "stripped — solvent re-placed de novo",
                "auto": "kept — starting waters retained"}
    # Block order left → right within a bar.
    BLOCKS = ["a_missed", "a_recovered", "orphan", "b_recovered", "b_missed"]

    def _derive(_df):
        """The five block widths per pair, in waters.

        The two recovered blocks are counted on the ground-truth side — A-only and B waters that
        the cross-refinement matched — so that each pairs exactly with its missed block and the
        four outer blocks partition A ∪ B. The predictor-side counts (n_pred_a_only, n_pred_b)
        would make the middle three sum to n_pred exactly instead, but then "missed" would no
        longer be the remainder of the block next to it, which is the comparison the row is for.
        """
        _out = _df[["reference", "donor"]].copy()
        _out["n_a_only"] = _df["n_a_only"]
        _out["n_b"] = _df["n_shared"] + _df["n_b_only"]
        _out["n_pred"] = _df["n_pred"]
        _out["a_recovered"] = _df["recall_a_only"] * _df["n_a_only"]
        _out["a_missed"] = _out["n_a_only"] - _out["a_recovered"]
        _out["b_recovered"] = _df["recall_shared"] * _df["n_shared"] + _df["recall_b_only"] * _df["n_b_only"]
        _out["b_missed"] = _out["n_b"] - _out["b_recovered"]
        _out["orphan"] = _df["n_pred_orphan"]
        return _out.set_index(["reference", "donor"])

    GROUNDING_LABEL = "PDB-REDO" if grounding.value else "self-refined"
    counts, _missing = {}, []
    for _v in VARIANTS:
        _path = _dir / f"starting_model_partition{grounding.value}_{_v}_1.0.csv"
        if _path.exists():
            counts[_v] = _derive(pd.read_csv(_path))
        else:
            _missing.append(_path.name)
    mo.stop(
        bool(_missing),
        mo.md("Missing: " + ", ".join(f"`{_m}`" for _m in _missing)
              + " — run `scripts/pairwise_water_metrics.py --partition --cutoff 1.0"
              + (" --ref-dir <cohort>/filtered_pdbs`." if grounding.value else "`.")),
    )

    pairs = counts["stripped"].index.intersection(counts["auto"].index)
    # One colour per structure, shared with the partition notebook so the two can be read together.
    _ids = sorted({_i for _k in pairs for _i in _k})
    STRUCT_COLORS = dict(zip(_ids, plt.get_cmap("Set1").colors))
    return BLOCKS, GROUNDING_LABEL, STRUCT_COLORS, VARIANTS, counts, pairs


@app.cell
def _(BLOCKS, GROUNDING_LABEL, STRUCT_COLORS, VARIANTS, counts, grouping, np, pairs, plt):
    # One row per ordered pair, blocked by the structure holding the selected role with a blank slot
    # between blocks. The orphan block straddles x = 0, so the two ground-truth sides grow outward
    # from the cross-refinement's own centre and a long saturated block on the left — template
    # sites recovered — is visible as an imbalance rather than as a number to compare.
    # Colour is the structure whose waters the block holds: A blocks take the starting model's
    # colour, B blocks the mtz's, so whichever role the grouping fixes, one side of the centre is
    # constant down a block and the other side is the partner.
    # The two panels share both axes, so a row is one horizontal line across the figure and the
    # protocols are compared by block length directly rather than through the tick labels.
    _role = grouping.value
    _slot = pairs.names.index(_role)
    _partner_slot = 1 - _slot
    _ids = sorted({_k[_slot] for _k in pairs})
    _tint = lambda _c, _towards_white=0.8: tuple(np.asarray(_c) * (1 - _towards_white) + _towards_white)

    _rows, _blocks = [], []
    _y = 0.0
    for _structure in _ids:
        _members = sorted((_k for _k in pairs if _k[_slot] == _structure), key=lambda _k: _k[_partner_slot])
        _blocks.append((_structure, _y, _y + len(_members) - 1))
        for _k in _members:
            _rows.append((_y, _k))
            _y += 1
        _y += 1

    def _edges(_row):
        """Left edge of each block, stacking outward from the orphan block's own half-width."""
        _widths = {_b: _row[_b] for _b in BLOCKS}
        _left = {"orphan": -_widths["orphan"] / 2}
        _left["a_recovered"] = _left["orphan"] - _widths["a_recovered"]
        _left["a_missed"] = _left["a_recovered"] - _widths["a_missed"]
        _left["b_recovered"] = _left["orphan"] + _widths["orphan"]
        _left["b_missed"] = _left["b_recovered"] + _widths["b_recovered"]
        return _widths, _left

    fig_bars, _axes = plt.subplots(1, len(VARIANTS), figsize=(19.0, 8.4), sharex=True, sharey=True)
    for _ax, (_variant, _title) in zip(_axes, VARIANTS.items()):
        for _y_pos, _k in _rows:
            _row = counts[_variant].loc[_k]
            _a_color, _b_color = STRUCT_COLORS[_k[0]], STRUCT_COLORS[_k[1]]
            _widths, _left = _edges(_row)
            _faces = {"a_missed": _tint(_a_color), "a_recovered": _a_color, "orphan": (1.0, 1.0, 1.0),
                      "b_recovered": _b_color, "b_missed": _tint(_b_color)}
            # Every block is drawn in waters on one shared axis, so length is the count and nothing
            # else — no percentages on the figure, they are in the table.
            for _block in BLOCKS:
                _ax.barh(_y_pos, _widths[_block], left=_left[_block], height=0.72,
                         color=_faces[_block], edgecolor="0.7" if _block == "orphan" else "white",
                         lw=0.9, zorder=3)
        _ax.axvline(0, color="0.45", lw=1, ls=(0, (3, 2)), zorder=1)
        _ax.set_title(_title, fontsize=12)
        _ax.xaxis.set_major_locator(plt.MultipleLocator(20))
        _ax.xaxis.set_major_formatter(lambda _t, _p: f"{abs(_t):.0f}")
        _ax.set_xlabel("waters   ·   ← A-only (starting model's private sites)      "
                       "B (mtz's own waters) →", fontsize=10.5)
        _ax.grid(axis="x", color="0.93", lw=0.8, zorder=0)
        _ax.set_axisbelow(True)
        _ax.tick_params(axis="y", length=0)
        for _side in ("top", "right", "left"):
            _ax.spines[_side].set_visible(False)

    # One window from the widest bar in either panel, so a block's length is comparable anywhere in
    # the figure — across rows and across the two protocols alike.
    _extents = [(_left["a_missed"], _left["b_missed"] + _widths["b_missed"])
                for _variant in VARIANTS for _, _k in _rows
                for _widths, _left in [_edges(counts[_variant].loc[_k])]]
    _axes[0].set_xlim(min(_e[0] for _e in _extents) - 2, max(_e[1] for _e in _extents) + 2)
    _axes[0].set_yticks([_y_pos for _y_pos, _ in _rows])
    _axes[0].set_yticklabels([_k[_partner_slot] for _, _k in _rows], fontsize=9.5)
    for _tick, (_y_pos, _k) in zip(_axes[0].get_yticklabels(), _rows):
        _tick.set_color(STRUCT_COLORS[_k[_partner_slot]])
    _axes[0].set_ylim(_rows[-1][0] + 0.9, -0.9)
    for _structure, _first, _last in _blocks:
        _axes[0].annotate(_structure, xy=(-0.062, (_first + _last) / 2), xycoords=("axes fraction", "data"),
                          ha="center", va="center", rotation=90, fontsize=11.5,
                          color=STRUCT_COLORS[_structure])

    _held, _partner = (("starting model (A)", "mtz (B)") if _role == "reference"
                       else ("mtz (B)", "starting model (A)"))
    _fill_legend = fig_bars.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor="0.45", edgecolor="white", label="recovered"),
                 plt.Rectangle((0, 0), 1, 1, facecolor=_tint((0.45, 0.45, 0.45)), edgecolor="white", label="missed"),
                 plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.7", label="orphan — near neither")],
        loc="lower left", bbox_to_anchor=(0.08, 0.0), fontsize=10, frameon=False, ncol=3)
    fig_bars.add_artist(_fill_legend)
    fig_bars.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=_c, edgecolor="white", label=_i)
                             for _i, _c in STRUCT_COLORS.items()],
                    title="waters of", loc="lower right", bbox_to_anchor=(0.92, 0.0), fontsize=10,
                    title_fontsize=10, frameon=False, ncol=len(STRUCT_COLORS))
    fig_bars.text(0.5, 0.105, f"every block length is waters on one shared axis   ·   block = the "
                  f"{_held} held fixed, row label = the {_partner}   ·   middle three blocks = "
                  "every water the cross-refinement placed   ·   outer two on each side = the "
                  "ground truth, exactly A-only (left) and B (right)", ha="center", fontsize=10,
                  color="0.3")
    # Why a row's outer blocks may not match across the two panels — worth saying on the figure,
    # since equal ground-truth lengths are the natural expectation.
    fig_bars.text(0.5, 0.082, "A-only and B are the same waters in both panels: the deposited "
                  "ground truth does not move with the protocol"
                  if GROUNDING_LABEL == "PDB-REDO" else
                  "A-only and B differ in length between the panels: the self-refined ground truth "
                  "is protocol-specific, so A's and B's own refinements place different waters — "
                  "switch to PDB-REDO to hold them fixed",
                  ha="center", fontsize=10, color="0.3")
    fig_bars.suptitle(f"Where each cross-refinement's waters land, grouped by {_held}   —   ground "
                      f"truth: {GROUNDING_LABEL} (1.0 Å, n = {len(pairs)} pairs)", fontsize=13)
    fig_bars.tight_layout(rect=(0.05, 0.145, 1, 0.96), w_pad=2.5)
    fig_bars
    return (fig_bars,)


@app.cell
def _(BLOCKS, VARIANTS, counts, grouping, mo, pairs, pd):
    # Every block in waters, both protocols, plus the three percentages the figure labels and the
    # residual between the middle three blocks and the cross-refinement's own water count. Protocol
    # is an index level rather than a column suffix — 5 blocks × 2 protocols would not fit across.
    _slot = pairs.names.index(grouping.value)
    _order = sorted(pairs, key=lambda _k: (_k[_slot], _k[1 - _slot]))
    _frames = []
    for _variant in VARIANTS:
        _rows = counts[_variant].loc[_order]
        _table = pd.DataFrame(index=_rows.index)
        _table["protocol"] = _variant
        for _block in BLOCKS:
            _table[_block.replace("_", " ")] = _rows[_block].round(1)
        _table["A-only recovered %"] = (_rows["a_recovered"] / _rows["n_a_only"] * 100).round(1)
        _table["B recovered %"] = (_rows["b_recovered"] / _rows["n_b"] * 100).round(1)
        _table["orphan %"] = (_rows["orphan"] / _rows["n_pred"] * 100).round(1)
        _table["middle three − n_pred"] = (
            _rows["a_recovered"] + _rows["orphan"] + _rows["b_recovered"] - _rows["n_pred"]
        ).round(1)
        _frames.append(_table.set_index("protocol", append=True))
    _both = pd.concat(_frames).sort_index(level=[0, 1], sort_remaining=False)
    _both = _both.rename_axis(index={"reference": "starting model (A)", "donor": "mtz (B)"})
    _residual = _both["middle three − n_pred"]
    mo.vstack([
        mo.md(f"Waters, both protocols. The last column is the one-to-one slack described above: "
              f"mean {_residual.mean():+.1f} waters, worst {_residual.abs().max():.0f}."),
        _both.reset_index(),
    ])
    return


@app.cell
def _(mo):
    save_ui = mo.ui.checkbox(label="Save figure to data/plots/")
    save_ui
    return (save_ui,)


@app.cell
def _(Path, fig_bars, grounding, grouping, mo, save_ui):
    _out = Path("data/plots")
    if save_ui.value:
        _out.mkdir(parents=True, exist_ok=True)
        _p = _out / f"partition_bars{grounding.value}_by_{grouping.value}_1.0.png"
        fig_bars.savefig(_p, dpi=200, bbox_inches="tight")
        _msg = mo.md(f"Saved → `{_p}`")
    else:
        _msg = mo.md(f"_Tick the box to write the figure to `{_out}/`._")
    _msg
    return


if __name__ == "__main__":
    app.run()
