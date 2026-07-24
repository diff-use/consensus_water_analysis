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
    `B_refined_by_A` uses B's diffraction data but A's model. We ask how many waters of the two
    independent self-refinements it recovers (recall, after Cα alignment into B's frame),
    split into three groups:

    - **shared** — in both `B_refined_by_B` and `A_refined_by_A` (conserved sites).
    - **B-only** — in `B_refined_by_B`, not `A_refined_by_A` → the **data donor's** own model-specific waters.
    - **A-only** — in `A_refined_by_A`, not `B_refined_by_B` → the **template's** model-specific waters.

    B-only and A-only are symmetric ("one structure's private waters relative to the other"),
    so comparing them for the same predictor is self-contained — no baseline needed. If the data
    drives the result, B-only ≫ A-only.

    The recall groups only score the cross-refinement against *real* self-refinement waters. To
    catch waters it *invents*, we also classify the cross-refinement's **own** waters
    (predictor-side): supported by B, template-only (A not B), or **orphan** (near neither —
    candidate bad waters, e.g. from a retained starting water). `similar_default` cohort, 1.0 Å,
    `auto` (starting waters kept) and `stripped` (re-placed de novo).

    Reads `starting_model_partition_<variant>_1.0.csv` (from
    `scripts/pairwise_water_metrics.py --partition`).
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
def _(Path, config, mo, pd):
    _dir = Path(f"{config.DATA_DIR}/hewls_65_subsampled_similar_default_phenix")
    GROUPS = {"recall_shared": "shared\n(conserved)", "recall_b_only": "B-only\n(data donor)", "recall_a_only": "A-only\n(template)"}
    frames, _missing = {}, []
    for _v in ("auto", "stripped"):
        _p = _dir / f"starting_model_partition_{_v}_1.0.csv"
        if _p.exists():
            frames[_v] = pd.read_csv(_p)
        else:
            _missing.append(_p.name)
    mo.stop(
        bool(_missing),
        mo.md("Missing: " + ", ".join(f"`{_m}`" for _m in _missing) + " — run `scripts/pairwise_water_metrics.py --partition`."),
    )
    return GROUPS, frames


@app.cell
def _(GROUPS, frames, np, plt):
    # Two panels, both fractions in [0, 1], both protocols. The 20 pairs aren't independent
    # (5 structures), so points are shown as a cloud (not a box); bold bar = mean.
    #   Left  (reference-side, recall): of each self-refinement group, what fraction the
    #         cross-refinement recovers → the data's own waters (B-only) beat the template's (A-only).
    #   Right (predictor-side, composition): of the cross-refinement's OWN waters, what fraction
    #         each self-refinement supports → "orphan" = neither B nor A (candidate bad waters).
    _shade = {"auto": "0.25", "stripped": "0.6"}
    _offset = {"auto": -0.16, "stripped": 0.16}
    _rng = np.random.default_rng(0)
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    _metrics = list(GROUPS)
    for _v, _df in frames.items():
        for _i, _m in enumerate(_metrics):
            _y = _df[_m].dropna().to_numpy()
            _x = _i + _offset[_v] + _rng.uniform(-0.05, 0.05, len(_y))
            ax_r.scatter(_x, _y, color=_shade[_v], s=20, alpha=0.5, zorder=2)
            ax_r.plot([_i + _offset[_v] - 0.11, _i + _offset[_v] + 0.11], [_y.mean(), _y.mean()],
                      color=_shade[_v], lw=3, zorder=3, label=_v if _i == 0 else None)
    ax_r.set_xticks(range(len(_metrics)))
    ax_r.set_xticklabels([GROUPS[_m] for _m in _metrics])
    ax_r.set_ylabel("recall of self-refinement waters")
    ax_r.set_ylim(-0.02, 1.02)
    ax_r.legend(title="protocol", loc="lower left")
    ax_r.set_title("Reference-side: data's waters recovered, not template's")

    # Composition: stacked mean fractions of the cross-refinement's own waters.
    _cats = [("n_pred_b", "supported by B (data)", "#4c72b0"),
             ("n_pred_a_only", "template-only (A not B)", "#dd8452"),
             ("n_pred_orphan", "orphan (neither)", "#c44e52")]
    _xpos = {"auto": 0, "stripped": 1}
    for _v, _df in frames.items():
        _bottom = 0.0
        for _col, _lab, _c in _cats:
            _frac = (_df[_col] / _df["n_pred"]).mean()
            ax_c.bar(_xpos[_v], _frac, bottom=_bottom, width=0.55, color=_c,
                     label=_lab if _v == "auto" else None)
            if _frac > 0.04:
                ax_c.text(_xpos[_v], _bottom + _frac / 2, f"{_frac:.0%}", ha="center", va="center",
                          fontsize=8, color="white")
            _bottom += _frac
    ax_c.set_xticks(list(_xpos.values()))
    ax_c.set_xticklabels(list(_xpos))
    ax_c.set_ylim(0, 1.0)
    ax_c.set_ylabel("fraction of cross-refined waters")
    ax_c.set_xlabel("protocol")
    ax_c.legend(loc="lower right", fontsize=8)
    ax_c.set_title("Predictor-side: how the cross-refined waters are supported")

    fig.suptitle("Cross-refinement follows the data, not the starting model (1.0 Å)", y=1.02)
    fig.tight_layout()
    fig
    return (fig,)


@app.cell
def _(GROUPS, frames, mo):
    # Recall (reference-side) means + consistency, and composition (predictor-side) fractions.
    _lines = []
    for _v, _df in frames.items():
        _m = {g: _df[g].mean() for g in GROUPS}
        _n_bgta = int((_df["recall_b_only"] > _df["recall_a_only"]).sum())
        _fa = (_df["n_pred_a_only"] / _df["n_pred"]).mean()
        _fo = (_df["n_pred_orphan"] / _df["n_pred"]).mean()
        _lines.append(
            f"- **{_v}**: recall — shared {_m['recall_shared']:.0%}, B-only {_m['recall_b_only']:.0%}, "
            f"A-only {_m['recall_a_only']:.0%} (B-only > A-only in {_n_bgta}/{len(_df)} pairs). "
            f"Of its own waters — template-only {_fa:.0%}, orphan {_fo:.0%}."
        )
    mo.md(
        "### Result (n = 5 structures, 20 non-independent pairs; descriptive)\n"
        + "\n".join(_lines)
        + "\n\n### Suggested manuscript text\n"
        + "> Refining each structure's diffraction data from a different structure's starting model "
        "reproduced the conserved water network almost completely (≈95% of waters shared between "
        "the two independent self-refinements were recovered) and recovered the data donor's own "
        "model-specific waters roughly twice as often as the starting model's (≈57% vs ≈30%), so "
        "the refined solvent structure is governed primarily by the diffraction data rather than "
        "the starting model. Conversely, of the waters the cross-refinement placed, ≈84–87% "
        "coincided with the data donor's own refinement, while only ≈6–8% were template-specific "
        "and ≈6–8% were supported by neither self-refinement; both of these residual fractions "
        "were slightly larger when the starting model's waters were retained (≈8% each) than when "
        "they were stripped and re-placed de novo (≈6%), indicating a small starting-model "
        "contribution of both carried-over and spurious waters."
    )
    return


@app.cell
def _(mo):
    save_ui = mo.ui.checkbox(label="Save figure to data/plots/")
    save_ui
    return (save_ui,)


@app.cell
def _(Path, fig, mo, save_ui):
    _out = Path("data/plots")
    if save_ui.value:
        _out.mkdir(parents=True, exist_ok=True)
        _p = _out / "partition_recall_1.0.png"
        fig.savefig(_p, dpi=200, bbox_inches="tight")
        _msg = mo.md(f"Saved → `{_p}`")
    else:
        _msg = mo.md(f"_Tick the box to write the figure to `{_out}/`._")
    _msg
    return


if __name__ == "__main__":
    app.run()
