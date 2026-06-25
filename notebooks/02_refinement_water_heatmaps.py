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
    # 02 — refinement + water-agreement heatmaps (combined)

    Coalesces notebooks **05** (scalar refinement metrics — `r_free`, `n_water`)
    and **06** (water-set agreement — precision / recall / chamfer) into one view,
    and adds a single **baseline selector** shared by both:

    - **original** — subtract each structure's *deposited* (PDB-REDO) value /
      *original-vs-original* agreement, exactly as 05/06 did.
    - **re-refined** — subtract the **self-refinement** instead: for scalar metrics
      the diagonal `<X>_refined_by_<X>`; for agreement the
      `self_refined_pairwise_metrics_<variant>` CSV (`<A>_refined_by_<A>` vs
      `<B>_refined_by_<B>`). The re-refined baseline **matches each panel's variant**,
      so a difference isolates the *cross-model* effect within one refinement protocol.

    In section B the **phenix matrix's ground truth tracks the baseline** so the Δ
    holds it fixed on both sides: original → `original-a`, re-refined → `a_refined_by_a`
    (the `phenix_pairwise_metrics_selfref_*` CSVs). Both Δ modes then vary only the
    predictor.

    Each section reads **baselines first** (the reference values and the per-structure
    self-refinement diagonal), then the full cross-refinement matrix, then the **Δ**
    that subtracts the baseline.

    This notebook reads only pre-computed artifacts (the phenix-cohort CSVs and
    `metadata.csv`) — no CIF parsing, alignment, or clustering here. All plotting
    comes from `cw.plots` (`to_matrix`, `make_panels`, `order_by_count`,
    `diagonal_matrix`); nothing is redefined. Notebooks 05 and 06 are left untouched.

    Generate the re-refined inputs with:
    `... --self-refined --results-dir <cohort>_phenix/refinement_results` (agreement
    reference) and `... --phenix --results-dir <...> --ref-from-self-refined`
    (self-grounded phenix matrix).
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    import config
    from cw.plots import diagonal_matrix, make_panels, order_by_count, to_matrix

    return (
        Path,
        config,
        diagonal_matrix,
        make_panels,
        np,
        order_by_count,
        pd,
        plt,
        sns,
        to_matrix,
    )


@app.cell
def _():
    # Refinement strategies; the re-refined baseline matches the panel's variant.
    VARIANTS = ["auto", "fixed", "stripped"]

    # Agreement-metric registry (shared by section B panels): label / cmap / range.
    METRICS = {
        "precision": {"label": "Precision", "cmap": "Blues", "vmin": 0, "vmax": 1},
        "recall": {"label": "Recall (coverage)", "cmap": "Greens", "vmin": 0, "vmax": 1},
        "f1": {"label": "F1", "cmap": "Purples", "vmin": 0, "vmax": 1},
        "matched_precision": {"label": "Matched precision", "cmap": "Blues", "vmin": 0, "vmax": 1},
        "matched_recall": {"label": "Matched recall", "cmap": "Greens", "vmin": 0, "vmax": 1},
        "chamfer": {"label": "Chamfer dist (Å)", "cmap": "rocket_r", "vmin": 0, "vmax": None},
        "rmsd_after": {"label": "RMSD after (Å)", "cmap": "rocket_r", "vmin": 0, "vmax": None},
        "max_cell_diff": {"label": "max cell diff (%)", "cmap": "rocket_r", "vmin": 0, "vmax": None},
    }

    # Scalar re-refinement metric -> the matching original column in metadata.csv.
    METRIC_TO_ORIGINAL = {"n_water": "num_water", "r_free": "r_free", "r_work": "r_work"}
    return METRICS, METRIC_TO_ORIGINAL, VARIANTS


@app.cell
def _(config, mo):
    COHORT = "hewls_65"
    subset  = f"{COHORT}_subsampled_similar_default"

    meta_stem_ui = mo.ui.text(
        value=subset,
        label="Scalar meta-CSV stem ({stem}_{variant}.csv)",
        full_width=True,
    )
    phenix_dir_ui = mo.ui.text(
        value=f"{config.DATA_DIR}/{subset}_phenix",
        label="Phenix cohort dir (holds meta + pairwise CSVs)",
        full_width=True,
    )
    metadata_ui = mo.ui.text(
        value=f"{config.DATA_DIR}/{COHORT}/metadata.csv",
        label="metadata.csv (original deposited values + water-count ordering)",
        full_width=True,
    )
    cutoff_ui = mo.ui.text(value="1.4", label="cutoff (Å, in pairwise-CSV filenames)")
    mo.vstack([phenix_dir_ui, meta_stem_ui, metadata_ui, cutoff_ui])
    return cutoff_ui, meta_stem_ui, metadata_ui, phenix_dir_ui


@app.cell
def _(Path, VARIANTS, meta_stem_ui, mo, phenix_dir_ui):
    _avail = [
        v for v in VARIANTS
        if (Path(phenix_dir_ui.value) / f"{meta_stem_ui.value}_{v}.csv").exists()
    ] or VARIANTS
    variant_ui = mo.ui.multiselect(options=_avail, value=_avail, label="Variants → one panel each")
    baseline_ui = mo.ui.radio(
        options=["original", "re-refined"],
        value="re-refined",
        label="Δ baseline",
        inline=True,
    )
    mo.vstack([mo.md(f"Discovered scalar variants: `{_avail}`"), variant_ui, baseline_ui])
    return baseline_ui, variant_ui


@app.cell
def _(Path, metadata_ui, pd):
    # pdb_id -> original deposited water count; the single ordering key for every
    # matrix axis so all panels stay subtractable. Empty if metadata is missing.
    if Path(metadata_ui.value).exists():
        _meta = pd.read_csv(metadata_ui.value)
        water_counts = pd.to_numeric(_meta.set_index("pdb_id")["num_water"], errors="coerce")
    else:
        water_counts = pd.Series(dtype=float)
    return (water_counts,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Pairwise alignment of the subsampled originals — RMSD + cell difference

    Independent of the phenix re-refinement: one combined heatmap over the
    subsampled *deposited* structures, read from `reference_pairwise_metrics.csv`.
    Both metrics are symmetric (`metric(a,b) == metric(b,a)`), so a single matrix
    carries both triangles — **lower = pairwise Cα RMSD after alignment (Å)**,
    **upper = max unit-cell edge difference (%)** — each on its own colour scale.
    Axes are sorted by original water count.
    """)
    return


@app.cell
def _(mo):
    pw_mask_diag_ui = mo.ui.switch(value=True, label="mask diagonal (self-comparisons) → NaN")
    pw_mask_diag_ui
    return (pw_mask_diag_ui,)


@app.cell
def _(
    Path,
    mo,
    np,
    order_by_count,
    pd,
    phenix_dir_ui,
    plt,
    pw_mask_diag_ui,
    sns,
    to_matrix,
    water_counts,
):
    _path = Path(phenix_dir_ui.value) / "reference_pairwise_metrics.csv"
    mo.stop(
        not _path.exists(),
        mo.md(f"`reference_pairwise_metrics.csv` not found in `{phenix_dir_ui.value}`."),
    )
    _ref = pd.read_csv(_path)
    _order = order_by_count(
        sorted(set(_ref["structure_ref"]) | set(_ref["structure_mobile"])), water_counts
    )
    _n = len(_order)

    _low, _up = "rmsd_after", "max_cell_diff"
    _rmsd = to_matrix(_ref, _low, _order, index="structure_ref", columns="structure_mobile")
    _cell = to_matrix(_ref, _up, _order, index="structure_ref", columns="structure_mobile")

    _diag = np.eye(_n, dtype=bool)
    _upper = np.triu(np.ones((_n, _n), dtype=bool), k=1)
    _mask = pw_mask_diag_ui.value

    # Two overlaid symmetric layers: RMSD on the lower triangle (+diagonal),
    # max cell diff on the upper triangle, each with its own colour scale.
    _fig, _ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(
        _rmsd, ax=_ax, cmap="mako_r",
        mask=_upper | _diag if _mask else _upper,
        annot=True, fmt=".2f", square=True, linewidths=0.5, linecolor="white", cbar=False,
    )
    sns.heatmap(
        _cell, ax=_ax, cmap="rocket_r", mask=~_upper,
        annot=True, fmt=".2f", square=True, linewidths=0.5, linecolor="white", cbar=False,
    )
    _ax.set_title(f"lower = {_low} (Å) · upper = {_up} (%)")
    _ax.set_xlabel("")
    _ax.set_ylabel("")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section A — scalar refinement metrics (`r_free`, `n_water`, `r_work`)

    Per-structure scalar values across the re-refinement matrix. We establish the
    **baseline** each structure is measured against (deposited value, and the
    self-refinement diagonal), then show the full cross-refinement matrix, then the
    **Δ** that isolates the cross-model effect. Metric / variants / Δ-baseline follow
    the selectors above.
    """)
    return


@app.cell
def _(Path, meta_stem_ui, pd, phenix_dir_ui, variant_ui):
    # Concatenate the per-variant scalar meta CSVs for the selected variants.
    _frames = []
    for _v in variant_ui.value:
        _p = Path(phenix_dir_ui.value) / f"{meta_stem_ui.value}_{_v}.csv"
        if _p.exists():
            _frames.append(pd.read_csv(_p))
    sa_df = pd.concat(_frames, ignore_index=True) if _frames else None
    return (sa_df,)


@app.cell
def _(mo, sa_df):
    mo.stop(sa_df is None, mo.md("No scalar meta CSVs found for the selected variants — check the cohort dir / stem above."))
    _key_cols = {"target_pdb", "ref_pdb", "variant"}
    _metrics = [c for c in sa_df.columns if c not in _key_cols]
    sa_metric_ui = mo.ui.dropdown(
        options=_metrics,
        value="n_water" if "n_water" in _metrics else _metrics[0],
        label="Scalar metric",
    )
    sa_metric_ui
    return (sa_metric_ui,)


@app.cell
def _(order_by_count, sa_df, water_counts):
    _ids = sorted(set(sa_df["target_pdb"]) | set(sa_df["ref_pdb"]))
    sa_order = order_by_count(_ids, water_counts)
    return (sa_order,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Reference scalar values (subtracted baseline)

    Every available baseline per `pdb_id`, side by side: the variant-independent
    deposited **original** value and each variant's **re-refined** self-refinement
    diagonal `<pdb>_refined_by_<pdb>`. The column(s) the Δ panels actually subtract —
    set by the **Δ baseline** selector above — are marked **★**; the rest are shown for
    reference only.
    """)
    return


@app.cell
def _(
    METRIC_TO_ORIGINAL,
    Path,
    baseline_ui,
    metadata_ui,
    mo,
    pd,
    sa_df,
    sa_metric_ui,
    sa_order,
    variant_ui,
):
    # Every available baseline per pdb_id, side by side: the variant-independent
    # deposited (original) value plus each variant's self-refinement diagonal. The Δ
    # panels subtract the ★-marked column(s) — `original`, or the variant-matched
    # `re-refined (<v>)` — set by the Δ-baseline selector.
    _metric = sa_metric_ui.value
    _orig_col = METRIC_TO_ORIGINAL.get(_metric)

    _cols = {}
    if _orig_col is not None and Path(metadata_ui.value).exists():
        _cols["original"] = pd.to_numeric(
            pd.read_csv(metadata_ui.value).set_index("pdb_id")[_orig_col], errors="coerce"
        ).reindex(sa_order)
    for _v in variant_ui.value:
        _d = sa_df[sa_df["variant"] == _v]
        _cols[f"re-refined ({_v})"] = (
            _d[_d["target_pdb"] == _d["ref_pdb"]].set_index("target_pdb")[_metric].reindex(sa_order)
        )

    _active = ["original"] if baseline_ui.value == "original" else [f"re-refined ({_v})" for _v in variant_ui.value]
    _active = [c for c in _active if c in _cols]

    _table = pd.DataFrame(_cols)
    _table.columns = [f"★ {c}" if c in _active else c for c in _table.columns]
    _table.index.name = "pdb_id"

    mo.vstack([
        mo.md(
            f"**Active Δ baseline: `{baseline_ui.value}`** — the Δ panels subtract the ★-marked "
            f"column(s): {', '.join(f'`{c}`' for c in _active) if _active else '_none available for this metric_'}. "
            "Other columns are shown for reference only."
        ),
        _table,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Self-refinement vs original — diagonal-only heatmaps

    Square `pdb × pdb` heatmaps that populate **only the leading diagonal** (each
    structure against itself), independent of the Δ-baseline selector. The first panel
    is the **original** deposited value per structure (`n_water` / `r_free` / `r_work`
    from `metadata.csv`); each following panel is the **Δ self-refined − original** for
    one variant — the self-refinement `<pdb>_refined_by_<pdb>` minus the deposited
    value, isolating the refinement protocol's own effect on each structure. Off-diagonal
    cells are blank by construction. Metric follows the **Scalar metric** dropdown above.
    """)
    return


@app.cell
def _(
    METRIC_TO_ORIGINAL,
    Path,
    diagonal_matrix,
    make_panels,
    metadata_ui,
    mo,
    np,
    pd,
    sa_df,
    sa_metric_ui,
    sa_order,
    variant_ui,
):
    _metric = sa_metric_ui.value
    _fmt = ".0f" if _metric == "n_water" else ".3f"
    _orig_col = METRIC_TO_ORIGINAL.get(_metric)
    mo.stop(_orig_col is None, mo.md(f"No original column mapped for `{_metric}`."))
    mo.stop(not Path(metadata_ui.value).exists(), mo.md(f"metadata.csv not found: `{metadata_ui.value}`"))

    _orig = pd.to_numeric(
        pd.read_csv(metadata_ui.value).set_index("pdb_id")[_orig_col], errors="coerce"
    ).reindex(sa_order)

    _label = f"PDB-REDO {_metric}"
    _panels = {_label: diagonal_matrix(_orig, sa_order)}
    _specs = {
        _label: {
            "label": _label, "cbar_label": _metric, "cmap": "viridis",
            "vmin": float(np.nanmin(_orig.to_numpy())), "vmax": float(np.nanmax(_orig.to_numpy())),
        }
    }
    for _v in variant_ui.value:
        _self = (
            sa_df[(sa_df["variant"] == _v) & (sa_df["target_pdb"] == sa_df["ref_pdb"])]
            .set_index("target_pdb")[_metric]
            .reindex(sa_order)
        )
        _delta = _self - _orig
        _key = f"phenix ({_v}) − PDB-REDO"
        _panels[_key] = diagonal_matrix(_delta, sa_order)
        _lim = float(np.nanmax(np.abs(_delta.to_numpy()))) or 1.0
        _specs[_key] = {"label": _key, "cbar_label": f"Δ{_metric}", "cmap": "RdBu_r", "vmin": -_lim, "vmax": _lim}

    make_panels(_panels, specs=_specs, fmt=_fmt, xlabel="", ylabel="")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Cross-refinement matrix

    Rows = **starting model** (`ref_pdb`), columns = **mtz data used** (`target_pdb`),
    both sorted by original water count. The leading diagonal is each structure's
    self-refinement (shown on its own above).
    """)
    return


@app.cell
def _(make_panels, sa_df, sa_metric_ui, sa_order, to_matrix, variant_ui):
    _metric = sa_metric_ui.value
    _fmt = ".0f" if _metric == "n_water" else ".3f"
    _panels = {
        v: to_matrix(sa_df[sa_df["variant"] == v], _metric, sa_order, index="ref_pdb", columns="target_pdb")
        for v in variant_ui.value
    }
    make_panels(
        _panels, cbar_label=_metric, cmap="viridis", shared_cbar=True, fmt=_fmt,
        xlabel="mtz data used", ylabel="starting model",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Δ vs baseline (cross − baseline)

    The cross-refinement matrix minus the selected baseline, subtracted column-wise
    (per mtz source / `target_pdb`):

    - **original** → deposited `metadata.csv` value of that structure (variant-independent).
    - **re-refined** → the self-refinement `<target>_refined_by_<target>` of the *same*
      variant (that variant's matrix diagonal for the column).

    In re-refined mode the diagonal is 0 by construction and is masked.
    """)
    return


@app.cell
def _(
    METRIC_TO_ORIGINAL,
    Path,
    baseline_ui,
    make_panels,
    metadata_ui,
    mo,
    pd,
    sa_df,
    sa_metric_ui,
    sa_order,
    to_matrix,
    variant_ui,
):
    # Δ vs baseline, subtracted column-wise (per mtz source / target_pdb):
    #   original   → deposited metadata.csv value of that structure (variant-independent).
    #   re-refined → the self-refinement value <target>_refined_by_<target> of the *same*
    #                variant (= that variant's matrix diagonal for the column).
    _metric = sa_metric_ui.value
    _fmt = ".0f" if _metric == "n_water" else ".3f"

    if baseline_ui.value == "original":
        _orig_col = METRIC_TO_ORIGINAL.get(_metric)
        mo.stop(_orig_col is None, mo.md(f"No original column mapped for `{_metric}`."))
        mo.stop(not Path(metadata_ui.value).exists(), mo.md(f"metadata.csv not found: `{metadata_ui.value}`"))
        _orig = pd.to_numeric(
            pd.read_csv(metadata_ui.value).set_index("pdb_id")[_orig_col], errors="coerce"
        ).reindex(sa_order)
        _subtrahend = {v: _orig for v in variant_ui.value}
    else:
        # Self-refinement diagonal of the *same* variant, reindexed to sa_order so
        # the column-wise .sub below keeps the matrix's column order (an unaligned
        # subtrahend would make pandas re-sort the columns, desyncing the axes).
        _subtrahend = {
            v: (
                lambda d: d[d["target_pdb"] == d["ref_pdb"]].set_index("target_pdb")[_metric].reindex(sa_order)
            )(sa_df[sa_df["variant"] == v])
            for v in variant_ui.value
        }

    _panels = {
        v: to_matrix(sa_df[sa_df["variant"] == v], _metric, sa_order, index="ref_pdb", columns="target_pdb").sub(
            _subtrahend[v], axis=1
        )
        for v in variant_ui.value
    }
    make_panels(
        _panels,
        cbar_label=f"Δ{_metric} (cross − {baseline_ui.value})",
        cmap="RdBu_r", center=0, shared_cbar=True, fmt=_fmt,
        mask_diagonal=baseline_ui.value == "re-refined",
        xlabel="mtz data used", ylabel="starting model",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section B — water-set agreement (precision / recall / chamfer)

    Each cell `(row=a, col=b)` is how well predictor `b`'s waters agree with reference
    `a`'s, after Cα alignment. As in Section A we lead with the **reference** agreement
    and the per-structure **self-refinement diagonal**, then the full **phenix**
    cross-refinement matrix (`<b>_refined_by_<a>` vs reference `a`), then the **Δ** that
    subtracts the selected baseline (re-refined matches the panel's variant).
    """)
    return


@app.cell
def _(Path, cutoff_ui, pd, phenix_dir_ui, variant_ui):
    # All section-B CSVs live in the phenix cohort dir. Missing files leave None / a
    # gap in the dict rather than stopping, so present panels still render. There are
    # two phenix matrices per variant — one grounded on original-a waters, one on the
    # self-refinement a_refined_by_a (--ref-from-self-refined) — so each Δ baseline can
    # hold its ground truth fixed.
    _dir = Path(phenix_dir_ui.value)
    _co = cutoff_ui.value

    _ref_path = _dir / "reference_pairwise_metrics.csv"
    sb_ref_orig = pd.read_csv(_ref_path) if _ref_path.exists() else None

    sb_phenix = {}
    sb_phenix_selfref = {}
    sb_selfref = {}
    for _v in variant_ui.value:
        for _name, _store in [
            (f"phenix_pairwise_metrics_{_v}_{_co}.csv", sb_phenix),
            (f"phenix_pairwise_metrics_selfref_{_v}_{_co}.csv", sb_phenix_selfref),
            (f"self_refined_pairwise_metrics_{_v}_{_co}.csv", sb_selfref),
        ]:
            _p = _dir / _name
            if _p.exists():
                _store[_v] = pd.read_csv(_p)
    return sb_phenix, sb_phenix_selfref, sb_ref_orig, sb_selfref


@app.cell
def _(METRICS, mo, sb_phenix, sb_ref_orig):
    _cols = set()
    for _d in [sb_ref_orig, *sb_phenix.values()]:
        if _d is not None:
            _cols |= set(_d.columns)
    _available = [k for k in METRICS if k in _cols] or list(METRICS)
    _default = [k for k in ["precision", "recall", "chamfer"] if k in _available]
    sb_metric_ui = mo.ui.multiselect(options=_available, value=_default, label="Agreement metrics → one panel each")
    sb_metric_ui
    return (sb_metric_ui,)


@app.cell
def _(order_by_count, sb_phenix, sb_ref_orig, water_counts):
    _ref = sb_ref_orig if sb_ref_orig is not None else (next(iter(sb_phenix.values()), None))
    if _ref is None:
        sb_order = []
    elif "structure_ref" in _ref.columns:
        sb_order = order_by_count(sorted(set(_ref["structure_ref"]) | set(_ref["structure_mobile"])), water_counts)
    else:
        sb_order = order_by_count(sorted(set(_ref["reference"]) | set(_ref["predictor"])), water_counts)
    return (sb_order,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Reference agreement (subtracted baseline)

    Both reference baselines are shown regardless of the selector: **original** =
    original-vs-original (`reference_pairwise_metrics.csv`) and **re-refined** =
    self-refinement vs self-refinement, one matrix per variant
    (`self_refined_pairwise_metrics_*`). The one the Δ panels actually subtract — set
    by the **Δ baseline** selector above — is marked **★**.
    """)
    return


@app.cell
def _(
    METRICS,
    baseline_ui,
    make_panels,
    mo,
    sb_metric_ui,
    sb_order,
    sb_ref_orig,
    sb_selfref,
    to_matrix,
):
    mo.stop(not sb_metric_ui.value, mo.md("Select at least one agreement metric above."))

    def _ref_panels(frame, mask_diag):
        return make_panels(
            {k: to_matrix(frame, k, sb_order, index="structure_ref", columns="structure_mobile") for k in sb_metric_ui.value},
            specs=METRICS, mask_diagonal=mask_diag,
            xlabel="predictor (mobile)", ylabel="ground truth (ref)",
        )

    # Show every reference baseline regardless of the selector — original-vs-original
    # and each per-variant self-refinement matrix — and ★-mark the one the Δ panels
    # actually subtract (the active Δ baseline).
    _orig_active = baseline_ui.value == "original"
    _blocks = [mo.md(
        f"**Active Δ baseline: `{baseline_ui.value}`** — the ★-marked reference below is the "
        "one the Δ panels subtract; the other is shown for reference only."
    )]

    _star = "★ " if _orig_active else ""
    if sb_ref_orig is not None:
        _blocks.append(mo.vstack([mo.md(f"**{_star}original reference — original-vs-original**"), _ref_panels(sb_ref_orig, True)]))
    else:
        _blocks.append(mo.md("_`reference_pairwise_metrics.csv` not found — original reference unavailable._"))

    _star = "" if _orig_active else "★ "
    if sb_selfref:
        for _v in sb_selfref:
            _blocks.append(mo.vstack([mo.md(f"**{_star}re-refined reference — `{_v}`**"), _ref_panels(sb_selfref[_v], True)]))
    else:
        _blocks.append(mo.md("_No `self_refined_pairwise_metrics_*` CSVs — re-refined reference unavailable (run `--self-refined`)._"))

    mo.vstack(_blocks)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Self-re-refinement vs PDB-REDO — diagonal-only agreement

    Square `pdb × pdb` heatmaps populated **only on the leading diagonal**: each cell
    `(a, a)` is the agreement metric between structure `a`'s **self-re-refinement**
    (`a_refined_by_a`) and its **deposited (PDB-REDO)** waters. This is exactly the
    leading diagonal of the original-grounded phenix matrix below
    (`phenix_pairwise_metrics_<variant>`), so it's independent of the Δ-baseline selector.
    One row of metric panels per variant; off-diagonal cells are blank. Metrics follow the
    **Agreement metrics** selector above.
    """)
    return


@app.cell
def _(
    METRICS,
    diagonal_matrix,
    make_panels,
    mo,
    sb_metric_ui,
    sb_order,
    sb_phenix,
    variant_ui,
):
    mo.stop(not sb_phenix, mo.md("No original-grounded phenix CSVs (`phenix_pairwise_metrics_<variant>`)."))
    mo.stop(not sb_metric_ui.value, mo.md("Select at least one agreement metric above."))

    _rows = []
    for _v in variant_ui.value:
        if _v not in sb_phenix:
            continue
        _df = sb_phenix[_v]
        _self = _df[_df["reference"] == _df["predictor"]].set_index("reference")
        _panels = {k: diagonal_matrix(_self[k], sb_order) for k in sb_metric_ui.value}
        _rows.append(mo.vstack([
            mo.md(f"**`{_v}` — self-re-refinement vs PDB-REDO (diagonal)**"),
            make_panels(_panels, specs=METRICS, xlabel="", ylabel="self-re-refinement vs PDB-REDO"),
        ]))
    mo.vstack(_rows)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### phenix cross-refinement matrix

    The grounding that matches the Δ baseline: original-`a` waters for **original**,
    `a_refined_by_a` for **re-refined**. Predictors are identical across groundings;
    only the ground-truth (reference-`a`) waters differ. The leading diagonal is the
    self-comparison shown on its own above.
    """)
    return


@app.cell
def _(
    METRICS,
    baseline_ui,
    make_panels,
    mo,
    sb_metric_ui,
    sb_order,
    sb_phenix,
    sb_phenix_selfref,
    to_matrix,
    variant_ui,
):
    # Show the phenix grounding that matches the Δ baseline: original-a waters for
    # "original", a_refined_by_a for "re-refined". Predictors are identical across
    # groundings; only the ground-truth (reference-a) waters differ.
    _src = sb_phenix if baseline_ui.value == "original" else sb_phenix_selfref
    _ground = "original-a" if baseline_ui.value == "original" else "a_refined_by_a"
    mo.stop(not _src, mo.md(f"⏳ No phenix CSVs with **{_ground}** ground truth — for re-refined run the script with `--ref-from-self-refined`."))
    mo.stop(not sb_metric_ui.value, mo.md("Select at least one agreement metric above."))
    mo.vstack([
        mo.vstack([
            mo.md(f"**phenix cross-refinement — `{v}`  (ground truth = `{_ground}`)**"),
            make_panels(
                {k: to_matrix(_src[v], k, sb_order, index="reference", columns="predictor") for k in sb_metric_ui.value},
                specs=METRICS, mask_diagonal=baseline_ui.value == "re-refined",
                xlabel="mtz used to re-refine with starting model", ylabel="starting model",
            ),
        ])
        for v in variant_ui.value if v in _src
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Δ agreement — phenix − selected reference

    `metric(phenix) − metric(reference)`, one row of metric panels per variant,
    diverging scale per panel (symmetric about 0). **Ground truth is held fixed**
    (original-a, or a_refined_by_a for re-refined), so the Δ isolates the predictor:
    `b_refined_by_a` (cross-model) vs the baseline predictor. For precision/recall/F1 a
    **positive** Δ (red) means the cross-refined predictor agrees with the reference
    waters *better* than the baseline does; for chamfer/RMSD (lower is better) positive
    = worse. For **re-refined** the diagonal is 0 by construction (self vs self on both
    sides); for **original** the diagonal is `agreement(original-a, a_refined_by_a) − 1`
    — the protocol's own effect on the self comparison.
    """)
    return


@app.cell
def _(
    baseline_ui,
    make_panels,
    mo,
    np,
    sb_metric_ui,
    sb_order,
    sb_phenix,
    sb_phenix_selfref,
    sb_ref_orig,
    sb_selfref,
    to_matrix,
    variant_ui,
):
    # Hold ground truth fixed on both sides of the subtraction:
    #   original   → phenix(orig-a, b_refined_by_a) − ref(orig-a, orig-b)
    #   re-refined → phenix(a_ref_by_a, b_refined_by_a) − ref(a_ref_by_a, b_ref_by_b)
    # so Δ isolates the predictor (cross-model vs baseline) alone.
    _orig = baseline_ui.value == "original"
    _phx_src = sb_phenix if _orig else sb_phenix_selfref
    mo.stop(
        not _phx_src,
        mo.md(f"⏳ No phenix CSVs for the **{baseline_ui.value}** grounding"
              + ("" if _orig else " — run the script with `--ref-from-self-refined`.")),
    )
    mo.stop(not sb_metric_ui.value, mo.md("Select at least one agreement metric above."))

    _rows = []
    for _v in variant_ui.value:
        if _v not in _phx_src:
            continue
        _ref = sb_ref_orig if _orig else sb_selfref.get(_v)
        if _ref is None:
            _rows.append(mo.md(f"_`{_v}`: baseline ({baseline_ui.value}) reference missing._"))
            continue
        _diffs, _specs = {}, {}
        for _k in sb_metric_ui.value:
            _phx = to_matrix(_phx_src[_v], _k, sb_order, index="reference", columns="predictor")
            _rm = to_matrix(_ref, _k, sb_order, index="structure_ref", columns="structure_mobile")
            _d = _phx - _rm
            _diffs[_k] = _d
            _lim = float(np.nanmax(np.abs(_d.to_numpy()))) or 1.0
            _specs[_k] = {"label": f"Δ {_k}", "cmap": "coolwarm", "vmin": -_lim, "vmax": _lim}
        _rows.append(mo.vstack([
            mo.md(f"**Δ `{_v}`  (phenix − {baseline_ui.value}, ground truth fixed)**"),
            make_panels(_diffs, specs=_specs, fmt="+.2f", mask_diagonal=not _orig, xlabel="mtz used to re-refine", ylabel="starting model"),
        ]))
    mo.vstack(_rows)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
