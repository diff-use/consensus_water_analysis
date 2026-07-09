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

    Coalesces deprecated notebooks **05** (scalar refinement metrics — `r_free`, `n_water`)
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

    **Inputs — order of operations.** One input file drives everything: a subset
    PDB-id list `data/<cohort_subset>.txt` (`LIST` below). Set `CUTOFF` (e.g. = 1.4)
    and `PHENIX_DIR` = `{DATA_DIR}/<cohort>_<strategy>_phenix` — the `STRATEGY`
    (default `default`) is appended to the cohort name, and this dir must match the
    **Phenix cohort dir** field. `<stem>` = the `PHENIX_DIR` basename without `_phenix`
    (= the **Scalar meta-CSV stem** field). `REF_DIR` = the parent cohort's deposited
    filtered CIFs, e.g. `{DATA_DIR}/hewls_65/filtered_pdbs`.

    0. **Vet the subset — no refinement needed.** Pairwise Cα-RMSD + max cell-diff over
       the *deposited (PDB-REDO)* structures, feeding the *Pairwise alignment* cell below:

       `uv run scripts/pairwise_water_metrics.py <LIST> -o <PHENIX_DIR>/reference_pairwise_metrics.csv --cutoff <CUTOFF>`
    1. **Refinement matrix** (heavy; Phenix env + deposited `.mtz` on disk):

       `PDBID_LIST=<LIST> STRATEGY=default JOBS=4 bash scripts/phenix/re-refine_all.sh`
       → writes `<PHENIX_DIR>/refinement_results/…`
    2. **Section A scalars** (one CSV per variant):

       `for v in auto stripped; do STRATEGY=default bash scripts/phenix/build_refinement_meta.sh <LIST> $v <PHENIX_DIR>/<stem>_$v.csv; done`

       <br>_`STRATEGY` must match step 1 — `build_refinement_meta.sh` now derives `<cohort>_<strategy>_phenix` exactly like `re-refine_all.sh` (no manual `COHORT_ID`)._
    3. **Section B agreement** (three families):

       `uv run scripts/pairwise_water_metrics.py --phenix       --results-dir <PHENIX_DIR>/refinement_results --ref-dir <REF_DIR> --cutoff <CUTOFF>` ·

       `uv run scripts/pairwise_water_metrics.py --self-refined --results-dir <PHENIX_DIR>/refinement_results --cutoff <CUTOFF>` ·

       `uv run scripts/pairwise_water_metrics.py --phenix       --results-dir <PHENIX_DIR>/refinement_results --cutoff <CUTOFF> --ref-from-self-refined`
    4. **`metadata.csv`** (water-count axis ordering + the *original* scalar baseline)
       from Stage-1 `scripts/build_metadata.py`. Optional — axes fall back to sorted-id
       order if it's missing.
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
        "precision": {"label": "Precision", "cmap": "viridis", "vmin": 0, "vmax": 1},
        "recall": {"label": "Recall (coverage)", "cmap": "viridis", "vmin": 0, "vmax": 1},
        "f1": {"label": "F1", "cmap": "viridis", "vmin": 0, "vmax": 1},
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
    # Axis-ordering key: None → deposited PDB-REDO num_water (metadata.csv); else the
    # phenix self-refinement n_water diagonal of the named variant (stripped or not).
    _sort_opts = {"PDB-REDO (num_water)": None}
    for _v in _avail:
        _sort_opts[f"phenix self-ref n_water ({_v})"] = _v
    sort_by_ui = mo.ui.dropdown(
        options=_sort_opts, value="PDB-REDO (num_water)", label="Sort axes by (ascending water count)"
    )
    mo.vstack([mo.md(f"Discovered scalar variants: `{_avail}`"), variant_ui, baseline_ui, sort_by_ui])
    return baseline_ui, sort_by_ui, variant_ui


@app.cell
def _(Path, meta_stem_ui, metadata_ui, pd, phenix_dir_ui, sort_by_ui):
    # pdb_id -> water count; the single ordering key for every matrix axis so all
    # panels stay subtractable. The "Sort axes by" selector picks the source:
    #   None    -> deposited PDB-REDO num_water (metadata.csv), the original behaviour.
    #   variant -> phenix self-refinement n_water diagonal (<pdb>_refined_by_<pdb>) from
    #              that variant's scalar meta CSV (stripped or not), independent of which
    #              variants are toggled for display.
    # Empty Series (→ sorted-id fallback in order_by_count) if the source is missing.
    _variant = sort_by_ui.value
    if _variant is not None:
        _p = Path(phenix_dir_ui.value) / f"{meta_stem_ui.value}_{_variant}.csv"
        if _p.exists():
            _d = pd.read_csv(_p)
            _self = _d[_d["mtz_source"] == _d["starting_model"]]
            water_counts = pd.to_numeric(_self.set_index("mtz_source")["n_water"], errors="coerce")
        else:
            water_counts = pd.Series(dtype=float)
    elif Path(metadata_ui.value).exists():
        _meta = pd.read_csv(metadata_ui.value)
        water_counts = pd.to_numeric(_meta.set_index("pdb_id")["num_water"], errors="coerce")
    else:
        water_counts = pd.Series(dtype=float)
    return (water_counts,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Pairwise alignment of the subsampled originals — RMSD + cell difference

    _Needs: `reference_pairwise_metrics.csv` (step 0 — no refinement)._

    Independent of the phenix re-refinement: one combined heatmap over the
    subsampled *deposited* structures, read from `reference_pairwise_metrics.csv`.
    Both metrics are symmetric (`metric(a,b) == metric(b,a)`), so a single matrix
    carries both triangles — **lower = pairwise Cα RMSD after alignment (Å)**,
    **upper = max unit-cell edge difference (%)** — each on its own colour scale.
    Axes follow the **Sort axes by** selector (default: original PDB-REDO water count;
    optionally a phenix self-refinement `n_water`, stripped or not).
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

    _Needs: `<stem>_<variant>.csv` (step 2) + `metadata.csv` (step 4, for the original baseline)._

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
    _key_cols = {"mtz_source", "starting_model", "variant"}
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
    _ids = sorted(set(sa_df["mtz_source"]) | set(sa_df["starting_model"]))
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
            _d[_d["mtz_source"] == _d["starting_model"]].set_index("mtz_source")[_metric].reindex(sa_order)
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

    Per-structure self-comparison (each structure against itself), independent of the
    Δ-baseline selector. The first panel is the **original** deposited value per
    structure (`n_water` / `r_free` / `r_work` from `metadata.csv`); each following panel
    is one variant's self-refinement `<pdb>_refined_by_<pdb>`. Metric follows the
    **Scalar metric** dropdown above.

    Two controls below:

    - **Diagonal layout** — `square` renders the full `pdb × pdb` matrix with only the
      leading diagonal filled (off-diagonal blank); `row` / `column` collapse it to a
      compact strip of just the diagonal values.
    - **phenix cells** — `delta` shows each variant as **Δ self-refined − original** (its
      own diverging scale per panel, deposited panel on its own viridis scale); `original`
      shows the **raw self-refined value** instead — same units as the deposited panel, so
      **all panels share one colorbar**.
    """)
    return


@app.cell
def _(mo):
    sa_diag_orient_ui = mo.ui.dropdown(
        options=["square", "row", "column"], value="square", label="Diagonal layout"
    )
    sa_diag_mode_ui = mo.ui.radio(
        options=["delta", "original"], value="delta", label="phenix cells", inline=True
    )
    mo.hstack([sa_diag_orient_ui, sa_diag_mode_ui], justify="start")
    return sa_diag_mode_ui, sa_diag_orient_ui


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
    sa_diag_mode_ui,
    sa_diag_orient_ui,
    sa_metric_ui,
    sa_order,
    variant_ui,
):
    _metric = sa_metric_ui.value
    _fmt = ".0f" if _metric == "n_water" else ".3f"
    _orig_col = METRIC_TO_ORIGINAL.get(_metric)
    mo.stop(_orig_col is None, mo.md(f"No original column mapped for `{_metric}`."))
    mo.stop(not Path(metadata_ui.value).exists(), mo.md(f"metadata.csv not found: `{metadata_ui.value}`"))

    # A diagonal Series → the panel matrix in the chosen layout: the full pdb×pdb matrix
    # with only the diagonal filled (square), or a compact 1×N / N×1 strip of just the
    # diagonal values (row / column).
    def _shape(series):
        _s = pd.Series(series).reindex(sa_order)
        if sa_diag_orient_ui.value == "row":
            return pd.DataFrame([_s.to_numpy()], index=[""], columns=list(sa_order))
        if sa_diag_orient_ui.value == "column":
            return pd.DataFrame(_s.to_numpy(), index=list(sa_order), columns=[""])
        return diagonal_matrix(_s, sa_order)

    _orig = pd.to_numeric(
        pd.read_csv(metadata_ui.value).set_index("pdb_id")[_orig_col], errors="coerce"
    ).reindex(sa_order)
    _self = {
        _v: sa_df[(sa_df["variant"] == _v) & (sa_df["mtz_source"] == sa_df["starting_model"])]
        .set_index("mtz_source")[_metric]
        .reindex(sa_order)
        for _v in variant_ui.value
    }
    _orig_label = f"PDB-REDO {_metric}"

    if sa_diag_mode_ui.value == "original":
        # Raw self-refined values beside the deposited value — same units, so every panel
        # shares one viridis colorbar.
        _panels = {_orig_label: _shape(_orig)}
        for _v in variant_ui.value:
            _panels[f"phenix ({_v})"] = _shape(_self[_v])
        _fig = make_panels(_panels, cbar_label=_metric, cmap="viridis", shared_cbar=True, fmt=_fmt, xlabel="", ylabel="")
    else:
        _panels = {_orig_label: _shape(_orig)}
        _specs = {
            _orig_label: {
                "label": _orig_label, "cbar_label": _metric, "cmap": "viridis",
                "vmin": float(np.nanmin(_orig.to_numpy())), "vmax": float(np.nanmax(_orig.to_numpy())),
            }
        }
        for _v in variant_ui.value:
            _delta = _self[_v] - _orig
            _key = f"phenix ({_v}) − PDB-REDO"
            _panels[_key] = _shape(_delta)
            _lim = float(np.nanmax(np.abs(_delta.to_numpy()))) or 1.0
            _specs[_key] = {"label": _key, "cbar_label": f"Δ{_metric}", "cmap": "RdBu_r", "vmin": -_lim, "vmax": _lim}
        _fig = make_panels(_panels, specs=_specs, fmt=_fmt, xlabel="", ylabel="")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Cross-refinement matrix

    Rows = **starting model** (`starting_model`), columns = **mtz data used** (`mtz_source`),
    both ordered by the **Sort axes by** selector. The leading diagonal is each structure's
    self-refinement (shown on its own above).
    """)
    return


@app.cell
def _(make_panels, sa_df, sa_metric_ui, sa_order, to_matrix, variant_ui):
    _metric = sa_metric_ui.value
    _fmt = ".0f" if _metric == "n_water" else ".3f"
    _panels = {
        v: to_matrix(sa_df[sa_df["variant"] == v], _metric, sa_order, index="starting_model", columns="mtz_source")
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
    (per mtz source / `mtz_source`):

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
    # Δ vs baseline, subtracted column-wise (per mtz source / mtz_source):
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
                lambda d: d[d["mtz_source"] == d["starting_model"]].set_index("mtz_source")[_metric].reindex(sa_order)
            )(sa_df[sa_df["variant"] == v])
            for v in variant_ui.value
        }

    _panels = {
        v: to_matrix(sa_df[sa_df["variant"] == v], _metric, sa_order, index="starting_model", columns="mtz_source").sub(
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
    ## Section B — water-set agreement (precision / recall / f1)

    _Needs: the step-3 agreement CSVs (+ `reference_pairwise_metrics.csv` for the original reference)._

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
    _default = [k for k in ["precision", "recall", "f1"] if k in _available]
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
    np,
    sb_metric_ui,
    sb_order,
    sb_ref_orig,
    sb_selfref,
    to_matrix,
):
    mo.stop(not sb_metric_ui.value, mo.md("Select at least one agreement metric above."))

    def _ref_mats(frame):
        return {k: to_matrix(frame, k, sb_order, index="structure_ref", columns="structure_mobile") for k in sb_metric_ui.value}

    # Build every reference block's matrices up front — original-vs-original and each
    # per-variant self-refinement matrix — so each metric's colorbar can be shared
    # *vertically* (one range per metric column, across original + all variant blocks)
    # instead of the fixed 0–1 in METRICS. ★-mark the one the Δ panels subtract.
    _orig_active = baseline_ui.value == "original"
    _named = []  # (star, title, matrices)
    if sb_ref_orig is not None:
        _named.append(("★ " if _orig_active else "", "original reference — original-vs-original", _ref_mats(sb_ref_orig)))
    for _v in sb_selfref:
        _named.append(("" if _orig_active else "★ ", f"re-refined reference — `{_v}`", _ref_mats(sb_selfref[_v])))

    # Per-metric shared (vmin, vmax) across all blocks, from the actual data range.
    _specs = {}
    for _k in sb_metric_ui.value:
        _vals = np.concatenate([mats[_k].to_numpy().ravel() for _, _, mats in _named]) if _named else np.array([])
        _vals = _vals[np.isfinite(_vals)]
        _specs[_k] = {**METRICS[_k], "vmin": float(_vals.min()), "vmax": float(_vals.max())} if _vals.size else METRICS[_k]

    def _ref_panels(mats):
        return make_panels(mats, specs=_specs, mask_diagonal=True, xlabel="predictor (mobile)", ylabel="ground truth (ref)")

    _blocks = [mo.md(
        f"**Active Δ baseline: `{baseline_ui.value}`** — the ★-marked reference below is the "
        "one the Δ panels subtract; the other is shown for reference only."
    )]

    if sb_ref_orig is not None:
        _star, _title, _mats = _named[0]
        _blocks.append(mo.vstack([mo.md(f"**{_star}{_title}**"), _ref_panels(_mats)]))
    else:
        _blocks.append(mo.md("_`reference_pairwise_metrics.csv` not found — original reference unavailable._"))

    if sb_selfref:
        for _star, _title, _mats in (_named[1:] if sb_ref_orig is not None else _named):
            _blocks.append(mo.vstack([mo.md(f"**{_star}{_title}**"), _ref_panels(_mats)]))
    else:
        _blocks.append(mo.md("_No `self_refined_pairwise_metrics_*` CSVs — re-refined reference unavailable (run `--self-refined`)._"))

    mo.vstack(_blocks)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Exploratory — precision vs Δ water count (predictor − reference)

    Does reference-agreement **precision** fall off linearly as the predictor's
    water count diverges from the reference's? Scatter of precision (original
    reference, `reference_pairwise_metrics.csv`) against
    `num_water(predictor) − num_water(reference)`, one point per off-diagonal
    `(reference, predictor)` pair, with an ordinary-least-squares fit and Pearson
    `r`. Self-comparisons (Δ = 0) are excluded. Water counts follow the **Sort
    axes by** selector, same as the heatmaps.
    """)
    return


@app.cell
def _(mo, np, plt, sb_ref_orig, water_counts):
    mo.stop(sb_ref_orig is None, mo.md("_`reference_pairwise_metrics.csv` not found — nothing to plot._"))
    mo.stop("precision" not in sb_ref_orig.columns, mo.md("_No `precision` column in the reference metrics._"))

    _pairs = sb_ref_orig[sb_ref_orig["structure_ref"] != sb_ref_orig["structure_mobile"]].copy()
    _pairs["reference_water"] = _pairs["structure_ref"].map(water_counts)
    _pairs["predictor_water"] = _pairs["structure_mobile"].map(water_counts)
    _pairs["delta_water"] = _pairs["predictor_water"] - _pairs["reference_water"]
    _pairs = _pairs.dropna(subset=["delta_water", "precision"])
    mo.stop(len(_pairs) < 2, mo.md("_Not enough paired points (need water counts for both members)._"))

    delta_water = _pairs["delta_water"].to_numpy(dtype=float)
    precision = _pairs["precision"].to_numpy(dtype=float)
    _slope, _intercept = np.polyfit(delta_water, precision, 1)
    pearson_r = float(np.corrcoef(delta_water, precision)[0, 1])

    _fig, _ax = plt.subplots(figsize=(6, 4.5))
    _ax.scatter(delta_water, precision, s=18, alpha=0.5, edgecolor="none")
    _line_x = np.linspace(delta_water.min(), delta_water.max(), 100)
    _ax.plot(
        _line_x, _slope * _line_x + _intercept, color="crimson", lw=1.5,
        label=(
            f"y = {_slope:.2e}·x + {_intercept:.3f}\n"
            f"r = {pearson_r:.3f}   r² = {pearson_r ** 2:.3f}   n = {len(_pairs)}"
        ),
    )
    _ax.axvline(0, color="gray", lw=0.7, ls="--")
    _ax.set_xlabel("Δ water count (predictor − reference)")
    _ax.set_ylabel("precision")
    _ax.set_title("Reference-agreement precision vs Δ water count")
    _ax.legend(loc="best", fontsize=8, frameon=False)
    _fig.tight_layout()
    _fig
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
    ### Δ self-refinement vs PDB-REDO — full pairwise matrix

    Generalizes the diagonal-only view above to **every pair**: cell `(a, b)` is
    `metric(a_refined_by_a ↔ b_refined_by_b) − metric(original-a ↔ original-b)` —
    the self-refined reference matrix (`self_refined_pairwise_metrics_<variant>`)
    minus the original reference matrix (`reference_pairwise_metrics.csv`),
    element-wise. Both sides are reference-vs-reference, so this isolates the effect
    of phenix self-re-refinement on the **pairwise water agreement between two
    structures**, off-diagonal included. The diagonal is 0 by construction (self vs
    self is identical agreement on both sides) and is masked. One row of metric
    panels per variant; diverging scale per metric, shared across variants. For
    precision/recall/F1 a **positive** Δ (red) means the self-refined pair agrees
    *better* than the deposited pair; for chamfer/RMSD positive = worse.
    """)
    return


@app.cell
def _(
    make_panels,
    mo,
    np,
    sb_metric_ui,
    sb_order,
    sb_ref_orig,
    sb_selfref,
    to_matrix,
    variant_ui,
):
    mo.stop(sb_ref_orig is None, mo.md("_`reference_pairwise_metrics.csv` not found — original reference unavailable._"))
    mo.stop(not sb_selfref, mo.md("_No `self_refined_pairwise_metrics_*` CSVs — re-refined reference unavailable (run `--self-refined`)._"))
    mo.stop(not sb_metric_ui.value, mo.md("Select at least one agreement metric above."))

    # Per-variant element-wise Δ: self-refined reference matrix − original reference
    # matrix. Both indexed by structure_ref/structure_mobile, so the subtraction is
    # cell-for-cell once each is reindexed to sb_order.
    _orig = {
        _k: to_matrix(sb_ref_orig, _k, sb_order, index="structure_ref", columns="structure_mobile")
        for _k in sb_metric_ui.value
    }
    _per_variant = []
    for _v in variant_ui.value:
        if _v not in sb_selfref:
            continue
        _diffs = {
            _k: to_matrix(sb_selfref[_v], _k, sb_order, index="structure_ref", columns="structure_mobile") - _orig[_k]
            for _k in sb_metric_ui.value
        }
        _per_variant.append((_v, _diffs))

    # Per-metric symmetric limit = max |Δ| across all variants present.
    _specs = {}
    for _k in sb_metric_ui.value:
        _lims = [
            float(np.nanmax(np.abs(d[_k].to_numpy())))
            for _, d in _per_variant
            if np.isfinite(d[_k].to_numpy()).any()
        ]
        _lim = (max(_lims) if _lims else 1.0) or 1.0
        _specs[_k] = {"label": f"Δ {_k}", "cmap": "coolwarm", "vmin": -_lim, "vmax": _lim}

    mo.vstack([
        mo.vstack([
            mo.md(f"**Δ `{_v}`  (self-refined ref − original ref, pairwise)**"),
            make_panels(_diffs, specs=_specs, fmt="+.2f", mask_diagonal=True, xlabel="structure b", ylabel="structure a"),
        ])
        for _v, _diffs in _per_variant
    ])
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
    ### phenix agreement — raw matrices (before Δ)

    The **un-subtracted** phenix matrices that feed the Δ panels below, so the
    absolute agreement is visible on its own. Cell `(a, b)` is
    `metric(reference-a ↔ b_refined_by_a)`, ground truth tracking the Δ baseline
    (`original-a` for **original**, `a_refined_by_a` for **re-refined**). Same source
    and layout as the Δ cell, minus the subtraction. Each metric's colour range is
    **shared across all variant blocks**, taken from the pooled data range (override
    either bound per metric in the control below). This is the same content as the
    *phenix cross-refinement matrix* cell higher up, placed here for direct
    comparison with the Δ.
    """)
    return


@app.cell
def _(mo, sb_metric_ui):
    # Optional manual override of the shared colour range, one (vmin, vmax) text pair
    # per selected metric. Blank = auto from the pooled data range across all blocks.
    raw_range_ui = mo.ui.dictionary({
        _k: mo.ui.dictionary({
            "vmin": mo.ui.text(value="", placeholder="auto", label=f"{_k} vmin"),
            "vmax": mo.ui.text(value="", placeholder="auto", label=f"{_k} vmax"),
        })
        for _k in sb_metric_ui.value
    })
    mo.vstack([
        mo.md("**Manual shared range** — blank = auto from the data range across all blocks."),
        raw_range_ui,
    ])
    return (raw_range_ui,)


@app.cell
def _(
    METRICS,
    baseline_ui,
    make_panels,
    mo,
    np,
    raw_range_ui,
    sb_metric_ui,
    sb_order,
    sb_phenix,
    sb_phenix_selfref,
    to_matrix,
    variant_ui,
):
    # Un-subtracted phenix matrices, same baseline-tracking source as the Δ cell:
    # original-a waters for "original", a_refined_by_a for "re-refined".
    _orig = baseline_ui.value == "original"
    _phx_src = sb_phenix if _orig else sb_phenix_selfref
    _ground = "original-a" if _orig else "a_refined_by_a"
    mo.stop(
        not _phx_src,
        mo.md(f"⏳ No phenix CSVs for the **{baseline_ui.value}** grounding"
              + ("" if _orig else " — run the script with `--ref-from-self-refined`.")),
    )
    mo.stop(not sb_metric_ui.value, mo.md("Select at least one agreement metric above."))

    # Build every present variant's matrices, then share (vmin, vmax) per metric across
    # all of them from the actual data range; a manual entry overrides either bound.
    _mats_by_variant = [
        (_v, {_k: to_matrix(_phx_src[_v], _k, sb_order, index="reference", columns="predictor") for _k in sb_metric_ui.value})
        for _v in variant_ui.value if _v in _phx_src
    ]

    def _override(_k, _bound, _auto):
        _raw = raw_range_ui.value.get(_k, {}).get(_bound, "") if raw_range_ui.value else ""
        try:
            return float(_raw) if str(_raw).strip() != "" else _auto
        except ValueError:
            return _auto

    _specs = {}
    for _k in sb_metric_ui.value:
        _vals = np.concatenate([m[_k].to_numpy().ravel() for _, m in _mats_by_variant]) if _mats_by_variant else np.array([])
        _vals = _vals[np.isfinite(_vals)]
        _auto_min = float(_vals.min()) if _vals.size else METRICS[_k].get("vmin")
        _auto_max = float(_vals.max()) if _vals.size else METRICS[_k].get("vmax")
        _specs[_k] = {**METRICS[_k], "vmin": _override(_k, "vmin", _auto_min), "vmax": _override(_k, "vmax", _auto_max)}

    mo.vstack([
        mo.vstack([
            mo.md(f"**phenix — `{_v}`  (ground truth = `{_ground}`)**"),
            make_panels(
                _mats, specs=_specs, mask_diagonal=not _orig,
                xlabel="mtz used to re-refine", ylabel="starting model",
            ),
        ])
        for _v, _mats in _mats_by_variant
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Δ agreement — phenix − per-panel reference

    `metric(phenix) − metric(reference)`, one row of metric panels per variant,
    diverging scale per panel (symmetric about 0). The phenix grounding still tracks the
    **Δ baseline** radio (original-a, or a_refined_by_a for re-refined), but the
    subtracted reference is now chosen **per panel** in the control below: the original
    reference or *any* variant's self-refinement — so cross pairings like
    `auto` cross-refine − `stripped` self-refinement are possible. For precision/recall/F1
    a **positive** Δ (red) means the phenix predictor agrees with the reference waters
    *better* than the baseline does; for chamfer/RMSD (lower is better) positive = worse.

    **Ground truth is held fixed** on both sides only when a re-refined panel subtracts
    its *own* variant's self-refinement; any cross-variant pairing (e.g. `auto` phenix −
    `stripped` self-refinement) subtracts against a different ground truth, so read the
    off-diagonal as a blended predictor + ground-truth effect. The **diagonal** is a
    separate matter: under the **re-refined** grounding it is 0 for *every* baseline
    (both sides collapse to a self-vs-self comparison → perfect → cancel) and is masked;
    under the **original** grounding it is `agreement(original-a, a_refined_by_a) − 1`
    and is shown.
    """)
    return


@app.cell
def _(baseline_ui, mo, sb_ref_orig, sb_selfref, variant_ui):
    # Per-panel Δ baseline: each phenix variant panel may subtract the original
    # reference OR any variant's self-refinement (e.g. auto phenix − stripped
    # self-refinement). Default reproduces the old ground-truth-fixed behaviour:
    # original ref for the "original" baseline, same-variant self-ref otherwise.
    _opts = {}
    if sb_ref_orig is not None:
        _opts["original"] = ("orig", None)
    for _w in sb_selfref:
        _opts[f"self-refined: {_w}"] = ("selfref", _w)

    def _default(_v):
        if baseline_ui.value == "original" and sb_ref_orig is not None:
            return "original"
        if _v in sb_selfref:
            return f"self-refined: {_v}"
        return next(iter(_opts), None)

    delta_baseline_ui = mo.ui.dictionary({
        _v: mo.ui.dropdown(options=_opts, value=_default(_v), label=f"`{_v}` panel −")
        for _v in variant_ui.value
    })
    mo.vstack([
        mo.md("**Per-panel Δ baseline** — the reference each phenix panel subtracts."),
        delta_baseline_ui,
    ])
    return (delta_baseline_ui,)


@app.cell
def _(
    baseline_ui,
    delta_baseline_ui,
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
    # Phenix grounding tracks the Δ-baseline radio (original-a vs a_refined_by_a); the
    # subtracted reference is chosen PER PANEL by the control above, so mismatched
    # pairings like auto-cross-refine − stripped-self-refine are possible. Under the
    # re-refined grounding the diagonal (i, i) is 0 by construction for ANY baseline:
    # the phenix diagonal is i's self-refinement vs itself and every reference diagonal
    # is also a self-vs-self comparison, so both sides are perfect and cancel. It is
    # therefore masked whenever the grounding is re-refined; under the original
    # grounding the diagonal is agreement(original-i, i_refined_by_i) − 1 and is shown.
    _orig = baseline_ui.value == "original"
    _phx_src = sb_phenix if _orig else sb_phenix_selfref
    mo.stop(
        not _phx_src,
        mo.md(f"⏳ No phenix CSVs for the **{baseline_ui.value}** grounding"
              + ("" if _orig else " — run the script with `--ref-from-self-refined`.")),
    )
    mo.stop(not sb_metric_ui.value, mo.md("Select at least one agreement metric above."))

    def _ref_frame(_choice):
        _kind, _key = _choice
        return sb_ref_orig if _kind == "orig" else sb_selfref.get(_key)

    def _label(_choice):
        _kind, _key = _choice
        return "original" if _kind == "orig" else f"self-refined: {_key}"

    # First pass: build every panel's Δ matrices (phenix − its chosen baseline). Second
    # pass renders them — the colour range is resolved per metric ACROSS panels, so the
    # same metric (e.g. Δ precision) shares one symmetric scale.
    _per_variant = []
    for _v in variant_ui.value:
        if _v not in _phx_src:
            continue
        _choice = delta_baseline_ui.value.get(_v)
        if _choice is None:
            _per_variant.append((_v, None, None, False))
            continue
        _ref = _ref_frame(_choice)
        _mask = not _orig
        if _ref is None:
            _per_variant.append((_v, None, _label(_choice), _mask))
            continue
        _diffs = {}
        for _k in sb_metric_ui.value:
            _phx = to_matrix(_phx_src[_v], _k, sb_order, index="reference", columns="predictor")
            _rm = to_matrix(_ref, _k, sb_order, index="structure_ref", columns="structure_mobile")
            _diffs[_k] = _phx - _rm
        _per_variant.append((_v, _diffs, _label(_choice), _mask))

    # Per-metric symmetric limit = max |Δ| over all panels that have data.
    _present = [d for _, d, _, _ in _per_variant if d is not None]
    _specs = {}
    for _k in sb_metric_ui.value:
        _lims = [
            float(np.nanmax(np.abs(d[_k].to_numpy())))
            for d in _present
            if np.isfinite(d[_k].to_numpy()).any()
        ]
        _lim = (max(_lims) if _lims else 1.0) or 1.0
        _specs[_k] = {"label": f"Δ {_k}", "cmap": "coolwarm", "vmin": -_lim, "vmax": _lim}

    _rows = []
    for _v, _diffs, _lbl, _mask in _per_variant:
        if _diffs is None:
            _rows.append(mo.md(
                f"_`{_v}`: baseline (`{_lbl}`) reference missing._" if _lbl else f"_`{_v}`: no baseline selected._"
            ))
            continue
        _rows.append(mo.vstack([
            mo.md(f"**Δ `{_v}`  (phenix − `{_lbl}`)**"),
            make_panels(_diffs, specs=_specs, fmt="+.2f", mask_diagonal=_mask, xlabel="mtz used to re-refine", ylabel="starting model"),
        ]))
    mo.vstack(_rows)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
