import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Isomorphous-subset finder + alignment-reference pick

    **Part 1 — minimal path** (reads `metadata.csv` only): survey the space
    groups, filter to an isomorphous subset within one space group, pick the
    best-resolution reference, and export the subset as a cohort `.txt`.

    **Part 2 — optional deep dive** (also reads `pairwise_metrics_<cutoff>.csv`
    from `scripts/pairwise_water_metrics.py`): clustermaps, below-threshold
    clusters, and cross-metric (conformation vs crystal-form) agreement — the
    evidence for *which* space group and tolerance to pick in Part 1. Run it only
    if you want to dig deeper before committing to a subset.
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path

    import gemmi

    import config
    from cw.metadata import max_cell_diff

    cohort_input = mo.ui.text(
        value="C001033",
        placeholder="<cohort>",
        label="Cohort name (drives the metadata.csv and output paths)",
        full_width=True,
    )
    cohort_input
    return Path, cohort_input, config, gemmi, max_cell_diff


@app.cell
def _(Path, cohort_input, config, mo):
    _default = str(Path(config.DATA_DIR) / cohort_input.value.strip() / "metadata.csv")
    csv_input = mo.ui.text(
        value=_default,
        placeholder="/path/to/<cohort>/metadata.csv",
        label="metadata.csv",
        full_width=True,
    )
    csv_input
    return (csv_input,)


@app.cell
def _(Path, csv_input, mo):
    import pandas as pd

    _csv = Path(csv_input.value.strip())
    mo.stop(not _csv.exists(), mo.md(f"**metadata.csv not found:** `{_csv}`"))

    df = pd.read_csv(_csv)
    # resolution / (deposited) r_work / r_free are "<missing>" when the value was
    # unavailable; coerce to NaN so numeric queries (e.g. `r_free <= 0.25`) don't
    # hit strings. deposited_* are absent from older metadata.csv, hence the guard.
    for _col in ("resolution", "r_work", "r_free", "deposited_r_work", "deposited_r_free"):
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")
    mo.md(f"Loaded **{len(df)}** structures from `{_csv.name}`.")
    return (df,)


@app.cell
def _(mo):
    mo.md("""
    ## 1 — Space-group counts
    """)
    return


@app.cell
def _(df, mo):
    sg_counts = (
        df.groupby("space_group")
        .agg(
            n_structures=("pdb_id", "size"),
            median_resolution=("resolution", "median"),
            best_resolution=("resolution", "min"),
        )
        .sort_values("n_structures", ascending=False)
    )
    mo.md(
        f"**{df['space_group'].nunique()}** distinct space groups across "
        f"{len(df)} structures."
    )
    return (sg_counts,)


@app.cell
def _(sg_counts):
    sg_counts
    return


@app.cell
def _(sg_counts):
    import matplotlib.pyplot as plt

    _fig, _ax = plt.subplots(figsize=(7, 0.4 * len(sg_counts) + 1.5))
    _ax.barh(sg_counts.index.astype(str), sg_counts["n_structures"], color="steelblue")
    _ax.invert_yaxis()
    _ax.set_xlabel("Structures")
    _ax.set_ylabel("Space group")
    _ax.set_title("Cohort structures per space group")
    _fig.tight_layout()
    _fig
    return (plt,)


@app.cell
def _(mo):
    mo.md("""
    ## 2 — Isomorphousness filter

    Pick the space group to work in (defaults to the most populated), a
    reference cell, and a tolerance. A structure is *isomorphous* if
    `max_cell_diff(cell, reference_cell)` is at or below the tolerance.

    **Reference cell** options:
    - *highest resolution* — anchor on the structure you'd most likely align to.
    - *median cell* — anchor on the cohort's central cell (robust to outliers).
    """)
    return


@app.cell
def _(mo, sg_counts):
    sg_dropdown = mo.ui.dropdown(
        options=sg_counts.index.astype(str).tolist(),
        value=str(sg_counts.index[0]),
        label="Space group",
    )
    ref_mode = mo.ui.dropdown(
        options=["median cell", "highest resolution"],
        value="median cell",
        label="Reference cell",
    )
    tol_slider = mo.ui.slider(
        start=0.0, stop=10.0, step=0.25, value=2.0, label="Tolerance (max cell diff %)",
        show_value=True,
    )
    mo.vstack([sg_dropdown, ref_mode, tol_slider])
    return ref_mode, sg_dropdown, tol_slider


@app.cell
def _(df, gemmi, max_cell_diff, mo, ref_mode, sg_dropdown, tol_slider):
    _cols = ["cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma"]

    in_sg = df[df["space_group"].astype(str) == sg_dropdown.value].copy()
    mo.stop(in_sg.empty, mo.md("No structures in that space group."))

    def _cell(row):
        return gemmi.UnitCell(*(float(row[c]) for c in _cols))

    if ref_mode.value == "highest resolution":
        _ref_row = in_sg.sort_values("resolution").iloc[0]
        ref_cell = _cell(_ref_row)
        ref_label = f"{_ref_row['pdb_id']} (res {_ref_row['resolution']:.2f} Å)"
    else:
        _med = in_sg[_cols].median()
        ref_cell = gemmi.UnitCell(*(float(_med[c]) for c in _cols))
        ref_label = "median cell"

    in_sg["max_cell_diff_pct"] = in_sg.apply(
        lambda r: max_cell_diff(_cell(r), ref_cell), axis=1
    )
    in_sg["isomorphous"] = in_sg["max_cell_diff_pct"] <= tol_slider.value

    _n_iso = int(in_sg["isomorphous"].sum())
    mo.md(
        f"Space group **{sg_dropdown.value}**: {len(in_sg)} structures.\n\n"
        f"Reference cell: **{ref_label}** — `{ref_cell}`\n\n"
        f"Within **{tol_slider.value}%** tolerance: "
        f"**{_n_iso} isomorphous** / {len(in_sg) - _n_iso} excluded."
    )
    return (in_sg,)


@app.cell
def _(in_sg, plt):
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.hist(in_sg["max_cell_diff_pct"], bins=40, color="steelblue")
    _ax.set_xlabel("max cell diff from reference (%)")
    _ax.set_ylabel("Structures")
    _ax.set_title("Unit-cell deviation within the selected space group")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2b — Additional column filters (optional)

    Narrow the isomorphous subset further with a boolean expression over the
    metadata columns — combine conditions with `and` / `or` / `not` and
    parentheses, e.g. `resolution <= 2.0 and deposited_r_free <= 0.2`.

    Uses `pandas.DataFrame.query` syntax. Rows with a `NaN` in a referenced
    column (missing resolution / R-free) never match, so they are dropped.
    Leave blank to keep the full isomorphous subset.
    """)
    return


@app.cell
def _(in_sg, mo):
    _iso = in_sg[in_sg["isomorphous"]]
    _numeric = [
        c
        for c in (
            "resolution",
            "r_work",
            "r_free",
            "deposited_r_work",
            "deposited_r_free",
            "num_water",
            "unit_cell_volume",
        )
        if c in _iso.columns and _iso[c].notna().any()
    ]
    _ranges = "\n".join(f"- `{c}`: {_iso[c].min():.3g} – {_iso[c].max():.3g}: median {_iso[c].median():.3g}" for c in _numeric)
    extra_query_input = mo.ui.text(
        value="",
        placeholder="resolution <= 2.0 and deposited_r_free <= 0.2",
        label="Additional filter (blank = keep all isomorphous)",
        full_width=True,
    )
    mo.vstack([
        extra_query_input,
        mo.md(
            f"**Columns:** `{'`, `'.join(in_sg.columns)}`\n\n"
            f"**Numeric ranges (isomorphous subset):**\n{_ranges}"
        ),
    ])
    return (extra_query_input,)


@app.cell
def _(df, extra_query_input, in_sg, mo):
    _n_iso = int(in_sg["isomorphous"].sum())  # Z: all isomorphous, criteria aside
    iso_filtered = in_sg[in_sg["isomorphous"]].copy()
    _expr = extra_query_input.value.strip()
    if _expr:
        try:
            _matched = df.query(_expr)  # X: whole cohort matching criteria
            iso_filtered = iso_filtered.query(_expr)  # Y: isomorphous AND match
        except Exception as _exc:
            mo.stop(True, mo.md(f"**Invalid query:** `{type(_exc).__name__}: {_exc}`"))
        _msg = (
            f"**{len(_matched)}** match the filtering criteria `{_expr}`, "
            f"among which **{len(iso_filtered)} / {_n_iso}** are isomorphous."
        )
    else:
        _msg = f"Isomorphous subset: **{_n_iso}** structures (no additional filter)."
    mo.md(_msg)
    return (iso_filtered,)


@app.cell
def _(mo):
    mo.md("""
    ## 3 — Alignment-reference candidates

    The filtered isomorphous subset sorted by resolution. The top row is the
    best-resolution structure — the natural alignment reference.
    """)
    return


@app.cell
def _(iso_filtered):
    _show = [
        c
        for c in (
            "pdb_id",
            "resolution",
            "r_free",
            "deposited_r_free",
            "max_cell_diff_pct",
            "num_water",
            "unit_cell_volume",
        )
        if c in iso_filtered.columns
    ]
    ref_candidates = iso_filtered.sort_values("resolution")[_show].reset_index(drop=True)
    ref_candidates
    return (ref_candidates,)


@app.cell
def _(mo, ref_candidates):
    mo.stop(ref_candidates.empty, mo.md("*No isomorphous structures at this tolerance.*"))
    _top = ref_candidates.iloc[0]
    mo.callout(
        mo.md(
            f"**Suggested reference: `{_top['pdb_id']}`** — "
            f"resolution {_top['resolution']:.2f} Å, "
            f"{int(_top['num_water'])} waters, "
            f"cell diff {_top['max_cell_diff_pct']:.2f}% from reference."
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4 — Export isomorphous cohort

    Write the isomorphous subset (Section 3) to a cohort `.txt` — one PDB ID per
    line — to feed the filter / align / cluster scripts, plus a sidecar `.yaml`
    recording the split criteria (space group, tolerance, reference) so the
    subset is reproducible. Set the path, then click the button (it only writes
    on click).
    """)
    return


@app.cell
def _(cohort_input, mo, ref_candidates):
    cohort_out_input = mo.ui.text(
        value=f"data/{cohort_input.value.strip()}_iso.txt",
        placeholder="data/<cohort>_iso.txt",
        label="Cohort .txt output path",
        full_width=True,
    )
    write_button = mo.ui.run_button(label=f"Write {len(ref_candidates)} IDs")
    mo.vstack([cohort_out_input, write_button])
    return cohort_out_input, write_button


@app.cell
def _(
    Path,
    cohort_out_input,
    csv_input,
    extra_query_input,
    iso_filtered,
    mo,
    ref_candidates,
    ref_mode,
    sg_dropdown,
    tol_slider,
    write_button,
):
    from datetime import datetime

    import yaml

    mo.stop(not write_button.value, mo.md("*Click the button above to write the cohort file.*"))
    mo.stop(ref_candidates.empty, mo.md("*No isomorphous structures to export.*"))

    _out = Path(cohort_out_input.value.strip())
    _ids = ref_candidates["pdb_id"].astype(str).tolist()
    _out.parent.mkdir(parents=True, exist_ok=True)
    _out.write_text("\n".join(_ids) + "\n")

    _extra_query = extra_query_input.value.strip() or None
    _cell_cols = ["cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma"]
    _meta = {
        "parent_cohort": Path(csv_input.value.strip()).parent.name,
        "subset_txt": _out.name,
        "split_criterion": "same_space_group_and_isomorphous_cell"
        + ("_and_column_filter" if _extra_query else ""),
        "space_group": sg_dropdown.value,
        "reference_cell_mode": ref_mode.value,
        "tolerance_pct": float(tol_slider.value),
        "additional_filter_query": _extra_query,
        "cell_ranges": {
            c: [float(iso_filtered[c].min()), float(iso_filtered[c].max())] for c in _cell_cols
        },
        "suggested_reference_pdb": str(ref_candidates.iloc[0]["pdb_id"]),
        "n_structures": len(_ids),
        "generated_by": "notebooks/optional_find_isomorphous_subset_and_align_ref.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _yaml_path = _out.with_suffix(".yaml")
    _yaml_path.write_text(yaml.safe_dump(_meta, sort_keys=False))

    mo.callout(
        mo.md(f"Wrote **{len(_ids)}** PDB IDs → `{_out}`\n\nProvenance → `{_yaml_path}`"),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    # Part 2 — Optional deep dive

    Everything below reads the **precomputed** pairwise matrix
    (`pairwise_metrics_<cutoff>.csv` from `scripts/pairwise_water_metrics.py`) on
    top of `metadata.csv`. It is diagnostic — evidence for the space group and
    tolerance chosen in Part 1 — and is safe to skip. Part 1 stands on its own
    and never depends on anything here.

    *(Export of a deep-dive-selected subset — a below-threshold cluster or a
    cross-metric-agreed set — is still to be designed; for now these sections
    only inform the Part 1 export.)*
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5 — Subgroup clustermaps (Cα RMSD & unit-cell difference)

    Hierarchically-clustered heatmaps of the **precomputed** pairwise matrix
    (`pairwise_metrics_<cutoff>.csv` from `scripts/pairwise_water_metrics.py`).
    Dark diagonal blocks are subgroups; the color strips flag each structure's
    space group and resolution, so a conformational block that lines up with one
    space-group color is really a crystal form, while one that cuts across
    colors is a genuine conformational split.

    - **Cα RMSD** (`rmsd_after`) → conformational / packing subgroups.
    - **max cell diff** (`max_cell_diff`) → crystal-form subgroups.

    The RMSD matrix is asymmetric (a→b ≠ b→a); it is symmetrized as (M + Mᵀ)/2
    before clustering. Heavy compute lives in the script — this only reads its CSV.
    """)
    return


@app.cell
def _(Path, config, csv_input, mo):
    _default = str(
        Path(csv_input.value.strip()).parent
        / f"pairwise_metrics_{config.CLUSTER_MEMBER_RADIUS}.csv"
    )
    pairwise_input = mo.ui.text(
        value=_default,
        placeholder="/path/to/<cohort>/pairwise_metrics_<cutoff>.csv",
        label="pairwise_metrics.csv (from scripts/pairwise_water_metrics.py)",
        full_width=True,
    )
    pairwise_input
    return (pairwise_input,)


@app.cell
def _(Path, mo, pairwise_input):
    import numpy as _np
    import pandas as _pd

    _pw_path = Path(pairwise_input.value.strip())
    mo.stop(
        not _pw_path.exists(),
        mo.md(
            f"**pairwise_metrics.csv not found:** `{_pw_path}`\n\n"
            "Generate it first (needs filtered CIFs):\n\n"
            "```\n"
            "uv run scripts/filter_waters.py data/<cohort>.txt -j 4\n"
            "uv run scripts/pairwise_water_metrics.py data/<cohort>.txt\n"
            "```"
        ),
    )
    _pw = _pd.read_csv(_pw_path)

    def _square(col):
        _m = _pw.pivot(index="structure_ref", columns="structure_mobile", values=col)
        return _m.reindex(index=_m.index, columns=_m.index)

    _rmsd = _square("rmsd_after")
    _cell = _square("max_cell_diff")
    rmsd_sym = (_rmsd + _rmsd.T) / 2.0
    cell_sym = (_cell + _cell.T) / 2.0
    mo.md(f"Loaded pairwise matrix for **{len(rmsd_sym)}** structures from `{_pw_path.name}`.")
    return cell_sym, rmsd_sym


@app.cell
def _(df):
    import matplotlib.patches as _mpatches
    import pandas as _pd
    import seaborn as _sns

    _meta = df.set_index("pdb_id")

    def subgroup_clustermap(matrix, title, cbar_label, cmap="rocket"):
        _ids = list(matrix.index)
        _mat = matrix.fillna(matrix.stack().max())

        _sg = _meta["space_group"].astype(str).reindex(_ids).fillna("?")
        _res = _pd.to_numeric(_meta["resolution"], errors="coerce").reindex(_ids)

        _sg_levels = sorted(_sg.unique())
        _sg_pal = dict(zip(_sg_levels, _sns.color_palette("tab10", len(_sg_levels))))
        _sg_colors = _sg.map(_sg_pal)

        _res_cmap = _sns.color_palette("viridis", as_cmap=True)
        if _res.notna().any():
            _lo, _hi = _res.min(), _res.max()
            _span = (_hi - _lo) or 1.0
            _res_colors = _pd.Series(
                [_res_cmap((v - _lo) / _span) if v == v else (0.85, 0.85, 0.85) for v in _res],
                index=_ids,
            )
        else:
            _res_colors = _pd.Series([(0.85, 0.85, 0.85)] * len(_ids), index=_ids)

        _row_colors = _pd.DataFrame(
            {"space group": _sg_colors, "resolution": _res_colors}, index=_ids
        )

        _g = _sns.clustermap(
            _mat,
            cmap=cmap,
            row_colors=_row_colors,
            col_colors=_row_colors,
            figsize=(11, 11),
            xticklabels=True,
            yticklabels=True,
            cbar_kws={"label": cbar_label},
        )
        _g.ax_heatmap.set_xlabel("")
        _g.ax_heatmap.set_ylabel("")
        _g.ax_heatmap.tick_params(labelsize=5)
        _g.fig.suptitle(title, y=1.02, fontsize=13)
        _handles = [_mpatches.Patch(color=_c, label=_l) for _l, _c in _sg_pal.items()]
        _g.ax_heatmap.legend(
            handles=_handles,
            title="space group",
            bbox_to_anchor=(1.28, 1.0),
            loc="upper left",
            fontsize=7,
            title_fontsize=8,
            frameon=False,
        )
        return _g.fig

    return (subgroup_clustermap,)


@app.cell
def _(rmsd_sym, subgroup_clustermap):
    subgroup_clustermap(rmsd_sym, "Cα RMSD (Å) — conformational subgroups", "Cα RMSD (Å)")
    return


@app.cell
def _(cell_sym, subgroup_clustermap):
    subgroup_clustermap(cell_sym, "max cell diff (%) — crystal-form subgroups", "max cell diff (%)")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6 — Below-threshold clusters

    Pick a metric (Cα RMSD or unit-cell difference) and a threshold, then cut the
    same pairwise matrix with **complete linkage** so every pair *inside* a
    cluster is ≤ threshold (not just the group mean). This cell reports **how many
    clusters** satisfy the threshold and **how many of those have more than 10
    members**, then gives a per-cluster summary — size, min / median / max of
    **both** metrics, resolution range, space groups, and PDB IDs — for each of
    those larger clusters.
    """)
    return


@app.cell
def _(cell_sym, mo, rmsd_sym):
    import numpy as _np

    def _pos_range(_m):
        _v = _m.to_numpy(dtype=float)
        _v = _v[~_np.isnan(_v)]
        _v = _v[_v > 0]
        return (_v.min(), _np.median(_v), _v.max()) if len(_v) else (0.0, 0.0, 1.0)

    _r = _pos_range(rmsd_sym)
    _c = _pos_range(cell_sym)

    threshold_metric = mo.ui.dropdown(
        options={"Cα RMSD (Å)": "rmsd", "max cell diff (%)": "cell"},
        value="Cα RMSD (Å)",
        label="Cluster on",
    )
    threshold_value = mo.ui.number(
        start=0.0,
        stop=float(max(_r[2], _c[2])),
        step=0.05,
        value=round(float(_r[1]), 2),
        label="Threshold (max pairwise value within a cluster)",
    )
    mo.vstack([
        threshold_metric,
        threshold_value,
        mo.md(
            "Observed off-diagonal ranges (min / median / max):\n\n"
            f"- Cα RMSD (Å): {_r[0]:.3f} / {_r[1]:.3f} / {_r[2]:.3f}\n"
            f"- max cell diff (%): {_c[0]:.2f} / {_c[1]:.2f} / {_c[2]:.2f}"
        ),
    ])
    return threshold_metric, threshold_value


@app.cell
def _(cell_sym, df, mo, rmsd_sym, threshold_metric, threshold_value):
    import numpy as _np
    import pandas as _pd
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    _min_members = 10  # only clusters with strictly more than this get a summary

    _matrix = rmsd_sym if threshold_metric.value == "rmsd" else cell_sym
    _thr = float(threshold_value.value)
    _ids = list(_matrix.index)

    _d = _matrix.to_numpy(dtype=float).copy()
    _d[_np.isnan(_d)] = _np.nanmax(_d)  # failed alignments → maximally far apart
    _d = (_d + _d.T) / 2.0
    _np.fill_diagonal(_d, 0.0)

    # Complete linkage: a cluster formed below height t has ALL its pairwise
    # distances ≤ t, so every cluster from a distance-cut at the threshold
    # satisfies the criterion by construction (singletons trivially).
    _labels = fcluster(linkage(squareform(_d, checks=False), method="complete"), t=_thr, criterion="distance")
    _vals, _counts = _np.unique(_labels, return_counts=True)

    _order = _np.argsort(_counts)[::-1]
    _big = [(_vals[i], int(_counts[i])) for i in _order if _counts[i] > _min_members]

    _meta = df.set_index("pdb_id")
    _metric_label = "Cα RMSD (Å)" if threshold_metric.value == "rmsd" else "max cell diff (%)"

    def _offdiag(_m, _members):
        _sub = _m.loc[_members, _members].to_numpy(dtype=float)
        _v = _sub[_np.triu_indices(len(_members), k=1)]
        return _v[~_np.isnan(_v)]

    def _stat(_v, _fmt):
        if len(_v) == 0:
            return "n/a"
        return f"min {_v.min():{_fmt}} / median {_np.median(_v):{_fmt}} / max {_v.max():{_fmt}}"

    def _summary(_rank, _members):
        _res = _pd.to_numeric(_meta["resolution"], errors="coerce").reindex(_members)
        _sg = _meta["space_group"].astype(str).reindex(_members)
        _res_line = (
            "n/a"
            if _res.notna().sum() == 0
            else f"min {_res.min():.2f} / median {_res.median():.2f} / max {_res.max():.2f}"
        )
        _sg_line = ", ".join(f"{_k} ×{_v}" for _k, _v in _sg.value_counts().items())
        return (
            f"#### Cluster {_rank} — {len(_members)} structures\n\n"
            "| metric | min / median / max |\n"
            "|---|---|\n"
            f"| Cα RMSD (Å) | {_stat(_offdiag(rmsd_sym, _members), '.3f')} |\n"
            f"| max cell diff (%) | {_stat(_offdiag(cell_sym, _members), '.2f')} |\n"
            f"| resolution (Å) | {_res_line} |\n\n"
            f"**Space groups:** {_sg_line}\n\n"
            f"**PDB IDs:** `{'`, `'.join(_members)}`\n"
        )

    _blocks = []
    for _rank, (_lab, _n) in enumerate(_big, 1):
        _members = [_ids[i] for i in range(len(_ids)) if _labels[i] == _lab]
        _blocks.append(_summary(_rank, _members))

    _header = (
        f"### Below-threshold clusters — {_metric_label} ≤ {_thr:g}\n\n"
        f"**{len(_vals)} clusters** satisfy the threshold (every pair within a cluster "
        f"≤ {_thr:g}; includes singletons), of which **{len(_big)}** have more than "
        f"{_min_members} members."
    )
    if not _big:
        _header += f"\n\n*No cluster has more than {_min_members} members at this threshold.*"

    _body = _header + "\n\n---\n\n" + "\n\n---\n\n".join(_blocks) if _blocks else _header
    mo.md(_body)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7 — Cross-metric agreement (conformation vs crystal form)

    Two partitions of the *same* structures: a **crystal-form** partition (space
    group first, then a within-SG unit-cell cut) and a **conformational**
    partition (a Cα-RMSD cut). This section asks *how much the two agree*, using
    the standard toolkit for comparing two clusterings.

    Cell diff is only comparable **within** a space group (different SGs constrain
    the cell differently), so it is never used across SGs — the crystal-form
    partition splits by space group first and only subdivides by cell *inside*
    each group.

    - **Threshold-free** — a Mantel test: Spearman correlation between the two
      pairwise matrices with a permutation p-value. The scatter shows the same
      pairs, split by whether the two structures share a space group.
    - **Threshold-based** — cut both into subgroups, then compare with a
      contingency / Jaccard heatmap, the chance-corrected **Adjusted Rand Index**,
      and normalized mutual information. An ARI sweep over the cutoff ranges
      (RMSD 0.5–1.0 Å, cell 2–5 %) shows whether the agreement is robust or an
      artifact of one hand-picked pair of thresholds.
    """)
    return


@app.cell
def _(mo, sg_counts):
    cross_metric_sg = mo.ui.dropdown(
        options=sg_counts.index.astype(str).tolist(),
        value=str(sg_counts.index[0]),
        label="Space group for the Mantel test (cell diff only meaningful within one SG)",
    )
    rmsd_cut = mo.ui.slider(
        start=0.4, stop=1.2, step=0.05, value=0.75, show_value=True,
        label="Cα RMSD cut (Å) — conformational subgroups",
    )
    cell_cut = mo.ui.slider(
        start=1.0, stop=6.0, step=0.25, value=3.0, show_value=True,
        label="within-SG cell-diff cut (%) — crystal-form subgroups",
    )
    min_subgroup_size = mo.ui.number(
        start=1, stop=50, step=1, value=5,
        label="Min subgroup size shown in the contingency / Jaccard heatmaps",
    )
    mo.vstack([cross_metric_sg, rmsd_cut, cell_cut, min_subgroup_size])
    return cell_cut, cross_metric_sg, min_subgroup_size, rmsd_cut


@app.cell
def _(cell_sym, df, plt, rmsd_sym):
    import numpy as _np
    from scipy.stats import spearmanr as _spearmanr

    _order = list(rmsd_sym.index)
    _space_group = (
        df.set_index("pdb_id")["space_group"].astype(str).reindex(_order).fillna("?").to_numpy()
    )
    _upper = _np.triu_indices(len(_order), k=1)
    _rmsd_pairs = rmsd_sym.to_numpy(dtype=float)[_upper]
    _cell_pairs = cell_sym.to_numpy(dtype=float)[_upper]
    _same_sg = (_space_group[:, None] == _space_group[None, :])[_upper]
    _valid = ~(_np.isnan(_rmsd_pairs) | _np.isnan(_cell_pairs))

    def _rho(_mask):
        _selected = _mask & _valid
        if _selected.sum() < 3:
            return None
        return _spearmanr(_rmsd_pairs[_selected], _cell_pairs[_selected]).statistic

    _rho_all = _rho(_np.ones_like(_valid))
    _rho_same = _rho(_same_sg)
    _rho_cross = _rho(~_same_sg)

    _fig, _ax = plt.subplots(figsize=(6.5, 5))
    _cross_mask = (~_same_sg) & _valid
    _same_mask = _same_sg & _valid
    _ax.scatter(
        _rmsd_pairs[_cross_mask], _cell_pairs[_cross_mask],
        s=10, alpha=0.3, color="0.6", label="different SG (cell diff not comparable)",
    )
    _ax.scatter(
        _rmsd_pairs[_same_mask], _cell_pairs[_same_mask],
        s=12, alpha=0.6, color="steelblue", label="same SG",
    )
    _ax.set_xlabel("Cα RMSD (Å)")
    _ax.set_ylabel("max cell diff (%)")

    def _fmt(_value):
        return "n/a" if _value is None else f"{_value:.2f}"

    _ax.set_title(
        "Pairwise RMSD vs cell diff\n"
        f"Spearman ρ — all {_fmt(_rho_all)} | same SG {_fmt(_rho_same)} | "
        f"different SG {_fmt(_rho_cross)}",
        fontsize=10,
    )
    _ax.legend(fontsize=8, frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(cell_sym, cross_metric_sg, df, mo, rmsd_sym):
    import numpy as _np
    from scipy.cluster.hierarchy import cophenet as _cophenet, linkage as _linkage
    from scipy.spatial.distance import squareform as _squareform
    from scipy.stats import spearmanr as _spearmanr

    _space_group = df.set_index("pdb_id")["space_group"].astype(str)
    _members = [
        _pdb for _pdb in rmsd_sym.index if _space_group.get(_pdb) == cross_metric_sg.value
    ]
    _rmsd_sub = rmsd_sym.loc[_members, _members]
    _cell_sub = cell_sym.loc[_members, _members]
    _keep = [
        _m for _m in _members
        if not (_rmsd_sub.loc[_m].isna().any() or _cell_sub.loc[_m].isna().any())
    ]
    _rmsd_sub = _rmsd_sub.loc[_keep, _keep]
    _cell_sub = _cell_sub.loc[_keep, _keep]

    def _condensed(_matrix):
        _d = _matrix.to_numpy(dtype=float)
        _d = (_d + _d.T) / 2.0
        _np.fill_diagonal(_d, 0.0)
        return _squareform(_d, checks=False)

    if len(_keep) < 5:
        _out = mo.md(
            f"**Mantel test — {cross_metric_sg.value}:** only {len(_keep)} clean "
            "structures; need at least 5."
        )
    else:
        _rmsd_condensed = _condensed(_rmsd_sub)
        _cell_condensed = _condensed(_cell_sub)
        _observed = _spearmanr(_rmsd_condensed, _cell_condensed).statistic

        _n = len(_keep)
        _upper = _np.triu_indices(_n, k=1)
        _cell_full = _cell_sub.to_numpy(dtype=float)
        _cell_full = (_cell_full + _cell_full.T) / 2.0
        _np.fill_diagonal(_cell_full, 0.0)
        _rng = _np.random.default_rng(0)
        _permutations = 999
        _count = 1
        for _ in range(_permutations):
            _perm = _rng.permutation(_n)
            _permuted = _cell_full[_np.ix_(_perm, _perm)][_upper]
            if abs(_spearmanr(_rmsd_condensed, _permuted).statistic) >= abs(_observed):
                _count += 1
        _p_value = _count / (_permutations + 1)

        _coph_rmsd = _cophenet(_linkage(_rmsd_condensed, method="complete"))
        _coph_cell = _cophenet(_linkage(_cell_condensed, method="complete"))
        _coph_corr = _spearmanr(_coph_rmsd, _coph_cell).statistic

        _out = mo.md(
            f"### Mantel test — {cross_metric_sg.value} ({_n} structures)\n\n"
            f"- **Matrix correlation** (Spearman ρ, RMSD vs cell diff): "
            f"**{_observed:.3f}**\n"
            f"- **Permutation p-value** ({_permutations} permutations): "
            f"**{_p_value:.3f}**\n"
            f"- **Cophenetic correlation** (RMSD tree vs cell tree): {_coph_corr:.3f}\n\n"
            "Threshold-free: no cut is chosen. A high ρ with small p means the two "
            "metrics order structure pairs consistently *within this space group*."
        )
    _out
    return


@app.cell
def _(cell_cut, cell_sym, df, mo, rmsd_cut, rmsd_sym):
    import numpy as _np
    import pandas as _pd
    from scipy.cluster.hierarchy import fcluster as _fcluster, linkage as _linkage
    from scipy.spatial.distance import squareform as _squareform
    from scipy.special import comb as _comb

    cross_metric_order = list(rmsd_sym.index)
    _space_group = (
        df.set_index("pdb_id")["space_group"].astype(str).reindex(cross_metric_order).fillna("?")
    )

    def _complete_linkage(matrix):
        _d = matrix.to_numpy(dtype=float).copy()
        _d[_np.isnan(_d)] = _np.nanmax(_d)
        _d = (_d + _d.T) / 2.0
        _np.fill_diagonal(_d, 0.0)
        return _linkage(_squareform(_d, checks=False), method="complete")

    _rmsd_linkage = _complete_linkage(rmsd_sym)

    _cell_linkages = {}
    for _sg_name, _idx in _space_group.groupby(_space_group).groups.items():
        _sg_members = list(_idx)
        _sg_linkage = _complete_linkage(cell_sym.loc[_sg_members, _sg_members]) if len(_sg_members) >= 2 else None
        _cell_linkages[_sg_name] = (_sg_members, _sg_linkage)

    def build_conformational_partition(rmsd_threshold):
        _labels = _fcluster(_rmsd_linkage, t=rmsd_threshold, criterion="distance")
        return _pd.Series(_labels, index=cross_metric_order)

    def build_crystal_form_partition(cell_threshold):
        _labels = _pd.Series(index=cross_metric_order, dtype=object)
        for _sg_name, (_sg_members, _sg_linkage) in _cell_linkages.items():
            if _sg_linkage is None:
                _local = [1] * len(_sg_members)
            else:
                _local = _fcluster(_sg_linkage, t=cell_threshold, criterion="distance")
            for _member, _sub in zip(_sg_members, _local):
                _labels[_member] = f"{_sg_name}#{int(_sub)}"
        return _labels

    def adjusted_rand_index(labels_a, labels_b):
        _table = _pd.crosstab(labels_a, labels_b).to_numpy(dtype=float)
        _n = _table.sum()
        _sum_cells = _comb(_table, 2).sum()
        _sum_a = _comb(_table.sum(axis=1), 2).sum()
        _sum_b = _comb(_table.sum(axis=0), 2).sum()
        _expected = _sum_a * _sum_b / _comb(_n, 2)
        _max_index = 0.5 * (_sum_a + _sum_b)
        if _max_index == _expected:
            return 1.0
        return float((_sum_cells - _expected) / (_max_index - _expected))

    def _normalized_mutual_info(labels_a, labels_b):
        _table = _pd.crosstab(labels_a, labels_b).to_numpy(dtype=float)
        _joint = _table / _table.sum()
        _pa = _joint.sum(axis=1)
        _pb = _joint.sum(axis=0)
        _expected = _np.outer(_pa, _pb)
        _nonzero = _joint > 0
        _mutual = float((_joint[_nonzero] * _np.log(_joint[_nonzero] / _expected[_nonzero])).sum())
        _entropy_a = float(-(_pa[_pa > 0] * _np.log(_pa[_pa > 0])).sum())
        _entropy_b = float(-(_pb[_pb > 0] * _np.log(_pb[_pb > 0])).sum())
        if _entropy_a == 0 or _entropy_b == 0:
            return 0.0
        return _mutual / _np.sqrt(_entropy_a * _entropy_b)

    conformational_labels = build_conformational_partition(float(rmsd_cut.value))
    crystal_form_labels = build_crystal_form_partition(float(cell_cut.value))

    _ari = adjusted_rand_index(crystal_form_labels, conformational_labels)
    _nmi = _normalized_mutual_info(crystal_form_labels, conformational_labels)

    mo.md(
        f"### Partition agreement — RMSD ≤ {float(rmsd_cut.value):g} Å, "
        f"cell ≤ {float(cell_cut.value):g} %\n\n"
        f"- **Crystal-form subgroups** (SG × within-SG cell cut): "
        f"**{crystal_form_labels.nunique()}**\n"
        f"- **Conformational subgroups** (RMSD cut): "
        f"**{conformational_labels.nunique()}**\n"
        f"- **Adjusted Rand Index**: **{_ari:.3f}**  (0 = chance, 1 = identical)\n"
        f"- **Normalized mutual information**: {_nmi:.3f}\n"
    )
    return (
        adjusted_rand_index,
        build_conformational_partition,
        build_crystal_form_partition,
        conformational_labels,
        crystal_form_labels,
    )


@app.cell
def _(conformational_labels, crystal_form_labels, min_subgroup_size, mo, plt):
    import pandas as _pd
    import seaborn as _sns

    _min_size = int(min_subgroup_size.value)
    _crystal_sizes = crystal_form_labels.value_counts()
    _conf_sizes = conformational_labels.value_counts()
    _crystal_keep = [_c for _c, _n in _crystal_sizes.items() if _n >= _min_size]
    _conf_keep = [_c for _c, _n in _conf_sizes.items() if _n >= _min_size]

    mo.stop(
        not _crystal_keep or not _conf_keep,
        mo.md(f"*No subgroup reaches {_min_size} members on both axes — lower the minimum size.*"),
    )

    _table = (
        _pd.crosstab(crystal_form_labels, conformational_labels)
        .reindex(index=_crystal_keep, columns=_conf_keep)
        .fillna(0)
    )
    _jaccard = _table.astype(float).copy()
    for _row in _table.index:
        for _col in _table.columns:
            _shared = _table.loc[_row, _col]
            _union = _crystal_sizes[_row] + _conf_sizes[_col] - _shared
            _jaccard.loc[_row, _col] = _shared / _union if _union else 0.0

    _fig, _axes = plt.subplots(
        1, 2,
        figsize=(max(9, 0.6 * len(_conf_keep) + 5), max(4, 0.4 * len(_crystal_keep) + 2)),
    )
    _sns.heatmap(
        _table.astype(int), ax=_axes[0], cmap="Blues", annot=True, fmt="d",
        cbar_kws={"label": "shared structures"}, linewidths=0.5, linecolor="white",
    )
    _axes[0].set_title(f"Contingency (subgroups ≥ {_min_size})")
    _axes[0].set_xlabel("conformational subgroup")
    _axes[0].set_ylabel("crystal-form subgroup")
    _sns.heatmap(
        _jaccard, ax=_axes[1], cmap="rocket", vmin=0, vmax=1, annot=True, fmt=".2f",
        cbar_kws={"label": "Jaccard"}, linewidths=0.5, linecolor="white",
    )
    _axes[1].set_title("Jaccard overlap")
    _axes[1].set_xlabel("conformational subgroup")
    _axes[1].set_ylabel("")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    adjusted_rand_index,
    build_conformational_partition,
    build_crystal_form_partition,
    plt,
):
    import numpy as _np
    import pandas as _pd
    import seaborn as _sns

    _rmsd_grid = _np.round(_np.arange(0.5, 1.001, 0.1), 2)
    _cell_grid = _np.round(_np.arange(2.0, 5.001, 0.5), 2)

    _conf_by = {float(_r): build_conformational_partition(float(_r)) for _r in _rmsd_grid}
    _cryst_by = {float(_c): build_crystal_form_partition(float(_c)) for _c in _cell_grid}

    _ari = _pd.DataFrame(index=_cell_grid, columns=_rmsd_grid, dtype=float)
    for _c in _cell_grid:
        for _r in _rmsd_grid:
            _ari.loc[_c, _r] = adjusted_rand_index(_cryst_by[float(_c)], _conf_by[float(_r)])

    _fig, _ax = plt.subplots(figsize=(7, 5))
    _sns.heatmap(
        _ari.astype(float), ax=_ax, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f",
        cbar_kws={"label": "Adjusted Rand Index"}, linewidths=0.5, linecolor="white",
    )
    _ax.set_xlabel("Cα RMSD cut (Å)")
    _ax.set_ylabel("within-SG cell-diff cut (%)")
    _ax.set_title("Partition agreement (ARI) across the cutoff ranges")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8 — Partition balance (volcano feasibility)

    Type a boolean expression over the metadata columns. **Arm A** = rows that
    match; **Arm B** = everyone else. A volcano comparing water-cluster
    conservedness between the two arms is only worth running when *both* arms
    are reasonably large — this cell just reports the split sizes so you can
    find a balanced partition.

    Uses `pandas.DataFrame.query` syntax: `and`, `or`, `not`, parentheses, and
    comparisons all work, e.g. `resolution >= 1.8 and r_free <= 0.25`. Rows with
    a `NaN` in a referenced column (missing resolution/R-free) never match the
    query, so they fall into Arm B.
    """)
    return


@app.cell
def _(df, mo):
    _numeric = [
        c for c in ("resolution", "r_work", "r_free", "num_water", "unit_cell_volume")
        if c in df.columns
    ]
    _ranges = "\n".join(
        f"- `{c}`: {df[c].min():.3g} – {df[c].max():.3g}"
        for c in _numeric
        if df[c].notna().any()
    )
    query_input = mo.ui.text(
        value="resolution >= 1.8 and r_free <= 0.25",
        placeholder="resolution >= 1.8 and r_free <= 0.25",
        label="Arm A query",
        full_width=True,
    )
    mo.vstack([
        query_input,
        mo.md(f"**Columns:** `{'`, `'.join(df.columns)}`\n\n**Numeric ranges:**\n{_ranges}"),
    ])
    return (query_input,)


@app.cell
def _(df, mo, query_input):
    _expr = query_input.value.strip()
    mo.stop(not _expr, mo.md("*Enter a query above to see the split.*"))

    try:
        _matches = df.query(_expr)
    except Exception as _exc:
        mo.stop(True, mo.md(f"**Invalid query:** `{type(_exc).__name__}: {_exc}`"))

    n_a = len(_matches)
    n_b = len(df) - n_a

    mo.md(
        f"**Arm A (matches):** {n_a} ({n_a / len(df):.0%}) &nbsp;|&nbsp; "
        f"**Arm B (rest):** {n_b} ({n_b / len(df):.0%})\n\n"
        f"Query: `{_expr}`"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
