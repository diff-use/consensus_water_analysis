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
    # Metadata volcano — cluster conservedness vs a metadata column

    Pick one numeric column; it drives the histogram and **two** volcanoes that
    relate each water cluster's conservedness to that column on its **continuous**
    scale (no threshold split). Per cluster the unit is per-structure binary
    membership — does each structure in the clustering universe contribute a
    within-cutoff water (1) or not (0)?

    - **B — point-biserial r**: `pointbiserialr(member, value)`. Scale-free,
      bounded −1..+1. With a fixed universe size r and its p are nearly locked,
      so this plot is close to a single curve.
    - **A — logistic β**: `logit P(member) = β₀ + β₁·z(value)`; β₁ = log-odds per
      1 SD. β and its Wald SE decouple, so effect and significance separate (at
      the cost of separation artifacts on sparse clusters).

    Both use Benjamini-Hochberg FDR and a two-gate hit call (effect size **and**
    `q`); the final cell cross-references the two. Column choices include the
    alignment RMSD metrics (merged from the report) and `control_random`, a
    deterministic per-structure value independent of biology — a leak-free
    pipeline should find no hits against it.
    """)
    return


@app.cell
def _():
    import hashlib
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import norm, pointbiserialr
    from sklearn.linear_model import LogisticRegression

    import config

    return (
        LogisticRegression,
        Path,
        config,
        hashlib,
        norm,
        np,
        pd,
        plt,
        pointbiserialr,
    )


@app.cell
def _(Path, config, mo):
    _iso = Path(config.DATA_DIR) / "carbonicanhydrase_000562_iso"
    members_input = mo.ui.text(
        value=str(_iso / "cluster_members.csv"),
        label="cluster_members.csv",
        full_width=True,
    )
    metadata_input = mo.ui.text(
        value=str(Path(config.DATA_DIR) / "carbonicanhydrase_000562" / "metadata.csv"),
        label="metadata.csv",
        full_width=True,
    )
    mo.vstack([members_input, metadata_input])
    return members_input, metadata_input


@app.cell
def _(Path, hashlib, members_input, metadata_input, mo, pd):
    _mp = Path(members_input.value.strip())
    _dp = Path(metadata_input.value.strip())
    mo.stop(not _mp.exists(), mo.md(f"**cluster_members.csv not found:** `{_mp}`"))
    mo.stop(not _dp.exists(), mo.md(f"**metadata.csv not found:** `{_dp}`"))

    members = pd.read_csv(_mp)
    meta = pd.read_csv(_dp)
    # Normalise the join key on both sides (filenames may differ in case).
    members["pdb_id"] = members["pdb_id"].astype(str).str.lower()
    meta["pdb_id"] = meta["pdb_id"].astype(str).str.lower()

    # Merge alignment RMSD (n_common_ca / rmsd_before / rmsd_after) from the report
    # next to the members CSV so they're correlatable metadata columns too.
    _reports = sorted(
        (Path(members_input.value.strip()).parent / "aligned_pdbs").glob(
            "alignment_report_*.csv"
        )
    )
    if _reports:
        _rep = pd.read_csv(_reports[0])
        _rep["pdb_id"] = _rep["pdb_id"].astype(str).str.lower()
        meta = meta.merge(
            _rep[["pdb_id", "n_common_ca", "rmsd_before", "rmsd_after"]],
            on="pdb_id",
            how="left",
        )

    # Synthetic negative control: a deterministic per-structure value in [0,1),
    # independent of biology. A leak-free pipeline finds no hits against it.
    meta["control_random"] = meta["pdb_id"].map(
        lambda s: int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    )
    mo.md(
        f"Loaded **{len(members):,}** member rows and **{len(meta)}** metadata rows "
        f"(metadata may cover more structures than were clustered)."
    )
    return members, meta


@app.cell
def _(np, plt):
    # Shared helpers — defined once, used by both approaches.
    def bh(pvals):
        p = np.asarray(pvals, dtype=float)
        m = len(p)
        order = np.argsort(p)
        q = p[order] * m / np.arange(1, m + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity from the top
        out = np.empty(m)
        out[order] = np.clip(q, 0, 1)
        return out

    def score(stats, effect_col, q_cut, effect_cut):
        sig = stats["q"] < q_cut
        big = stats[effect_col].abs() >= effect_cut
        out = stats.copy()
        out["category"] = np.select(
            [sig & big, sig & ~big],
            ["hit", "significant, small effect"],
            default="ns",
        )
        out["direction"] = np.where(
            stats[effect_col] >= 0, "rises_with_value", "falls_with_value"
        )
        return out

    def volcano(scored, effect_col, xlabel, title, q_cut, effect_cut, xlim=None,
                color_values=None, color_label=None, cmap="viridis"):
        y = -np.log10(scored["q"])
        # Continuous colouring widens the figure to make room for the colorbar.
        fig, ax = plt.subplots(figsize=(3.8, 3) if color_values is not None else (3, 3))
        if color_values is None:
            palette = {
                "ns": "lightgray",
                "significant, small effect": "steelblue",
                "hit": "orange",
            }
            for cat, color in palette.items():
                mask = scored["category"] == cat
                ax.scatter(
                    scored.loc[mask, effect_col], y[mask], s=20, alpha=0.6,
                    color=color, edgecolor="k" if cat == "hit" else "none",
                )
        else:
            sc = ax.scatter(
                scored[effect_col], y, s=20, alpha=0.6,
                c=np.asarray(color_values, dtype=float), cmap=cmap, edgecolor="none",
            )
            # Ring the hits so they stay legible under any continuous scale.
            _hit = scored["category"] == "hit"
            if _hit.any():
                ax.scatter(
                    scored.loc[_hit, effect_col], y[_hit], s=24,
                    facecolor="none", edgecolor="k", linewidths=0.8,
                )
            fig.colorbar(sc, ax=ax).set_label(color_label or "")
        ax.axhline(-np.log10(q_cut), ls="--", color="tomato", lw=1)
        ax.axvline(effect_cut, ls="--", color="0.5", lw=1)
        ax.axvline(-effect_cut, ls="--", color="0.5", lw=1)
        ax.axvline(0, color="0.85", lw=0.6)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("−log10(q)")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    return bh, score, volcano


@app.cell
def _(mo):
    color_dd = mo.ui.dropdown(
        options=["category", "cluster_occupancy", "median edia", "median b_factor"],
        value="category",
        label="Colour volcano points by",
    )
    color_dd
    return (color_dd,)


@app.cell
def _(members):
    # Per-cluster median water metrics, keyed by cluster_id, mirroring the cluster
    # analysis notebook (within-cutoff members of real clusters only).
    _assigned = members[(members["within_cutoff"]) & (members["cluster_id"] >= 0)]
    cluster_metrics = _assigned.groupby("cluster_id")[["edia", "b_factor"]].median()
    return (cluster_metrics,)


@app.cell
def _(cluster_metrics, color_dd):
    # Map the dropdown choice to (color array aligned to scored rows, label, cmap).
    # "category" returns None, which keeps volcano's gray/blue/orange palette.
    _median_specs = {
        "median edia": ("edia", "magma"),
        "median b_factor": ("b_factor", "cividis"),
    }

    def color_for(scored):
        choice = color_dd.value
        if choice == "category":
            return None, None, None
        if choice == "cluster_occupancy":
            return scored["occ"].to_numpy(), "cluster occupancy", "viridis"
        column, cmap = _median_specs[choice]
        values = scored["cluster_id"].map(cluster_metrics[column]).to_numpy(dtype=float)
        return values, f"median {column}", cmap

    return (color_for,)


@app.cell
def _(meta, mo, pd):
    # One column drives the histogram and both volcanoes; histogram scope is the
    # user's choice because metadata.csv may cover more structures than were
    # aligned / clustered.
    _numeric_cols = [
        c
        for c in meta.columns
        if c != "pdb_id" and pd.to_numeric(meta[c], errors="coerce").notna().any()
    ]
    col_dd = mo.ui.dropdown(
        options=_numeric_cols,
        value="resolution" if "resolution" in _numeric_cols else _numeric_cols[0],
        label="Metadata column",
    )
    hist_scope = mo.ui.radio(
        options=["clustering universe", "all metadata.csv"],
        value="clustering universe",
        label="Histogram scope",
    )
    mo.vstack([col_dd, hist_scope])
    return col_dd, hist_scope


@app.cell
def _(col_dd):
    col = col_dd.value
    return (col,)


@app.cell
def _(col, hist_scope, members, meta, mo, pd, plt):
    _series = pd.to_numeric(meta.set_index("pdb_id")[col], errors="coerce").dropna()
    if hist_scope.value == "clustering universe":
        _series = _series[_series.index.isin(set(members["pdb_id"]))]
    mo.stop(_series.empty, mo.md(f"*No `{col}` values for the selected scope.*"))

    # Clip the x-range to 1–99% so a few outliers don't squash the bulk.
    _lo, _hi = float(_series.quantile(0.01)), float(_series.quantile(0.99))
    if _lo == _hi:
        _lo, _hi = float(_series.min()), float(_series.max())

    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.hist(_series, bins=50, range=(_lo, _hi), color="steelblue")
    _ax.axvline(_series.median(), color="k", ls="--", lw=1)
    _ax.set_xlabel(col)
    _ax.set_ylabel("structures")
    _ax.set_title(f"{col} — {hist_scope.value} (n={len(_series)}, 1–99%)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(col, members, meta, mo, pd):
    # Shared membership over the clustering universe (structures with a value):
    # a clusters × structures 0/1 matrix both approaches iterate.
    _universe = pd.Index(sorted(set(members["pdb_id"])), name="pdb_id")
    _vals = (
        pd.to_numeric(meta.set_index("pdb_id")[col], errors="coerce")
        .reindex(_universe)
        .dropna()
    )
    mo.stop(_vals.empty, mo.md(f"*No `{col}` values over the clustering universe.*"))
    _structures = _vals.index
    metadata_values = _vals.to_numpy()

    _kept = members[(members["within_cutoff"]) & (members["cluster_id"] >= 0)]
    _kept = _kept[_kept["pdb_id"].isin(_structures)]
    member_matrix = (
        _kept.assign(_present=1)
        .pivot_table(index="cluster_id", columns="pdb_id", values="_present",
                     aggfunc="max", fill_value=0)
        .reindex(columns=_structures, fill_value=0)
    )
    mo.md(
        f"Clustering universe: **{len(_universe)}** structures, "
        f"**{len(_structures)}** with a `{col}` value; "
        f"**{member_matrix.shape[0]}** clusters."
    )
    return member_matrix, metadata_values


@app.cell
def _(bh, member_matrix, metadata_values, mo, pd, pointbiserialr):
    _n = member_matrix.shape[1]
    _rows, _skipped = [], 0
    for _cid, _row in member_matrix.iterrows():
        _member = _row.to_numpy()
        _n_present = int(_member.sum())
        # pointbiserialr is undefined for a constant membership vector.
        if _n_present == 0 or _n_present == _n:
            _skipped += 1
            continue
        _r, _p = pointbiserialr(_member, metadata_values)
        _rows.append({
            "cluster_id": int(_cid),
            "n_present": _n_present,
            "occ": _n_present / _n,
            "r": float(_r),
            "p": float(_p),
        })

    stats_corr = pd.DataFrame(_rows)
    if not stats_corr.empty:
        stats_corr["q"] = bh(stats_corr["p"].to_numpy())
    mo.md(
        f"### B — point-biserial r\n\nScored **{len(stats_corr)}** clusters "
        f"({_skipped} skipped: present in all or none)."
    )
    return (stats_corr,)


@app.cell
def _(mo):
    q_corr = mo.ui.number(
        value=0.05, start=1e-6, stop=1.0, step=0.005, label="q cutoff (FDR)"
    )
    effect_corr = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.3, label="|r| cutoff", show_value=True
    )
    mo.vstack([q_corr, effect_corr])
    return effect_corr, q_corr


@app.cell
def _(effect_corr, mo, q_corr, score, stats_corr):
    mo.stop(stats_corr.empty, mo.md("*No clusters to score.*"))
    scored_corr = score(stats_corr, "r", float(q_corr.value), float(effect_corr.value))
    _n_hit = int((scored_corr["category"] == "hit").sum())
    _n_small = int((scored_corr["category"] == "significant, small effect").sum())
    mo.md(
        f"**{_n_hit} hits** at q<{float(q_corr.value):g}, |r|≥{float(effect_corr.value):g} "
        f"— {_n_small} significant but sub-threshold."
    )
    return (scored_corr,)


@app.cell
def _(col, color_for, effect_corr, q_corr, scored_corr, volcano):
    _c, _clabel, _cmap = color_for(scored_corr)
    volcano(
        scored_corr, "r", "point-biserial r", f"{col} (point-biserial)",
        float(q_corr.value), float(effect_corr.value), xlim=(-1, 1),
        color_values=_c, color_label=_clabel, cmap=_cmap,
    )
    return


@app.cell
def _(mo):
    min_count = mo.ui.number(
        value=3, start=1, stop=50, step=1, label="min class count (separation filter)"
    )
    q_logit = mo.ui.number(
        value=0.05, start=1e-6, stop=1.0, step=0.005, label="q cutoff (FDR)"
    )
    effect_logit = mo.ui.slider(
        start=0.0, stop=3.0, step=0.05, value=1.5, label="|β| cutoff", show_value=True
    )
    mo.vstack([min_count, q_logit, effect_logit])
    return effect_logit, min_count, q_logit


@app.cell
def _(
    LogisticRegression,
    bh,
    member_matrix,
    metadata_values,
    min_count,
    mo,
    norm,
    np,
    pd,
):
    _std = metadata_values.std(ddof=0)
    mo.stop(_std == 0, mo.md("*Metadata column is constant — nothing to regress on.*"))
    _z = (metadata_values - metadata_values.mean()) / _std
    _feature = _z.reshape(-1, 1)
    _design = np.column_stack([np.ones(_z.size), _z])  # intercept + standardised value
    _n = member_matrix.shape[1]
    _mc = int(min_count.value)

    _rows, _skipped = [], 0
    for _cid, _row in member_matrix.iterrows():
        _y = _row.to_numpy()
        _n_present = int(_y.sum())
        if min(_n_present, _n - _n_present) < _mc:
            _skipped += 1
            continue
        _fit = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000).fit(_feature, _y)
        _beta = float(_fit.coef_[0, 0])
        _prob = _fit.predict_proba(_feature)[:, 1]
        _weight = _prob * (1.0 - _prob)
        try:
            _cov = np.linalg.inv(_design.T @ (_weight[:, None] * _design))
            _se = float(np.sqrt(_cov[1, 1]))
        except np.linalg.LinAlgError:
            _skipped += 1
            continue
        if not np.isfinite(_se) or _se == 0:
            _skipped += 1
            continue
        _rows.append({
            "cluster_id": int(_cid),
            "n_present": _n_present,
            "occ": _n_present / _n,
            "beta": _beta,
            "odds_ratio": float(np.exp(_beta)),
            "se": _se,
            "p": float(2.0 * norm.sf(abs(_beta / _se))),
        })

    stats_logit = pd.DataFrame(_rows)
    if not stats_logit.empty:
        stats_logit["q"] = bh(stats_logit["p"].to_numpy())
    mo.md(
        f"### A — logistic β\n\nScored **{len(stats_logit)}** clusters "
        f"({_skipped} skipped: <{_mc} per class, or singular fit)."
    )
    return (stats_logit,)


@app.cell
def _(effect_logit, mo, q_logit, score, stats_logit):
    mo.stop(stats_logit.empty, mo.md("*No clusters to score.*"))
    scored_logit = score(stats_logit, "beta", float(q_logit.value), float(effect_logit.value))
    _n_hit = int((scored_logit["category"] == "hit").sum())
    _n_small = int((scored_logit["category"] == "significant, small effect").sum())
    mo.md(
        f"**{_n_hit} hits** at q<{float(q_logit.value):g}, |β|≥{float(effect_logit.value):g} "
        f"— {_n_small} significant but sub-threshold."
    )
    return (scored_logit,)


@app.cell
def _(col, color_for, effect_logit, q_logit, scored_logit, volcano):
    _c, _clabel, _cmap = color_for(scored_logit)
    volcano(
        scored_logit, "beta", "logistic β (log-odds per 1 SD)", f"{col} (logistic β)",
        float(q_logit.value), float(effect_logit.value),
        color_values=_c, color_label=_clabel, cmap=_cmap,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Common hits across approaches

    Cross-reference the two hit calls. Pick the **hit set** — clusters called a
    *hit* in **both** approaches (the trustworthy intersection), in **either**, or
    in one alone — and how to rank them. Sort keys come from the approach you
    select: `q` ascending and/or `|effect|` descending (`|r|` for point-biserial,
    `|β|` for logistic). Requires both volcanoes above to have been scored.
    """)
    return


@app.cell
def _(mo):
    hit_set = mo.ui.radio(
        options=[
            "both approaches (intersection)",
            "either approach (union)",
            "point-biserial only",
            "logistic only",
        ],
        value="both approaches (intersection)",
        label="Hit set",
    )
    sort_order = mo.ui.dropdown(
        options=["q, then effect size", "effect size, then q", "q only", "effect size only"],
        value="q, then effect size",
        label="Sort by",
    )
    sort_source = mo.ui.radio(
        options=["point-biserial", "logistic"],
        value="point-biserial",
        label="Rank by this approach's q / |effect|",
    )
    mo.vstack([hit_set, sort_order, sort_source])
    return hit_set, sort_order, sort_source


@app.cell
def _(hit_set, mo, scored_corr, scored_logit, sort_order, sort_source):
    mo.stop(
        scored_corr.empty or scored_logit.empty,
        mo.md("*Need both approaches scored — run the volcano cells above first.*"),
    )

    _pb = scored_corr[["cluster_id", "n_present", "occ", "r", "q", "category", "direction"]].rename(
        columns={"q": "q_pb", "category": "cat_pb", "direction": "dir_pb"}
    )
    _lg = scored_logit[["cluster_id", "beta", "odds_ratio", "q", "category", "direction"]].rename(
        columns={"q": "q_logit", "category": "cat_logit", "direction": "dir_logit"}
    )
    # Logistic clusters are a subset of point-biserial's (its min-class filter is
    # stricter), so a left merge keeps n_present / occ on every row.
    _table = _pb.merge(_lg, on="cluster_id", how="left")

    _is_pb = _table["cat_pb"].eq("hit")
    _is_lg = _table["cat_logit"].eq("hit")
    _table = _table[
        {
            "both approaches (intersection)": _is_pb & _is_lg,
            "either approach (union)": _is_pb | _is_lg,
            "point-biserial only": _is_pb,
            "logistic only": _is_lg,
        }[hit_set.value]
    ].copy()

    _qcol = "q_pb" if sort_source.value == "point-biserial" else "q_logit"
    _table["abs_effect"] = (
        _table["r"] if sort_source.value == "point-biserial" else _table["beta"]
    ).abs()
    _by, _asc = {
        "q, then effect size": ([_qcol, "abs_effect"], [True, False]),
        "effect size, then q": (["abs_effect", _qcol], [False, True]),
        "q only": ([_qcol], [True]),
        "effect size only": (["abs_effect"], [False]),
    }[sort_order.value]
    _table = _table.sort_values(_by, ascending=_asc, na_position="last")

    _cols = [
        "cluster_id", "n_present", "occ",
        "r", "q_pb", "cat_pb", "dir_pb",
        "beta", "odds_ratio", "q_logit", "cat_logit", "dir_logit",
    ]
    mo.vstack([
        mo.md(
            f"**{len(_table)}** clusters — *{hit_set.value}*, sorted by "
            f"*{sort_order.value}* using *{sort_source.value}* metrics."
        ),
        _table[_cols].reset_index(drop=True),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Metadata pairplot

    Scatter-matrix of any numeric metadata columns (resolution, r_free,
    rmsd_after, `control_random`, …) to eyeball their pairwise relationships and
    marginal distributions. Scope is independent of the histogram above. Rows are
    dropped where any selected column is missing.
    """)
    return


@app.cell
def _(meta, mo, pd):
    _numeric_cols = [
        c
        for c in meta.columns
        if c != "pdb_id" and pd.to_numeric(meta[c], errors="coerce").notna().any()
    ]
    _default = [c for c in ["resolution", "r_free", "rmsd_after"] if c in _numeric_cols]
    pair_cols = mo.ui.multiselect(
        options=_numeric_cols,
        value=_default or _numeric_cols[:3],
        label="Pairplot columns",
    )
    pair_scope = mo.ui.radio(
        options=["clustering universe", "all metadata.csv"],
        value="clustering universe",
        label="Scope",
    )
    mo.vstack([pair_cols, pair_scope])
    return pair_cols, pair_scope


@app.cell
def _(members, meta, mo, np, pair_cols, pair_scope, pd):
    import seaborn as sns

    _cols = list(pair_cols.value)
    mo.stop(len(_cols) < 2, mo.md("*Pick at least two columns for a pairplot.*"))

    _df = meta.set_index("pdb_id")[_cols].apply(pd.to_numeric, errors="coerce")
    if pair_scope.value == "clustering universe":
        _df = _df[_df.index.isin(set(members["pdb_id"]))]
    _df = _df.dropna()
    mo.stop(_df.empty, mo.md("*No rows with all selected columns present.*"))

    # Rescale large-magnitude columns (e.g. unit cell volume) and fold the power
    # of 10 into the column name, so ticks stay short (1.28) without a floating
    # ×10^n offset that would collide with the panel above.
    _renamed = {}
    for _c in _cols:
        _peak = _df[_c].abs().max()
        if np.isfinite(_peak) and _peak >= 1e4:
            _exp = int(np.floor(np.log10(_peak)))
            _df[_c] = _df[_c] / 10 ** _exp
            _renamed[_c] = f"{_c} (×10^{_exp})"
    _df = _df.rename(columns=_renamed)

    _grid = sns.pairplot(_df, diag_kind="hist", plot_kws={"s": 10, "alpha": 0.3, "edgecolor":'k'}, corner=True)
    _grid.figure.suptitle(f"{pair_scope.value} (n={len(_df)})", y=1.02)
    _grid.figure
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
