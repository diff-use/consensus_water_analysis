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
    # Starting-model lineage forest

    For one cohort, trace where each structure came from: its refinement
    starting model, then that model's starting model, and so on until the chain
    stops (`<missing>`, free text with no PDB code, an ambiguous / conflicting
    parent, or a cycle). A node keeps at most one parent, so every lineage is a
    single-rooted tree; the cohort as a whole is a *forest*.

    - **First hop** for each cohort member comes from its `metadata.csv`
      `starting_model` column — no network.
    - **Out-of-cohort ancestors** are resolved on demand via the RCSB Data API
      (each fetched at most once). Tracing is gated behind a button so it only
      hits the network when you ask it to.

    The tree-building logic lives in `cw.lineage`; everything that draws the
    figure is defined in this notebook.
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path

    import config

    _default = str(Path(config.DATA_DIR) / "hewls_65" / "metadata.csv")
    csv_input = mo.ui.text(
        value=_default,
        placeholder="/path/to/<cohort>/metadata.csv",
        label="metadata.csv",
        full_width=True,
    )
    csv_input
    return Path, csv_input


@app.cell
def _(Path, csv_input, mo):
    import pandas as pd

    from cw.lineage import parse_codes

    _csv = Path(csv_input.value.strip())
    mo.stop(not _csv.exists(), mo.md(f"**metadata.csv not found:** `{_csv}`"))

    meta = pd.read_csv(_csv)
    meta["pdb_id"] = meta["pdb_id"].astype(str).str.lower()
    _starting = dict(zip(meta["pdb_id"], meta["starting_model"].astype(str)))

    # First hop per cohort member, straight from the CSV (no network). The cohort
    # is exactly the set of structures with a metadata row.
    cohort_first_hop = {pid: parse_codes(_starting[pid]) for pid in meta["pdb_id"]}
    _n_parent = sum(1 for _codes, _status in cohort_first_hop.values() if _status == "ok")
    mo.md(
        f"Cohort **{_csv.parent.name}** — {len(cohort_first_hop)} members, "
        f"**{_n_parent}** with a valid first-hop parent."
    )
    return (cohort_first_hop,)


@app.cell
def _(mo):
    mo.md("""
    ## 1 — Load or trace the forest

    If a saved `lineage.csv` already exists at the path below, it is loaded
    directly — no network. Otherwise click **Trace** to follow every member's
    parent chain out of the cohort via the RCSB Data API (the only network step);
    save it below to skip the lookups next time.
    """)
    return


@app.cell
def _(Path, csv_input, mo):
    lineage_path_input = mo.ui.text(
        value=str(Path(csv_input.value.strip()).parent / "lineage.csv"),
        placeholder="/path/to/<cohort>/lineage.csv",
        label="lineage.csv path (loaded if present, else written here)",
        full_width=True,
    )
    trace_button = mo.ui.run_button(label="Trace lineages (RCSB lookups)")
    mo.vstack([lineage_path_input, trace_button])
    return lineage_path_input, trace_button


@app.cell
def _(Path, cohort_first_hop, lineage_path_input, mo, trace_button):
    import pandas

    from cw.lineage import to_dataframe, trace_forest
    from cw.metadata import fetch_starting_model

    _path = Path(lineage_path_input.value.strip())
    if _path.exists():
        lineage_df = pandas.read_csv(_path)
        lineage_df["parent"] = lineage_df["parent"].fillna("")
        _status = mo.md(f"Loaded saved lineage from `{_path}`.")
    else:
        mo.stop(
            not trace_button.value,
            mo.md(f"No saved `{_path.name}` yet — click **Trace lineages** to build it."),
        )
        with mo.status.spinner(title="Tracing lineages via RCSB…"):
            _nodes = trace_forest(cohort_first_hop, fetch_starting_model)
            lineage_df = to_dataframe(_nodes)
        _status = mo.md("Traced via RCSB — save it below to skip the lookups next time.")
    _status
    return (lineage_df,)


@app.cell
def _(lineage_df, mo):
    _sizes = lineage_df.groupby("lineage_id").size()
    _n_external = int((lineage_df["in_cohort"] == 0).sum())
    mo.md(
        f"**{len(lineage_df)}** nodes ({_n_external} out-of-cohort ancestors) in "
        f"**{_sizes.size}** lineages "
        f"({int((_sizes >= 2).sum())} multi-node, {int((_sizes == 1).sum())} singletons)."
    )
    return


@app.cell
def _(lineage_df):
    lineage_df
    return


@app.cell
def _(mo):
    mo.md("""
    ### Optionally save `lineage.csv`

    Persist the lineage table to the path set above so the next run loads it
    instead of hitting RCSB. Writes only on click.
    """)
    return


@app.cell
def _(mo):
    write_csv_button = mo.ui.run_button(label="Write lineage.csv")
    write_csv_button
    return (write_csv_button,)


@app.cell
def _(Path, lineage_df, lineage_path_input, mo, write_csv_button):
    mo.stop(not write_csv_button.value, mo.md("*Click the button above to write the CSV.*"))
    _out = Path(lineage_path_input.value.strip())
    _out.parent.mkdir(parents=True, exist_ok=True)
    lineage_df.to_csv(_out, index=False)
    mo.callout(mo.md(f"Wrote **{len(lineage_df)}** rows → `{_out}`"), kind="success")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 — Draw the forest

    Blue = cohort node, grey = out-of-cohort ancestor. Every node is the same
    size; a wide or deep tree spans more grid columns / a taller row instead of
    shrinking. Multi-node lineages each get a panel; single-node cohort lineages
    (invalid starting model, no descendants) collapse into one shared panel —
    unless there are more than *max singletons* of them, in which case they are
    omitted. Highlight PDB IDs (space- or comma-separated) to give them a red
    node edge.
    """)
    return


@app.cell
def _(mo):
    max_singletons_slider = mo.ui.slider(
        start=0, stop=100, step=5, value=20, label="Max singletons to show", show_value=True
    )
    highlight_input = mo.ui.text(
        value="", #9lmk, 8dyz, 9rvi, 9i6p, 6ybf
        placeholder="e.g. 3k34 1cil",
        label="Highlight PDB IDs",
        full_width=True,
    )
    mo.vstack([max_singletons_slider, highlight_input])
    return highlight_input, max_singletons_slider


@app.function
# Figure-specific drawing. Lives in the notebook, not cw/. Nested helpers keep it
# self-contained (no cross-cell namespace); the keyword args expose the knobs most
# worth tweaking — colors, node size, and packing width.
def make_forest_figure(
    df,
    *,
    max_singletons=20,
    highlight=(),
    in_cohort_color="#2b7bba",
    external_color="#cccccc",
    highlight_color="red",
    node_size=480.0,
    node_font=6.0,
    n_cols=6,
):
    """Render every lineage tree in one figure.

    Blue = cohort node, grey = out-of-cohort ancestor. Every node is the same
    size; a wide or deep tree spans more grid columns / a taller row instead of
    shrinking. Multi-node lineages each get a panel; single-node cohort lineages
    (invalid starting model, no descendants) collapse into one shared panel —
    unless there are more than ``max_singletons`` of them, in which case they
    are omitted. ``n_cols`` is the base-column grid the panels pack into (a small
    tree takes one narrow column, a wide tree spans several).

    Returns ``(fig, notes)`` where ``notes`` is a list of human-readable strings
    (omitted singletons, unusable highlight IDs) for the caller to display.
    """
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Panel geometry, in inches per data-unit, so node spacing stays constant
    # across panels while each row is only as tall as its deepest tree.
    col_width, scale_h, scale_v = 2.0, 0.5, 0.5
    pad, title_in, legend_in = 0.5, 0.35, 0.5

    def build_children(rows):
        children = defaultdict(list)
        for pid, info in rows.items():
            if info["parent"]:
                children[info["parent"]].append(pid)
        for k in children:
            children[k].sort()
        return children

    def layout_tree(roots, children):
        """Hierarchical layout: leaves get sequential x; a parent is centered
        over its children; y = -depth."""
        pos = {}
        next_x = [0.0]

        def place(node, depth):
            kids = children.get(node, [])
            if not kids:
                x = next_x[0]
                next_x[0] += 1.0
                pos[node] = (x, -float(depth))
                return x
            kid_xs = [place(k, depth + 1) for k in kids]
            x = sum(kid_xs) / len(kid_xs)
            pos[node] = (x, -float(depth))
            return x

        for root in roots:
            place(root, 0)
            next_x[0] += 0.5
        return pos

    def tree_layout(members, rows):
        member_set = set(members)
        all_children = build_children(rows)
        children = {
            n: [k for k in all_children.get(n, []) if k in member_set] for n in members
        }
        roots = sorted(
            m for m in members if not rows[m]["parent"] or rows[m]["parent"] not in member_set
        ) or sorted(members)
        return layout_tree(roots, children)

    def spans(pos):
        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1

    def panel_grid(w_span, h_span):
        """(column-span, row-height-in-inches) for a panel of the given extent."""
        col_span = int(np.clip(np.ceil((w_span + 2 * pad) * scale_h / col_width), 1, n_cols))
        height_in = (h_span + 2 * pad) * scale_v + title_in
        return col_span, height_in

    def strip_axes(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def draw_nodes(ax, pos, rows, title, highlight):
        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        colors = [in_cohort_color if rows[n]["in_cohort"] else external_color for n in pos]
        edges = [highlight_color if n in highlight else "black" for n in pos]
        widths = [1.8 if n in highlight else 0.6 for n in pos]
        ax.scatter(xs, ys, s=node_size, c=colors, edgecolors=edges, linewidths=widths, zorder=2)
        for node, (x, y) in pos.items():
            ax.text(x, y, node, ha="center", va="center", fontsize=node_font, zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        strip_axes(ax)

    def draw_tree(ax, lineage_id, members, rows, pos, highlight):
        for child in members:
            parent = rows[child]["parent"]
            if parent in pos and child in pos:
                (x1, y1), (x2, y2) = pos[parent], pos[child]
                ax.plot([x1, x2], [y1, y2], "-", color="black", lw=0.7, zorder=1)
        in_coh = sum(1 for m in members if rows[m]["in_cohort"])
        draw_nodes(ax, pos, rows, f"{lineage_id}  ({in_coh}/{len(members)})", highlight)

    def singleton_layout(members):
        ncol = max(1, int(np.ceil(np.sqrt(len(members)))))
        return {pid: (i % ncol, -(i // ncol)) for i, pid in enumerate(members)}

    notes = []
    rows = {}
    lineages = defaultdict(list)
    for r in df.itertuples(index=False):
        parent = "" if pd.isna(r.parent) else str(r.parent)
        rows[r.pdb_id] = {"parent": parent, "in_cohort": bool(r.in_cohort)}
        lineages[r.lineage_id].append(r.pdb_id)

    trees = sorted(
        (L for L, m in lineages.items() if len(m) >= 2),
        key=lambda L: (-sum(rows[p]["in_cohort"] for p in lineages[L]), L),
    )
    singletons = sorted(m[0] for L, m in lineages.items() if len(m) == 1)

    show_singletons = 0 < len(singletons) <= max_singletons
    if len(singletons) > max_singletons:
        notes.append(f"omitted {len(singletons)} singleton cohort PDBs (> max singletons)")

    highlight = {str(h).lower() for h in highlight}
    if highlight:
        drawn = {m for L in trees for m in lineages[L]}
        if show_singletons:
            drawn.update(singletons)
        absent = sorted(highlight - set(rows))
        hidden = sorted(highlight & set(singletons) - drawn)
        if absent:
            notes.append(f"highlight not in the forest, ignored: {', '.join(absent)}")
        if hidden:
            notes.append(f"highlight singleton(s) not shown: {', '.join(hidden)}")

    # Build a draw closure + grid footprint (column-span, row-height) per panel.
    panels = []
    for lineage_id in trees:
        members = lineages[lineage_id]
        pos = tree_layout(members, rows)
        col_span, height_in = panel_grid(*spans(pos))
        panels.append(
            (
                lambda ax, L=lineage_id, m=members, p=pos: draw_tree(
                    ax, L, m, rows, p, highlight
                ),
                col_span,
                height_in,
            )
        )
    if show_singletons:
        pos = singleton_layout(singletons)
        col_span, height_in = panel_grid(*spans(pos))
        panels.append(
            (
                lambda ax, m=singletons, p=pos: draw_nodes(
                    ax, p, rows, f"singletons ({len(m)})", highlight
                ),
                col_span,
                height_in,
            )
        )
    if not panels:
        raise ValueError("no lineages to plot")

    # First-fit pack panels into rows of n_cols base columns; each row is only
    # as tall as its tallest panel, so several small lineages share a short row.
    placements = []  # (draw, row, col, col_span)
    row_heights = []
    row, col = 0, 0
    for draw, col_span, height_in in panels:
        if col + col_span > n_cols:
            row += 1
            col = 0
        if row >= len(row_heights):
            row_heights.append(0.0)
        row_heights[row] = max(row_heights[row], height_in)
        placements.append((draw, row, col, col_span))
        col += col_span

    n_rows = len(row_heights)
    fig_h = sum(row_heights) + legend_in
    fig = plt.figure(figsize=(n_cols * col_width, fig_h))
    gridspec = fig.add_gridspec(
        n_rows,
        n_cols,
        height_ratios=row_heights,
        top=1.0 - 0.2 / fig_h,
        bottom=legend_in / fig_h,
        left=0.01,
        right=0.99,
        hspace=0.45,
        wspace=0.15,
    )
    for draw, r, c, col_span in placements:
        draw(fig.add_subplot(gridspec[r, c : c + col_span]))

    fig.legend(
        handles=[
            plt.Line2D(
                [], [], marker="o", linestyle="", markerfacecolor=in_cohort_color,
                markeredgecolor="black", markersize=10, label="in cohort",
            ),
            plt.Line2D(
                [], [], marker="o", linestyle="", markerfacecolor=external_color,
                markeredgecolor="black", markersize=10, label="external (ancestor only)",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.5 * legend_in / fig_h),
    )
    return fig, notes


@app.cell
def _(highlight_input, lineage_df, max_singletons_slider, mo):
    _highlight = highlight_input.value.replace(",", " ").split()
    # Other knobs — in_cohort_color, external_color, highlight_color, node_size,
    # node_font, n_cols — can be passed here to restyle the figure.
    forest_fig, _notes = make_forest_figure(
        lineage_df, max_singletons=max_singletons_slider.value, highlight=_highlight
    )
    _msg = mo.md("\n".join(f"- {n}" for n in _notes)) if _notes else None
    mo.vstack([m for m in (_msg, forest_fig) if m is not None])
    return (forest_fig,)


@app.cell
def _(mo):
    mo.md("""
    ### Optionally save the figure

    Default is `<cohort>/plots/lineage.png`. Writes only on click.
    """)
    return


@app.cell
def _(Path, csv_input, mo):
    png_out_input = mo.ui.text(
        value=str(Path(csv_input.value.strip()).parent / "plots" / "lineage.png"),
        placeholder="/path/to/<cohort>/plots/lineage.png",
        label="Figure output path",
        full_width=True,
    )
    save_png_button = mo.ui.run_button(label="Save figure")
    mo.vstack([png_out_input, save_png_button])
    return png_out_input, save_png_button


@app.cell
def _(Path, forest_fig, mo, png_out_input, save_png_button):
    mo.stop(not save_png_button.value, mo.md("*Click the button above to save the figure.*"))
    _out = Path(png_out_input.value.strip())
    _out.parent.mkdir(parents=True, exist_ok=True)
    forest_fig.savefig(_out, dpi=150, bbox_inches="tight")
    mo.callout(mo.md(f"Saved figure → `{_out}`"), kind="success")
    return


if __name__ == "__main__":
    app.run()
