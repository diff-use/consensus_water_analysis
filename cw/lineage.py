"""Starting-model lineage forest — tree-building only.

Traces, for each cohort PDB, its refinement starting model, then that model's
starting model, and so on until a terminal result (``<missing>``, free text with
no PDB code, a conflict / ambiguous parent, or a cycle) — following ancestors out
of the cohort via the RCSB Data API. A node keeps at most one parent (ambiguous
parents are terminal), so every lineage is a single-rooted tree.

Pure functions only: heavy work (RCSB lookups) is injected as a ``fetch`` callable
by the caller. This module produces the lineage *table* (``to_dataframe``);
rendering the forest lives in ``notebooks/optional_starting_model_lineage.py``.

Ports:
  - ``_union_find_lineages`` → porting_reference/lineage/build_lineage_graph.py
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import pandas as pd

from cw.metadata import _pdb_codes

# Follow the parent chain only when the starting model resolves to exactly one
# unambiguous PDB code.
_FOLLOW_STATUS = "ok"


def parse_codes(text: str) -> tuple[list[str], str]:
    """``(codes, status)`` from a metadata.csv ``starting_model`` string.

    Mirrors ``cw.metadata.resolve_starting_model`` so a cohort member's first hop
    needs no network. From the CSV alone a field-disagreement conflict is
    indistinguishable from a legitimately multi-parent value, so both collapse to
    ``'conflict'`` — either way the chain stops (ambiguous parent is terminal).
    """
    if not text or str(text).strip() in ("", "<missing>"):
        return [], "missing"
    codes = _pdb_codes([str(text)])
    if not codes:
        return [], "text"
    if len(codes) > 1:
        return codes, "conflict"
    return codes, "ok"


def trace_forest(
    cohort_first_hop: dict[str, tuple[list[str], str]],
    fetch: Callable[[str], tuple[list[str], str]],
) -> dict[str, dict]:
    """Trace every cohort PDB upward until terminal.

    ``cohort_first_hop`` maps each cohort PDB (lowercase) to its ``(codes, status)``
    parsed from metadata.csv — used for the first hop so cohort members are never
    re-fetched. ``fetch(pdb_id) -> (codes, status)`` resolves out-of-cohort
    ancestors (RCSB); each ancestor is fetched at most once.

    Returns ``{pdb_id: {"in_cohort": bool, "parent": str | None,
    "terminal_reason": str | None}}``. ``parent`` is the followed PDB code (or
    None); ``terminal_reason`` is set only on nodes whose chain stops here, one of
    ``'ok'`` unused, ``'missing'`` | ``'text'`` | ``'conflict'`` | ``'cycle'``.
    """
    nodes: dict[str, dict] = {}
    cache: dict[str, tuple[list[str], str]] = {}

    def resolve(pdb_id: str, first_hop: bool) -> tuple[list[str], str]:
        if first_hop and pdb_id in cohort_first_hop:
            return cohort_first_hop[pdb_id]
        if pdb_id not in cache:
            cache[pdb_id] = fetch(pdb_id)
        return cache[pdb_id]

    for seed in cohort_first_hop:
        path: list[str] = []
        current = seed
        first_hop = True
        while True:
            node = nodes.get(current)
            if node is None:
                node = {
                    "in_cohort": current in cohort_first_hop,
                    "parent": None,
                    "terminal_reason": None,
                }
                nodes[current] = node
            if current in path:  # cycle within this walk
                node["terminal_reason"] = "cycle"
                break
            path.append(current)
            if node["parent"] is not None:  # already traced onward
                current = node["parent"]
                first_hop = False
                continue
            if node["terminal_reason"] is not None:  # already known terminal
                break
            codes, status = resolve(current, first_hop)
            if status == _FOLLOW_STATUS and codes[0] != current:
                node["parent"] = codes[0]
                current = codes[0]
                first_hop = False
            elif status == _FOLLOW_STATUS:  # self-referential
                node["terminal_reason"] = "cycle"
                break
            else:
                node["terminal_reason"] = status
                break
    return nodes


def _union_find_lineages(nodes: dict[str, dict]) -> dict[str, list[str]]:
    """Connected components over child→parent edges. Ported from build_lineage_graph.py."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pid in nodes:
        parent.setdefault(pid, pid)
    for pid, node in nodes.items():
        if node["parent"]:
            parent.setdefault(node["parent"], node["parent"])
            union(pid, node["parent"])

    components: dict[str, list[str]] = defaultdict(list)
    for n in parent:
        components[find(n)].append(n)
    return components


def to_dataframe(nodes: dict[str, dict]) -> pd.DataFrame:
    """Assemble the lineage table (one row per node) with stable lineage IDs.

    Columns: ``pdb_id, in_cohort, parent, terminal_reason, lineage_id,
    lineage_size_total, lineage_size_in_cohort``.
    """
    components = _union_find_lineages(nodes)

    def sort_key(item: tuple[str, list[str]]) -> tuple:
        members = item[1]
        in_coh = sum(1 for m in members if nodes[m]["in_cohort"])
        return (-in_coh, -len(members), min(members))

    node_to_lineage: dict[str, str] = {}
    lineage_size: dict[str, tuple[int, int]] = {}
    for i, (_root, members) in enumerate(sorted(components.items(), key=sort_key)):
        lineage_id = f"L{i:03d}"
        in_coll = sum(1 for m in members if nodes[m]["in_cohort"])
        for m in members:
            node_to_lineage[m] = lineage_id
        lineage_size[lineage_id] = (len(members), in_coll)

    records = []
    for pid in sorted(nodes):
        lineage_id = node_to_lineage[pid]
        size_total, size_in_cohort = lineage_size[lineage_id]
        records.append(
            {
                "pdb_id": pid,
                "in_cohort": int(nodes[pid]["in_cohort"]),
                "parent": nodes[pid]["parent"] or "",
                "terminal_reason": nodes[pid]["terminal_reason"] or "",
                "lineage_id": lineage_id,
                "lineage_size_total": size_total,
                "lineage_size_in_cohort": size_in_cohort,
            }
        )
    return pd.DataFrame.from_records(records)
