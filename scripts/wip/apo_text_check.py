"""Soft-check an apo call against the RCSB free text for each entry.

The geometric apo/holo call (``apo_holo_*.py``) sees only what is *modelled*. A
soak that produced weak or disordered density is deposited with no ligand in the
coordinates and is therefore indistinguishable from a genuine apo crystal on
geometry alone — but the deposition free text usually still says so ("in complex
with", "soaked with", a compound name in the title). This script reads that text
and flags the mismatch.

It is deliberately a *soft* check: the output is per-structure evidence for human
review, not a reclassification. Nothing is dropped and no CSV the pipeline
consumes is modified.

Signals, per entry:
  - ligand language in title / keywords / citation title  → suspicious
  - apo language ("apo", "unbound", "ligand-free")        → reassuring
  - soak language in the crystallization details          → suspicious
  - RCSB ``nonpolymer_bound_components`` that are absent
    from the modelled het inventory                       → suspicious
  - deposited binding-affinity data on the entry          → suspicious

Verdict is the combination: ``review`` when ligand/soak evidence appears with no
apo language, ``apo-consistent`` when apo language is explicit, else ``neutral``.

NOTE ON THE NETWORK RULE: CLAUDE.md sanctions RCSB Data API calls for Stage 1
metadata only (experiment_condition / starting_model). This script adds free-text
fields from the same already-cached endpoint (``cw.metadata._fetch_rcsb_entry``),
one request per structure, for QC of the apo cohort. No structure or EDIA file is
ever fetched.

Usage:
    uv run scripts/wip/apo_text_check.py <apo_holo.csv> [--calls apo,apo-peripheral-ligand]
                                         [--csv out.csv]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # config, cw

import pandas as pd

from cw.metadata import _fetch_rcsb_entry

# Word stems that suggest something was meant to be bound. Matched on word
# boundaries against lowercased text, so "bound" does not fire on "boundary" and
# "complex" does not fire inside "complexity".
#
# These are scanned against the ENTRY TITLE only (plus the crystallization details
# for the soak set). The struct_keywords text is UniProt-style annotation
# boilerplate — a DJ-1 entry carries "Ubl conjugation", "Oxidation", "Chaperone"
# whether or not anything is bound — and the citation title describes the paper,
# not the entry, so a campaign titled "…inhibitors of X" would flag its own apo
# reference. Both are recorded for review but neither sets the verdict.
LIGAND_PATTERNS = [
    r"\bin complex with\b",
    r"\bcomplex(?:ed|es)?\b",
    r"\bbound\b",
    r"\bbinding\b",
    r"\bligand(?:ed)?\b",
    r"\binhibitor\b",
    r"\bsubstrate\b",
    r"\bco-?crystal\w*\b",
    r"\bsoak(?:ed|ing)?\b",
    r"\bbacksoak\w*\b",
    r"\bfragment\b",
    r"\badduct\b",
    r"\bcovalent(?:ly)?\b",
    r"\bconjugat\w*\b",
    r"\bholo\b",
    r"\bcompound \w+\b",
    r"\bmodifier\b",
    r"\bmixing with\b",
    r"\bmixed with\b",
    r"\bincubated with\b",
    r"\bmodification\b",
    r"\breacted with\b",
    r"\btreated with\b",
]

# Explicit statements that the entry is ligand-free.
APO_PATTERNS = [
    r"\bapo\b",
    r"\bapo-?form\b",
    r"\bunbound\b",
    r"\bunliganded\b",
    r"\bligand-?free\b",
    r"\bfree enzyme\b",
    r"\bunmodified\b",
    r"\breduced form\b",
    r"\bno mixing\b",
    r"\bwithout ligand\b",
    r"\bnative\b",
]

# Soak language specific to the crystallization protocol.
SOAK_PATTERNS = [
    r"\bsoak(?:ed|ing)?\b",
    r"\bco-?crystalli[sz]",
    r"\bincubat\w*\b",
    r"\bmm compound\b",
    r"\bdmso stock\b",
    r"\btitrat\w*\b",
]

# Crystallization additives that are not evidence of a ligand soak, so their
# presence in the details string must not trip the soak scan.
BENIGN_SOAK_CONTEXT = re.compile(
    r"\bsoak\w*\s+(?:in|with)\s+(?:cryo|paraton|paratone|oil|"
    r"glycerol|peg|liquid nitrogen)",
    re.I,
)


def matches(text: str, patterns: list[str]) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    return sorted({m.group(0) for p in patterns for m in re.finditer(p, lowered)})


@dataclass
class Check:
    pdb_id: str
    call: str
    title: str = ""
    keywords: str = ""
    citation: str = ""
    crystal_grow: str = ""
    rcsb_bound: list[str] = field(default_factory=list)
    unmodelled_bound: list[str] = field(default_factory=list)
    affinity: bool = False
    ligand_terms: list[str] = field(default_factory=list)
    apo_terms: list[str] = field(default_factory=list)
    soak_terms: list[str] = field(default_factory=list)
    citation_terms: list[str] = field(default_factory=list)  # informational only
    site_het: str = ""  # "COMP at d A" from the apo/holo CSV, if present
    fetched: bool = True

    @property
    def flags(self) -> list[str]:
        out = []
        if self.ligand_terms:
            out.append("ligand-language")
        if self.soak_terms:
            out.append("soak-language")
        if self.unmodelled_bound:
            out.append("rcsb-ligand-not-in-coords")
        if self.affinity:
            out.append("affinity-data")
        if self.apo_terms:
            out.append("apo-language")
        if not self.fetched:
            out.append("no-rcsb-entry")
        return out

    @property
    def verdict(self) -> str:
        if not self.fetched:
            return "unchecked"
        suspicious = self.ligand_terms or self.soak_terms or self.unmodelled_bound or self.affinity
        if suspicious and not self.apo_terms:
            return "review"
        if self.apo_terms:
            return "apo-consistent"
        return "neutral"


def modelled_het(all_het: object) -> set[str]:
    """Comp_ids from the apo_holo.csv ``all_het`` column ('COMP:n|COMP:n')."""
    if not isinstance(all_het, str) or not all_het:
        return set()
    return {part.split(":")[0] for part in all_het.split("|") if part}


def check(pdb_id: str, call: str, all_het: object, site_het: str = "") -> Check:
    entry = _fetch_rcsb_entry(pdb_id)
    if entry is None:
        return Check(pdb_id, call, site_het=site_het, fetched=False)

    struct = entry.get("struct") or {}
    keywords_block = entry.get("struct_keywords") or {}
    entry_info = entry.get("rcsb_entry_info") or {}

    title = struct.get("title") or ""
    keywords = " ; ".join(
        v for v in (keywords_block.get("pdbx_keywords"), keywords_block.get("text")) if v
    )
    citation = (entry.get("rcsb_primary_citation") or {}).get("title") or ""
    crystal_grow = " ; ".join(
        (g or {}).get("pdbx_details") or "" for g in (entry.get("exptl_crystal_grow") or [])
    ).strip(" ;")

    rcsb_bound = sorted(entry_info.get("nonpolymer_bound_components") or [])
    unmodelled = sorted(set(rcsb_bound) - modelled_het(all_het))

    grow_for_scan = "" if BENIGN_SOAK_CONTEXT.search(crystal_grow) else crystal_grow
    return Check(
        pdb_id,
        call,
        title=title,
        keywords=keywords,
        citation=citation,
        crystal_grow=crystal_grow,
        rcsb_bound=rcsb_bound,
        unmodelled_bound=unmodelled,
        site_het=site_het,
        affinity=bool(entry.get("rcsb_binding_affinity")),
        ligand_terms=matches(title, LIGAND_PATTERNS),
        apo_terms=matches(title, APO_PATTERNS),
        soak_terms=matches(grow_for_scan, SOAK_PATTERNS),
        citation_terms=matches(citation, LIGAND_PATTERNS),
    )


def report(checks: list[Check]) -> None:
    verdicts = Counter(c.verdict for c in checks)
    print(f"checked {len(checks)} structures against the RCSB free text\n")
    print("--- Verdict ---")
    for verdict in ("apo-consistent", "neutral", "review", "unchecked"):
        if verdicts[verdict]:
            print(f"  {verdict:<16}{verdicts[verdict]:>4}")
    print()

    flags = Counter(f for c in checks for f in c.flags)
    print("--- Signals (a structure can carry several) ---")
    for flag, n in flags.most_common():
        print(f"  {flag:<28}{n:>4}")
    print()

    for verdict in ("review", "neutral", "apo-consistent", "unchecked"):
        group = [c for c in checks if c.verdict == verdict]
        if not group:
            continue
        print(f"--- {verdict} ({len(group)}) ---")
        for c in group:
            print(f"  {c.pdb_id}  [{c.call}]  {c.title}")
            if c.ligand_terms:
                print(f"       ligand terms : {', '.join(c.ligand_terms)}")
            if c.apo_terms:
                print(f"       apo terms    : {', '.join(c.apo_terms)}")
            if c.soak_terms:
                print(f"       soak terms   : {', '.join(c.soak_terms)}")
            if c.unmodelled_bound:
                print(f"       RCSB bound but not modelled: {', '.join(c.unmodelled_bound)}")
            if c.affinity:
                print("       deposited binding-affinity data")
            if c.site_het:
                print(f"       modelled in site: {c.site_het}")
            if c.citation_terms:
                print(f"       (citation only) : {', '.join(c.citation_terms)}")
        print()


def write_csv(checks: list[Check], path: Path) -> None:
    pd.DataFrame(
        [
            {
                "pdb_id": c.pdb_id,
                "call": c.call,
                "verdict": c.verdict,
                "flags": "|".join(c.flags),
                "title": c.title,
                "keywords": c.keywords,
                "citation_title": c.citation,
                "crystal_grow": c.crystal_grow,
                "ligand_terms": "|".join(c.ligand_terms),
                "apo_terms": "|".join(c.apo_terms),
                "soak_terms": "|".join(c.soak_terms),
                "citation_terms": "|".join(c.citation_terms),
                "site_het": c.site_het,
                "rcsb_bound_components": "|".join(c.rcsb_bound),
                "unmodelled_bound": "|".join(c.unmodelled_bound),
                "affinity_data": int(c.affinity),
            }
            for c in checks
        ]
    ).to_csv(path, index=False)
    print(f"wrote {len(checks)} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("apo_holo", type=Path, help="apo_holo.csv from an apo_holo_*.py run")
    ap.add_argument(
        "--calls",
        default="apo,apo-peripheral-ligand",
        help="comma-separated calls to check (default: %(default)s)",
    )
    ap.add_argument("--csv", type=Path, help="write per-structure rows to this CSV")
    args = ap.parse_args()

    table = pd.read_csv(args.apo_holo)
    wanted = [c.strip() for c in args.calls.split(",") if c.strip()]
    subset = table[table.call.isin(wanted)]
    if subset.empty:
        print(f"no rows with call in {wanted}")
        sys.exit(1)

    has_site_het = {"het_comp", "het_dist", "site_het"} <= set(subset.columns)
    checks = [
        check(
            row.pdb_id,
            row.call,
            row.all_het,
            site_het=(
                f"{row.het_comp} at {row.het_dist:.2f} A"
                if has_site_het and row.site_het and isinstance(row.het_comp, str)
                else ""
            ),
        )
        for row in subset.itertuples()
    ]
    report(checks)
    if args.csv:
        write_csv(checks, args.csv)


if __name__ == "__main__":
    main()
