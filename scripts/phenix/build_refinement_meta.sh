#!/usr/bin/env bash
# Build a meta CSV of final refinement stats across all pairwise refinements for a
# cohort. For every (target, reference) pair drawn from a PDB-id list, find the
# pairwise phenix log for the given variant, cache its parsed per-log CSV beside the
# log, and collect the final ("end") row into one combined CSV keyed by the pair.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER="$SCRIPT_DIR/parse_phenix_log.sh"
source "$SCRIPT_DIR/setup_env.sh"

usage() {
    echo "usage: $(basename "$0") <pdb_list.txt> <fixed|auto|stripped> <meta_out.csv>" >&2
    echo "  DATA_DIR env overrides the data root (default: $DATA_DIR)" >&2
    echo "  COHORT_ID env overrides the cohort name (default: <pdb_list> basename)" >&2
    echo "  STRATEGY env selects the refinement tree: rigid|default (default: default)" >&2
    exit 1
}

[ "$#" -eq 3 ] || usage
txt="$1"; variant="$2"; out="$3"
[ -r "$txt" ] || { echo "error: cannot read pdb list: $txt" >&2; exit 1; }
case "$variant" in fixed|auto|stripped) ;; *) echo "error: variant must be fixed, auto, or stripped" >&2; exit 1;; esac

COHORT_ID="${COHORT_ID:-$(basename "$txt" .txt)}"
STRATEGY="${STRATEGY:-default}"
case "$STRATEGY" in rigid|default) ;; *) echo "error: STRATEGY must be 'rigid' or 'default'" >&2; exit 1 ;; esac
RESULTS_DIR="$DATA_DIR/${COHORT_ID}_${STRATEGY}_phenix/refinement_results"

# PDB ids: last whitespace field of each non-empty line, lowercased (handles both
# "3ATN" and "1\t3ATN" cohort formats).
ids=()
while IFS= read -r id; do ids+=("$id"); done < <(awk 'NF {print tolower($NF)}' "$txt")
[ "${#ids[@]}" -gt 0 ] || { echo "error: no pdb ids found in $txt" >&2; exit 1; }

mkdir -p "$(dirname "$out")"
echo "target_pdb,ref_pdb,variant,r_work,r_free,n_water" > "$out"

found=0; skipped=0
for a in "${ids[@]}"; do
    for b in "${ids[@]}"; do
        stem="${a}_refined_by_${b}_${variant}_001"
        log="$RESULTS_DIR/$a/refined_by_${b}_${variant}/${stem}.log"
        if [ ! -f "$log" ]; then
            echo "skip: no log for $a refined by $b ($variant)" >&2
            skipped=$((skipped + 1))
            continue
        fi

        csv="${log%.log}.csv"
        [ -f "$csv" ] || bash "$PARSER" "$log" > "$csv"

        lastrow="$(tail -n 1 "$csv")"
        if [[ "$lastrow" == stage,* || -z "$lastrow" ]]; then
            echo "warn: no data rows in $csv, skipping $a/$b" >&2
            skipped=$((skipped + 1))
            continue
        fi
        IFS=, read -r _stage rwork rfree nwater <<< "$lastrow"
        echo "$a,$b,$variant,$rwork,$rfree,$nwater" >> "$out"
        found=$((found + 1))
    done
done

echo "wrote $out: $found refinements ($skipped pairs skipped)" >&2
