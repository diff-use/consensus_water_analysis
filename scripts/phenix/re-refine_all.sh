#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.sh"

if [ -n "$PHENIX_ENV_PATH" ]; then
    source "$PHENIX_ENV_PATH"
fi

export NPROC="${NPROC:-1}"
export NUMCYCLE="${NUMCYCLE:-10}"
export MAPCUTOFF="${MAPCUTOFF:-3.5}"
export JOBS="${JOBS:-4}"
export STRATEGY="${STRATEGY:-rigid}"

PDBID_LIST="${PDBID_LIST:-$DATA_DIR/hewls_65_subsampled.txt}"
export COHORT_ID="${COHORT_ID:-$(basename "${PDBID_LIST}" .txt)}"

OUT_DIR="${DATA_DIR}/${COHORT_ID}_${STRATEGY}_phenix/refinement_results"
mkdir -p "$OUT_DIR"

IDS=$(tr '[:upper:]' '[:lower:]' < "$PDBID_LIST" | awk 'NF')

# Pass 1: strip waters from every reference once, up front. Cheap, and doing it
# before the parallel matrix means no two jobs race to write the same file.
echo "Stripping waters from all references"
for REF_PDBID in $IDS; do
    STRIPPED_DIR="${OUT_DIR}/${REF_PDBID}"
    STRIPPED_CIFFILE="${STRIPPED_DIR}/${REF_PDBID}_final_waterstripped.cif"
    if [ -f "$STRIPPED_CIFFILE" ]; then
        continue
    fi
    CIFFILE="${ALL_PDB_REDO_DIR}/${REF_PDBID}/${REF_PDBID}_final.cif"
    mkdir -p "$STRIPPED_DIR"
    ( cd "$STRIPPED_DIR" && phenix.pdbtools "${CIFFILE}" remove="resname HOH" output.suffix="_waterstripped" > /dev/null 2>&1 )
done

# Pass 2: full N×N refinement matrix, JOBS refinements running at a time.
for PDBID in $IDS; do
    for REF_PDBID in $IDS; do
        printf '%s\t%s\n' "$PDBID" "$REF_PDBID"
    done
done | xargs -P "$JOBS" -n2 "${SCRIPT_DIR}/re-refine.sh"
