#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.sh"

export NPROC="${NPROC:-1}"
export NUMCYCLE="${NUMCYCLE:-10}"
export MAPCUTOFF="${MAPCUTOFF:-3.5}"
export JOBS="${JOBS:-4}"
export STRATEGY="${STRATEGY:-default}"

PDBID_LIST="${PDBID_LIST:-$DATA_DIR/hewls_65_subsampled.txt}"
export COHORT_ID="${COHORT_ID:-$(basename "${PDBID_LIST}" .txt)}"

OUT_DIR="${DATA_DIR}/${COHORT_ID}_${STRATEGY}_phenix/refinement_results"
mkdir -p "$OUT_DIR"

IDS=$(tr '[:upper:]' '[:lower:]' < "$PDBID_LIST" | awk 'NF')

# Pass 0: default align every off-diagonal starting model onto its MTZ-source
# model BEFORE sourcing Phenix, so the alignment runs in the project's own Python
# env (biotite/numpy/gemmi — no Phenix). The starting model (REF_PDBID) is fed to
# phenix.refine against PDBID's data, which lives in PDBID's deposited frame; an
# unaligned offset can defeat default refinement (e.g. 5kxn). Aligning here puts
# the model in the data's frame and adopts PDBID's crystal symmetry. The diagonal
# (PDBID == REF_PDBID) is identity and skipped. Aligned cif paths are unique per
# pair, so parallel writes don't race. re-refine.sh consumes these (and prepares
# the per-pair waterkept/waterstripped cifs), so no per-reference precompute pass
# is needed.
REPORT="${OUT_DIR}/alignment_report.csv"
echo "pdb_id,ref_pdbid,n_common_ca,rmsd_before,rmsd_after,status" > "$REPORT"

align_pair() {
    PDBID="$1"; REF_PDBID="$2"
    OUT="${OUT_DIR}/${PDBID}/prepared_models/${REF_PDBID}_aligned.cif"
    [ -f "$OUT" ] && return 0
    mkdir -p "$(dirname "$OUT")"
    ( cd "$PROJECT_ROOT" && env -u PYTHONPATH -u PYTHONHOME uv run scripts/phenix/align_starting_model.py \
        --mobile "${ALL_PDB_REDO_DIR}/${REF_PDBID}/${REF_PDBID}_final.cif" \
        --reference "${ALL_PDB_REDO_DIR}/${PDBID}/${PDBID}_final.cif" \
        --reference-mtz "${ALL_PDB_REDO_DIR}/${PDBID}/${PDBID}_final.mtz" \
        --pdb-id "$PDBID" --ref-pdb-id "$REF_PDBID" \
        --out "$OUT" >> "$REPORT" )
}
export -f align_pair
export OUT_DIR ALL_PDB_REDO_DIR PROJECT_ROOT REPORT

echo "Aligning starting models onto MTZ-source frames"
for PDBID in $IDS; do
    for REF_PDBID in $IDS; do
        [ "$PDBID" != "$REF_PDBID" ] && printf '%s\t%s\n' "$PDBID" "$REF_PDBID"
    done
done | xargs -P "$JOBS" -n2 bash -c 'align_pair "$@"' _

if [ -n "$PHENIX_ENV_PATH" ]; then
    source "$PHENIX_ENV_PATH"
fi

# Pass 1: full N×N refinement matrix, JOBS refinements running at a time.
# re-refine.sh picks up each pair's aligned cif and prepares the waterkept/
# waterstripped variants per pair.
for PDBID in $IDS; do
    for REF_PDBID in $IDS; do
        printf '%s\t%s\n' "$PDBID" "$REF_PDBID"
    done
done | xargs -P "$JOBS" -n2 "${SCRIPT_DIR}/re-refine.sh"
