#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# ===============================================================================
# Normally invoked by re-refine_all.sh. To run this script standalone, uncomment
# the following block and set the environment variables.
# ===============================================================================
# source PHENIX_ENV_PATH
# export NPROC="${NPROC:-4}"
# export NUMCYCLE="${NUMCYCLE:-4}"
# export MAPCUTOFF="${MAPCUTOFF:-3.5}"
# export COHORT_ID=hewls_65_subsampled
# source "$SCRIPT_DIR/setup_env.sh"

echo "NPROC: ${NPROC} NUMCYCLE: ${NUMCYCLE} MAPCUTOFF: ${MAPCUTOFF}"

PDBID=${1:-"3atn"}
REF_PDBID=${2:-$PDBID}

# OUT_DIR: cohort-scoped output tree under the pipeline data root. STRATEGY selects
# the eff template and isolates each strategy's results in its own
# <cohort>_<strategy>_phenix tree.
: "${COHORT_ID:?not set — re-refine_all.sh exports it; set it for standalone runs}"
STRATEGY="${STRATEGY:-default}"
case "$STRATEGY" in
    rigid)   EFF_TEMPLATE="${SCRIPT_DIR}/refine_template_rigid.eff";   PHENIX_COHORT="${COHORT_ID}_rigid" ;;
    default) EFF_TEMPLATE="${SCRIPT_DIR}/refine_template_default.eff"; PHENIX_COHORT="${COHORT_ID}_default" ;;
    *) echo "error: STRATEGY must be 'rigid' or 'default'" >&2; exit 1 ;;
esac
OUT_DIR="${DATA_DIR}/${PHENIX_COHORT}_phenix/refinement_results"

if [ ! -d "$DATA_DIR" ]; then
    echo "error: DATA_DIR does not exist: $DATA_DIR" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

export MTZFILE="${ALL_PDB_REDO_DIR}/${PDBID}/${PDBID}_final.mtz"


REF_CIF="${ALL_PDB_REDO_DIR}/${REF_PDBID}/${REF_PDBID}_final.cif"
PDBID_CIF="${ALL_PDB_REDO_DIR}/${PDBID}/${PDBID}_final.cif"
SUFFIX="refined_by_${REF_PDBID}"
OUTPREFIX="${PDBID}_${SUFFIX}"

# Per-pair model prep, stored under the MTZ source's own folder so filenames are
# unique across parallel jobs. The starting model (REF_PDBID) is default aligned
# onto PDBID's deposited frame (the MTZ frame) so refinement doesn't start
# misplaced; re-refine_all.sh does this up front in Pass 0. The diagonal
# (PDBID == REF_PDBID) needs no alignment — the deposited model already matches its
# own data. For standalone runs that skipped Pass 0, align inline (env -u PYTHONPATH
# keeps the sourced Phenix env out of `uv run`); fall back to the raw model if
# alignment is skipped (too few common Cα) or errors.
PREP_DIR="${OUT_DIR}/${PDBID}/prepared_models"
mkdir -p "$PREP_DIR"
if [ "$PDBID" = "$REF_PDBID" ]; then
    START_CIF="$REF_CIF"
else
    START_CIF="${PREP_DIR}/${REF_PDBID}_aligned.cif"
    if [ ! -f "$START_CIF" ]; then
        echo "Aligning ${REF_PDBID} onto ${PDBID} (inline)"
        ( cd "$PROJECT_ROOT" && env -u PYTHONPATH -u PYTHONHOME uv run scripts/phenix/align_starting_model.py \
            --mobile "$REF_CIF" --reference "$PDBID_CIF" --reference-mtz "$MTZFILE" \
            --pdb-id "$PDBID" --ref-pdb-id "$REF_PDBID" --out "$START_CIF" )
        if [ ! -f "$START_CIF" ]; then
            echo "warning: alignment unavailable for ${REF_PDBID} -> ${PDBID}; using raw model"
            START_CIF="$REF_CIF"
        fi
    fi
fi

# phenix.pdbtools re-emits the model without the deposited TLS metadata that
# phenix.refine asserts on. WATERKEPT feeds auto/fixed; WATERSTRIPPED additionally
# removes waters and feeds stripped.
WATERKEPT_CIFFILE="${PREP_DIR}/${REF_PDBID}_waterkept.cif"
WATERSTRIPPED_CIFFILE="${PREP_DIR}/${REF_PDBID}_waterstripped.cif"
if [ ! -f "$WATERKEPT_CIFFILE" ]; then
    echo "Preparing ${REF_PDBID} starting model for ${PDBID} (waters kept)"
    ( cd "$PREP_DIR" && phenix.pdbtools "${START_CIF}" output.file_name="${REF_PDBID}_waterkept.cif" > /dev/null 2>&1 )
fi
if [ ! -f "$WATERSTRIPPED_CIFFILE" ]; then
    echo "Preparing ${REF_PDBID} starting model for ${PDBID} (waters stripped)"
    ( cd "$PREP_DIR" && phenix.pdbtools "${START_CIF}" remove="resname HOH" output.file_name="${REF_PDBID}_waterstripped.cif" > /dev/null 2>&1 )
fi

echo "================================================"
echo "Refining ${PDBID} mtz using ${REF_PDBID} as starting model"
echo "================================================"
for tag in auto stripped; do
# for tag in fixed auto stripped; do
    EFF_CIFFILE=$WATERKEPT_CIFFILE
    EFF_OUTPREFIX=${OUTPREFIX}_${tag}
    ORDERED_SOLVENT=""
    if [ "$tag" = "fixed" ]; then
        ORDERED_SOLVENT="ordered_solvent=false"
    elif [ "$tag" = "stripped" ]; then
        EFF_CIFFILE=$WATERSTRIPPED_CIFFILE
    fi
    echo "Tag: ${tag}, cif stem: $(basename "${EFF_CIFFILE}" .cif), ordered_solvent: ${ORDERED_SOLVENT}"
    refine_dir="${OUT_DIR}/${PDBID}/${SUFFIX}_${tag}"
    if [ -f "${refine_dir}/${EFF_OUTPREFIX}_001.log" ] && grep "Final R-work" "${refine_dir}/${EFF_OUTPREFIX}_001.log"; then
        echo "Already refined, skipping"
        continue
    fi
    mkdir -p $refine_dir
    cd $refine_dir
    phenix.refine ${EFF_CIFFILE} ${EFF_TEMPLATE} ${ORDERED_SOLVENT} output.prefix=${EFF_OUTPREFIX} > /dev/null 2>&1
    grep "Final R-work" ${EFF_OUTPREFIX}_001.log
done
