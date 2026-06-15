#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
STRATEGY="${STRATEGY:-rigid}"
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


CIFFILE="${ALL_PDB_REDO_DIR}/${REF_PDBID}/${REF_PDBID}_final.cif"
SUFFIX="refined_by_${REF_PDBID}"
OUTPREFIX="${PDBID}_${SUFFIX}"

# Reference cifs live in the reference's own subfolder. phenix.pdbtools re-emits
# the model without the deposited TLS metadata that phenix.refine asserts on, so
# auto/stripped/fixed share a consistent baseline. WATERKEPT feeds auto/fixed;
# WATERSTRIPPED additionally removes waters. re-refine_all.sh precomputes both;
# this block only runs for standalone invocations.
REF_DIR="${OUT_DIR}/${REF_PDBID}"
WATERKEPT_CIFFILE="${REF_DIR}/${REF_PDBID}_final_waterkept.cif"
WATERSTRIPPED_CIFFILE="${REF_DIR}/${REF_PDBID}_final_waterstripped.cif"
mkdir -p "$REF_DIR"
if [ ! -f "$WATERKEPT_CIFFILE" ]; then
    echo "Preparing ${REF_PDBID} reference (waters kept)"
    ( cd "$REF_DIR" && phenix.pdbtools "${CIFFILE}" output.file_name="${REF_PDBID}_final_waterkept.cif" > /dev/null 2>&1 )
fi
if [ ! -f "$WATERSTRIPPED_CIFFILE" ]; then
    echo "Preparing ${REF_PDBID} reference (waters stripped)"
    ( cd "$REF_DIR" && phenix.pdbtools "${CIFFILE}" remove="resname HOH" output.file_name="${REF_PDBID}_final_waterstripped.cif" > /dev/null 2>&1 )
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
