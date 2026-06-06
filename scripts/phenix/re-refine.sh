#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EFF_TEMPLATE="${SCRIPT_DIR}/refine_template.eff"

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

# OUT_DIR: cohort-scoped output tree under the pipeline data root.
: "${COHORT_ID:?not set — re-refine_all.sh exports it; set it for standalone runs}"
OUT_DIR="${DATA_DIR}/${COHORT_ID}_phenix/refinement_results"

if [ ! -d "$DATA_DIR" ]; then
    echo "error: DATA_DIR does not exist: $DATA_DIR" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

export MTZFILE="${ALL_PDB_REDO_DIR}/${PDBID}/${PDBID}_final.mtz"


CIFFILE="${ALL_PDB_REDO_DIR}/${REF_PDBID}/${REF_PDBID}_final.cif"
SUFFIX="refined_by_${REF_PDBID}"
OUTPREFIX="${PDBID}_${SUFFIX}"

# The water-stripped reference cif lives in the reference structure's own subfolder.
STRIPPED_DIR="${OUT_DIR}/${REF_PDBID}"
STRIPPED_CIFFILE="${STRIPPED_DIR}/${REF_PDBID}_final_waterstripped.cif"
if [ ! -f "$STRIPPED_CIFFILE" ]; then
    echo "Stripping water from ${REF_PDBID}"
    mkdir -p "$STRIPPED_DIR"
    cd "$STRIPPED_DIR"
        phenix.pdbtools "${CIFFILE}" remove="resname HOH" output.suffix="_waterstripped" > /dev/null 2>1
        echo "Finished stripping water from ${REF_PDBID}, entering ${PDBID} output directory"
else
    echo "Water-stripped CIF file already exists: ${STRIPPED_CIFFILE}"
fi

echo "================================================"
echo "Refining ${PDBID} using ${REF_PDBID} as reference"
echo "================================================"
for tag in fixed auto stripped; do
    EFF_CIFFILE=$CIFFILE
    EFF_OUTPREFIX=${OUTPREFIX}_${tag}
    ORDERED_SOLVENT=""
    if [ "$tag" = "fixed" ]; then
        ORDERED_SOLVENT="ordered_solvent=false"
    elif [ "$tag" = "stripped" ]; then
        EFF_CIFFILE=$STRIPPED_CIFFILE
    fi
    echo "Tag: ${tag}, cif stem: $(basename "${EFF_CIFFILE}" .cif), ordered_solvent: ${ORDERED_SOLVENT}"
    refine_dir="${OUT_DIR}/${PDBID}/${SUFFIX}_${tag}"
    if [ -f "${refine_dir}/${EFF_OUTPREFIX}_001.log" ] && grep "Final R-work" "${refine_dir}/${EFF_OUTPREFIX}_001.log"; then
        echo "Already refined, skipping"
        continue
    fi
    mkdir -p $refine_dir
    cd $refine_dir
    phenix.refine ${EFF_CIFFILE} ${EFF_TEMPLATE} ${ORDERED_SOLVENT} output.prefix=${EFF_OUTPREFIX} > /dev/null 2>1
    grep "Final R-work" ${EFF_OUTPREFIX}_001.log
done
