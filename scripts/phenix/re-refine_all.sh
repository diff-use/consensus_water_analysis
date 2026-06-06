#!/bin/bash
source $PHENIX_ENV_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.sh"

export NPROC="${NPROC:-8}"
export NUMCYCLE="${NUMCYCLE:-10}"
export MAPCUTOFF="${MAPCUTOFF:-3.5}"

PDBID_LIST="${PDBID_LIST:-$DATA_DIR/hewls_65_subsampled.txt}"
export COHORT_ID="${COHORT_ID:-$(basename "${PDBID_LIST}" .txt)}"

for PDBID in $(cat ${PDBID_LIST}); do
    for REF_PDBID in $(cat ${PDBID_LIST}); do
            PDBID=$(echo "${PDBID}" | tr '[:upper:]' '[:lower:]')
            REF_PDBID=$(echo "${REF_PDBID}" | tr '[:upper:]' '[:lower:]')
            "${SCRIPT_DIR}/re-refine.sh" ${PDBID} ${REF_PDBID}
    done
done
