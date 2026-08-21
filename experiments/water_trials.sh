#!/usr/bin/env bash
# Water-treatment matrix for one structure, run against the local phenix install:
# starting waters kept/stripped x the ordered_solvent settings in $ORDERED_SOLVENT
# x the strategies in $STRATEGIES. Reuses scripts/phenix/refine_template_*.eff and
# scripts/phenix/parse_phenix_log.sh.
#
# Arg 2 selects the starting model: data/${PDBID}_${VARIANT}.cif, defaulting to
# the _final.cif. The reflection data is always ${PDBID}_final.mtz.
#
#   ./experiments/water_trials.sh 3b3a                            # 8 trials, occupancies refined
#   STRATEGIES=rigid NO_OCC=1 ./experiments/water_trials.sh 3b3a  # 4 rigid trials, occupancies fixed
#   STRATEGIES=adponly ORDERED_SOLVENT=false ./experiments/water_trials.sh 3b3a t0.50
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PHENIX_DIR="$PROJECT_ROOT/scripts/phenix"

PDBID="${1:-3b3a}"
VARIANT="${2:-final}"
export NPROC="${NPROC:-4}"
export NUMCYCLE="${NUMCYCLE:-10}"
export MAPCUTOFF="${MAPCUTOFF:-3.5}"
STRATEGIES="${STRATEGIES:-default rigid}"
ORDERED_SOLVENT="${ORDERED_SOLVENT:-true false}"

# config.py's local-mode value; the committed pod config leaves PHENIX_ENV_PATH empty.
PHENIX_ENV_PATH="${PHENIX_ENV_PATH:-/Users/dorismai/Applications/phenix-2.0-5936/phenix_env.sh}"
if [ -n "$PHENIX_ENV_PATH" ]; then
    # shellcheck disable=SC1090
    source "$PHENIX_ENV_PATH"
fi

IN_DIR="$PROJECT_ROOT/data"
if [ "$VARIANT" = "final" ]; then
    OUT_DIR="$IN_DIR/${PDBID}_water_trials"
else
    OUT_DIR="$IN_DIR/${PDBID}_${VARIANT}_water_trials"
fi
export MTZFILE="${IN_DIR}/${PDBID}_final.mtz"
DEPOSITED_CIF="${IN_DIR}/${PDBID}_${VARIANT}.cif"

PREP_DIR="${OUT_DIR}/prepared_models"
mkdir -p "$PREP_DIR"

# phenix.pdbtools re-emits the model without the deposited TLS metadata that
# phenix.refine asserts on. Same prep as scripts/phenix/re-refine.sh.
KEPT_CIF="${PREP_DIR}/${PDBID}_waterkept.cif"
STRIPPED_CIF="${PREP_DIR}/${PDBID}_waterstripped.cif"
if [ ! -f "$KEPT_CIF" ]; then
    echo "Preparing ${PDBID} starting model (waters kept)"
    ( cd "$PREP_DIR" && phenix.pdbtools "$DEPOSITED_CIF" \
        output.file_name="$(basename "$KEPT_CIF")" > pdbtools_waterkept.log 2>&1 )
fi
if [ ! -f "$STRIPPED_CIF" ]; then
    echo "Preparing ${PDBID} starting model (waters stripped)"
    ( cd "$PREP_DIR" && phenix.pdbtools "$DEPOSITED_CIF" remove="resname HOH" \
        output.file_name="$(basename "$STRIPPED_CIF")" > pdbtools_waterstripped.log 2>&1 )
fi

echo "NPROC: ${NPROC} NUMCYCLE: ${NUMCYCLE} MAPCUTOFF: ${MAPCUTOFF} STRATEGIES: ${STRATEGIES} ORDERED_SOLVENT: ${ORDERED_SOLVENT} NO_OCC: ${NO_OCC:-0}"

# NO_OCC=1 removes occupancy refinement for BOTH protein and water: drop
# `occupancies` from the strategy (kills the altloc constrained groups and the
# partial-occupancy protein atoms) and turn off ordered_solvent's own water
# occupancy refinement, which the strategy does not govern.
TAG_SUFFIX=""
if [ "${NO_OCC:-0}" = "1" ]; then
    TAG_SUFFIX="_noocc"
fi

strategy_without_occupancies() {
    case "$1" in
        rigid)   echo "rigid_body+individual_adp" ;;
        default) echo "individual_sites+individual_sites_real_space+individual_adp" ;;
    esac
}

for strategy in $STRATEGIES; do
    EFF_TEMPLATE="${PHENIX_DIR}/refine_template_${strategy}.eff"
    for waters in kept stripped; do
        if [ "$waters" = "kept" ]; then MODEL="$KEPT_CIF"; else MODEL="$STRIPPED_CIF"; fi
        for ordered_solvent in $ORDERED_SOLVENT; do
            TAG="${waters}_os${ordered_solvent}_${strategy}${TAG_SUFFIX}"
            PREFIX="${PDBID}_${TAG}"
            REFINE_DIR="${OUT_DIR}/${TAG}"
            OVERRIDES=(ordered_solvent="$ordered_solvent")
            if [ "${NO_OCC:-0}" = "1" ]; then
                OVERRIDES+=(refinement.refine.strategy="$(strategy_without_occupancies "$strategy")")
                OVERRIDES+=(refinement.ordered_solvent.refine_occupancies=False)
            fi
            if [ -f "${REFINE_DIR}/${PREFIX}_001.log" ] && \
               grep -q "Final R-work" "${REFINE_DIR}/${PREFIX}_001.log"; then
                echo "== ${TAG}: already refined, skipping"
                continue
            fi
            mkdir -p "$REFINE_DIR"
            echo "== ${TAG}: $(basename "$MODEL") + $(basename "$EFF_TEMPLATE") ${OVERRIDES[*]}"
            ( cd "$REFINE_DIR" && phenix.refine "$MODEL" "$EFF_TEMPLATE" \
                "${OVERRIDES[@]}" output.prefix="$PREFIX" \
                > phenix.stdout 2>&1 ) || echo "   FAILED (see ${REFINE_DIR}/phenix.stdout)"
            grep "Final R-work" "${REFINE_DIR}/${PREFIX}_001.log" 2>/dev/null || true
        done
    done
done

# Summary covers every trial dir present, not just this run's. What was actually
# refined is read back from the log's own flag block rather than inferred from the
# tag, so command-line strategy overrides can't drift from the reported columns.
SUMMARY="${OUT_DIR}/summary.csv"
echo "trial,waters,ordered_solvent,strategy,coords,occupancies,r_work_start,r_free_start,n_water_start,r_work_final,r_free_final,n_water_final" > "$SUMMARY"
read_flag() { awk -v k="$2" '$1==k && $2=="=" {print $3; exit}' "$1"; }
for d in "${OUT_DIR}"/*/; do
    TAG="$(basename "$d")"
    [ "$TAG" = "prepared_models" ] && continue
    case "$TAG" in
        *_noocc) BASE="${TAG%_noocc}" ;;
        *)       BASE="$TAG" ;;
    esac
    WATERS="${BASE%%_*}"
    REST="${BASE#*_}"
    OS="${REST%%_*}"; OS="${OS#os}"
    STRAT="${REST#*_}"
    LOG="${d}${PDBID}_${TAG}_001.log"
    [ -f "$LOG" ] || { echo "${TAG},${WATERS},${OS},${STRAT},,,,,,,," >> "$SUMMARY"; continue; }
    if [ "$(read_flag "$LOG" individual_sites)" = "True" ]; then COORDS=individual
    elif [ "$(read_flag "$LOG" rigid_body)" = "True" ]; then COORDS=rigid
    else COORDS=fixed; fi
    if [ "$(read_flag "$LOG" occupancies)" = "True" ]; then OCC=refined; else OCC=fixed; fi
    "${PHENIX_DIR}/parse_phenix_log.sh" "$LOG" | awk -F, -v t="$TAG" -v w="$WATERS" \
        -v o="$OS" -v s="$STRAT" -v x="$COORDS" -v c="$OCC" '
        $1 == "0"   { rw0 = $2; rf0 = $3; nw0 = $4 }
        $1 == "end" { rwe = $2; rfe = $3; nwe = $4 }
        END { printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", t, w, o, s, x, c, rw0, rf0, nw0, rwe, rfe, nwe }
    ' >> "$SUMMARY"
done

echo
column -s, -t "$SUMMARY"
echo
echo "summary: $SUMMARY"
