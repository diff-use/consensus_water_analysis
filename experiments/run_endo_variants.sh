#!/usr/bin/env bash
cd /home/jovyan/workspace
source .venv/bin/activate
DD=$(python -c "import config;print(config.DATA_DIR)")
REF=5r32
MCS=20
MS=20
ORDER=(
  endothiapepsin_000240_iso_edia0.4
  endothiapepsin_000240_iso_edia0.6
  endothiapepsin_000240_iso_edia0.8
  endothiapepsin_000240_iso_bfactor_z1.0water
  endothiapepsin_000240_iso_bfactor_z1.5water
  endothiapepsin_000240_iso_bfactor_z2.0water
)
declare -A FLAGS=(
  [endothiapepsin_000240_iso_edia0.4]="--edia-cutoff 0.4"
  [endothiapepsin_000240_iso_edia0.6]="--edia-cutoff 0.6"
  [endothiapepsin_000240_iso_edia0.8]="--edia-cutoff 0.8"
  [endothiapepsin_000240_iso_bfactor_z1.0water]="--bfactor-cutoff 1.0 --bfactor-mode zscore --bfactor-population water"
  [endothiapepsin_000240_iso_bfactor_z1.5water]="--bfactor-cutoff 1.5 --bfactor-mode zscore --bfactor-population water"
  [endothiapepsin_000240_iso_bfactor_z2.0water]="--bfactor-cutoff 2.0 --bfactor-mode zscore --bfactor-population water"
)
mkdir -p logs
for v in "${ORDER[@]}"; do
  LOG=logs/${v}.log
  echo "=== $(date) START $v flags=${FLAGS[$v]} ===" | tee -a "$LOG"
  if (
    set -e
    python scripts/filter_waters.py data/${v}.txt ${FLAGS[$v]} -j 4
    python scripts/align_structures.py data/${v}.txt --reference $REF -j 4
    python scripts/find_clustering_hyperparameters.py data/${v}.txt
    python scripts/cluster_waters.py data/${v}.txt --min-cluster-size $MCS --min-samples $MS -o "$DD/${v}/min_cluster_size_${MCS}_min_samples_${MS}"
  ) >>"$LOG" 2>&1; then
    echo "=== $(date) DONE $v ===" | tee -a "$LOG"
  else
    echo "=== $(date) FAILED $v ===" | tee -a "$LOG"
  fi
done
echo "=== $(date) ALL DONE ==="
