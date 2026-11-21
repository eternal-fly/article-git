#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ubuntu/Workspace/Fei/wlf/QRC/BFS-QRC/FINAL/17_hamiltonian_setting_fair_migration"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/qisten/bin/python"
LOG_PATH="$ROOT_DIR/logs/q12_p_by_t_cpu_serial.log"

cd "$ROOT_DIR"
mkdir -p logs
exec > >(tee -a "$LOG_PATH") 2>&1

export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24

echo "[$(date --iso-8601=seconds)] Q12 Hamiltonian feature and p-by-T CPU queue started"
for DATASET_VALUE in BACE BBBP; do
  for SEED_VALUE in 0 10 20 30 40 50; do
    echo "[$(date --iso-8601=seconds)] begin $DATASET_VALUE/general Q12 seed=$SEED_VALUE"
    "$PYTHON_BIN" run_qrc_features.py fixed-q \
      --device cpu \
      --datasets "$DATASET_VALUE" \
      --variants canonical_general \
      --q-values 12 \
      --t-values 2,4,6,8,10,12,16,20,24,30 \
      --seeds "$SEED_VALUE" \
      --batch-size 32
    echo "[$(date --iso-8601=seconds)] complete $DATASET_VALUE/general Q12 seed=$SEED_VALUE"
  done

  echo "[$(date --iso-8601=seconds)] begin $DATASET_VALUE linear p-by-T readouts"
  "$PYTHON_BIN" evaluate_p_by_t.py \
    --datasets "$DATASET_VALUE" \
    --t-values 6,12,20,30 \
    --seeds 0,10,20,30,40,50 \
    --decoders linear \
    --linear-device cpu
  echo "[$(date --iso-8601=seconds)] complete $DATASET_VALUE linear p-by-T readouts"

  echo "[$(date --iso-8601=seconds)] begin $DATASET_VALUE RBF-SVR p-by-T readouts"
  "$PYTHON_BIN" evaluate_p_by_t.py \
    --datasets "$DATASET_VALUE" \
    --t-values 6,12,20,30 \
    --seeds 0,10,20,30,40,50 \
    --decoders rbf_svr \
    --linear-device cpu
  echo "[$(date --iso-8601=seconds)] complete $DATASET_VALUE RBF-SVR p-by-T readouts"
done
echo "[$(date --iso-8601=seconds)] Q12 Hamiltonian feature and p-by-T CPU queue complete"

