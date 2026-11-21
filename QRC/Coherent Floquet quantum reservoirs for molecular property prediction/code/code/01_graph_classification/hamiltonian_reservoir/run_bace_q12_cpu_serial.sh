#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ubuntu/Workspace/Fei/wlf/QRC/BFS-QRC/FINAL/17_hamiltonian_setting_fair_migration"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/qisten/bin/python"
LOG_PATH="$ROOT_DIR/logs/bace_general_q12_cpu_serial.log"

cd "$ROOT_DIR"
mkdir -p logs
exec > >(tee -a "$LOG_PATH") 2>&1

export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24

echo "[$(date --iso-8601=seconds)] CPU serial Hamiltonian QRC stage started"
for SEED_VALUE in 0 10 20 30 40 50; do
  echo "[$(date --iso-8601=seconds)] begin BACE/general Q12 seed=$SEED_VALUE"
  "$PYTHON_BIN" run_qrc_features.py fixed-q \
    --device cpu \
    --datasets BACE \
    --variants canonical_general \
    --q-values 12 \
    --t-values 2,4,6,8,10,12,16,20,24,30 \
    --seeds "$SEED_VALUE" \
    --batch-size 32
  echo "[$(date --iso-8601=seconds)] complete BACE/general Q12 seed=$SEED_VALUE"
done
echo "[$(date --iso-8601=seconds)] CPU serial Hamiltonian QRC stage complete"

