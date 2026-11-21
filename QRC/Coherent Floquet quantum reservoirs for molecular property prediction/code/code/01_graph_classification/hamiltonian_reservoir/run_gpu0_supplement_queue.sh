#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ubuntu/Workspace/Fei/wlf/QRC/BFS-QRC/FINAL/17_hamiltonian_setting_fair_migration"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/qisten/bin/python"
LOG_PATH="$ROOT_DIR/logs/gpu0_supplement_queue.log"

cd "$ROOT_DIR"
mkdir -p logs
exec > >(tee -a "$LOG_PATH") 2>&1

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

echo "[$(date --iso-8601=seconds)] GPU0 supplement queue started"
"$PYTHON_BIN" -m pytest -q tests

echo "[$(date --iso-8601=seconds)] stage 1/5: Q12 global-prior suffix banks"
"$PYTHON_BIN" run_qrc_features.py fixed-q \
  --device cuda:0 \
  --datasets BACE,BBBP \
  --variants canonical_global_prior \
  --q-values 12 \
  --t-values 2,4,6,8,10,12,16,20,24,30 \
  --seeds 0,10,20,30,40,50 \
  --batch-size 256

echo "[$(date --iso-8601=seconds)] stage 2/5: Q12 global-prior p-by-T, linear and SVR"
"$PYTHON_BIN" evaluate_p_by_t.py \
  --datasets BACE,BBBP \
  --token-variant canonical_global_prior \
  --t-values 6,12,20,30 \
  --seeds 0,10,20,30,40,50 \
  --decoders linear,rbf_svr \
  --linear-device cuda:0

echo "[$(date --iso-8601=seconds)] stage 3/5: full 11x11 W-g suffix grid"
"$PYTHON_BIN" run_qrc_features.py wg \
  --device cuda:0 \
  --wg-values 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --seeds 0,10,20,30,40,50 \
  --batch-size 256

echo "[$(date --iso-8601=seconds)] stage 4/5: closed/open W-g linear readouts"
"$PYTHON_BIN" evaluate_wg.py \
  --g-values 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --W-values 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --seeds 0,10,20,30,40,50 \
  --device cuda:0

echo "[$(date --iso-8601=seconds)] stage 5/5: Q16 general/global-prior suffix banks"
"$PYTHON_BIN" run_qrc_features.py fixed-q \
  --device cuda:0 \
  --datasets BACE,BBBP \
  --variants canonical_general,canonical_global_prior \
  --q-values 16 \
  --t-values 2,4,6,8,10,12,16,20,24,30 \
  --seeds 0,10,20,30,40,50 \
  --batch-size 32

echo "[$(date --iso-8601=seconds)] GPU0 supplement queue complete"
