#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ubuntu/Workspace/Fei/wlf/QRC/BFS-QRC/FINAL/17_hamiltonian_setting_fair_migration"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/qisten/bin/python"
WORKING_BUS="00000000:06:00.0"

cd "$ROOT_DIR"
mkdir -p logs

while true; do
  ACTIVE_BUSES="$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader 2>/dev/null || true)"
  if ! grep -Fq "$WORKING_BUS" <<<"$ACTIVE_BUSES"; then
    break
  fi
  echo "[$(date --iso-8601=seconds)] waiting: working RTX 4090 is occupied"
  sleep 60
done

echo "[$(date --iso-8601=seconds)] starting Hamiltonian BACE/general Q12 fixed-T suffix banks"
export PYTHONPATH="$ROOT_DIR"
export CUDA_VISIBLE_DEVICES=0
"$PYTHON_BIN" run_qrc_features.py fixed-q \
  --device cuda:0 \
  --datasets BACE \
  --variants canonical_general \
  --q-values 12 \
  --t-values 2,4,6,8,10,12,16,20,24,30 \
  --seeds 0,10,20,30,40,50
echo "[$(date --iso-8601=seconds)] Hamiltonian BACE/general Q12 feature stage complete"

