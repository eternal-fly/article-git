#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ubuntu/Workspace/Fei/wlf/QRC/BFS-QRC/FINAL/17_hamiltonian_setting_fair_migration"
RUN_LOG="$ROOT_DIR/logs/gpu0_supplement_queue.log"
MONITOR_LOG="$ROOT_DIR/logs/gpu0_supplement_monitor.log"
SESSION_NAME="qrc_ham_supp_gpu0"

cd "$ROOT_DIR"
mkdir -p logs

monitor_previous=""
while true; do
  monitor_now="$(date --iso-8601=seconds)"
  monitor_stage="$(rg '\] stage [1-5]/5|GPU0 supplement queue complete' "$RUN_LOG" | tail -n 1 || true)"
  # Some output directories do not exist until their stage starts.  Keep those
  # expected "zero so far" states from terminating the monitor under pipefail.
  monitor_global="$(find results/p_by_T_global_prior_diagnostic -name metrics.csv 2>/dev/null | wc -l || true)"
  monitor_wg="$(find cache/qrc_suffix_banks/BACE/canonical_general/Q12 -path '*/g*_W*/T30_seed*.npz' 2>/dev/null | wc -l || true)"
  monitor_wg_eval="$(find results/w_g_heatmaps -name metrics.csv 2>/dev/null | wc -l || true)"
  monitor_q16="$(find cache/qrc_suffix_banks -path '*/Q16/g0.840_W0.500/T*_seed*.npz' 2>/dev/null | wc -l || true)"
  monitor_gpu="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | sed -n '1p' || true)"
  monitor_status="$monitor_stage | global_eval=$monitor_global/96 | wg_banks=$monitor_wg/732 | wg_eval=$monitor_wg_eval/732 | q16=$monitor_q16/240 | gpu0=$monitor_gpu"

  if [[ "$monitor_status" != "$monitor_previous" ]]; then
    echo "$monitor_now $monitor_status" >> "$MONITOR_LOG"
    monitor_previous="$monitor_status"
  fi

  if rg -q 'GPU0 supplement queue complete' "$RUN_LOG"; then
    echo "$monitor_now COMPLETE" >> "$MONITOR_LOG"
    exit 0
  fi
  if rg -q 'Traceback|OutOfMemory|CUDA out of memory|Killed|^ERROR' "$RUN_LOG"; then
    echo "$monitor_now NEEDS_ATTENTION: error marker in run log" >> "$MONITOR_LOG"
    exit 2
  fi
  if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "$monitor_now NEEDS_ATTENTION: tmux session $SESSION_NAME is absent" >> "$MONITOR_LOG"
    exit 3
  fi
  sleep 60
done
