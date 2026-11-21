#!/usr/bin/env bash
set -euo pipefail

scan_dir="/home/ubuntu/Workspace/Fei/wlf/QRC/BFS-QRC/FINAL/16_dissipative_protocol_migration"
python_bin="/home/ubuntu/miniconda3/envs/qisten/bin/python"
log_path="${scan_dir}/logs/bace_general_g_w_heatmap_gpu0.log"
runner="${scan_dir}/run_bace_general_g_w_heatmap.py"

mkdir -p "${scan_dir}/logs" "${scan_dir}/.matplotlib"
export MPLCONFIGDIR="${scan_dir}/.matplotlib"

while true; do
    timestamp="$(date --iso-8601=seconds)"
    if ! nvidia-smi -i 0 --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
        echo "[${timestamp}] waiting: physical GPU0 is not visible to NVML" | tee -a "${log_path}"
        sleep 60
        continue
    fi
    gpu_pids="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
    if [[ -n "${gpu_pids}" ]]; then
        echo "[${timestamp}] waiting: physical GPU0 has compute process(es): ${gpu_pids//$'\n'/,}" | tee -a "${log_path}"
        sleep 60
        continue
    fi
    # Give pre-existing GPU0 waiters one full polling interval to claim the card.
    echo "[${timestamp}] GPU0 appears free; rechecking after 45 s grace period" | tee -a "${log_path}"
    sleep 45
    if ! nvidia-smi -i 0 --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
        continue
    fi
    gpu_pids="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
    if [[ -z "${gpu_pids}" ]]; then
        break
    fi
done

echo "[$(date --iso-8601=seconds)] starting BACE/general Q12/T30 fixed-p g-by-W grid on physical GPU0" | tee -a "${log_path}"
cd "${scan_dir}"
export CUDA_VISIBLE_DEVICES=0

"${python_bin}" "${runner}" features --mode full --device cuda:0 2>&1 | tee -a "${log_path}"
"${python_bin}" "${runner}" evaluate --mode full 2>&1 | tee -a "${log_path}"
"${python_bin}" "${runner}" summarize --strict 2>&1 | tee -a "${log_path}"
"${python_bin}" "${runner}" plot 2>&1 | tee -a "${log_path}"
"${python_bin}" "${runner}" audit --strict 2>&1 | tee -a "${log_path}"

echo "[$(date --iso-8601=seconds)] complete" | tee -a "${log_path}"
