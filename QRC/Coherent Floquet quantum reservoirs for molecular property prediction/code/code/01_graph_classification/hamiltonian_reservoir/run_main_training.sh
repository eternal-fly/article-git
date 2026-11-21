#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH=.

/home/ubuntu/miniconda3/envs/qisten/bin/python -u evaluate_main_qrc.py \
  --decoders linear --linear-device cuda:0

/home/ubuntu/miniconda3/envs/qisten/bin/python -u evaluate_main_qrc.py \
  --decoders rbf_svr --linear-device cpu

/home/ubuntu/miniconda3/envs/qisten/bin/python -u assemble_main_results.py
/home/ubuntu/miniconda3/envs/qisten/bin/python -m pytest -q tests
