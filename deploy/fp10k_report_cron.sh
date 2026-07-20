#!/bin/bash
set -u
cd /home/trader/MT5 || exit 1
export MT5_BRIDGE_PORT=18812
PY=/home/trader/miniconda3/envs/envmt5/bin/python
$PY scripts/challenge_daily_report.py \
  --config configs/v5_fp_flex_10k.json \
  --state  data/v5_runs/fp10k_state.json \
  --port 18812 >> data/v5_runs/fp10k-report.log 2>&1
