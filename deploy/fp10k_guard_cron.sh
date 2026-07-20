#!/bin/bash
# FundingPips Flex 10K — real-time PROTECTOR (guards + flatten on breach, no reconcile).
set -u
cd /home/trader/MT5 || exit 1
export MT5_BRIDGE_PORT=18812
PY=/home/trader/miniconda3/envs/envmt5/bin/python
$PY scripts/v5_basket_challenge_exec.py \
  --config configs/v5_fp_flex_10k.json \
  --state  data/v5_runs/fp10k_state.json \
  --paper-csv data/v5_runs/fp10k_guard_log.csv \
  --guard-only --live --execute >> data/v5_runs/fp10k-guard.log 2>&1
