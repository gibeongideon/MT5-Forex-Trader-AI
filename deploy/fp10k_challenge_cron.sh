#!/bin/bash
# FundingPips Flex 10K — hourly RECONCILE (XAU micro + ETH + DJI30, magic 360562).
# Instance 1: prefix ~/.mt5, Xvfb :99, bridge 18812. Guard-first executor.
set -u
cd /home/trader/MT5 || exit 1
export MT5_BRIDGE_PORT=18812
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/fp10k-cron.log
echo "=== $(date -u '+%F %T UTC') fp10k reconcile ===" >> "$LOG"
$PY scripts/v5_basket_challenge_exec.py \
  --config configs/v5_fp_flex_10k.json \
  --state  data/v5_runs/fp10k_state.json \
  --paper-csv data/v5_runs/fp10k_live_log.csv \
  --live --execute >> "$LOG" 2>&1
