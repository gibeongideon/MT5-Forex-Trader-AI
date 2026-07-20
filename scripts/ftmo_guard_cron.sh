#!/bin/bash
# FTMO real-time PROTECTOR — evaluate guards + flatten on breach, no reconcile.
set -u
cd /home/rock/Desktop/2026_Projects/Trader36/MT5 || exit 1
export DISPLAY=:0
CONDA=/home/rock/anaconda3/bin/conda
$CONDA run -n envmt5 python scripts/v5_basket_challenge_exec.py \
  --config configs/v5_ftmo_challenge.json \
  --state data/v5_runs/ftmo_challenge_state.json \
  --paper-csv data/v5_runs/ftmo_challenge_guard_log.csv \
  --guard-only --live --execute >> data/v5_runs/ftmo-guard-cron.log 2>&1
