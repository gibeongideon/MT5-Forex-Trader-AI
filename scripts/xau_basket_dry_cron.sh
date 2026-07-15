#!/bin/bash
# Basket challenge DRY-RUN pass (plan-only, no orders). Connects to the
# PERSISTENT mt5-terminal.service bridge (does NOT launch/kill MT5). Safe:
# executor never sends orders while fp_symbol mappings are null / no --execute.
set -u
cd /home/rock/Desktop/2026_Projects/Trader36/MT5 || exit 1
LOG=data/v5_runs/v5-basket-dry-cron.log
CONDA=/home/rock/anaconda3/bin/conda
echo "=== $(date -u '+%F %T UTC') basket dry wake ===" >> "$LOG"
$CONDA run -n envmt5 python scripts/v5_basket_challenge_exec.py \
  --state data/v5_runs/basket_challenge_dry_state.json \
  --paper-csv data/v5_runs/basket_challenge_dry_log.csv >> "$LOG" 2>&1
echo "$(date -u '+%F %T UTC') basket dry pass done" >> "$LOG"
