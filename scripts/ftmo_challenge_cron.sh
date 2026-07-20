#!/bin/bash
# FTMO 2-Step challenge — hourly RECONCILE pass (focused XAU+BTC+NDX book).
# Guard-first executor; --live --execute (safe: on a DEMO/investor acct nothing
# sends; the spread/rule guards protect a funded acct). Ensures the local MT5
# terminal + bridge are up first.
set -u
cd /home/rock/Desktop/2026_Projects/Trader36/MT5 || exit 1
export DISPLAY=:0
CONDA=/home/rock/anaconda3/bin/conda
LOG=data/v5_runs/ftmo-challenge-cron.log
echo "=== $(date -u '+%F %T UTC') ftmo reconcile wake ===" >> "$LOG"
./start_mt5.sh >> "$LOG" 2>&1
ready=0
for i in $(seq 1 24); do
  if $CONDA run -n envmt5 python -c "
from src.core.mt5_connector import MT5Connector
c=MT5Connector(); c.connect(); ok=c.account_info() is not None; c.disconnect()
import sys; sys.exit(0 if ok else 1)" >> "$LOG" 2>&1; then ready=1; break; fi
  sleep 5
done
[ "$ready" -ne 1 ] && { echo "$(date -u '+%F %T UTC') bridge not ready — skip" >> "$LOG"; exit 1; }
$CONDA run -n envmt5 python scripts/v5_basket_challenge_exec.py \
  --config configs/v5_ftmo_challenge.json \
  --state data/v5_runs/ftmo_challenge_state.json \
  --paper-csv data/v5_runs/ftmo_challenge_live_log.csv \
  --live --execute >> "$LOG" 2>&1
echo "$(date -u '+%F %T UTC') ftmo reconcile done" >> "$LOG"
