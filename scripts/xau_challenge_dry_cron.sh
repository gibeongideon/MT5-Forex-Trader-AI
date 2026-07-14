#!/bin/bash
# DRY-RUN wrapper for the FundingPips challenge bot on the CURRENT terminal.
# NO ORDERS ARE EVER SENT: --execute is deliberately absent, so the executor
# only plans (send=False) and appends one row per pass to the dry-run CSV.
# Own state file — the real challenge_state.json stays untouched.
set -u
cd /home/rock/Desktop/2026_Projects/Trader36/MT5 || exit 1
export DISPLAY=:0
LOG=data/v5_runs/v5-xau-challenge-dry-cron.log
CONDA=/home/rock/anaconda3/bin/conda

echo "=== $(date -u '+%F %T UTC') challenge DRY wake ===" >> "$LOG"
./start_mt5.sh >> "$LOG" 2>&1

ready=0
for i in $(seq 1 24); do
  if $CONDA run -n envmt5 python -c "
from src.core.mt5_connector import MT5Connector
c=MT5Connector(); c.connect()
ok = c.account_info() is not None
c.disconnect()
import sys; sys.exit(0 if ok else 1)
" >> "$LOG" 2>&1; then ready=1; break; fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "$(date -u '+%F %T UTC') bridge not ready — skipping" >> "$LOG"
  exit 1
fi

# --live acknowledges the real (cent) account for PLANNING only; no --execute.
$CONDA run -n envmt5 python scripts/v5_xau_challenge.py \
  --live --save-data \
  --state data/v5_runs/challenge_dryrun_state.json \
  --paper-csv data/v5_runs/challenge_dryrun_log.csv >> "$LOG" 2>&1
echo "$(date -u '+%F %T UTC') challenge DRY pass done" >> "$LOG"
