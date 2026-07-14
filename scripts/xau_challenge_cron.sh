#!/bin/bash
# Wrapper for the FundingPips challenge bot. Mirrors xau_dual_cron.sh:
# terminal+bridge up, bridge answering, then ONE guard-first pass.
# NOTE: point start_mt5.sh / .env at the FUNDINGPIPS terminal profile before
# enabling the timer (never the cent-account terminal).
set -u
cd /home/rock/Desktop/2026_Projects/Trader36/MT5 || exit 1
export DISPLAY=:0
LOG=data/v5_runs/v5-xau-challenge-cron.log
CONDA=/home/rock/anaconda3/bin/conda

echo "=== $(date -u '+%F %T UTC') challenge wake ===" >> "$LOG"
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

$CONDA run -n envmt5 python scripts/v5_xau_challenge.py \
  --live --execute --save-data >> "$LOG" 2>&1
echo "$(date -u '+%F %T UTC') challenge pass done" >> "$LOG"
