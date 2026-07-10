#!/bin/bash
# Robust wrapper for the XAUUSD LIVE bot (real cent account).
# Mirror of xau_cron.sh, but runs the live executor with the --live + --execute
# double lock. Ensures the MT5 terminal + rpyc bridge are up and ANSWERING
# before the reconcile pass, so a cold start never fires against a dead bridge.
#
# Deliberately does NOT pass --force-min-lot: on a real account, if the engine
# cannot size ~1% at the broker minimum lot, the pass SKIPS the trade rather
# than forcing an over-risk fill.
#
# Idempotent: start_mt5.sh skips an already-running terminal/bridge.
set -u
cd /home/rock/Desktop/2026_Projects/Trader36/MT5 || exit 1
export DISPLAY=:0
LOG=data/v5_runs/v5-xau-live-cron.log
CONDA=/home/rock/anaconda3/bin/conda

echo "=== $(date -u '+%F %T UTC') live cron wake ===" >> "$LOG"
./start_mt5.sh >> "$LOG" 2>&1

# Poll the bridge until MT5 answers (account_info returns), up to ~2 min.
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
  echo "$(date -u '+%F %T UTC') bridge not ready after 2 min — skipping this pass" >> "$LOG"
  exit 1
fi

$CONDA run -n envmt5 python scripts/v5_xau_live.py \
  --live --execute --save-data >> "$LOG" 2>&1
echo "$(date -u '+%F %T UTC') live pass done" >> "$LOG"
