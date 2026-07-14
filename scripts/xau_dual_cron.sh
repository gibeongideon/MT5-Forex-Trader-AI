#!/bin/bash
# Robust wrapper for the TWO H4 XAUUSD bots (2026-07-14 research promotion):
#   ls    = long/short trend+breakout ensemble (magic 360541)
#   champ = LONG-ONLY concentrated blend champion (magic 360542)
#
# Mirror of xau_live_cron.sh (which it replaces): ensures the MT5 terminal +
# rpyc bridge are up and ANSWERING, then runs one reconcile pass per bot,
# sequentially, with the --live + --execute double lock. No --force-min-lot:
# if ~1% risk cannot be sized at the broker minimum lot, the pass SKIPS.
#
# Idempotent: start_mt5.sh skips an already-running terminal/bridge.
set -u
cd /home/rock/Desktop/2026_Projects/Trader36/MT5 || exit 1
export DISPLAY=:0
LOG=data/v5_runs/v5-xau-dual-cron.log
CONDA=/home/rock/anaconda3/bin/conda

echo "=== $(date -u '+%F %T UTC') dual cron wake ===" >> "$LOG"
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

# --save-data on the first pass only (one H4 CSV refresh per wake).
$CONDA run -n envmt5 python scripts/v5_xau_dual.py \
  --bot ls --live --execute --save-data >> "$LOG" 2>&1
echo "$(date -u '+%F %T UTC') ls pass done" >> "$LOG"

$CONDA run -n envmt5 python scripts/v5_xau_dual.py \
  --bot champ --live --execute >> "$LOG" 2>&1
echo "$(date -u '+%F %T UTC') champ pass done" >> "$LOG"
