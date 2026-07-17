#!/bin/bash
# Wrapper for the FAST intraday XAUUSD trend bot (magic 360543, M30 cadence).
# Mirror of xau_dual_cron.sh. The executor's spread_guard ABORTS unless the
# live gold spread <= max_spread_usd (config, $0.24) — so this is SAFE to run
# even if pointed at the wide-spread cent account: it will refuse and exit.
#
# Deploy ONLY on a raw/ECN gold account (spread <= $0.24). Set
# bots.fast.symbol_override in configs/v5_xau_fast.json to the raw symbol
# (e.g. XAUUSDb) if resolve_symbol does not pick it automatically.
set -u
cd /home/rock/Desktop/2026_Projects/Trader36/MT5 || exit 1
export DISPLAY=:0
LOG=data/v5_runs/v5-xau-fast-cron.log
CONDA=/home/rock/anaconda3/bin/conda

echo "=== $(date -u '+%F %T UTC') fast cron wake ===" >> "$LOG"
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
  echo "$(date -u '+%F %T UTC') bridge not ready after 2 min — skipping" >> "$LOG"
  exit 1
fi

# --force-min-lot: this local demo is small (~$1k), so 0.5% risk rounds below
# the 0.01 min lot and would never open. Forcing the min lot lets the PAPER
# demo actually trade (real risk % is higher, irrelevant on a demo). REMOVE
# this flag on a properly-sized live raw account so sizing honours risk_frac.
$CONDA run -n envmt5 python scripts/v5_xau_fast.py \
  --bot fast --live --execute --force-min-lot --save-data >> "$LOG" 2>&1
echo "$(date -u '+%F %T UTC') fast pass done" >> "$LOG"
