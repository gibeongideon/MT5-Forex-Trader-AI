#!/bin/bash
# FundingPips Flex 10K — hourly RECONCILE (XAU micro + ETH + DJI30, magic 360562).
# Instance 1: prefix ~/.mt5, Xvfb :99, bridge 18812. Guard-first executor.
#
# READINESS GATE (2026-07-20): after a reboot/terminal restart MT5 needs ~1-2 min
# before symbols resolve. Without this the executor sees symbol_info=None, prints
# "in sync - nothing to do" and SILENTLY DOES NOT TRADE while looking healthy in
# the log. So wait until the bridge answers AND the book's symbols really resolve.
set -u
cd /home/trader/MT5 || exit 1
export MT5_BRIDGE_PORT=18812
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/fp10k-cron.log
echo "=== $(date -u '+%F %T UTC') fp10k reconcile ===" >> "$LOG"

ready=0
for i in $(seq 1 24); do
  if $PY -c "
import sys
from mt5linux import MetaTrader5
m = MetaTrader5(host='localhost', port=18812)
if not m.initialize() or m.account_info() is None:
    sys.exit(1)
for s in ('XAUUSDmicro','ETHUSD','DJI30'):
    m.symbol_select(s, True)
    i = m.symbol_info(s); t = m.symbol_info_tick(s)
    if i is None or t is None or not getattr(t,'bid',0):
        sys.exit(1)
m.shutdown()
" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "$(date -u '+%F %T UTC') NOT READY (bridge/symbols) — skipping pass" >> "$LOG"
  exit 1
fi

$PY scripts/v5_basket_challenge_exec.py \
  --config configs/v5_fp_flex_10k.json \
  --state  data/v5_runs/fp10k_state.json \
  --paper-csv data/v5_runs/fp10k_live_log.csv \
  --live --execute >> "$LOG" 2>&1
